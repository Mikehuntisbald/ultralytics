#!/usr/bin/env python3
"""Expand a YOLO26-PS 2.5D pose head while preserving the narrow-head function."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import torch_safe_load


POSE_PREFIXES = ("model.29.cv4", "model.29.one2one_cv4")


def copy_overlap(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Copy the common slice of src into dst and leave the rest unchanged."""
    slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
    dst[slices] = src[slices].to(device=dst.device, dtype=dst.dtype)
    return dst


def zero_extra_output_inputs(state: dict[str, torch.Tensor], old_width: int) -> None:
    """Make newly added hidden channels initially invisible to the final pose projection."""
    for branch in ("cv4", "one2one_cv4"):
        for level in range(4):
            key = f"model.29.{branch}.{level}.2.weight"
            if key in state and state[key].ndim == 4 and state[key].shape[1] > old_width:
                state[key][:, old_width:] = 0


def expand_pose_head(model_yaml: Path, source: Path, out: Path) -> None:
    """Build the target YAML and copy source checkpoint weights into it."""
    ckpt, _ = torch_safe_load(source)
    src_model = (ckpt.get("ema") or ckpt["model"]).float()
    src_state = src_model.state_dict()

    target = YOLO(str(model_yaml)).model.float()
    dst_state = target.state_dict()

    old_width = None
    for key, src in src_state.items():
        dst = dst_state.get(key)
        if dst is None:
            continue
        if dst.shape == src.shape:
            dst_state[key] = src.to(device=dst.device, dtype=dst.dtype)
        elif key.startswith(POSE_PREFIXES):
            if key.endswith(".0.conv.weight") and old_width is None:
                old_width = src.shape[0]
            dst_state[key] = copy_overlap(dst.clone(), src)

    if old_width is None:
        raise RuntimeError(f"No source pose head weights found in {source}")
    zero_extra_output_inputs(dst_state, old_width)
    target.load_state_dict(dst_state, strict=False)

    new_ckpt = copy.deepcopy(ckpt)
    new_ckpt["model"] = target.half()
    new_ckpt["ema"] = None
    new_ckpt["optimizer"] = None
    new_ckpt["train_args"] = {**ckpt.get("train_args", {}), "model": str(model_yaml)}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, out)
    print(f"saved {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expand_pose_head(args.model, args.source, args.out)


if __name__ == "__main__":
    main()
