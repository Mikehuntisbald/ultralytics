#!/usr/bin/env python3
"""Merge Stage C det/pose weights with the Stage D independent segmentation branch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/26/yolo26s-ps25d-yolo-pose-merged-segbranch.yaml"
DEFAULT_C = ROOT / "runs/detect/yolo26ps_c_pose25d_fixedlr_1p46e5_guard_fixedcls/weights/best.pt"


def checkpoint_model(path: Path):
    """Load a checkpoint model on CPU."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float()
    return ckpt, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", type=Path, default=DEFAULT_C, help="Stage C full det/pose checkpoint")
    parser.add_argument("--d", type=Path, required=True, help="Stage D segbranch checkpoint")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Merged model YAML")
    parser.add_argument("--out", type=Path, required=True, help="Output merged checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, c_model = checkpoint_model(args.c)
    d_ckpt, d_model = checkpoint_model(args.d)
    merged = YOLO(str(args.model)).model.float()

    c_sd = c_model.state_dict()
    d_sd = d_model.state_dict()
    m_sd = merged.state_dict()
    transfer: dict[str, torch.Tensor] = {}
    c_copied = d_copied = 0

    merged_final = max(int(k.split(".")[1]) for k in m_sd if k.startswith("model.") and k.split(".")[1].isdigit())
    c_final = max(int(k.split(".")[1]) for k in c_sd if k.startswith("model.") and k.split(".")[1].isdigit())

    for key, value in c_sd.items():
        parts = key.split(".", 2)
        new_key = key
        if len(parts) >= 3 and parts[0] == "model" and parts[1].isdigit() and int(parts[1]) == c_final:
            new_key = f"model.{merged_final}.{parts[2]}"
        if new_key in m_sd and m_sd[new_key].shape == value.shape:
            transfer[new_key] = value
            c_copied += 1

    # Stage D segbranch layers 20..31 become the appended merged seg neck.
    seg_offset = merged_final - 12 - 20
    for key, value in d_sd.items():
        parts = key.split(".", 2)
        if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
            continue
        layer = int(parts[1])
        if 20 <= layer <= 31:
            new_key = f"model.{layer + seg_offset}.{parts[2]}"
        elif layer == 32:
            new_key = f"model.{merged_final}.seg_head.{parts[2]}"
        else:
            continue
        if new_key in m_sd and m_sd[new_key].shape == value.shape:
            transfer[new_key] = value
            d_copied += 1

    missing = [k for k in m_sd if k not in transfer]
    merged.load_state_dict(transfer, strict=False)
    merged.names = getattr(c_model, "names", getattr(merged, "names", None))
    merged.args = getattr(c_model, "args", {})
    merged.yaml_file = str(args.model)
    merged.eval()

    out_ckpt = dict(d_ckpt)
    out_ckpt["model"] = merged.half()
    out_ckpt["ema"] = None
    out_ckpt["train_args"] = {
        **dict(getattr(c_model, "args", {}) or {}),
        "model": str(args.model),
        "stage_c": str(args.c),
        "stage_d_segbranch": str(args.d),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.out)
    print(
        {
            "out": str(args.out),
            "c_copied": c_copied,
            "d_copied": d_copied,
            "target_keys": len(m_sd),
            "missing": len(missing),
            "missing_sample": missing[:20],
        }
    )


if __name__ == "__main__":
    main()
