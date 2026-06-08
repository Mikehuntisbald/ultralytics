#!/usr/bin/env python3
"""Merge e9 pose weights with a strong Stage A detector and seed a human-centric det head."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import torch_safe_load


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/26/yolo26s-ps25d.yaml"
DEFAULT_E9 = (
    ROOT
    / "runs/detect/yolo26ps_c_parallelposeneck_poseonly_from_e9_assignonly_b48_acc1_d0_2p862e4/weights/last.pt"
)
DEFAULT_STRONG_DET = (
    ROOT / "official_run/stage_a_detection_det_recover_best_reinitopt_fillvram/epoch5_before_objects365_only/best.pt"
)
DEFAULT_OUT = ROOT / "runs/detect/yolo26ps_e9_strongdet_humanhead_init/weights/last.pt"
DET_HEAD_PREFIXES = ("cv2", "cv3", "one2one_cv2", "one2one_cv3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--e9", type=Path, default=DEFAULT_E9)
    parser.add_argument("--strong-det", type=Path, default=DEFAULT_STRONG_DET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--save-best-copy", action="store_true", default=True)
    return parser.parse_args()


def load_state(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    ckpt, _ = torch_safe_load(path)
    model = (ckpt.get("ema") or ckpt["model"]).float()
    return ckpt, model.state_dict()


def final_layer(state: dict[str, torch.Tensor]) -> int:
    layers = []
    for key in state:
        match = re.match(r"model\.(\d+)\.", key)
        if match:
            layers.append(int(match.group(1)))
    if not layers:
        raise RuntimeError("No numeric model layer prefixes found in checkpoint state_dict")
    return max(layers)


def copy_matching(target: dict[str, torch.Tensor], key: str, value: torch.Tensor) -> bool:
    dst = target.get(key)
    if dst is None or dst.shape != value.shape:
        return False
    target[key] = value.to(device=dst.device, dtype=dst.dtype)
    return True


def merge_weights(target_state: dict[str, torch.Tensor], e9_state: dict[str, torch.Tensor], det_state: dict[str, torch.Tensor]):
    copied_e9 = copied_det_group = copied_det_head = skipped = 0

    for key, value in e9_state.items():
        if copy_matching(target_state, key, value):
            copied_e9 += 1

    strong_final = final_layer(det_state)
    target_final = final_layer(target_state)
    target_head = f"model.{target_final}."
    strong_head = f"model.{strong_final}."

    for key, value in det_state.items():
        match = re.match(r"model\.(\d+)\.", key)
        if not match:
            continue
        layer = int(match.group(1))
        if 0 <= layer <= 28:
            if copy_matching(target_state, key, value):
                copied_det_group += 1
            else:
                skipped += 1
            continue
        if not key.startswith(strong_head):
            continue
        suffix = key[len(strong_head) :]
        if not suffix.startswith(DET_HEAD_PREFIXES):
            continue
        new_key = target_head + suffix
        if copy_matching(target_state, new_key, value):
            copied_det_head += 1
        else:
            skipped += 1

    return {
        "copied_e9": copied_e9,
        "copied_strong_backbone_det_neck": copied_det_group,
        "copied_strong_det_head": copied_det_head,
        "skipped_shape_or_missing": skipped,
        "strong_final_layer": strong_final,
        "target_final_layer": target_final,
    }


def merge(args: argparse.Namespace) -> None:
    e9_ckpt, e9_state = load_state(args.e9)
    _, det_state = load_state(args.strong_det)

    target = YOLO(str(args.model)).model.float()
    target_state = target.state_dict()
    stats = merge_weights(target_state, e9_state, det_state)
    target.load_state_dict(target_state, strict=False)
    head = target.model[-1]
    if not hasattr(head, "copy_human_head_from_det"):
        raise RuntimeError("Target head does not support human det initialization")
    head.copy_human_head_from_det()

    new_ckpt = copy.deepcopy(e9_ckpt)
    new_ckpt["model"] = target.half()
    new_ckpt["ema"] = None
    new_ckpt["optimizer"] = None
    new_ckpt["epoch"] = -1
    new_ckpt["best_fitness"] = None
    new_ckpt["train_args"] = {
        **dict(e9_ckpt.get("train_args", {})),
        "model": str(args.model),
        "merged_from_e9": str(args.e9),
        "merged_strong_det": str(args.strong_det),
        "human_det_head": "copied_from_merged_det_head_person_face_rows",
    }
    new_ckpt["merge_stats"] = stats

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, args.out)
    print(f"saved {args.out}")
    print(stats)
    if args.save_best_copy:
        best = args.out.with_name("best.pt")
        torch.save(new_ckpt, best)
        print(f"saved {best}")


def main() -> None:
    merge(parse_args())


if __name__ == "__main__":
    main()
