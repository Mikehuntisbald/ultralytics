#!/usr/bin/env python3
"""Build a quality-filtered Stage B COCO pose split.

The original Stage B COCO-WholeBody split contains many person instances with
zero or very few visible COCO-17 keypoints. This script creates a pose-focused
view that keeps only instances with enough visible 2D body keypoints, while
leaving the original prepared dataset untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML


DATA_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI")
BASE_YAML = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d.yaml"
OUT_YAML = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d_poseq_min8.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--base-yaml", type=Path, default=BASE_YAML)
    parser.add_argument("--out-yaml", type=Path, default=OUT_YAML)
    parser.add_argument("--min-visible", type=int, default=8)
    parser.add_argument("--suffix", default="poseq_min8")
    return parser.parse_args()


def visible_count(inst: dict[str, Any]) -> int:
    """Return the number of visible 2D body keypoints on an instance."""
    return sum(1 for k in inst.get("body_kpts_2d") or [] if len(k) >= 3 and float(k[2]) > 0)


def filter_record(record: dict[str, Any], min_visible: int) -> dict[str, Any] | None:
    """Keep only pose-quality instances from a unified record."""
    kept = []
    for inst in record.get("instances", []):
        flags = inst.get("flags") or {}
        if not flags.get("has_body2d") or visible_count(inst) < min_visible:
            continue
        item = dict(inst)
        item["flags"] = dict(flags)
        item["flags"]["has_bbox"] = True
        item["flags"]["has_body2d"] = True
        item["flags"]["has_body3d"] = False
        item["flags"]["has_person_mask"] = False
        kept.append(item)

    if not kept:
        return None

    out = dict(record)
    out["instances"] = kept
    task_flags = dict(record.get("task_flags") or {})
    task_flags["has_det"] = True
    task_flags["has_pose2d"] = True
    task_flags["has_pose3d"] = False
    task_flags["has_person_mask"] = False
    task_flags["has_scene_seg"] = False
    out["task_flags"] = task_flags
    out["pose_quality_filter"] = {"min_visible": int(min_visible)}
    return out


def build_split(data_root: Path, split: str, suffix: str, min_visible: int) -> tuple[int, int, int]:
    """Write filtered manifest and image list for one split."""
    in_manifest = data_root / "manifests" / f"stage_b_{split}.jsonl"
    out_manifest = data_root / "manifests" / f"stage_b_{suffix}_{split}.jsonl"
    out_txt = data_root / f"stage_b_{suffix}_{split}.txt"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    kept_records = 0
    kept_instances = 0
    with in_manifest.open(encoding="utf-8") as src, out_manifest.open("w", encoding="utf-8") as mf, out_txt.open(
        "w", encoding="utf-8"
    ) as txt:
        for line in src:
            if not line.strip():
                continue
            total_records += 1
            record = json.loads(line)
            filtered = filter_record(record, min_visible)
            if filtered is None:
                continue
            kept_records += 1
            kept_instances += len(filtered["instances"])
            txt.write(str(Path(filtered["image"]).resolve()) + "\n")
            mf.write(json.dumps(filtered, separators=(",", ":")) + "\n")
    return total_records, kept_records, kept_instances


def write_yaml(args: argparse.Namespace) -> None:
    """Write a dataset YAML pointing at the filtered split."""
    data = YAML.load(args.base_yaml)
    data["train"] = f"stage_b_{args.suffix}_train.txt"
    data["val"] = f"stage_b_{args.suffix}_val.txt"
    data.pop("unified_labels", None)
    data["unified_manifest"] = {
        "train": f"manifests/stage_b_{args.suffix}_train.jsonl",
        "val": f"manifests/stage_b_{args.suffix}_val.jsonl",
    }
    YAML.save(args.out_yaml, data)


def main() -> None:
    """Build the filtered split and dataset YAML."""
    args = parse_args()
    args.data_root = args.data_root.resolve()
    stats = {}
    for split in ("train", "val"):
        stats[split] = build_split(args.data_root, split, args.suffix, args.min_visible)
    write_yaml(args)

    print(f"Wrote dataset YAML: {args.out_yaml}")
    for split, (total_records, kept_records, kept_instances) in stats.items():
        print(
            f"{split}: kept {kept_records}/{total_records} images "
            f"with {kept_instances} instances at min_visible={args.min_visible}"
        )


if __name__ == "__main__":
    main()
