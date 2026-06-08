#!/usr/bin/env python3
"""Build a stricter clean 2D-pose target split for YOLO26-PS Stage C.

The split is intentionally derived from the existing targetclean manifests so it
can be compared against the previous probe without changing the original data.
It keeps easy-to-audit, GT-only filters: no prediction-error filtering.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML


DATA_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI")
DEFAULT_TEMPLATE_YAML = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d_targetclean.yaml"
DEFAULT_OUT_YAML = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d_targetstrict.yaml"
SPLITS = ("train", "val")
COCO_BODY_REQUIRED = (5, 6, 11, 12)
AGORA_BODY_REQUIRED = (5, 6, 11, 12)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--template-yaml", type=Path, default=DEFAULT_TEMPLATE_YAML)
    parser.add_argument("--out-yaml", type=Path, default=DEFAULT_OUT_YAML)
    parser.add_argument("--src-dir", default="manifests/derived_clean")
    parser.add_argument("--dst-dir", default="manifests/derived_strict")
    parser.add_argument("--src-name", default="targetclean")
    parser.add_argument("--dst-name", default="targetstrict")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--coco-max-instances", type=int, default=2)
    parser.add_argument("--coco-min-visible", type=int, default=16)
    parser.add_argument("--coco-min-height", type=float, default=120.0)
    parser.add_argument("--coco-min-width", type=float, default=45.0)
    parser.add_argument("--coco-min-aspect", type=float, default=1.35)
    parser.add_argument("--coco-max-aspect", type=float, default=3.8)
    parser.add_argument("--coco-min-yspan", type=float, default=0.72)
    parser.add_argument("--coco-max-yspan", type=float, default=1.08)
    parser.add_argument("--coco-min-torso", type=float, default=0.18)
    parser.add_argument("--coco-max-overlap", type=float, default=0.10)
    parser.add_argument("--agora-max-instances", type=int, default=3)
    parser.add_argument("--agora-min-visible", type=int, default=13)
    parser.add_argument("--agora-min-height", type=float, default=150.0)
    parser.add_argument("--agora-min-width", type=float, default=45.0)
    parser.add_argument("--agora-min-aspect", type=float, default=1.45)
    parser.add_argument("--agora-max-aspect", type=float, default=3.7)
    parser.add_argument("--agora-min-yspan", type=float, default=0.80)
    parser.add_argument("--agora-max-yspan", type=float, default=0.95)
    parser.add_argument("--agora-min-torso", type=float, default=0.22)
    parser.add_argument("--agora-max-overlap", type=float, default=0.08)
    parser.add_argument("--keep-3dpw", action="store_true", help="keep a strict 3DPW subset instead of excluding 3DPW")
    return parser.parse_args()


def canonical_source(value: Any) -> str:
    """Normalize source names."""
    aliases = {
        "coco": "coco_wholebody",
        "coco_whole_body": "coco_wholebody",
        "coco_wholebody": "coco_wholebody",
        "agora": "agora",
        "3dpw": "3dpw",
        "3d_pw": "3dpw",
    }
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return aliases.get(text, text)


def visible_count(inst: dict[str, Any]) -> int:
    """Count visible 2D keypoints."""
    return sum(1 for point in inst.get("body_kpts_2d") or [] if len(point) >= 3 and float(point[2]) > 0)


def joint_visible(inst: dict[str, Any], index: int) -> bool:
    """Return whether a joint is present and visible."""
    kpts = inst.get("body_kpts_2d") or []
    return index < len(kpts) and len(kpts[index]) >= 3 and float(kpts[index][2]) > 0


def all_joints_visible(inst: dict[str, Any], indices: tuple[int, ...]) -> bool:
    """Return whether all requested joints are visible."""
    return all(joint_visible(inst, i) for i in indices)


def bbox_xyxy(inst: dict[str, Any]) -> list[float] | None:
    """Return an xyxy box if available."""
    box = inst.get("bbox") or []
    if len(box) < 4:
        return None
    return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]


def box_area(box: list[float]) -> float:
    """Return xyxy box area."""
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def box_iou(a: list[float], b: list[float]) -> float:
    """Return IoU for two xyxy boxes."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (box_area(a) + box_area(b) - inter + 1e-9)


def yspan_ratio(inst: dict[str, Any], height: float) -> float:
    """Return visible-keypoint vertical span normalized by box height."""
    ys = [float(p[1]) for p in inst.get("body_kpts_2d") or [] if len(p) >= 3 and float(p[2]) > 0]
    return (max(ys) - min(ys)) / max(height, 1.0) if ys else 0.0


def torso_ratio(inst: dict[str, Any], height: float) -> float:
    """Return shoulder-to-hip vertical separation normalized by box height."""
    if not all_joints_visible(inst, (5, 6, 11, 12)):
        return -1.0
    kpts = inst.get("body_kpts_2d") or []
    shoulder_y = (float(kpts[5][1]) + float(kpts[6][1])) * 0.5
    hip_y = (float(kpts[11][1]) + float(kpts[12][1])) * 0.5
    return (hip_y - shoulder_y) / max(height, 1.0)


def in_frame(box: list[float], width: float, height: float, margin: float) -> bool:
    """Return whether a box is away from the image boundary."""
    return box[0] >= margin and box[1] >= margin and box[2] <= width - margin and box[3] <= height - margin


def base_pose_instances(record: dict[str, Any]) -> list[tuple[int, dict[str, Any], list[float]]]:
    """Return pose-capable person instances with boxes."""
    out = []
    for idx, inst in enumerate(record.get("instances") or []):
        flags = inst.get("flags") or {}
        box = bbox_xyxy(inst)
        if inst.get("category") != "person" or box is None:
            continue
        if not flags.get("has_body2d", visible_count(inst) > 0):
            continue
        out.append((idx, inst, box))
    return out


def reject(reason_counts: Counter[str], reason: str) -> bool:
    """Record a reject reason and return False for concise filter code."""
    reason_counts[reason] += 1
    return False


def passes_coco(
    inst: dict[str, Any],
    box: list[float],
    max_overlap: float,
    record: dict[str, Any],
    args: argparse.Namespace,
    reason_counts: Counter[str],
) -> bool:
    """Return whether a COCO instance should remain in the strict split."""
    width = float(record.get("width") or 0)
    height_img = float(record.get("height") or 0)
    w = max(box[2] - box[0], 1.0)
    h = max(box[3] - box[1], 1.0)
    aspect = h / w
    yspan = yspan_ratio(inst, h)
    torso = torso_ratio(inst, h)
    if visible_count(inst) < args.coco_min_visible:
        return reject(reason_counts, "coco_visible")
    if not all_joints_visible(inst, COCO_BODY_REQUIRED):
        return reject(reason_counts, "coco_required_joints")
    if not in_frame(box, width, height_img, args.margin):
        return reject(reason_counts, "coco_boundary")
    if h < args.coco_min_height or w < args.coco_min_width:
        return reject(reason_counts, "coco_size")
    if not (args.coco_min_aspect <= aspect <= args.coco_max_aspect):
        return reject(reason_counts, "coco_aspect")
    if not (args.coco_min_yspan <= yspan <= args.coco_max_yspan):
        return reject(reason_counts, "coco_yspan")
    if torso < args.coco_min_torso:
        return reject(reason_counts, "coco_torso")
    if max_overlap > args.coco_max_overlap:
        return reject(reason_counts, "coco_overlap")
    return True


def passes_agora(
    inst: dict[str, Any],
    box: list[float],
    max_overlap: float,
    record: dict[str, Any],
    args: argparse.Namespace,
    reason_counts: Counter[str],
) -> bool:
    """Return whether an AGORA instance should remain in the strict split."""
    width = float(record.get("width") or 0)
    height_img = float(record.get("height") or 0)
    w = max(box[2] - box[0], 1.0)
    h = max(box[3] - box[1], 1.0)
    aspect = h / w
    yspan = yspan_ratio(inst, h)
    torso = torso_ratio(inst, h)
    if visible_count(inst) < args.agora_min_visible:
        return reject(reason_counts, "agora_visible")
    if not all_joints_visible(inst, AGORA_BODY_REQUIRED):
        return reject(reason_counts, "agora_required_joints")
    if not in_frame(box, width, height_img, args.margin):
        return reject(reason_counts, "agora_boundary")
    if h < args.agora_min_height or w < args.agora_min_width:
        return reject(reason_counts, "agora_size")
    if not (args.agora_min_aspect <= aspect <= args.agora_max_aspect):
        return reject(reason_counts, "agora_aspect")
    if not (args.agora_min_yspan <= yspan <= args.agora_max_yspan):
        return reject(reason_counts, "agora_yspan")
    if torso < args.agora_min_torso:
        return reject(reason_counts, "agora_torso")
    if max_overlap > args.agora_max_overlap:
        return reject(reason_counts, "agora_overlap")
    return True


def filter_record(
    record: dict[str, Any], args: argparse.Namespace, reason_counts: Counter[str]
) -> tuple[dict[str, Any] | None, str]:
    """Filter a manifest record and return the filtered record plus source."""
    source = canonical_source(record.get("source"))
    if source == "3dpw" and not args.keep_3dpw:
        reason_counts["3dpw_excluded"] += len(record.get("instances") or [])
        return None, source
    if source not in {"coco_wholebody", "agora"}:
        reason_counts[f"{source or 'unknown'}_source"] += 1
        return None, source

    pose_instances = base_pose_instances(record)
    max_instances = args.coco_max_instances if source == "coco_wholebody" else args.agora_max_instances
    if len(pose_instances) > max_instances:
        reason_counts[f"{source}_crowd_record"] += len(pose_instances)
        return None, source

    boxes = [box for _, _, box in pose_instances]
    kept_instances = []
    for idx, inst, box in pose_instances:
        max_overlap = max([box_iou(box, other) for other in boxes if other is not box] or [0.0])
        if source == "coco_wholebody":
            keep = passes_coco(inst, box, max_overlap, record, args, reason_counts)
        else:
            keep = passes_agora(inst, box, max_overlap, record, args, reason_counts)
        if keep:
            kept_instances.append(deepcopy(inst))

    if not kept_instances:
        reason_counts[f"{source}_empty_record"] += 1
        return None, source

    out = deepcopy(record)
    out["source"] = source
    out["instances"] = kept_instances
    out["task_flags"] = dict(out.get("task_flags") or {})
    out["task_flags"]["has_det"] = any((inst.get("flags") or {}).get("has_bbox", bool(inst.get("bbox"))) for inst in kept_instances)
    out["task_flags"]["has_pose2d"] = any((inst.get("flags") or {}).get("has_body2d", False) for inst in kept_instances)
    out["task_flags"]["has_pose3d"] = any((inst.get("flags") or {}).get("has_body3d", False) for inst in kept_instances)
    out["task_flags"]["has_person_mask"] = any((inst.get("flags") or {}).get("has_person_mask", False) for inst in kept_instances)
    return out, source


def process_split(split: str, args: argparse.Namespace) -> dict[str, Any]:
    """Create one filtered manifest split and image list."""
    src_manifest = args.root / args.src_dir / f"stage_c_{split}_{args.src_name}.jsonl"
    dst_manifest = args.root / args.dst_dir / f"stage_c_{split}_{args.dst_name}.jsonl"
    dst_list = args.root / f"stage_c_{split}_{args.dst_name}.txt"
    if not src_manifest.exists():
        raise FileNotFoundError(src_manifest)
    dst_manifest.parent.mkdir(parents=True, exist_ok=True)

    records_by_source: Counter[str] = Counter()
    instances_by_source: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    total_records = 0
    total_instances = 0
    out_records = []

    with src_manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_records += 1
            record = json.loads(line)
            total_instances += len(record.get("instances") or [])
            filtered, source = filter_record(record, args, reason_counts)
            if filtered is None:
                continue
            records_by_source[source] += 1
            instances_by_source[source] += len(filtered.get("instances") or [])
            out_records.append(filtered)

    with dst_manifest.open("w", encoding="utf-8") as f:
        for record in out_records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with dst_list.open("w", encoding="utf-8") as f:
        for record in out_records:
            f.write(str(record["image"]) + "\n")

    return {
        "split": split,
        "source_manifest": str(src_manifest),
        "manifest": str(dst_manifest),
        "list": str(dst_list),
        "input_records": total_records,
        "input_instances": total_instances,
        "records": len(out_records),
        "instances": sum(instances_by_source.values()),
        "records_by_source": dict(sorted(records_by_source.items())),
        "instances_by_source": dict(sorted(instances_by_source.items())),
        "reject_reasons": dict(sorted(reason_counts.items())),
    }


def write_data_yaml(args: argparse.Namespace) -> None:
    """Write the dataset YAML for the generated split."""
    data = YAML.load(args.template_yaml)
    data["path"] = str(args.root)
    data["train"] = f"stage_c_train_{args.dst_name}.txt"
    data["val"] = f"stage_c_val_{args.dst_name}.txt"
    data["unified_manifest"] = {
        "train": f"{args.dst_dir}/stage_c_train_{args.dst_name}.jsonl",
        "val": f"{args.dst_dir}/stage_c_val_{args.dst_name}.jsonl",
    }
    YAML.save(args.out_yaml, data)


def main() -> None:
    """Build all splits and write a summary."""
    args = parse_args()
    summaries = [process_split(split, args) for split in SPLITS]
    write_data_yaml(args)
    summary = {
        "dst_name": args.dst_name,
        "data_yaml": str(args.out_yaml),
        "rules": {k: v for k, v in vars(args).items() if k not in {"root", "template_yaml", "out_yaml"}},
        "splits": summaries,
    }
    summary_path = args.root / args.dst_dir / f"stage_c_{args.dst_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
