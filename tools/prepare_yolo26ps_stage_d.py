#!/usr/bin/env python3
"""Prepare YOLO26s-PS-2.5D Stage D person-mask data.

Stage D trains the person instance-mask branch while keeping pose supervision
alive. This script builds the unified-schema train/val lists expected by
``yolo26ps_stage_d_person_mask.yaml`` from:

- COCO person mask annotations, with polygon masks
- OCHuman COCO-format annotations, with compressed RLE masks
- Stage C pose/2.5D manifests
- A small Stage A detection guard subset for source-aware validation/sampling
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


VISION_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
OUT = VISION_ROOT / "YOLO26PS_STAGE_MULTI"
STAGE_A = VISION_ROOT / "YOLO26PS_STAGE_A"
COCO_MASKS = VISION_ROOT / "COCO_person_masks" / "annotations"
COCO_IMAGES = VISION_ROOT / "COCO_2017"
OCHUMAN = VISION_ROOT / "OCHuman" / "OCHuman"
DET_GUARD_WEIGHTS = {"objects365": 5.0, "crowdhuman": 3.0, "wider_face": 3.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-root", type=Path, default=VISION_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stage-a", type=Path, default=STAGE_A)
    parser.add_argument("--max-coco-mask-train", type=int, default=0, help="optional COCO person-mask train cap")
    parser.add_argument("--max-coco-mask-val", type=int, default=0, help="optional COCO person-mask val cap")
    parser.add_argument("--max-ochuman-train", type=int, default=0, help="optional OCHuman train cap")
    parser.add_argument("--max-ochuman-val", type=int, default=0, help="optional OCHuman val cap")
    parser.add_argument("--max-stage-c-train", type=int, default=0, help="optional Stage C pose train cap")
    parser.add_argument("--max-stage-c-val", type=int, default=0, help="optional Stage C pose val cap")
    parser.add_argument("--det-train-limit", type=int, default=60000, help="Stage A detection guard train cap")
    parser.add_argument("--det-val-limit", type=int, default=5000, help="Stage A detection guard val cap")
    parser.add_argument("--smoke-per-source", type=int, default=16, help="write small balanced smoke lists with N images/source")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


def read_lines(path: Path, limit: int = 0) -> list[str]:
    if not path.exists():
        return []
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
                if limit and len(lines) >= limit:
                    break
    return lines


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_list(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [str(x) for x in lines if str(x).strip()]
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def cap(records: list[dict], limit: int) -> list[dict]:
    return records[:limit] if limit and len(records) > limit else records


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = map(float, box[:4])
    return [x, y, x + w, y + h]


def coco17(keypoints: list[float] | None) -> list[list[float]]:
    values = [float(x) for x in (keypoints or [])[: 17 * 3]]
    values += [0.0] * max(0, 17 * 3 - len(values))
    return [values[i : i + 3] for i in range(0, 17 * 3, 3)]


def polygon_area(poly: list[float]) -> float:
    pts = [float(x) for x in poly]
    if len(pts) < 6:
        return 0.0
    xs = pts[0::2]
    ys = pts[1::2]
    return abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs)))) * 0.5


def largest_polygon(segmentation: object) -> list[list[float]] | None:
    if not isinstance(segmentation, list) or not segmentation:
        return None
    if all(isinstance(x, (int, float)) for x in segmentation):
        segmentation = [segmentation]
    polygons = [poly for poly in segmentation if isinstance(poly, list) and len(poly) >= 6]
    if not polygons:
        return None
    poly = max(polygons, key=polygon_area)
    return [[float(poly[i]), float(poly[i + 1])] for i in range(0, len(poly) - 1, 2)]


def compressed_rle(segmentation: object) -> dict | None:
    if not isinstance(segmentation, dict):
        return None
    counts = segmentation.get("counts")
    size = segmentation.get("size")
    if counts is None or size is None:
        return None
    return {"counts": counts, "size": size}


def make_mask_instance(ann: dict, source: str, include_keypoints: bool) -> dict | None:
    bbox = ann.get("bbox") or []
    if len(bbox) < 4 or bbox[2] <= 1 or bbox[3] <= 1:
        return None
    segmentation = ann.get("segmentation")
    person_mask = largest_polygon(segmentation) or compressed_rle(segmentation)
    if person_mask is None:
        return None
    inst = {
        "category": "person",
        "bbox": xywh_to_xyxy(bbox),
        "person_mask": person_mask,
        "flags": {
            "has_bbox": True,
            "has_body2d": False,
            "has_body3d": False,
            "has_person_mask": True,
        },
    }
    if include_keypoints:
        kpts = coco17(ann.get("keypoints"))
        inst["body_kpts_2d"] = kpts
        inst["flags"]["has_body2d"] = any(k[2] > 0 for k in kpts)
    return inst


def make_record(image: dict, image_path: Path, source: str, instances: list[dict]) -> dict:
    return {
        "image": str(image_path.resolve()),
        "source": source,
        "width": int(image["width"]),
        "height": int(image["height"]),
        "instances": instances,
        "task_flags": {
            "has_det": bool(instances),
            "has_pose2d": any(inst["flags"].get("has_body2d", False) for inst in instances),
            "has_pose3d": False,
            "has_person_mask": any(inst["flags"].get("has_person_mask", False) for inst in instances),
            "has_scene_seg": False,
        },
    }


def bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def copy_instance_supervision(dst: dict, src: dict) -> None:
    flags = dst.setdefault("flags", {})
    src_flags = src.get("flags") or {}
    for key in ("body_kpts_2d", "body_kpts_3d", "person_mask"):
        if key not in dst and key in src:
            dst[key] = src[key]
    flags["has_body2d"] = bool(flags.get("has_body2d") or src_flags.get("has_body2d") or dst.get("body_kpts_2d"))
    flags["has_body3d"] = bool(flags.get("has_body3d") or src_flags.get("has_body3d") or dst.get("body_kpts_3d"))
    flags["has_person_mask"] = bool(
        flags.get("has_person_mask") or src_flags.get("has_person_mask") or dst.get("person_mask") is not None
    )


def merge_instance_lists(base: list[dict], incoming: list[dict]) -> list[dict]:
    merged = list(base)
    for inst in incoming:
        best_i, best_iou = -1, 0.0
        for i, cur in enumerate(merged):
            if cur.get("category") != inst.get("category"):
                continue
            iou = bbox_iou_xyxy(cur.get("bbox") or [], inst.get("bbox") or [])
            if iou > best_iou:
                best_i, best_iou = i, iou
        if best_i >= 0 and best_iou >= 0.45:
            cur = merged[best_i]
            cur_has_mask = bool((cur.get("flags") or {}).get("has_person_mask") or cur.get("person_mask") is not None)
            new_has_mask = bool((inst.get("flags") or {}).get("has_person_mask") or inst.get("person_mask") is not None)
            if new_has_mask and not cur_has_mask:
                copy_instance_supervision(inst, cur)
                merged[best_i] = inst
            else:
                copy_instance_supervision(cur, inst)
        else:
            merged.append(inst)
    return merged


def refresh_task_flags(record: dict) -> None:
    instances = record.get("instances") or []
    record["task_flags"] = {
        "has_det": bool(instances),
        "has_pose2d": any((inst.get("flags") or {}).get("has_body2d", False) for inst in instances),
        "has_pose3d": any((inst.get("flags") or {}).get("has_body3d", False) for inst in instances),
        "has_person_mask": any((inst.get("flags") or {}).get("has_person_mask", False) for inst in instances),
        "has_scene_seg": bool(record.get("scene_seg")),
    }


def merge_records_by_image(records: list[dict]) -> list[dict]:
    priority = {
        "coco_person_mask": 70,
        "ochuman": 65,
        "agora": 60,
        "3dpw": 55,
        "coco_wholebody": 50,
    }
    merged: dict[str, dict] = {}
    order: list[str] = []
    for record in records:
        key = str(Path(record["image"]).resolve())
        if key not in merged:
            merged[key] = record
            order.append(key)
            continue
        cur = merged[key]
        cur["instances"] = merge_instance_lists(cur.get("instances") or [], record.get("instances") or [])
        cur_source = str(cur.get("source", ""))
        new_source = str(record.get("source", ""))
        if priority.get(new_source, 0) > priority.get(cur_source, 0):
            cur["source"] = new_source
        refresh_task_flags(cur)
    return [merged[key] for key in order]


def prepare_coco_person_masks(annotation_path: Path, image_root: Path, source: str, limit: int = 0) -> list[dict]:
    if not annotation_path.exists() or not image_root.exists():
        return []
    data = read_json(annotation_path)
    images = {int(im["id"]): im for im in data.get("images", [])}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", 1)) != 1 or int(ann.get("iscrowd", 0)):
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    records = []
    for image_id in sorted(anns_by_image):
        image = images.get(image_id)
        if not image:
            continue
        image_path = image_root / image["file_name"]
        if not image_path.exists():
            continue
        instances = [make_mask_instance(ann, source, include_keypoints=False) for ann in anns_by_image[image_id]]
        instances = [inst for inst in instances if inst is not None]
        if not instances:
            continue
        records.append(make_record(image, image_path, source, instances))
        if limit and len(records) >= limit:
            break
    return records


def prepare_ochuman(annotation_path: Path, image_root: Path, limit: int = 0) -> list[dict]:
    if not annotation_path.exists() or not image_root.exists():
        return []
    data = read_json(annotation_path)
    images = {int(im["id"]): im for im in data.get("images", [])}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", 1)) != 1 or int(ann.get("iscrowd", 0)):
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    records = []
    for image_id in sorted(anns_by_image):
        image = images.get(image_id)
        if not image:
            continue
        image_path = image_root / image["file_name"]
        if not image_path.exists():
            continue
        instances = [make_mask_instance(ann, "ochuman", include_keypoints=True) for ann in anns_by_image[image_id]]
        instances = [inst for inst in instances if inst is not None]
        if not instances:
            continue
        records.append(make_record(image, image_path, "ochuman", instances))
        if limit and len(records) >= limit:
            break
    return records


def line_source(line: str) -> str:
    text = str(line).lower()
    if "objects365" in text or "object365" in text:
        return "objects365"
    if "crowdhuman" in text:
        return "crowdhuman"
    if "wider" in text:
        return "wider_face"
    return "unknown_det"


def detection_guard_lines(stage_a: Path, split: str, limit: int) -> list[str]:
    if limit <= 0:
        return []
    total_weight = sum(DET_GUARD_WEIGHTS.values())
    targets = {k: int(round(limit * v / total_weight)) for k, v in DET_GUARD_WEIGHTS.items()}
    delta = limit - sum(targets.values())
    if delta:
        targets["objects365"] = targets.get("objects365", 0) + delta
    buckets = {k: [] for k in targets}
    with (stage_a / ("train.txt" if split == "train" else "val.txt")).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            source = line_source(line)
            if source in buckets and len(buckets[source]) < targets[source]:
                buckets[source].append(line)
                if all(len(buckets[k]) >= targets[k] for k in targets):
                    break
    out = []
    for source in ("objects365", "crowdhuman", "wider_face"):
        out.extend(buckets.get(source, []))
    return out


def balanced_records(records: list[dict], per_source: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        source = str(record.get("source", ""))
        if len(buckets[source]) < per_source:
            buckets[source].append(record)
    out = []
    for source in sorted(buckets):
        out.extend(buckets[source])
    return out


def balanced_det_lines(lines: list[str], per_source: int) -> list[str]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        source = line_source(line)
        if len(buckets[source]) < per_source:
            buckets[source].append(line)
    out = []
    for source in ("objects365", "crowdhuman", "wider_face"):
        out.extend(buckets.get(source, []))
    return out


def source_counts(records: Iterable[dict], extra_lines: Iterable[str] = ()) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get("source", ""))] += 1
    for line in extra_lines:
        counts[line_source(line)] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    stage_c_manifest = out / "manifests"
    stage_c_train = cap(read_jsonl(stage_c_manifest / "stage_c_train.jsonl"), args.max_stage_c_train)
    stage_c_val = cap(read_jsonl(stage_c_manifest / "stage_c_val.jsonl"), args.max_stage_c_val)
    for rec in stage_c_train + stage_c_val:
        rec.setdefault("source", "coco_wholebody")

    coco_train = prepare_coco_person_masks(
        args.vision_root / "COCO_person_masks/annotations/person_instances_train2017.json",
        args.vision_root / "COCO_2017/train2017",
        "coco_person_mask",
        args.max_coco_mask_train,
    )
    coco_val = prepare_coco_person_masks(
        args.vision_root / "COCO_person_masks/annotations/person_instances_val2017.json",
        args.vision_root / "COCO_2017/val2017",
        "coco_person_mask",
        args.max_coco_mask_val,
    )
    ochuman_root = args.vision_root / "OCHuman/OCHuman"
    ochuman_train = prepare_ochuman(
        ochuman_root / "annotations/ochuman_coco_format_val_range_0.00_1.00.json",
        ochuman_root / "images",
        args.max_ochuman_train,
    )
    ochuman_val = prepare_ochuman(
        ochuman_root / "annotations/ochuman_coco_format_test_range_0.00_1.00.json",
        ochuman_root / "images",
        args.max_ochuman_val,
    )

    det_train = detection_guard_lines(args.stage_a, "train", args.det_train_limit)
    det_val = detection_guard_lines(args.stage_a, "val", args.det_val_limit)

    train_records = merge_records_by_image(stage_c_train + coco_train + ochuman_train)
    val_records = merge_records_by_image(stage_c_val + coco_val + ochuman_val)
    train_lines = [str(Path(r["image"]).resolve()) for r in train_records] + det_train
    val_lines = [str(Path(r["image"]).resolve()) for r in val_records] + det_val

    write_jsonl(out / "manifests" / "stage_d_train.jsonl", train_records)
    write_jsonl(out / "manifests" / "stage_d_val.jsonl", val_records)
    write_list(out / "stage_d_train.txt", train_lines)
    write_list(out / "stage_d_val.txt", val_lines)

    if args.smoke_per_source > 0:
        smoke_train_records = balanced_records(train_records, args.smoke_per_source)
        smoke_val_records = balanced_records(val_records, args.smoke_per_source)
        smoke_train_det = balanced_det_lines(det_train, args.smoke_per_source)
        smoke_val_det = balanced_det_lines(det_val, args.smoke_per_source)
        write_jsonl(out / "manifests" / "stage_d_smoke_train.jsonl", smoke_train_records)
        write_jsonl(out / "manifests" / "stage_d_smoke_val.jsonl", smoke_val_records)
        write_list(
            out / "stage_d_smoke_train.txt",
            [str(Path(r["image"]).resolve()) for r in smoke_train_records] + smoke_train_det,
        )
        write_list(
            out / "stage_d_smoke_val.txt",
            [str(Path(r["image"]).resolve()) for r in smoke_val_records] + smoke_val_det,
        )

    summary = {
        "train_total": len(train_lines),
        "val_total": len(val_lines),
        "manifest_train": len(train_records),
        "manifest_val": len(val_records),
        "train_sources": source_counts(train_records, det_train),
        "val_sources": source_counts(val_records, det_val),
        "smoke_per_source": args.smoke_per_source,
        "note": "OCHuman provides val/test annotations only here; val is used as Stage D train and test as Stage D validation.",
    }
    (out / "stage_d_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
