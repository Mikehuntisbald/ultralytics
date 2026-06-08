#!/usr/bin/env python3
"""Append OCHuman pose-only replay records to the Stage C det-replay dataset.

This keeps the existing Stage C pose manifests intact and writes new
``*_ochuman`` lists/manifests. Detection replay lines are taken from the prepared
Stage A image lists so the standard ``/images/`` -> ``/labels/`` YOLO lookup
finds real det labels. OCHuman contributes person boxes and COCO17 2D keypoints
only; mask fields are intentionally omitted so Stage C does not turn into a
mask-training stage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


VISION_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
OUT = VISION_ROOT / "YOLO26PS_STAGE_MULTI"
OCHUMAN = VISION_ROOT / "OCHuman" / "OCHuman"
STAGE_A = VISION_ROOT / "YOLO26PS_STAGE_A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--image-root", type=Path, default=OCHUMAN / "images")
    parser.add_argument(
        "--train-ann",
        type=Path,
        default=OCHUMAN / "annotations" / "ochuman_coco_format_val_range_0.00_1.00.json",
        help="OCHuman val annotations are used as Stage C train replay.",
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=OCHUMAN / "annotations" / "ochuman_coco_format_test_range_0.00_1.00.json",
        help="OCHuman test annotations are used as Stage C validation replay.",
    )
    parser.add_argument("--stage-a", type=Path, default=STAGE_A)
    parser.add_argument("--base-train-manifest", type=Path, default=OUT / "manifests" / "stage_c_train.jsonl")
    parser.add_argument("--base-val-manifest", type=Path, default=OUT / "manifests" / "stage_c_val.jsonl")
    parser.add_argument("--train-list-name", default="stage_c_detreplay_light_ochuman_train.txt")
    parser.add_argument("--val-list-name", default="stage_c_detreplay_ochuman_val.txt")
    parser.add_argument("--train-manifest-name", default="manifests/stage_c_train_ochuman.jsonl")
    parser.add_argument("--val-manifest-name", default="manifests/stage_c_val_ochuman.jsonl")
    parser.add_argument("--summary-name", default="stage_c_detreplay_ochuman_summary.json")
    parser.add_argument("--max-ochuman-train", type=int, default=0)
    parser.add_argument("--max-ochuman-val", type=int, default=0)
    parser.add_argument("--min-visible-kpts", type=int, default=1)
    parser.add_argument("--det-train-objects365", type=int, default=92000)
    parser.add_argument("--det-train-crowdhuman", type=int, default=2000)
    parser.add_argument("--det-train-wider-face", type=int, default=6000)
    parser.add_argument("--det-val-objects365", type=int, default=4600)
    parser.add_argument("--det-val-crowdhuman", type=int, default=100)
    parser.add_argument("--det-val-wider-face", type=int, default=300)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_list(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [str(line) for line in lines if str(line).strip()]
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def xywh_to_xyxy(box: list[float], width: int, height: int) -> list[float] | None:
    if len(box) < 4:
        return None
    x, y, w, h = map(float, box[:4])
    if w <= 1 or h <= 1:
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def coco17(keypoints: list[float] | None) -> tuple[list[list[float]], int]:
    values = [float(x) for x in (keypoints or [])[: 17 * 3]]
    values += [0.0] * max(0, 17 * 3 - len(values))
    out = []
    visible = 0
    for i in range(0, 17 * 3, 3):
        x, y, v = values[i : i + 3]
        if v > 0 and x > 0 and y > 0:
            out.append([round(x, 3), round(y, 3), float(v)])
            visible += 1
        else:
            out.append([0.0, 0.0, 0.0])
    return out, visible


def make_instance(ann: dict, width: int, height: int, min_visible_kpts: int) -> dict | None:
    bbox = xywh_to_xyxy(ann.get("bbox") or [], width, height)
    if bbox is None:
        return None
    kpts, visible = coco17(ann.get("keypoints"))
    has_body2d = visible >= min_visible_kpts
    return {
        "category": "person",
        "bbox": bbox,
        "body_kpts_2d": kpts,
        "flags": {
            "has_bbox": True,
            "has_body2d": has_body2d,
            "has_body3d": False,
            "has_person_mask": False,
        },
    }


def make_record(image: dict, image_path: Path, instances: list[dict]) -> dict:
    return {
        "image": str(image_path.resolve()),
        "source": "ochuman",
        "width": int(image["width"]),
        "height": int(image["height"]),
        "instances": instances,
        "task_flags": {
            "has_det": bool(instances),
            "has_pose2d": any(inst["flags"]["has_body2d"] for inst in instances),
            "has_pose3d": False,
            "has_person_mask": False,
            "has_scene_seg": False,
        },
    }


def prepare_ochuman(annotation_path: Path, image_root: Path, limit: int, min_visible_kpts: int) -> list[dict]:
    data = read_json(annotation_path)
    images = {int(image["id"]): image for image in data.get("images", [])}
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
        width, height = int(image["width"]), int(image["height"])
        instances = [make_instance(ann, width, height, min_visible_kpts) for ann in anns_by_image[image_id]]
        instances = [inst for inst in instances if inst is not None]
        if not instances:
            continue
        records.append(make_record(image, image_path, instances))
        if limit and len(records) >= limit:
            break
    return records


def source_from_line(line: str) -> str:
    text = str(line).lower()
    if "objects365" in text or "object365" in text:
        return "objects365"
    if "crowdhuman" in text:
        return "crowdhuman"
    if "wider" in text:
        return "wider_face"
    if "ochuman" in text:
        return "ochuman"
    if "3dpw" in text:
        return "3dpw"
    if "agora" in text:
        return "agora"
    if "coco" in text:
        return "coco_wholebody"
    return "unknown"


def yolo_label_path_for_image(line: str) -> Path:
    path = str(Path(line))
    if "/images/" in path:
        return Path(path.replace("/images/", "/labels/")).with_suffix(".txt")
    return Path(path).with_suffix(".txt")


def select_det_replay_lines(stage_a: Path, split: str, targets: dict[str, int]) -> list[str]:
    lines = read_lines(stage_a / f"{split}.txt")
    buckets = {source: [] for source in targets}
    for line in lines:
        source = source_from_line(line)
        if source not in buckets or len(buckets[source]) >= targets[source]:
            continue
        if not yolo_label_path_for_image(line).exists():
            continue
        buckets[source].append(line)
        if all(len(buckets[source]) >= targets[source] for source in targets):
            break

    missing = {source: targets[source] - len(buckets[source]) for source in targets if len(buckets[source]) < targets[source]}
    if missing:
        print(f"WARNING: det replay targets not fully met for {split}: {missing}")
    out = []
    for source in ("objects365", "crowdhuman", "wider_face"):
        out.extend(buckets.get(source, []))
    return out


def source_counts(records: Iterable[dict], extra_lines: Iterable[str] = ()) -> dict[str, int]:
    counts = Counter()
    for record in records:
        counts[str(record.get("source", ""))] += 1
    for line in extra_lines:
        counts[source_from_line(line)] += 1
    return dict(sorted(counts.items()))


def lines_not_in_manifest(lines: Iterable[str], records: Iterable[dict]) -> list[str]:
    manifest_images = {str(record["image"]) for record in records if record.get("image")}
    return [line for line in lines if str(line) not in manifest_images]


def append_records_by_image(base_records: list[dict], extra_records: list[dict]) -> list[dict]:
    seen = {str(Path(record["image"]).resolve()) for record in base_records if record.get("image")}
    out = list(base_records)
    for record in extra_records:
        key = str(Path(record["image"]).resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def main() -> None:
    args = parse_args()
    train_records_base = read_jsonl(args.base_train_manifest)
    val_records_base = read_jsonl(args.base_val_manifest)
    det_train = select_det_replay_lines(
        args.stage_a,
        "train",
        {
            "objects365": args.det_train_objects365,
            "crowdhuman": args.det_train_crowdhuman,
            "wider_face": args.det_train_wider_face,
        },
    )
    det_val = select_det_replay_lines(
        args.stage_a,
        "val",
        {
            "objects365": args.det_val_objects365,
            "crowdhuman": args.det_val_crowdhuman,
            "wider_face": args.det_val_wider_face,
        },
    )

    ochuman_train = prepare_ochuman(
        args.train_ann,
        args.image_root,
        args.max_ochuman_train,
        args.min_visible_kpts,
    )
    ochuman_val = prepare_ochuman(
        args.val_ann,
        args.image_root,
        args.max_ochuman_val,
        args.min_visible_kpts,
    )

    train_records = append_records_by_image(train_records_base, ochuman_train)
    val_records = append_records_by_image(val_records_base, ochuman_val)
    train_lines = dedupe_keep_order([record["image"] for record in train_records] + det_train)
    val_lines = dedupe_keep_order([record["image"] for record in val_records] + det_val)

    train_manifest = args.out / args.train_manifest_name
    val_manifest = args.out / args.val_manifest_name
    train_list = args.out / args.train_list_name
    val_list = args.out / args.val_list_name
    summary_path = args.out / args.summary_name

    write_jsonl(train_manifest, train_records)
    write_jsonl(val_manifest, val_records)
    write_list(train_list, train_lines)
    write_list(val_list, val_lines)

    train_det_only_lines = lines_not_in_manifest(train_lines, train_records)
    val_det_only_lines = lines_not_in_manifest(val_lines, val_records)
    summary = {
        "train_list": str(train_list),
        "val_list": str(val_list),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "base_train_manifest_records": len(train_records_base),
        "base_val_manifest_records": len(val_records_base),
        "det_train_lines": len(det_train),
        "det_val_lines": len(det_val),
        "ochuman_train_records": len(ochuman_train),
        "ochuman_val_records": len(ochuman_val),
        "train_total_lines": len(train_lines),
        "val_total_lines": len(val_lines),
        "train_manifest_records": len(train_records),
        "val_manifest_records": len(val_records),
        "train_sources": source_counts(train_records, train_det_only_lines),
        "val_sources": source_counts(val_records, val_det_only_lines),
        "note": "OCHuman val is used for train replay and OCHuman test is used for validation replay; mask fields are omitted.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
