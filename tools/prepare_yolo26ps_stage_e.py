#!/usr/bin/env python3
"""Prepare YOLO26s-PS-2.5D Stage E scene-segmentation data.

Stage E adds ADEChallengeData2016 semantic scene masks on top of the already
prepared Stage D multi-task set. ADE records are scene-only: they set
``has_scene_seg=true`` and keep detection/pose/mask flags false, so the unified
loss gates detector supervision off instead of treating ADE images as
background negatives.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image


VISION_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
OUT = VISION_ROOT / "YOLO26PS_STAGE_MULTI"
ADE = VISION_ROOT / "ADEChallengeData2016"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-root", type=Path, default=VISION_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ade", type=Path, default=ADE)
    parser.add_argument("--max-ade-train", type=int, default=0, help="optional ADE train image cap")
    parser.add_argument("--max-ade-val", type=int, default=0, help="optional ADE val image cap")
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [str(line) for line in lines if str(line).strip()]
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def cap(values: list, limit: int) -> list:
    return values[:limit] if limit and len(values) > limit else values


def ade_records(root: Path, split: str, limit: int = 0) -> list[dict]:
    image_root = root / "images" / split
    mask_root = root / "annotations" / split
    records = []
    for image_path in sorted(image_root.glob("*.jpg")):
        mask_path = mask_root / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except OSError:
            continue
        records.append(
            {
                "image": str(image_path.resolve()),
                "source": "ade20k",
                "width": int(width),
                "height": int(height),
                "instances": [],
                "scene_seg": str(mask_path.resolve()),
                "task_flags": {
                    "has_det": False,
                    "has_pose2d": False,
                    "has_pose3d": False,
                    "has_person_mask": False,
                    "has_scene_seg": True,
                },
            }
        )
        if limit and len(records) >= limit:
            break
    return records


def record_source(record: dict) -> str:
    source = str(record.get("source") or "").lower()
    if source:
        return source
    if record.get("scene_seg") or (record.get("task_flags") or {}).get("has_scene_seg"):
        return "ade20k"
    if (record.get("task_flags") or {}).get("has_person_mask"):
        return "coco_person_mask"
    if (record.get("task_flags") or {}).get("has_pose3d"):
        return "pose3d"
    if (record.get("task_flags") or {}).get("has_pose2d"):
        return "pose2d"
    return "unknown"


def line_source(line: str) -> str:
    text = line.lower()
    if "adechallengedata2016" in text or "ade20k" in text or "adechallenge" in text:
        return "ade20k"
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
    if "coco_person_mask" in text:
        return "coco_person_mask"
    if "coco" in text:
        return "coco_wholebody"
    return "unknown"


def source_counts(records: Iterable[dict], lines: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    manifest_images = {str(Path(record["image"]).resolve()) for record in records if record.get("image")}
    for record in records:
        counts[record_source(record)] += 1
    for line in lines:
        key = str(Path(line).resolve())
        if key not in manifest_images:
            counts[line_source(line)] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    out = args.out
    manifests = out / "manifests"

    stage_d_train_lines = read_lines(out / "stage_d_train.txt")
    stage_d_val_lines = read_lines(out / "stage_d_val.txt")
    stage_d_train_records = read_jsonl(manifests / "stage_d_train.jsonl")
    stage_d_val_records = read_jsonl(manifests / "stage_d_val.jsonl")

    ade_train = ade_records(args.ade, "training", args.max_ade_train)
    ade_val = ade_records(args.ade, "validation", args.max_ade_val)

    train_records = stage_d_train_records + ade_train
    val_records = stage_d_val_records + ade_val
    train_lines = stage_d_train_lines + [record["image"] for record in ade_train]
    val_lines = stage_d_val_lines + [record["image"] for record in ade_val]

    write_jsonl(manifests / "stage_e_train.jsonl", train_records)
    write_jsonl(manifests / "stage_e_val.jsonl", val_records)
    write_lines(out / "stage_e_train.txt", train_lines)
    write_lines(out / "stage_e_val.txt", val_lines)

    summary = {
        "train_total": len(train_lines),
        "val_total": len(val_lines),
        "manifest_train": len(train_records),
        "manifest_val": len(val_records),
        "ade_train": len(ade_train),
        "ade_val": len(ade_val),
        "train_sources": source_counts(train_records, train_lines),
        "val_sources": source_counts(val_records, val_lines),
        "note": "ADEChallengeData2016 uses raw masks 0..150; data yaml maps 0 to ignore 255 and 1..150 to 0..149.",
    }
    (out / "stage_e_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
