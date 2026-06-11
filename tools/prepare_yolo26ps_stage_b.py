#!/usr/bin/env python3
"""Prepare YOLO26s-PS-2.5D Stage B pose2d data.

This creates the unified-schema Stage B files expected by
``ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d.yaml``. COCO-WholeBody
records provide person bbox + COCO-17 pose2d supervision. Stage A detection
sources are included through their existing YOLO txt labels and are parsed by
the unified dataloader fallback.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DATASETS = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
OUT = DATASETS / "YOLO26PS_STAGE_MULTI"
STAGE_A = DATASETS / "YOLO26PS_STAGE_A"
COCO_WHOLEBODY = DATASETS / "COCO-WholeBody" / "coco_wholebody_train_v1.0.json"
COCO_IMAGES = DATASETS / "COCO_2017" / "train2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=DATASETS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stage-a", type=Path, default=STAGE_A)
    parser.add_argument("--max-coco-train", type=int, help="optional COCO-WholeBody train cap for quick checks")
    parser.add_argument("--max-coco-val", type=int, default=5000, help="COCO-WholeBody validation subset size")
    parser.add_argument("--det-train-limit", type=int, default=0, help="optional Stage A detection train subset size")
    parser.add_argument("--det-val-limit", type=int, default=5000, help="Stage A detection validation subset size")
    return parser.parse_args()


def coco17(keypoints: list[float]) -> list[list[float]]:
    values = [float(x) for x in keypoints[: 17 * 3]]
    values += [0.0] * max(0, 17 * 3 - len(values))
    return [values[i : i + 3] for i in range(0, 17 * 3, 3)]


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = map(float, box[:4])
    return [x, y, x + w, y + h]


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")


def append_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_coco_wholebody(coco_json: Path) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in data.get("images", [])}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", 1)) != 1 or int(ann.get("iscrowd", 0)):
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) < 4:
            continue
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            continue
        anns_by_image[int(ann["image_id"])].append(ann)
    return images, anns_by_image


def make_coco_record(image: dict, annotations: list[dict], image_path: Path) -> dict:
    instances = []
    for ann in annotations:
        kpts = coco17(ann.get("keypoints") or [])
        instances.append(
            {
                "category": "person",
                "bbox": xywh_to_xyxy(ann["bbox"]),
                "body_kpts_2d": kpts,
                "flags": {
                    "has_bbox": True,
                    "has_body2d": any(k[2] > 0 for k in kpts),
                    "has_body3d": False,
                    "has_person_mask": False,
                },
            }
        )
    return {
        "image": str(image_path.resolve()),
        "source": "coco_wholebody",
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


def prepare_coco_wholebody(out: Path, coco_json: Path, image_root: Path, max_train: int | None, max_val: int) -> tuple[list[str], list[str]]:
    images, anns_by_image = load_coco_wholebody(coco_json)
    image_ids = sorted(i for i, anns in anns_by_image.items() if anns)
    if max_train:
        image_ids = image_ids[:max_train]
    val_count = min(max_val, max(1, len(image_ids) // 20))
    val_ids = set(image_ids[-val_count:])
    train_lines, val_lines = [], []
    manifest_train, manifest_val = [], []
    for image_id in image_ids:
        image = images.get(image_id)
        if not image:
            continue
        image_path = image_root / image["file_name"]
        if not image_path.exists():
            continue
        split = "val" if image_id in val_ids else "train"
        record = make_coco_record(image, anns_by_image[image_id], image_path)
        label_path = out / "labels" / "stage_b" / split / "coco_wholebody" / Path(image["file_name"]).with_suffix(".json")
        write_json(label_path, record)
        line = str(image_path.resolve())
        if split == "val":
            val_lines.append(line)
            manifest_val.append(record)
        else:
            train_lines.append(line)
            manifest_train.append(record)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "manifests" / "stage_b_train.jsonl").write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in manifest_train), encoding="utf-8"
    )
    (out / "manifests" / "stage_b_val.jsonl").write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in manifest_val), encoding="utf-8"
    )
    return train_lines, val_lines


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def detection_source_lines(stage_a: Path, split: str, train_limit: int, val_limit: int) -> list[str]:
    lines = read_lines(stage_a / ("train.txt" if split == "train" else "val.txt"))
    if split == "train" and train_limit > 0:
        return lines[:train_limit]
    if split == "val" and val_limit > 0:
        return lines[:val_limit]
    return lines


def main() -> None:
    args = parse_args()
    datasets = args.datasets
    out = args.out
    stage_a = args.stage_a
    coco_json = datasets / "COCO-WholeBody" / "coco_wholebody_train_v1.0.json"
    coco_images = datasets / "COCO_2017" / "train2017"
    out.mkdir(parents=True, exist_ok=True)

    coco_train, coco_val = prepare_coco_wholebody(out, coco_json, coco_images, args.max_coco_train, args.max_coco_val)
    det_train = detection_source_lines(stage_a, "train", args.det_train_limit, args.det_val_limit)
    det_val = detection_source_lines(stage_a, "val", args.det_train_limit, args.det_val_limit)
    train = coco_train + det_train
    val = coco_val + det_val
    append_lines(out / "stage_b_train.txt", train)
    append_lines(out / "stage_b_val.txt", val)
    summary = {
        "coco_wholebody_train": len(coco_train),
        "coco_wholebody_val": len(coco_val),
        "det_train": len(det_train),
        "det_val": len(det_val),
        "train_total": len(train),
        "val_total": len(val),
    }
    (out / "stage_b_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
