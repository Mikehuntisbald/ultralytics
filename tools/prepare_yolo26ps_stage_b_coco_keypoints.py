#!/usr/bin/env python3
"""Prepare YOLO26-PS Stage B data from COCO 2017 person keypoints only.

The source annotations follow the same bbox/keypoint filtering conventions as
Ultralytics' COCO pose converter, but the output is YOLO26-PS unified schema so
the multi-task loss receives has_pose2d and instance_flags correctly.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DATASETS = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
OUT = DATASETS / "YOLO26PS_STAGE_MULTI"
COCO = DATASETS / "COCO_2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=COCO, help="COCO 2017 root with annotations/train2017/val2017")
    parser.add_argument("--out", type=Path, default=OUT, help="YOLO26PS_STAGE_MULTI output root")
    parser.add_argument("--max-train", type=int, default=0, help="optional cap for quick checks")
    parser.add_argument("--max-val", type=int, default=0, help="optional cap for quick checks")
    return parser.parse_args()


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = map(float, box[:4])
    return [x, y, x + w, y + h]


def coco17(keypoints: list[float]) -> list[list[float]]:
    values = [float(x) for x in keypoints[: 17 * 3]]
    values += [0.0] * max(0, 17 * 3 - len(values))
    return [values[i : i + 3] for i in range(0, 17 * 3, 3)]


def load_person_keypoints(annotation_file: Path) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in data.get("images", [])}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if int(ann.get("category_id", 1)) != 1 or int(ann.get("iscrowd", 0)):
            continue
        if int(ann.get("num_keypoints", 0)) <= 0:
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) < 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            continue
        if ann.get("keypoints") is None:
            continue
        anns_by_image[int(ann["image_id"])].append(ann)
    return images, anns_by_image


def make_record(image: dict, annotations: list[dict], image_path: Path) -> dict:
    instances = []
    for ann in annotations:
        kpts = coco17(ann.get("keypoints") or [])
        has_body2d = any(k[2] > 0 for k in kpts)
        if not has_body2d:
            continue
        instances.append(
            {
                "category": "person",
                "bbox": xywh_to_xyxy(ann["bbox"]),
                "body_kpts_2d": kpts,
                "flags": {
                    "has_bbox": True,
                    "has_body2d": True,
                    "has_body3d": False,
                    "has_person_mask": False,
                },
            }
        )
    return {
        "image": str(image_path.resolve()),
        "source": "coco_keypoints",
        "width": int(image["width"]),
        "height": int(image["height"]),
        "instances": instances,
        "task_flags": {
            "has_det": bool(instances),
            "has_pose2d": bool(instances),
            "has_pose3d": False,
            "has_person_mask": False,
            "has_scene_seg": False,
        },
    }


def write_split(coco_root: Path, out: Path, split: str, cap: int) -> dict:
    annotation_file = coco_root / "annotations" / f"person_keypoints_{split}2017.json"
    image_root = coco_root / f"{split}2017"
    images, anns_by_image = load_person_keypoints(annotation_file)
    image_ids = sorted(i for i, anns in anns_by_image.items() if anns)
    if cap > 0:
        image_ids = image_ids[:cap]

    lines, manifest = [], []
    missing_images = 0
    instances = 0
    for image_id in image_ids:
        image = images.get(image_id)
        if not image:
            continue
        image_path = image_root / image["file_name"]
        if not image_path.exists():
            missing_images += 1
            continue
        record = make_record(image, anns_by_image[image_id], image_path)
        if not record["instances"]:
            continue
        label_path = out / "labels" / "stage_b_coco_keypoints" / split / "coco_keypoints" / Path(image["file_name"]).with_suffix(".json")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
        lines.append(str(image_path.resolve()))
        manifest.append(record)
        instances += len(record["instances"])

    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "manifests" / f"stage_b_coco_keypoints_{split}.jsonl").write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in manifest), encoding="utf-8"
    )
    (out / f"stage_b_coco_keypoints_{split}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return {
        f"{split}_images": len(lines),
        f"{split}_instances": instances,
        f"{split}_missing_images": missing_images,
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    summary.update(write_split(args.coco, args.out, "train", args.max_train))
    summary.update(write_split(args.coco, args.out, "val", args.max_val))
    (args.out / "stage_b_coco_keypoints_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
