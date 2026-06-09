#!/usr/bin/env python3
"""Small source-aware 2D pose sanity check for YOLO26-PS checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from tools.visualize_yolo26ps_stage import active_tasks_for_source, prepare_predictions, run_inference, source_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI/manifests/stage_c_val.jsonl"),
    )
    parser.add_argument("--source", default="coco_wholebody")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[576, 768])
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--min-iou", type=float, default=0.50)
    parser.add_argument("--min-kpts", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-vis", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, help="Optional JSON summary path")
    return parser.parse_args()


def normalize_imgsz(value: list[int]) -> list[int]:
    return [value[0], value[0]] if len(value) == 1 else [value[0], value[1]]


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def canonical_source(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def valid_pose_instances(record: dict[str, Any], min_kpts: int) -> list[dict[str, Any]]:
    out = []
    for inst in record.get("instances", []):
        flags = inst.get("flags") or {}
        kpts = inst.get("body_kpts_2d") or []
        visible = sum(1 for k in kpts if len(k) >= 3 and float(k[2]) > 0)
        if flags.get("has_body2d") and visible >= min_kpts and len(inst.get("bbox") or []) >= 4:
            out.append(inst)
    return out


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    wanted = canonical_source(args.source)
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            source = canonical_source(record.get("source") or source_from_path(record.get("image", "")))
            if source != wanted:
                continue
            if not Path(record.get("image", "")).exists():
                continue
            instances = valid_pose_instances(record, args.min_kpts)
            if instances:
                record["source"] = source
                record["_pose_instances"] = instances
                records.append(record)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    return records[: args.samples]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    args.imgsz = normalize_imgsz(args.imgsz)
    records = load_records(args)
    model = YOLO(str(args.weights))
    if records:
        head = model.model.model[-1]
        if hasattr(head, "set_active_tasks"):
            head.set_active_tasks(active_tasks_for_source(args.source, args, head))
        head.max_det = args.max_det
        model.predict(
            records[0]["image"],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            save=False,
            verbose=False,
            device=args.device,
        )
        predictor = model.predictor
        head = model.model.model[-1]
        if hasattr(head, "set_active_tasks"):
            head.set_active_tasks(active_tasks_for_source(args.source, args, head))
        head.max_det = args.max_det
        model.model.eval()
    mpjpes: list[float] = []
    ious: list[float] = []
    confs: list[float] = []
    matched = det_images = failed_images = 0

    for record in records:
        image = cv2.imread(record["image"])
        if image is None:
            failed_images += 1
            continue
        with torch.no_grad():
            deploy, ratio_pad, im = run_inference(model, predictor, image, args)
            preds = prepare_predictions(deploy, im, image.shape[:2], ratio_pad, args.source, args)

        if len(preds["boxes"]):
            det_images += 1
        used: set[int] = set()
        for inst in record["_pose_instances"]:
            gt_box = [float(x) for x in inst["bbox"][:4]]
            gt_kpts = torch.tensor(inst["body_kpts_2d"], dtype=torch.float32)
            visible = gt_kpts[:, 2] > 0
            best: tuple[float, int, torch.Tensor] | None = None
            for j, (box, pose) in enumerate(zip(preds["boxes"], preds["pose"])):
                if j in used:
                    continue
                iou = box_iou([float(x) for x in box.tolist()], gt_box)
                if best is None or iou > best[0]:
                    best = (iou, j, pose)
            if best is None or best[0] < args.min_iou:
                continue
            iou, j, pose = best
            used.add(j)
            dist = (pose[visible, :2] - gt_kpts[visible, :2]).norm(dim=1)
            mpjpes.append(float(dist.mean()))
            ious.append(iou)
            confs.append(float(pose[:, 3].mean()))
            matched += 1

    summary = {
        "weights": str(args.weights),
        "manifest": str(args.manifest),
        "source": args.source,
        "records": len(records),
        "det_images": det_images,
        "matched_instances": matched,
        "failed_images": failed_images,
        "iou_median": median(ious) if ious else None,
        "mpjpe_mean_px": mean(mpjpes) if mpjpes else None,
        "mpjpe_median_px": median(mpjpes) if mpjpes else None,
        "mpjpe_p90_px": sorted(mpjpes)[int(0.9 * (len(mpjpes) - 1))] if mpjpes else None,
        "pose_conf_mean": mean(confs) if confs else None,
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
