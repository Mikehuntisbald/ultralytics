#!/usr/bin/env python3
"""Evaluate YOLO26-PS pose OKS AP and root-relative z error."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from tools.eval_yolo26ps_pose2d import box_iou, valid_pose_instances
from tools.visualize_yolo26ps_stage import active_tasks_for_source, prepare_predictions, run_inference, source_from_path


DEFAULT_MANIFEST = Path(
    "/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI/manifests/stage_c_val_ochuman.jsonl"
)
COCO_OKS_SIGMAS = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
    dtype=np.float32,
) / 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sources", default="coco_wholebody,3dpw,agora")
    parser.add_argument("--samples", type=int, default=0, help="0 means all matching records")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[576, 768])
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--min-kpts", type=int, default=1)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-vis", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def normalize_imgsz(value: list[int]) -> list[int]:
    return [value[0], value[0]] if len(value) == 1 else [value[0], value[1]]


def canonical_source(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = {canonical_source(x) for x in args.sources.split(",") if canonical_source(x)}
    records: list[dict[str, Any]] = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            source = canonical_source(record.get("source") or source_from_path(record.get("image", "")))
            if source not in wanted or not Path(record.get("image", "")).exists():
                continue
            instances = valid_pose_instances(record, args.min_kpts)
            if not instances:
                continue
            record["source"] = source
            record["_pose_instances"] = instances
            records.append(record)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    return records[: args.samples] if args.samples and args.samples > 0 else records


def bbox_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def oks_score(pred_pose: torch.Tensor, inst: dict[str, Any]) -> float:
    gt = np.asarray(inst["body_kpts_2d"], dtype=np.float32).reshape(-1, 3)
    visible = gt[:, 2] > 0
    if not bool(visible.any()):
        return 0.0
    pred = pred_pose.detach().cpu().numpy()
    d2 = ((pred[:, 0] - gt[:, 0]) ** 2 + (pred[:, 1] - gt[:, 1]) ** 2)[visible]
    area = max(bbox_area([float(x) for x in inst["bbox"][:4]]), 1.0)
    sigmas = COCO_OKS_SIGMAS[visible]
    e = d2 / ((2 * sigmas) ** 2 * area * 2 + 1e-9)
    return float(np.exp(-e).mean())


def ap_from_detections(dets: list[tuple[float, float, int]], total_gt: int, threshold: float) -> float:
    if total_gt <= 0:
        return 0.0
    dets = sorted(dets, key=lambda x: x[0], reverse=True)
    tp = np.zeros(len(dets), dtype=np.float32)
    fp = np.zeros(len(dets), dtype=np.float32)
    used: set[int] = set()
    for i, (_score, oks, gt_id) in enumerate(dets):
        if oks >= threshold and gt_id >= 0 and gt_id not in used:
            tp[i] = 1.0
            used.add(gt_id)
        else:
            fp[i] = 1.0
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / max(float(total_gt), 1.0)
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "rmse": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    args.imgsz = normalize_imgsz(args.imgsz)
    records = load_records(args)
    model = YOLO(str(args.weights))
    if not records:
        raise SystemExit("No records found.")

    head = model.model.model[-1]
    if hasattr(head, "set_active_tasks"):
        head.set_active_tasks(active_tasks_for_source(records[0]["source"], args, head))
    head.max_det = args.max_det
    model.predict(records[0]["image"], imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=args.max_det, save=False, verbose=False, device=args.device)
    predictor = model.predictor
    model.model.eval()

    dets: list[tuple[float, float, int]] = []
    z_errors: list[float] = []
    z_abs_by_source: dict[str, list[float]] = defaultdict(list)
    source_gt: dict[str, int] = defaultdict(int)
    total_gt = 0
    next_gt_id = 0
    matched_for_z = 0

    for record in records:
        image = cv2.imread(record["image"])
        if image is None:
            continue
        source = record["source"]
        if hasattr(head, "set_active_tasks"):
            head.set_active_tasks(active_tasks_for_source(source, args, head))
        with torch.no_grad():
            deploy, ratio_pad, im = run_inference(model, predictor, image, args)
            preds = prepare_predictions(deploy, im, image.shape[:2], ratio_pad, source, args)

        insts = record["_pose_instances"]
        gt_ids = list(range(next_gt_id, next_gt_id + len(insts)))
        next_gt_id += len(insts)
        total_gt += len(insts)
        source_gt[source] += len(insts)

        per_pred_best: list[tuple[int, float, float]] = []
        for pred_i, (box, pose, score) in enumerate(zip(preds["boxes"], preds["pose"], preds["scores"])):
            best_gt = -1
            best_oks = 0.0
            best_iou = 0.0
            for local_gt, inst in enumerate(insts):
                iou = box_iou([float(x) for x in box.tolist()], [float(x) for x in inst["bbox"][:4]])
                if iou < 0.1:
                    continue
                oks = oks_score(pose, inst)
                if oks > best_oks:
                    best_gt, best_oks, best_iou = gt_ids[local_gt], oks, iou
            dets.append((float(score), best_oks, best_gt))
            per_pred_best.append((best_gt, best_oks, best_iou))

        used_local: set[int] = set()
        for local_gt, inst in enumerate(insts):
            body3d = inst.get("body_kpts_3d") or []
            flags = inst.get("flags") or {}
            if not flags.get("has_body3d") or not body3d:
                continue
            candidates = []
            for pred_i, (box, pose) in enumerate(zip(preds["boxes"], preds["pose"])):
                if pred_i in used_local:
                    continue
                iou = box_iou([float(x) for x in box.tolist()], [float(x) for x in inst["bbox"][:4]])
                if iou >= 0.5:
                    candidates.append((iou, pred_i, pose))
            if not candidates:
                continue
            _iou, pred_i, pose = max(candidates, key=lambda x: x[0])
            used_local.add(pred_i)
            gt3d = np.asarray(body3d, dtype=np.float32).reshape(-1, 4)
            valid = gt3d[:, 3] > 0
            if not bool(valid.any()):
                continue
            pred_z = pose.detach().cpu().numpy()[:, 2]
            err = np.abs(pred_z[valid] - gt3d[:, 2][valid])
            z_errors.extend(float(x) for x in err)
            z_abs_by_source[source].extend(float(x) for x in err)
            matched_for_z += 1

    thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    ap_by_thr = {f"AP{int(t * 100)}": ap_from_detections(dets, total_gt, t) for t in thresholds}
    summary = {
        "weights": str(args.weights),
        "manifest": str(args.manifest),
        "sources": sorted({r["source"] for r in records}),
        "records": len(records),
        "gt_instances": total_gt,
        "detections": len(dets),
        "pose_AP50": ap_by_thr["AP50"],
        "pose_AP75": ap_by_thr["AP75"],
        "pose_AP50_95": float(np.mean(list(ap_by_thr.values()))) if ap_by_thr else 0.0,
        "pose_AP_by_threshold": ap_by_thr,
        "z_rel_abs_error": summarize(z_errors),
        "z_rel_matched_instances": matched_for_z,
        "z_rel_abs_error_by_source": {k: summarize(v) for k, v in sorted(z_abs_by_source.items())},
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
