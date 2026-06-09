#!/usr/bin/env python3
"""Visualize matched GT-vs-pred 2D pose overlays for YOLO26-PS checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
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
from tools.visualize_yolo26ps_stage import (
    COCO17_SKELETON,
    active_tasks_for_source,
    make_contact_sheet,
    prepare_predictions,
    put_label,
    run_inference,
    source_from_path,
)


DEFAULT_MANIFEST = Path(
    "/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI/manifests/stage_c_val_ochuman.jsonl"
)
GT_COLOR = (80, 255, 120)
PRED_COLOR = (255, 80, 210)
MISS_COLOR = (40, 40, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sources", default="coco_wholebody,ochuman,3dpw,agora")
    parser.add_argument("--samples-per-source", type=int, default=4)
    parser.add_argument(
        "--image-stems",
        default="",
        help="Optional comma-separated image stems or filenames to visualize instead of random source samples.",
    )
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[576, 768])
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--min-iou", type=float, default=0.50)
    parser.add_argument("--min-kpts", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-vis", type=int, default=30)
    parser.add_argument("--pose-conf", type=float, default=0.20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, default=ROOT / "examples")
    parser.add_argument("--name", default="")
    return parser.parse_args()


def normalize_imgsz(value: list[int]) -> list[int]:
    return [value[0], value[0]] if len(value) == 1 else [value[0], value[1]]


def canonical_source(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def record_source(record: dict[str, Any]) -> str:
    """Return explicit source, falling back to the image path for older manifests."""
    return canonical_source(record.get("source", "")) or canonical_source(source_from_path(record.get("image", "")))


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = [canonical_source(x) for x in args.sources.split(",") if canonical_source(x)]
    wanted_stems = {
        Path(x.strip()).stem
        for x in str(args.image_stems or "").split(",")
        if x.strip()
    }
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            source = record_source(record)
            if source not in wanted:
                continue
            if not Path(record.get("image", "")).exists():
                continue
            if wanted_stems and Path(record.get("image", "")).stem not in wanted_stems:
                continue
            instances = valid_pose_instances(record, args.min_kpts)
            if not instances:
                continue
            record["source"] = source
            record["_pose_instances"] = instances
            by_source[source].append(record)

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    for source in wanted:
        group = by_source.get(source, [])
        if wanted_stems:
            records.extend(group)
        else:
            rng.shuffle(group)
            records.extend(group[: args.samples_per_source])
    return records


def draw_box(img: np.ndarray, box: list[float] | torch.Tensor, color: tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_pose(
    img: np.ndarray,
    kpts: np.ndarray,
    visible: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 3,
    thickness: int = 2,
) -> None:
    for a, b in COCO17_SKELETON:
        if a >= len(kpts) or b >= len(kpts) or not (visible[a] and visible[b]):
            continue
        p1 = (int(round(float(kpts[a, 0]))), int(round(float(kpts[a, 1]))))
        p2 = (int(round(float(kpts[b, 0]))), int(round(float(kpts[b, 1]))))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    for i, (x, y) in enumerate(kpts[:, :2]):
        if i < len(visible) and visible[i]:
            cv2.circle(img, (int(round(float(x))), int(round(float(y)))), radius, color, -1, cv2.LINE_AA)


def match_instances(record: dict[str, Any], preds: dict[str, Any], min_iou: float) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    used: set[int] = set()
    for gt_i, inst in enumerate(record["_pose_instances"]):
        gt_box = [float(x) for x in inst["bbox"][:4]]
        best: tuple[float, int] | None = None
        for pred_i, box in enumerate(preds["boxes"]):
            if pred_i in used:
                continue
            iou = box_iou([float(x) for x in box.tolist()], gt_box)
            if best is None or iou > best[0]:
                best = (iou, pred_i)
        if best is None or best[0] < min_iou:
            matches.append({"gt_i": gt_i, "pred_i": None, "iou": best[0] if best else 0.0, "inst": inst})
            continue
        used.add(best[1])
        matches.append({"gt_i": gt_i, "pred_i": best[1], "iou": best[0], "inst": inst})
    return matches


def draw_matches(
    image: np.ndarray,
    record: dict[str, Any],
    preds: dict[str, Any],
    matches: list[dict[str, Any]],
    pose_conf: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vis = image.copy()
    stats: list[dict[str, Any]] = []
    for item in matches:
        gt_i = int(item["gt_i"])
        pred_i = item["pred_i"]
        inst = item["inst"]
        gt_box = [float(x) for x in inst["bbox"][:4]]
        gt_kpts = np.asarray(inst["body_kpts_2d"], dtype=np.float32).reshape(-1, 3)
        gt_visible = gt_kpts[:, 2] > 0
        draw_box(vis, gt_box, GT_COLOR, 2)
        draw_pose(vis, gt_kpts[:, :2], gt_visible, GT_COLOR, radius=3, thickness=2)

        entry: dict[str, Any] = {
            "gt_i": gt_i,
            "pred_i": pred_i,
            "iou": round(float(item["iou"]), 5),
            "gt_visible": int(gt_visible.sum()),
            "matched": pred_i is not None,
        }
        label_color = MISS_COLOR
        label = f"GT{gt_i} no match"
        if pred_i is not None:
            pred_box = preds["boxes"][pred_i]
            pred_pose = preds["pose"][pred_i].detach().cpu().numpy()
            pred_visible = pred_pose[:, 3] >= pose_conf
            draw_box(vis, pred_box, PRED_COLOR, 2)
            draw_pose(vis, pred_pose[:, :2], pred_visible, PRED_COLOR, radius=2, thickness=2)
            dist = np.linalg.norm(pred_pose[gt_visible, :2] - gt_kpts[gt_visible, :2], axis=1)
            mpjpe = float(dist.mean()) if len(dist) else 0.0
            entry.update(
                {
                    "mpjpe_px": round(mpjpe, 3),
                    "pred_visible": int(pred_visible.sum()),
                    "pred_conf_mean": round(float(pred_pose[:, 3].mean()), 5),
                }
            )
            label_color = PRED_COLOR
            label = f"GT{gt_i}->P{pred_i} iou={entry['iou']:.2f} e={mpjpe:.1f} k={int(pred_visible.sum())}"
        x1, y1, _x2, _y2 = gt_box
        put_label(vis, label, (int(round(x1)), int(round(y1)) - 4), label_color, 0.45)
        stats.append(entry)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 44), (25, 25, 25), -1)
    title = f"{record.get('source')} | {Path(record['image']).name} | GT green, pred magenta | matches={sum(s['matched'] for s in stats)}/{len(stats)}"
    cv2.putText(vis, title[:160], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    return vis, stats


def main() -> None:
    args = parse_args()
    args.imgsz = normalize_imgsz(args.imgsz)
    records = load_records(args)
    if not records:
        raise SystemExit("No pose records found for requested sources.")

    out_name = args.name or f"yolo26ps_pose2d_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    head = model.model.model[-1]
    if hasattr(head, "set_active_tasks"):
        head.set_active_tasks(active_tasks_for_source("coco_wholebody", args, head))
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
        head.set_active_tasks(active_tasks_for_source("coco_wholebody", args, head))
    head.max_det = args.max_det
    model.model.eval()

    saved: list[Path] = []
    summary: dict[str, Any] = {
        "weights": str(args.weights),
        "manifest": str(args.manifest),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "pose_conf": args.pose_conf,
        "min_iou": args.min_iou,
        "images": [],
    }
    all_mpjpe: list[float] = []

    for index, record in enumerate(records):
        image = cv2.imread(record["image"])
        if image is None:
            summary["images"].append({"image": record["image"], "error": "cv2.imread failed"})
            continue
        with torch.no_grad():
            deploy, ratio_pad, im = run_inference(model, predictor, image, args)
            preds = prepare_predictions(deploy, im, image.shape[:2], ratio_pad, canonical_source(record["source"]), args)
        matches = match_instances(record, preds, args.min_iou)
        vis, stats = draw_matches(image, record, preds, matches, args.pose_conf)
        out_path = out_dir / f"{index:02d}_{canonical_source(record['source'])}_{Path(record['image']).stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        saved.append(out_path)
        all_mpjpe.extend(float(s["mpjpe_px"]) for s in stats if s.get("matched") and "mpjpe_px" in s)
        summary["images"].append(
            {
                "source": record.get("source"),
                "image": record["image"],
                "output": str(out_path),
                "num_gt": len(record["_pose_instances"]),
                "num_pred": int(len(preds["boxes"])),
                "matches": stats,
            }
        )
        matched = sum(1 for s in stats if s.get("matched"))
        print(f"{index + 1:02d}/{len(records)} {record.get('source'):14s} gt={len(stats):2d} matched={matched:2d} -> {out_path.name}")

    contact = out_dir / "contact_sheet.jpg"
    make_contact_sheet(saved, contact)
    summary["contact_sheet"] = str(contact)
    summary["mpjpe_mean_px"] = mean(all_mpjpe) if all_mpjpe else None
    summary["mpjpe_median_px"] = median(all_mpjpe) if all_mpjpe else None
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved contact sheet: {contact}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
