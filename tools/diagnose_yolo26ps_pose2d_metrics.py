#!/usr/bin/env python3
"""Diagnose YOLO26-PS 2D pose errors by source and scale."""

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
from ultralytics.utils import ops

from tools.eval_yolo26ps_pose2d import box_iou, valid_pose_instances
from tools.visualize_yolo26ps_stage import active_tasks_for_source, prepare_predictions, run_inference, source_from_path


DEFAULT_MANIFEST = Path(
    "/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI/manifests/stage_c_val_ochuman.jsonl"
)
COCO17_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
BOX_H_BINS = (
    (0.0, 96.0, "h000_096"),
    (96.0, 160.0, "h096_160"),
    (160.0, 256.0, "h160_256"),
    (256.0, 384.0, "h256_384"),
    (384.0, float("inf"), "h384_plus"),
)
IOU_BINS = (
    (0.50, 0.65, "iou50_65"),
    (0.65, 0.75, "iou65_75"),
    (0.75, 0.85, "iou75_85"),
    (0.85, float("inf"), "iou85_plus"),
)
VISIBLE_KPT_BINS = (
    (0, 11, "k008_010"),
    (11, 15, "k011_014"),
    (15, 18, "k015_017"),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sources", default="coco_wholebody,ochuman,3dpw,agora")
    parser.add_argument("--samples-per-source", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[576, 768])
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--min-iou", type=float, default=0.50)
    parser.add_argument("--min-kpts", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-vis", type=int, default=300)
    parser.add_argument("--worst-limit", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, help="Optional JSON summary path")
    return parser.parse_args()


def normalize_imgsz(value: list[int]) -> list[int]:
    """Normalize image size to [height, width]."""
    return [int(value[0]), int(value[0])] if len(value) == 1 else [int(value[0]), int(value[1])]


def canonical_source(value: str) -> str:
    """Normalize source names."""
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def record_source(record: dict[str, Any]) -> str:
    """Return explicit source, falling back to image path."""
    return canonical_source(record.get("source", "")) or canonical_source(source_from_path(record.get("image", "")))


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load balanced records by requested source."""
    wanted = [canonical_source(x) for x in args.sources.split(",") if canonical_source(x)]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            source = record_source(record)
            if source not in wanted or not Path(record.get("image", "")).exists():
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
        rng.shuffle(group)
        records.extend(group[: args.samples_per_source])
    return records


def original_to_input_xy(xy: torch.Tensor, ratio_pad: tuple[Any, Any] | None) -> torch.Tensor:
    """Map original-image xy coordinates into model-input letterbox coordinates."""
    out = xy.clone()
    if ratio_pad is None:
        return out
    ratio, pad = ratio_pad
    if isinstance(ratio, (tuple, list)):
        gain_w = float(ratio[0])
        gain_h = float(ratio[1] if len(ratio) > 1 else ratio[0])
    else:
        gain_w = gain_h = float(ratio)
    if isinstance(pad, (tuple, list)):
        pad_x = float(pad[0])
        pad_y = float(pad[1] if len(pad) > 1 else pad[0])
    else:
        pad_x = pad_y = float(pad)
    out[..., 0] = out[..., 0] * gain_w + pad_x
    out[..., 1] = out[..., 1] * gain_h + pad_y
    return out


def summarize_values(values: list[float]) -> dict[str, float | None]:
    """Return common summary stats for a list."""
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None}
    sorted_values = sorted(float(v) for v in values)
    return {
        "mean": float(mean(sorted_values)),
        "median": float(median(sorted_values)),
        "p90": float(np.quantile(sorted_values, 0.90)),
        "p95": float(np.quantile(sorted_values, 0.95)),
    }


def range_bin(value: float, bins: tuple[tuple[float, float, str], ...]) -> str:
    """Return the configured range-bin name for a value."""
    for lo, hi, name in bins:
        if lo <= value < hi:
            return name
    return bins[-1][2]


def visibility_name(value: float) -> str:
    """Return a COCO-style keypoint visibility bucket name."""
    return "v2_visible" if value >= 2 else "v1_labeled"


def summarize_grouped(values: dict[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
    """Summarize named lists while preserving group counts."""
    out = {}
    for name in sorted(values):
        stats = summarize_values(values[name])
        out[name] = {"count": len(values[name]), **stats}
    return out


def summarize_keypoints(values: list[list[float]]) -> dict[str, dict[str, float | int | None]]:
    """Summarize per-COCO17-keypoint arrays."""
    out = {}
    for i, arr in enumerate(values):
        name = COCO17_NAMES[i] if i < len(COCO17_NAMES) else f"kpt_{i}"
        stats = summarize_values(arr)
        out[name] = {"index": i, "count": len(arr), **stats}
    return out


def raw_pose_predictions_with_anchor(
    model: YOLO,
    raw: dict[str, torch.Tensor],
    im: torch.Tensor,
    image_shape: tuple[int, int],
    ratio_pad: tuple[Any, Any],
    source: str,
    args: argparse.Namespace,
) -> list[dict[str, torch.Tensor]]:
    """Return raw decoded poses and their anchor centers, aligned with filtered/NMS detections."""
    head = model.model.model[-1]
    if isinstance(raw, dict) and "one2one" in raw:
        raw = raw["one2one"]
    if "pose25d" not in raw:
        return []
    boxes = head._get_decode_xyxy(raw).permute(0, 2, 1)
    scores = raw["scores"].sigmoid().permute(0, 2, 1)
    _score, cls, idx = head.get_topk_index(scores, head.max_det)
    anchor_xy = (head.anchors.T * head.strides.T).to(boxes.dtype)
    anchor_xy = anchor_xy.gather(dim=0, index=idx[0].repeat(1, 2))
    pose = head.pose_decode(raw["pose25d"]).gather(
        dim=1,
        index=idx.view(idx.shape[0], idx.shape[1], 1, 1).repeat(1, 1, *head.kpt_shape),
    )
    det = torch.cat([boxes.gather(dim=1, index=idx.repeat(1, 1, 4)), _score, cls], dim=-1)[0]
    pose = pose[0]
    classes = det[:, 5].round().long()
    keep = det[:, 4] >= args.conf
    selected = {0} if source not in {"objects365", "wider_face"} else None
    if selected is not None:
        cls_keep = torch.zeros_like(keep, dtype=torch.bool)
        for cls_id in selected:
            cls_keep |= classes == cls_id
        keep &= cls_keep

    boxes_in = det[keep, :4].clone()
    scores_in = det[keep, 4].clone()
    classes = classes[keep].clone()
    pose = pose[keep].clone()
    anchor_xy = anchor_xy[keep].clone()
    if boxes_in.numel():
        h_in, w_in = im.shape[2:]
        boxes_in[:, [0, 2]] = boxes_in[:, [0, 2]].clamp(0, w_in)
        boxes_in[:, [1, 3]] = boxes_in[:, [1, 3]].clamp(0, h_in)
        valid = (boxes_in[:, 2] > boxes_in[:, 0] + 1) & (boxes_in[:, 3] > boxes_in[:, 1] + 1)
        boxes_in, scores_in, classes, pose, anchor_xy = (
            boxes_in[valid],
            scores_in[valid],
            classes[valid],
            pose[valid],
            anchor_xy[valid],
        )
    if boxes_in.numel():
        from tools.visualize_yolo26ps_stage import nms_class_aware

        keep_idx = nms_class_aware(boxes_in, scores_in, classes, args.iou)
        keep_idx = keep_idx[scores_in[keep_idx].argsort(descending=True)][: args.max_vis]
        pose = pose[keep_idx]
        anchor_xy = anchor_xy[keep_idx]
    out: list[dict[str, torch.Tensor]] = []
    if pose.numel():
        pose_orig = pose.detach().cpu().float().clone()
        pose_orig[..., :2] = ops.scale_coords(im.shape[2:], pose_orig[..., :2], image_shape, ratio_pad=ratio_pad)
        anchor_input = anchor_xy.detach().cpu().float()
        anchor_orig = ops.scale_coords(im.shape[2:], anchor_input.clone(), image_shape, ratio_pad=ratio_pad)
        out = [{"pose": p, "anchor_input": ai, "anchor_orig": ao} for p, ai, ao in zip(pose_orig, anchor_input, anchor_orig)]
    return out


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run diagnostics."""
    args.imgsz = normalize_imgsz(args.imgsz)
    records = load_records(args)
    model = YOLO(str(args.weights))
    predictor = None
    if records:
        head = model.model.model[-1]
        if hasattr(head, "set_active_tasks"):
            head.set_active_tasks(active_tasks_for_source(records[0]["source"], args, head))
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
            head.set_active_tasks(active_tasks_for_source(records[0]["source"], args, head))
        head.max_det = args.max_det
        model.model.eval()

    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "records": 0,
            "gt_instances": 0,
            "matched_instances": 0,
            "det_images": 0,
            "failed_images": 0,
            "mpjpe_px": [],
            "mpjpe_input_px": [],
            "mpjpe_raw_pose_px": [],
            # Deprecated compatibility alias for older diagnostics. This is raw decoded pose MPJPE, not anchor-center MPJPE.
            "mpjpe_raw_anchor_px": [],
            "mpjpe_box_norm": [],
            "mpjpe_box_h_norm": [],
            "mpjpe_box_sqrt_area_norm": [],
            "anchor_center_norm": [],
            "anchor_inside_pose_radius": [],
            "iou": [],
            "gt_box_h": [],
            "gt_box_w": [],
            "raw_minus_deploy_mpjpe_px": [],
            "per_keypoint_px": [[] for _ in COCO17_NAMES],
            "per_keypoint_raw_pose_px": [[] for _ in COCO17_NAMES],
            "visibility_px": defaultdict(list),
            "visibility_raw_pose_px": defaultdict(list),
            "box_h_bins_px": defaultdict(list),
            "iou_bins_px": defaultdict(list),
            "visible_kpt_bins_px": defaultdict(list),
            "pose_conf": [],
            "worst": [],
        }
    )

    for record in records:
        source = canonical_source(record["source"])
        bucket = buckets[source]
        bucket["records"] += 1
        bucket["gt_instances"] += len(record["_pose_instances"])
        image = cv2.imread(record["image"])
        if image is None:
            bucket["failed_images"] += 1
            continue
        head = model.model.model[-1]
        if hasattr(head, "set_active_tasks"):
            head.set_active_tasks(active_tasks_for_source(source, args, head))
        with torch.no_grad():
            deploy, ratio_pad, im = run_inference(model, predictor, image, args)
            raw = model.model(im)[1]
            preds = prepare_predictions(deploy, im, image.shape[:2], ratio_pad, source, args)
            raw_poses = raw_pose_predictions_with_anchor(model, raw, im, image.shape[:2], ratio_pad, source, args)
        if len(preds["boxes"]):
            bucket["det_images"] += 1

        used: set[int] = set()
        for gt_i, inst in enumerate(record["_pose_instances"]):
            gt_box = [float(x) for x in inst["bbox"][:4]]
            gt_w = max(gt_box[2] - gt_box[0], 1.0)
            gt_h = max(gt_box[3] - gt_box[1], 1.0)
            gt_scale = max((gt_w * gt_h) ** 0.5, 1.0)
            gt_kpts = torch.tensor(inst["body_kpts_2d"], dtype=torch.float32)
            visible = gt_kpts[:, 2] > 0
            best: tuple[float, int, torch.Tensor] | None = None
            for pred_i, (box, pose) in enumerate(zip(preds["boxes"], preds["pose"])):
                if pred_i in used:
                    continue
                iou = box_iou([float(x) for x in box.tolist()], gt_box)
                if best is None or iou > best[0]:
                    best = (iou, pred_i, pose)
            if best is None or best[0] < args.min_iou:
                continue

            iou, pred_i, pose = best
            used.add(pred_i)
            dist = (pose[visible, :2] - gt_kpts[visible, :2]).norm(dim=1)
            raw_info = raw_poses[pred_i] if pred_i < len(raw_poses) else None
            raw_pose = raw_info["pose"] if raw_info is not None else None
            dist_raw = (
                (raw_pose[visible, :2] - gt_kpts[visible, :2]).norm(dim=1) if raw_pose is not None else torch.empty(0)
            )
            raw_deploy_delta = None
            if dist_raw.numel():
                raw_deploy_delta = float(dist_raw.mean() - dist.mean())
            anchor_norm = None
            anchor_inside_radius = None
            if raw_info is not None:
                gt_center = torch.tensor([(gt_box[0] + gt_box[2]) * 0.5, (gt_box[1] + gt_box[3]) * 0.5])
                gt_wh = torch.tensor([gt_w, gt_h]).clamp(min=1.0)
                anchor_delta = (raw_info["anchor_orig"] - gt_center) / gt_wh
                anchor_norm = float(anchor_delta.norm())
                radius = float(getattr(args, "pose_anchor_radius", 0.65))
                anchor_inside_radius = float(anchor_norm <= radius)
            gt_input_xy = original_to_input_xy(gt_kpts[visible, :2], ratio_pad)
            pred_input_xy = original_to_input_xy(pose[visible, :2], ratio_pad)
            dist_input = (pred_input_xy - gt_input_xy).norm(dim=1)
            mpjpe = float(dist.mean())
            mpjpe_input = float(dist_input.mean())
            mpjpe_raw_pose = float(dist_raw.mean()) if dist_raw.numel() else None
            bucket["matched_instances"] += 1
            bucket["mpjpe_px"].append(mpjpe)
            bucket["mpjpe_input_px"].append(mpjpe_input)
            if mpjpe_raw_pose is not None:
                bucket["mpjpe_raw_pose_px"].append(mpjpe_raw_pose)
                bucket["mpjpe_raw_anchor_px"].append(mpjpe_raw_pose)
                bucket["raw_minus_deploy_mpjpe_px"].append(float(mpjpe_raw_pose - mpjpe))
            if anchor_norm is not None:
                bucket["anchor_center_norm"].append(anchor_norm)
                bucket["anchor_inside_pose_radius"].append(anchor_inside_radius)
            bucket["mpjpe_box_norm"].append(float((dist / torch.tensor([gt_w, gt_h]).norm()).mean()))
            bucket["mpjpe_box_h_norm"].append(mpjpe / gt_h)
            bucket["mpjpe_box_sqrt_area_norm"].append(mpjpe / gt_scale)
            bucket["iou"].append(float(iou))
            bucket["gt_box_h"].append(gt_h)
            bucket["gt_box_w"].append(gt_w)
            bucket["pose_conf"].append(float(pose[:, 3].mean()))
            visible_indices = torch.where(visible)[0].tolist()
            for local_i, kpt_i in enumerate(visible_indices):
                bucket["per_keypoint_px"][int(kpt_i)].append(float(dist[local_i]))
                bucket["visibility_px"][visibility_name(float(gt_kpts[int(kpt_i), 2]))].append(float(dist[local_i]))
                if dist_raw.numel():
                    bucket["per_keypoint_raw_pose_px"][int(kpt_i)].append(float(dist_raw[local_i]))
                    bucket["visibility_raw_pose_px"][visibility_name(float(gt_kpts[int(kpt_i), 2]))].append(
                        float(dist_raw[local_i])
                    )
            bucket["box_h_bins_px"][range_bin(gt_h, BOX_H_BINS)].append(mpjpe)
            bucket["iou_bins_px"][range_bin(float(iou), IOU_BINS)].append(mpjpe)
            bucket["visible_kpt_bins_px"][range_bin(float(len(visible_indices)), VISIBLE_KPT_BINS)].append(mpjpe)
            bucket["worst"].append(
                {
                    "mpjpe_px": round(mpjpe, 3),
                    "mpjpe_input_px": round(mpjpe_input, 3),
                    "mpjpe_raw_pose_px": round(mpjpe_raw_pose, 3) if mpjpe_raw_pose is not None else None,
                    "mpjpe_raw_anchor_px": round(mpjpe_raw_pose, 3) if mpjpe_raw_pose is not None else None,
                    "raw_minus_deploy_mpjpe_px": round(raw_deploy_delta, 3) if raw_deploy_delta is not None else None,
                    "anchor_center_norm": round(anchor_norm, 4) if anchor_norm is not None else None,
                    "anchor_inside_pose_radius": bool(anchor_inside_radius) if anchor_inside_radius is not None else None,
                    "iou": round(float(iou), 4),
                    "image": record["image"],
                    "gt_i": gt_i,
                    "pred_i": int(pred_i),
                    "gt_box": [round(float(x), 2) for x in gt_box],
                }
            )

    summary: dict[str, Any] = {
        "weights": str(args.weights),
        "manifest": str(args.manifest),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "min_iou": args.min_iou,
        "min_kpts": args.min_kpts,
        "samples_per_source": args.samples_per_source,
        "metric_notes": {
            "mpjpe_raw_pose_px": "MPJPE from the raw head pose_decode path before deployment bbox-normalized pose conversion.",
            "mpjpe_raw_anchor_px": "Deprecated alias of mpjpe_raw_pose_px kept for compatibility; it is not anchor-center MPJPE.",
            "anchor_center_norm": "L2 distance from selected anchor center to GT box center, normalized by GT box width/height.",
        },
        "sources": {},
    }
    all_values: dict[str, list[float]] = defaultdict(list)
    for source, bucket in sorted(buckets.items()):
        source_out = {
            "records": int(bucket["records"]),
            "det_images": int(bucket["det_images"]),
            "gt_instances": int(bucket["gt_instances"]),
            "matched_instances": int(bucket["matched_instances"]),
            "match_ratio": float(bucket["matched_instances"] / max(bucket["gt_instances"], 1)),
            "failed_images": int(bucket["failed_images"]),
            "mpjpe_px": summarize_values(bucket["mpjpe_px"]),
            "mpjpe_input_px": summarize_values(bucket["mpjpe_input_px"]),
            "mpjpe_raw_pose_px": summarize_values(bucket["mpjpe_raw_pose_px"]),
            "mpjpe_raw_anchor_px": summarize_values(bucket["mpjpe_raw_anchor_px"]),
            "mpjpe_box_norm": summarize_values(bucket["mpjpe_box_norm"]),
            "mpjpe_box_h_norm": summarize_values(bucket["mpjpe_box_h_norm"]),
            "mpjpe_box_sqrt_area_norm": summarize_values(bucket["mpjpe_box_sqrt_area_norm"]),
            "anchor_center_norm": summarize_values(bucket["anchor_center_norm"]),
            "anchor_inside_pose_radius": summarize_values(bucket["anchor_inside_pose_radius"]),
            "iou": summarize_values(bucket["iou"]),
            "gt_box_h": summarize_values(bucket["gt_box_h"]),
            "gt_box_w": summarize_values(bucket["gt_box_w"]),
            "raw_minus_deploy_mpjpe_px": summarize_values(bucket["raw_minus_deploy_mpjpe_px"]),
            "per_keypoint_px": summarize_keypoints(bucket["per_keypoint_px"]),
            "per_keypoint_raw_pose_px": summarize_keypoints(bucket["per_keypoint_raw_pose_px"]),
            "visibility_px": summarize_grouped(bucket["visibility_px"]),
            "visibility_raw_pose_px": summarize_grouped(bucket["visibility_raw_pose_px"]),
            "box_h_bins_px": summarize_grouped(bucket["box_h_bins_px"]),
            "iou_bins_px": summarize_grouped(bucket["iou_bins_px"]),
            "visible_kpt_bins_px": summarize_grouped(bucket["visible_kpt_bins_px"]),
            "pose_conf": summarize_values(bucket["pose_conf"]),
            "worst": sorted(bucket["worst"], key=lambda x: x["mpjpe_px"], reverse=True)[: max(int(args.worst_limit), 0)],
        }
        summary["sources"][source] = source_out
        for key in (
            "mpjpe_px",
            "mpjpe_input_px",
            "mpjpe_raw_pose_px",
            "mpjpe_raw_anchor_px",
            "mpjpe_box_norm",
            "mpjpe_box_h_norm",
            "mpjpe_box_sqrt_area_norm",
            "anchor_center_norm",
            "anchor_inside_pose_radius",
            "raw_minus_deploy_mpjpe_px",
        ):
            all_values[key].extend(float(x) for x in bucket[key])

    summary["overall"] = {key: summarize_values(values) for key, values in sorted(all_values.items())}
    return summary


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    summary = evaluate(args)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
