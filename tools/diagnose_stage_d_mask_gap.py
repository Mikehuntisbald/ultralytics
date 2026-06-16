#!/usr/bin/env python3
"""Compare Stage D person-mask predictions against a reference segmentation model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import box_iou, mask_iou

from tools.eval_yolo26ps_stage_d_mask import (
    DEFAULT_DATA,
    YOLO26PSStageDMaskValidator,
    normalize_imgsz,
)


DEFAULT_STUDENT = (
    ROOT
    / "runs/detect/yolo26ps_d_seghead_maskonly_cv4proto_bnlock_from_e2best_lr1e5_b96_s30000_setsid-2/weights/best.pt"
)
DEFAULT_REFERENCE = ROOT / "pretrains/yolo26s-seg.pt"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", default="576,768")
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--source-filter", default="", help="only keep images whose path contains this text")
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--hard-topn", type=int, default=0, help="write the top-N gap images to a sampler hard list")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/detect/stage_d_mask_gap_diagnosis")
    return parser.parse_args()


def make_val_args(args: argparse.Namespace, weights: Path, name: str) -> Any:
    """Return an Ultralytics config namespace for mask validation helpers."""
    return get_cfg(
        overrides={
            "model": str(weights),
            "data": str(args.data),
            "imgsz": normalize_imgsz(args.imgsz),
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "split": "val",
            "task": "segment",
            "mode": "val",
            "project": str(args.out),
            "name": name,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "overlap_mask": True,
            "val_samples": args.val_samples,
            "classes": [0],
        }
    )


def apply_source_filter(args: argparse.Namespace, data: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    """Return a dataset config whose split path is pre-filtered by image path text."""
    source_filter = str(args.source_filter or "").strip().lower()
    filtered_data = dict(data)
    if isinstance(data.get("unified_manifest"), dict):
        filtered_data["unified_manifest"] = dict(data["unified_manifest"])
    if isinstance(data.get("unified_labels"), dict):
        filtered_data["unified_labels"] = dict(data["unified_labels"])
    # DetectionValidator always builds validation-mode datasets. Map val-mode metadata to the requested split so
    # train diagnostics still read the train manifest and labels.
    if args.split != "val":
        filtered_data["val"] = data.get(args.split)
        if isinstance(filtered_data.get("unified_manifest"), dict):
            filtered_data["unified_manifest"]["val"] = filtered_data["unified_manifest"].get(args.split)
        if isinstance(filtered_data.get("unified_labels"), dict):
            filtered_data["unified_labels"]["val"] = filtered_data["unified_labels"].get(args.split)

    if not source_filter:
        return filtered_data, None

    split_path = Path(data.get(args.split, ""))
    if not split_path.exists():
        raise FileNotFoundError(f"Cannot source-filter missing split list: {split_path}")
    lines = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = [line for line in lines if source_filter in line.replace("\\", "/").lower()]
    if not keep:
        raise SystemExit(f"source-filter={args.source_filter!r} matched 0 images in {split_path}")

    args.out.mkdir(parents=True, exist_ok=True)
    safe_filter = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in source_filter)
    filtered_path = args.out / f"{split_path.stem}.{safe_filter}.txt"
    filtered_path.write_text("\n".join(keep) + "\n", encoding="utf-8")

    filtered_data[args.split] = str(filtered_path)
    filtered_data["val"] = str(filtered_path)
    return filtered_data, filtered_path


@torch.no_grad()
def setup_model_and_validator(
    weights: Path,
    args: argparse.Namespace,
    data: dict[str, Any],
    device: torch.device,
    name: str,
) -> tuple[YOLO, SegmentationValidator]:
    """Load a model and a validator with the Stage D person-mask filtering."""
    model = YOLO(str(weights))
    model.model.to(device).eval()
    if hasattr(model.model, "set_head_attr"):
        model.model.set_head_attr(max_det=args.max_det, agnostic_nms=False)
    val_args = make_val_args(args, weights, name)
    validator = YOLO26PSStageDMaskValidator(args=val_args, _callbacks=model.callbacks, decode_head="seg")
    validator.device = device
    validator.data = data
    validator.stride = max(int(model.model.stride.max()), 32)
    validator.init_metrics(model.model)
    return model, validator


def prepare_mask_batch(
    validator: YOLO26PSStageDMaskValidator,
    preds: list[dict[str, torch.Tensor]],
    batch: dict[str, Any],
    si: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None:
    """Return one image's person-mask GT and predictions in validator metric coordinates."""
    has_mask = batch.get("has_person_mask")
    if not (torch.is_tensor(has_mask) and si < has_mask.numel() and bool(has_mask[si])):
        return None
    pred = preds[si]
    keep = pred["cls"].long().eq(0)
    pred = {
        **pred,
        "bboxes": pred["bboxes"][keep],
        "conf": pred["conf"][keep],
        "cls": pred["cls"][keep] * 0,
        "masks": pred["masks"][keep],
    }
    local = dict(batch)
    local["batch_idx"] = batch["batch_idx"].clone()
    prepared = validator._prepare_batch(si, local)
    return prepared, pred


def best_iou_stats(gt: dict[str, torch.Tensor], pred: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Compute best-match box and mask IoU statistics for one image."""
    n_gt = int(gt["cls"].shape[0])
    n_pred = int(pred["cls"].shape[0])
    out: dict[str, Any] = {
        "gt": n_gt,
        "pred": n_pred,
        "box_best": [],
        "mask_best": [],
        "tp50": 0,
        "tp75": 0,
        "tp90": 0,
    }
    if n_gt == 0 or n_pred == 0:
        return out

    b_iou = box_iou(gt["bboxes"], pred["bboxes"]).detach().cpu()
    m_iou = mask_iou(gt["masks"].flatten(1), pred["masks"].flatten(1).float()).detach().cpu()
    out["box_best"] = b_iou.max(dim=1).values.tolist()
    out["mask_best"] = m_iou.max(dim=1).values.tolist()
    for threshold, key in ((0.50, "tp50"), (0.75, "tp75"), (0.90, "tp90")):
        candidates = torch.nonzero(m_iou >= threshold, as_tuple=False)
        if candidates.numel() == 0:
            continue
        scores = m_iou[candidates[:, 0], candidates[:, 1]]
        order = scores.argsort(descending=True)
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for idx in candidates[order]:
            g = int(idx[0])
            p = int(idx[1])
            if g in used_gt or p in used_pred:
                continue
            used_gt.add(g)
            used_pred.add(p)
        out[key] = len(used_gt)
    return out


def summarize(values: list[float]) -> dict[str, float]:
    """Return compact distribution stats."""
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """Aggregate per-image diagnostics."""
    mask_values = [x for row in rows for x in row[f"{prefix}_mask_best"]]
    box_values = [x for row in rows for x in row[f"{prefix}_box_best"]]
    gt_total = sum(int(row["gt"]) for row in rows)
    pred_total = sum(int(row[f"{prefix}_pred"]) for row in rows)
    return {
        "gt": gt_total,
        "pred": pred_total,
        "mask_best": summarize(mask_values),
        "box_best": summarize(box_values),
        "mask_recall50": sum(int(row[f"{prefix}_tp50"]) for row in rows) / max(gt_total, 1),
        "mask_recall75": sum(int(row[f"{prefix}_tp75"]) for row in rows) / max(gt_total, 1),
        "mask_recall90": sum(int(row[f"{prefix}_tp90"]) for row in rows) / max(gt_total, 1),
    }


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path, hard_topn: int = 0) -> None:
    """Write CSV and JSON reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = [
        "image",
        "gt",
        "student_pred",
        "reference_pred",
        "student_mask_mean",
        "reference_mask_mean",
        "mask_mean_gap",
        "student_box_mean",
        "reference_box_mean",
        "student_tp50",
        "student_tp75",
        "student_tp90",
        "reference_tp50",
        "reference_tp75",
        "reference_tp90",
    ]
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    if hard_topn > 0:
        hard = rows[:hard_topn]
        (out_dir / f"hard_images_top{len(hard)}.txt").write_text(
            "\n".join(str(row["image"]) for row in hard) + ("\n" if hard else ""),
            encoding="utf-8",
        )


@torch.no_grad()
def main() -> None:
    """Run the gap diagnosis."""
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() and torch.cuda.is_available() else args.device)
    data = check_det_dataset(str(args.data))
    data, filtered_path = apply_source_filter(args, data)
    student, student_validator = setup_model_and_validator(args.student, args, data, device, "student")
    reference, reference_validator = setup_model_and_validator(args.reference, args, data, device, "reference")
    dataloader = student_validator.get_dataloader(data.get(student_validator.args.split), args.batch)

    rows: list[dict[str, Any]] = []
    seen = 0
    for batch in dataloader:
        batch = student_validator.preprocess(batch)
        student_preds = student_validator.postprocess(student.model(batch["img"]))
        reference_preds = reference_validator.postprocess(reference.model(batch["img"]))
        for si, image in enumerate(batch["im_file"]):
            if seen >= args.max_images:
                break
            s_pair = prepare_mask_batch(student_validator, student_preds, batch, si)
            r_pair = prepare_mask_batch(reference_validator, reference_preds, batch, si)
            if s_pair is None or r_pair is None:
                continue
            gt_s, pred_s = s_pair
            gt_r, pred_r = r_pair
            s = best_iou_stats(gt_s, pred_s)
            r = best_iou_stats(gt_r, pred_r)
            student_mask_mean = mean(s["mask_best"]) if s["mask_best"] else 0.0
            reference_mask_mean = mean(r["mask_best"]) if r["mask_best"] else 0.0
            row = {
                "image": image,
                "gt": s["gt"],
                "student_pred": s["pred"],
                "reference_pred": r["pred"],
                "student_mask_best": s["mask_best"],
                "reference_mask_best": r["mask_best"],
                "student_box_best": s["box_best"],
                "reference_box_best": r["box_best"],
                "student_mask_mean": student_mask_mean,
                "reference_mask_mean": reference_mask_mean,
                "mask_mean_gap": reference_mask_mean - student_mask_mean,
                "student_box_mean": mean(s["box_best"]) if s["box_best"] else 0.0,
                "reference_box_mean": mean(r["box_best"]) if r["box_best"] else 0.0,
                "student_tp50": s["tp50"],
                "student_tp75": s["tp75"],
                "student_tp90": s["tp90"],
                "reference_tp50": r["tp50"],
                "reference_tp75": r["tp75"],
                "reference_tp90": r["tp90"],
            }
            rows.append(row)
            seen += 1
        if seen >= args.max_images:
            break

    rows.sort(key=lambda x: x["mask_mean_gap"], reverse=True)
    summary = {
        "student": str(args.student),
        "reference": str(args.reference),
        "split": args.split,
        "source_filter": args.source_filter,
        "filtered_path": str(filtered_path) if filtered_path else "",
        "images": len(rows),
        "student_summary": aggregate(rows, "student"),
        "reference_summary": aggregate(rows, "reference"),
        "largest_gaps": [
            {
                k: row[k]
                for k in (
                    "image",
                    "gt",
                    "student_pred",
                    "reference_pred",
                    "student_mask_mean",
                    "reference_mask_mean",
                    "mask_mean_gap",
                    "student_box_mean",
                    "reference_box_mean",
                )
            }
            for row in rows[:20]
        ],
    }
    write_outputs(rows, summary, args.out, args.hard_topn)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
