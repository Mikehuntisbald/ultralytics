#!/usr/bin/env python3
"""Evaluate YOLO26-PS Stage D checkpoints with standard instance-mask metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import LOGGER, nms, ops


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_d_person_mask_person_only.yaml"


class YOLO26PSStageDMaskValidator(SegmentationValidator):
    """Segmentation validator that unwraps YOLO26-PS adapter segment inference outputs."""

    def __init__(self, *args, decode_head: str = "seg", **kwargs):
        """Initialize the Stage D mask validator with an explicit decode branch."""
        super().__init__(*args, **kwargs)
        self.decode_head = normalize_decode_head(decode_head)

    def init_metrics(self, model):
        """Initialize segmentation metrics and keep a handle to the adapter head for raw human-det decoding."""
        super().init_metrics(model)
        layers = getattr(model, "model", None)
        if hasattr(layers, "model"):
            layers = layers.model
        self.head = layers[-1]
        self.nc = 1
        self.names = {0: "Person"}
        self.metrics.names = self.names

    def _raw_one2many(self, preds):
        """Return the raw one-to-many dict for branch-specific mask decoding."""
        raw = preds[1] if isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[1], dict) else None
        if raw is None and isinstance(preds, dict):
            raw = preds
        if isinstance(raw, dict) and "one2many" in raw:
            raw = raw["one2many"]
        return raw

    def _raw_prediction_tensor(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build an NMS input tensor from raw boxes/scores and mask coefficients."""
        use_human = self.decode_head == "human" and "human_boxes" in raw and "human_scores" in raw
        decode_raw = raw
        if use_human:
            decode_raw = {
                **raw,
                "boxes": raw["human_boxes"],
                "scores": raw["human_scores"][:, :1],
                "feats": raw.get("human_feats", raw.get("feats")),
            }
        self.head._get_decode_boxes(decode_raw)
        boxes = self.head.decode_bboxes(
            self.head.dfl(decode_raw["boxes"]), self.head.anchors.unsqueeze(0), xywh=False
        )
        boxes = (boxes * self.head.strides).permute(0, 2, 1)
        scores = decode_raw["scores"][:, :1].sigmoid().permute(0, 2, 1)
        mask_coef = raw["mask_coefficient"].permute(0, 2, 1)
        return torch.cat((ops.xyxy2xywh(boxes), scores, mask_coef), dim=2).permute(0, 2, 1)

    def postprocess(self, preds):
        """Decode boxes and aligned mask coefficients from raw one-to-many Stage D outputs."""
        raw = self._raw_one2many(preds)
        if not isinstance(raw, dict) or "mask_coefficient" not in raw:
            return super().postprocess(preds[0] if isinstance(preds, tuple) else preds)
        proto = raw.get("proto")
        if isinstance(proto, tuple):
            proto = proto[0]
        if proto is None:
            raise RuntimeError("Stage D mask validator requires proto outputs.")

        pred = self._raw_prediction_tensor(raw)
        outputs = nms.non_max_suppression(
            pred,
            self.args.conf,
            self.args.iou,
            nc=1,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=False,
        )
        imgsz = [4 * x for x in proto.shape[2:]]
        processed = []
        for i, x in enumerate(outputs):
            pred_i = {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5] * 0, "extra": x[:, 6:]}
            coefficient = pred_i["extra"]
            pred_i["masks"] = (
                self.process(proto[i], coefficient, pred_i["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred_i["bboxes"].device,
                )
            )
            processed.append(pred_i)
        return processed

    def _mask_image_enabled(self, batch: dict[str, Any], si: int) -> bool:
        """Return whether this validation image has person-mask supervision."""
        has_mask = batch.get("has_person_mask")
        return bool(torch.is_tensor(has_mask) and si < has_mask.numel() and bool(has_mask[si]))

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare only person instances with mask supervision for Stage D mask metrics."""
        prepared = super()._prepare_batch(si, batch)
        if prepared["cls"].numel() == 0:
            return prepared

        keep = prepared["cls"].long().eq(int(self.data.get("person_cls", 0)))
        flags = batch.get("instance_flags")
        idx = batch["batch_idx"] == si
        if torch.is_tensor(flags) and flags.numel():
            flags_i = flags[idx].to(keep.device).bool()
            if flags_i.ndim == 2 and flags_i.shape[0] == keep.shape[0] and flags_i.shape[1] > 3:
                keep &= flags_i[:, 3]
        prepared["cls"] = prepared["cls"][keep] * 0
        prepared["bboxes"] = prepared["bboxes"][keep]
        prepared["masks"] = prepared["masks"][keep]
        return prepared

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics only on images that carry person-mask labels."""
        mask_preds = []
        mask_indices = []
        for si, pred in enumerate(preds):
            if not self._mask_image_enabled(batch, si):
                continue
            keep = pred["cls"].long().eq(0)
            pred = {
                **pred,
                "bboxes": pred["bboxes"][keep],
                "conf": pred["conf"][keep],
                "cls": pred["cls"][keep] * 0,
                "masks": pred["masks"][keep],
            }
            mask_indices.append(si)
            mask_preds.append(pred)

        if not mask_preds:
            return
        local_batch = dict(batch)
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[:1] == (len(preds),):
                local_batch[key] = value[mask_indices]
            elif key in {"im_file", "ori_shape", "ratio_pad"} and isinstance(value, (list, tuple)):
                local_batch[key] = [value[i] for i in mask_indices]
        old_batch_idx = batch["batch_idx"]
        selected = torch.zeros_like(old_batch_idx, dtype=torch.bool)
        for old_i in mask_indices:
            selected |= old_batch_idx == old_i
        local_batch["batch_idx"] = old_batch_idx[selected].clone()
        for new_i, old_i in enumerate(mask_indices):
            local_batch["batch_idx"][local_batch["batch_idx"] == old_i] = new_i
        inst_count = old_batch_idx.shape[0]
        for key in ("cls", "bboxes", "segments", "keypoints", "body_kpts_3d", "instance_flags"):
            value = batch.get(key)
            if torch.is_tensor(value) and value.shape[:1] == (inst_count,):
                local_batch[key] = value[selected]
        super().update_metrics(mask_preds, local_batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", default="576,768")
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--name", default="yolo26ps_stage_d_mask_val")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--decode-head",
        choices=("human", "seg"),
        default="seg",
        help="for adapter heads, decode boxes/scores from frozen human_det or the original seg det head",
    )
    return parser.parse_args()


def normalize_imgsz(value: str | int) -> list[int] | int:
    if isinstance(value, int):
        return value
    parts = [int(x) for x in str(value).replace("x", ",").split(",") if x.strip()]
    return parts[0] if len(parts) == 1 else parts


def normalize_decode_head(value: Any) -> str:
    """Normalize person-mask decode head names."""
    head = str(value or "seg").strip().lower()
    if head in {"seg", "segment", "mask", "det", "default"}:
        return "seg"
    if head in {"human", "human_det", "person", "person_face"}:
        return "human"
    return "seg"


@torch.no_grad()
def run_direct(model: YOLO, validator: YOLO26PSStageDMaskValidator, args: argparse.Namespace):
    """Run validation directly on the PyTorch model so raw one-to-many outputs are not fused away."""
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() and torch.cuda.is_available() else args.device)
    model.model.to(device).eval()
    if hasattr(model.model, "set_head_attr"):
        model.model.set_head_attr(max_det=args.max_det, agnostic_nms=False)
    validator.device = device
    validator.data = check_det_dataset(str(args.data))
    validator.stride = max(int(model.model.stride.max()), 32)
    validator.dataloader = validator.get_dataloader(validator.data.get(validator.args.split), validator.args.batch)
    validator.init_metrics(model.model)
    validator.jdict = []
    for batch in validator.dataloader:
        batch = validator.preprocess(batch)
        preds = model.model(batch["img"])
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    validator.finalize_metrics()
    validator.print_results()
    return stats


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    val_args = get_cfg(
        overrides={
            "model": str(args.weights),
            "data": str(args.data),
            "imgsz": normalize_imgsz(args.imgsz),
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "split": "val",
            "task": "segment",
            "mode": "val",
            "project": str(args.project),
            "name": args.name,
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
    validator = YOLO26PSStageDMaskValidator(args=val_args, _callbacks=model.callbacks, decode_head=args.decode_head)
    stats = run_direct(model, validator, args)
    LOGGER.info(f"Mask mAP50: {stats.get('metrics/mAP50(M)', float('nan')):.6f}")
    LOGGER.info(f"Mask mAP50-95: {stats.get('metrics/mAP50-95(M)', float('nan')):.6f}")


if __name__ == "__main__":
    main()
