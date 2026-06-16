#!/usr/bin/env python3
"""Compare Stage D det-person and human_det-person detection metrics on the same val split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, nms, ops


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_d_person_mask_person_only.yaml"


def first_prediction_tensor(value):
    """Return the first batched prediction tensor from nested model outputs."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_prediction_tensor(item)
            if torch.is_tensor(found):
                return found
    return None


class DetPersonValidator(DetectionValidator):
    """Detection validator that unwraps adapter segment outputs and keeps the normal det/person branch."""

    def postprocess(self, preds):
        preds = first_prediction_tensor(preds)
        if preds is None:
            raise RuntimeError("Could not find detection prediction tensor in model outputs.")
        return super().postprocess(preds)


class HumanDetPersonValidator(DetectionValidator):
    """Detection validator that decodes and evaluates the YOLO26-PS human_det person branch."""

    def init_metrics(self, model):
        super().init_metrics(model)
        layers = getattr(model, "model", None)
        if hasattr(layers, "model"):
            layers = layers.model
        self.head = layers[-1]
        self.nc = 1
        self.names = {0: "Person"}
        self.metrics.names = self.names

    def postprocess(self, preds):
        raw = preds[1] if isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[1], dict) else None
        if raw is None and isinstance(preds, dict):
            raw = preds
        if raw is None or "human_boxes" not in raw or "human_scores" not in raw:
            raise RuntimeError("Checkpoint did not produce human_boxes/human_scores; cannot evaluate human_det.")
        boxes = self.head.decode_bboxes(self.head.dfl(raw["human_boxes"]), self.head.anchors.unsqueeze(0)) * self.head.strides
        pred = torch.cat((boxes, raw["human_scores"].sigmoid()), dim=1)
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
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5] * 0, "extra": x[:, 6:]} for x in outputs]

    def postprocess_raw(self, raw):
        """Postprocess a raw prediction dict from direct PyTorch forward."""
        if "human_boxes" not in raw and "one2many" in raw:
            raw = raw["one2many"]
        if "human_boxes" not in raw or "human_scores" not in raw:
            raise RuntimeError("Checkpoint did not produce human_boxes/human_scores; cannot evaluate human_det.")
        boxes = self.head.decode_bboxes(self.head.dfl(raw["human_boxes"]), self.head.anchors.unsqueeze(0)) * self.head.strides
        pred = torch.cat((boxes, raw["human_scores"].sigmoid()), dim=1)
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
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5] * 0, "extra": x[:, 6:]} for x in outputs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", default="576,768")
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name", default="det_vs_human_det")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    return parser.parse_args()


def normalize_imgsz(value: str) -> list[int] | int:
    parts = [int(x) for x in str(value).replace("x", ",").split(",") if x.strip()]
    return parts[0] if len(parts) == 1 else parts


def run_one(model: YOLO, args: argparse.Namespace, branch: str):
    head = model.model.model[-1]
    if hasattr(head, "set_active_tasks"):
        head.set_active_tasks({"det", "seg", "human_det"} if branch == "human_det" else {"det", "seg"})
    val_args = get_cfg(
        overrides={
            "model": str(args.weights),
            "data": str(args.data),
            "imgsz": normalize_imgsz(args.imgsz),
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "split": "val",
            "task": "detect",
            "mode": "val",
            "project": str(args.project),
            "name": f"{args.name}_{branch}",
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "val_samples": args.val_samples,
            "classes": [0],
            "verbose": False,
        }
    )
    validator_cls = HumanDetPersonValidator if branch == "human_det" else DetPersonValidator
    validator = validator_cls(args=val_args, _callbacks=model.callbacks)
    if branch == "human_det":
        return run_human_det_direct(model, validator, args)
    return validator(model=model.model)


@torch.no_grad()
def run_human_det_direct(model: YOLO, validator: HumanDetPersonValidator, args: argparse.Namespace):
    """Run human_det eval directly on the PyTorch model so raw dict outputs are preserved."""
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() and torch.cuda.is_available() else args.device)
    model.model.to(device).eval()
    validator.device = device
    validator.data = validator.data or __import__("ultralytics.data.utils", fromlist=["check_det_dataset"]).check_det_dataset(
        str(args.data)
    )
    validator.stride = max(int(model.model.stride.max()), 32)
    validator.dataloader = validator.dataloader or validator.get_dataloader(
        validator.data.get(validator.args.split), validator.args.batch
    )
    validator.init_metrics(model.model)
    validator.jdict = []
    for batch in validator.dataloader:
        batch = validator.preprocess(batch)
        outputs = model.model(batch["img"])
        raw = outputs[1] if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict) else outputs
        preds = validator.postprocess_raw(raw)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    validator.finalize_metrics()
    validator.print_results()
    return stats


def main() -> None:
    args = parse_args()
    rows = {}
    for branch in ("det", "human_det"):
        model = YOLO(str(args.weights))
        stats = run_one(model, args, branch)
        rows[branch] = stats
        LOGGER.info(
            "%s: P=%.6f R=%.6f mAP50=%.6f mAP50-95=%.6f",
            branch,
            stats.get("metrics/precision(B)", float("nan")),
            stats.get("metrics/recall(B)", float("nan")),
            stats.get("metrics/mAP50(B)", float("nan")),
            stats.get("metrics/mAP50-95(B)", float("nan")),
        )


if __name__ == "__main__":
    main()
