#!/usr/bin/env python3
"""Visualize OCHuman masks and compare pretrained/Stage-D checkpoints on the same images."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.utils.nms import TorchNMS

try:
    from pycocotools import mask as mask_utils
except Exception as exc:  # pragma: no cover - import guard for CLI use
    raise RuntimeError("pycocotools is required to decode OCHuman RLE masks") from exc


DEFAULT_MANIFEST = (
    Path("/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI/manifests/stage_d_val.jsonl")
)
DEFAULT_PRETRAIN = ROOT / "pretrains/yolo26s-seg.pt"
DEFAULT_LAST = ROOT / "runs/detect/yolo26ps_d_fullseghead_from_adapter_e12_s100k_b96_lr1e4_trend-2/weights/last.pt"
DEFAULT_OUT = ROOT / "examples/ochuman_pretrain_last_compare"

WHITE = (240, 240, 240)
GRAY = (170, 170, 170)
BLACK = (18, 18, 18)
PALETTE = (
    (60, 220, 255),
    (80, 255, 120),
    (255, 180, 70),
    (180, 120, 255),
    (255, 100, 180),
    (90, 180, 255),
    (220, 220, 80),
    (80, 210, 210),
    (80, 120, 255),
    (120, 255, 210),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pretrain", type=Path, default=DEFAULT_PRETRAIN)
    parser.add_argument("--last", type=Path, default=DEFAULT_LAST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[576, 768])
    parser.add_argument("--vis-conf", type=float, default=0.25)
    parser.add_argument("--metric-conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--last-decode-head",
        choices=("seg", "human"),
        default="seg",
        help="Stage-D branch used for boxes/scores when decoding last.pt masks.",
    )
    return parser.parse_args()


def normalize_imgsz(value: list[int]) -> int | list[int]:
    if len(value) == 1:
        return int(value[0])
    return [int(value[0]), int(value[1])]


def load_ochuman_records(manifest: Path, samples: int, seed: int) -> list[dict[str, Any]]:
    records = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("source") != "ochuman":
                continue
            if not any(inst.get("person_mask") for inst in record.get("instances", [])):
                continue
            image = Path(record["image"])
            if image.exists():
                records.append(record)
    if not records:
        raise RuntimeError(f"No OCHuman person-mask records found in {manifest}")
    rng = random.Random(seed)
    return rng.sample(records, min(samples, len(records)))


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    clean = dict(rle)
    if isinstance(clean.get("counts"), str):
        clean["counts"] = clean["counts"].encode("ascii")
    mask = mask_utils.decode(clean)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def decode_polygon(poly: list[Any], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = poly if poly and isinstance(poly[0], list) else [poly]
    for item in polygons:
        pts = np.asarray(item, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def gt_masks_and_boxes(record: dict[str, Any], shape: tuple[int, int]) -> tuple[list[np.ndarray], list[list[float]]]:
    height, width = shape
    masks: list[np.ndarray] = []
    boxes: list[list[float]] = []
    for inst in record.get("instances", []):
        if inst.get("category") != "person":
            continue
        raw_mask = inst.get("person_mask")
        if isinstance(raw_mask, dict):
            mask = decode_rle(raw_mask)
        elif isinstance(raw_mask, list):
            mask = decode_polygon(raw_mask, height, width)
        else:
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(mask)
        if inst.get("bbox") is not None:
            boxes.append([float(x) for x in inst["bbox"][:4]])
    return masks, boxes


def nms_class_aware(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, iou: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    max_coord = boxes.max()
    return TorchNMS.nms(boxes + classes.to(boxes).view(-1, 1) * (max_coord + 1), scores, iou)


class PredictorBase:
    label: str

    def predict(
        self, image: np.ndarray, image_path: Path, conf: float
    ) -> tuple[list[np.ndarray], list[list[float]], list[float]]:
        raise NotImplementedError


class PretrainPredictor(PredictorBase):
    def __init__(self, weights: Path, imgsz: int | list[int], iou: float, device: str):
        self.weights = weights
        self.imgsz = imgsz
        self.iou = iou
        self.device = device
        self.label = weights.name
        self.model = YOLO(str(weights))

    def predict(
        self, image: np.ndarray, image_path: Path, conf: float
    ) -> tuple[list[np.ndarray], list[list[float]], list[float]]:
        height, width = image.shape[:2]
        results = self.model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            conf=conf,
            iou=self.iou,
            classes=[0],
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return [], [], []

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        keep = cls == 0
        boxes = boxes_xyxy[keep].tolist()
        confs = scores[keep].tolist()

        masks: list[np.ndarray] = []
        if result.masks is not None:
            raw_masks = result.masks.data.detach().cpu().numpy()[keep]
            for raw in raw_masks:
                mask = raw > 0.5
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(
                        bool
                    )
                masks.append(mask)
        return masks, boxes, [float(x) for x in confs]


class StageDPredictor(PredictorBase):
    def __init__(
        self,
        weights: Path,
        imgsz: int | list[int],
        iou: float,
        max_det: int,
        device: str,
        decode_head: str,
    ):
        self.weights = weights
        self.imgsz = [imgsz, imgsz] if isinstance(imgsz, int) else list(imgsz)
        self.iou = iou
        self.max_det = max_det
        self.device_arg = device
        self.decode_head = decode_head
        self.label = f"{weights.name}:{decode_head}"
        self.model = YOLO(str(weights))
        self.head = self.model.model.model[-1]
        if hasattr(self.head, "set_active_tasks"):
            self.head.set_active_tasks({"det", "seg"})
        if hasattr(self.head, "person_mask_decode_head"):
            self.head.person_mask_decode_head = decode_head
        else:
            setattr(self.head, "person_mask_decode_head", decode_head)
        setattr(self.head, "use_human_det_branch", decode_head == "human")
        self.head.max_det = max_det
        self.device = torch.device(f"cuda:{device}" if str(device).isdigit() and torch.cuda.is_available() else device)
        self.model.model.to(self.device).eval()
        self.predictor = DetectionPredictor(
            overrides={
                "model": str(weights),
                "imgsz": self.imgsz,
                "conf": 0.001,
                "iou": iou,
                "max_det": max_det,
                "save": False,
                "verbose": False,
                "device": device,
            }
        )
        self.predictor.device = self.device
        self.predictor.imgsz = self.imgsz
        self.predictor.model = SimpleNamespace(
            fp16=False,
            stride=self.model.model.stride,
            input_stride=getattr(self.model.model, "input_stride", self.model.model.stride),
            format="pt",
            dynamic=False,
        )

    @torch.no_grad()
    def predict(
        self, image: np.ndarray, image_path: Path, conf: float
    ) -> tuple[list[np.ndarray], list[list[float]], list[float]]:
        height, width = image.shape[:2]
        if hasattr(self.head, "set_active_tasks"):
            self.head.set_active_tasks({"det", "seg"})
        self.head.max_det = self.max_det
        self.head.person_mask_decode_head = self.decode_head
        setattr(self.head, "use_human_det_branch", self.decode_head == "human")

        self.predictor.batch_ratio_pad = None
        im = self.predictor.preprocess([image])
        ratio_pad = self.predictor.batch_ratio_pad[0] if getattr(self.predictor, "batch_ratio_pad", None) else None
        raw = self.model.model(im)
        deploy = raw
        if (
            isinstance(raw, (tuple, list))
            and len(raw) == 2
            and isinstance(raw[0], (tuple, list))
            and len(raw[0]) >= 5
        ):
            deploy = raw[0]
        if (
            isinstance(raw, (tuple, list))
            and len(raw) == 2
            and isinstance(raw[0], (tuple, list))
            and len(raw[0]) == 2
            and torch.is_tensor(raw[0][0])
            and torch.is_tensor(raw[0][1])
        ):
            deploy = self._segment_to_deploy(raw[0][0], raw[0][1])
        if not (isinstance(deploy, (tuple, list)) and len(deploy) >= 5):
            raise RuntimeError(f"Unexpected Stage-D output type: {type(raw)}")
        det_out, _pose25d, mask_coef, proto, _scene = deploy[:5]
        det = det_out[0]
        coef = mask_coef[0]
        classes = det[:, 5].round().long()
        keep = (det[:, 4] >= conf) & classes.eq(0)
        boxes = det[keep, :4].clone()
        scores = det[keep, 4].clone()
        classes = classes[keep].clone()
        coef = coef[keep].clone()

        if boxes.numel():
            h_in, w_in = im.shape[2:]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w_in)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h_in)
            valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes, scores, classes, coef = boxes[valid], scores[valid], classes[valid], coef[valid]
        if boxes.numel():
            keep_idx = nms_class_aware(boxes, scores, classes, self.iou)
            keep_idx = keep_idx[scores[keep_idx].argsort(descending=True)][: self.max_det]
            boxes, scores, coef = boxes[keep_idx], scores[keep_idx], coef[keep_idx]

        if not boxes.numel():
            return [], [], []

        boxes_orig = ops.scale_boxes(im.shape[2:], boxes.clone(), (height, width), ratio_pad=ratio_pad).detach().cpu()
        masks = []
        if torch.is_tensor(proto) and proto.shape[-1] > 0 and proto.shape[-2] > 0:
            proto_tensor = proto[0] if not isinstance(proto, (tuple, list)) else proto[0][0]
            masks_in = ops.process_mask(proto_tensor, coef.float(), boxes, im.shape[2:], upsample=True)
            masks_scaled = ops.scale_masks(masks_in[:, None].float(), (height, width), ratio_pad=ratio_pad)[:, 0]
            masks = [(m.detach().cpu().numpy() > 0.5) for m in masks_scaled]
        return masks, boxes_orig.numpy().tolist(), [float(x) for x in scores.detach().cpu().tolist()]

    def _segment_to_deploy(
        self, pred: torch.Tensor, proto: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(proto, (tuple, list)):
            proto = proto[0]
        nm = int(proto.shape[1]) if torch.is_tensor(proto) and proto.ndim == 4 else 32
        if pred.shape[-1] == 6 + nm:
            boxes = pred[..., :4]
            scores = pred[..., 4:5]
            cls = pred[..., 5:6]
            coeff = pred[..., 6:]
        else:
            pred = pred.permute(0, 2, 1).contiguous()
            nc = pred.shape[-1] - 4 - nm
            boxes = ops.xywh2xyxy(pred[..., :4])
            scores, cls = pred[..., 4 : 4 + nc].max(dim=-1, keepdim=True)
            coeff = pred[..., 4 + nc :]
        det = torch.cat([boxes, scores, cls], dim=-1)
        pose25d = pred.new_zeros((pred.shape[0], pred.shape[1], 17, 4))
        scene_seg = pred.new_zeros((pred.shape[0], 0, 0, 0))
        return det, pose25d, coeff, proto, scene_seg


def best_iou_summary(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> dict[str, Any]:
    if not gt_masks or not pred_masks:
        return {"mean_best_gt_iou": 0.0, "best_gt_iou": [0.0 for _ in gt_masks]}
    best = []
    for gt in gt_masks:
        scores = []
        for pred in pred_masks:
            inter = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()
            scores.append(float(inter / union) if union else 0.0)
        best.append(max(scores) if scores else 0.0)
    return {"mean_best_gt_iou": float(np.mean(best)) if best else 0.0, "best_gt_iou": [float(x) for x in best]}


def mask_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter / union) if union else 0.0


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def ap_for_threshold(records: list[dict[str, Any]], model_key: str, iou_thr: float) -> float:
    gt_total = sum(len(r["gt_masks"]) for r in records)
    if gt_total == 0:
        return 0.0
    preds = []
    matched = {}
    for image_i, record in enumerate(records):
        for pred_i, score in enumerate(record["predictions"][model_key]["scores"]):
            preds.append((float(score), image_i, pred_i))
        matched[image_i] = np.zeros(len(record["gt_masks"]), dtype=bool)
    preds.sort(key=lambda x: x[0], reverse=True)
    tp = np.zeros(len(preds), dtype=np.float32)
    fp = np.zeros(len(preds), dtype=np.float32)
    for i, (_score, image_i, pred_i) in enumerate(preds):
        gt_masks = records[image_i]["gt_masks"]
        pred_mask = records[image_i]["predictions"][model_key]["masks"][pred_i]
        if not gt_masks:
            fp[i] = 1.0
            continue
        ious = np.array([mask_iou(gt, pred_mask) for gt in gt_masks], dtype=np.float32)
        best = int(ious.argmax()) if ious.size else -1
        if best >= 0 and ious[best] >= iou_thr and not matched[image_i][best]:
            tp[i] = 1.0
            matched[image_i][best] = True
        else:
            fp[i] = 1.0
    if len(preds) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(gt_total, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
    return voc_ap(recall, precision)


def evaluate(records: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    ap_by_thr = {f"{thr:.2f}": ap_for_threshold(records, model_key, thr) for thr in thresholds}
    gt_total = sum(len(r["gt_masks"]) for r in records)
    pred_total = sum(len(r["predictions"][model_key]["masks"]) for r in records)
    best_ious = []
    for record in records:
        pred_masks = record["predictions"][model_key]["masks"]
        for gt in record["gt_masks"]:
            best_ious.append(max([mask_iou(gt, pred) for pred in pred_masks], default=0.0))
    return {
        "images": len(records),
        "gt_instances": gt_total,
        "pred_instances": pred_total,
        "mask_mAP50": ap_by_thr["0.50"],
        "mask_mAP50_95": float(np.mean(list(ap_by_thr.values()))),
        "ap_by_iou": ap_by_thr,
        "mean_best_gt_iou": float(np.mean(best_ious)) if best_ious else 0.0,
    }


def draw_boxes(image: np.ndarray, boxes: list[list[float]], colors: list[tuple[int, int, int]]) -> None:
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, str(i), (x1 + 3, max(16, y1 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def overlay_instances(
    image: np.ndarray,
    masks: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    alpha: float = 0.42,
) -> np.ndarray:
    out = image.copy()
    for i, mask in enumerate(masks):
        if not mask.any():
            continue
        color = colors[i % len(colors)]
        out[mask] = (out[mask].astype(np.float32) * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(
            np.uint8
        )
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def overlay_gt_pred_instances(
    image: np.ndarray,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
) -> np.ndarray:
    out = image.copy()
    for i, mask in enumerate(gt_masks):
        color = PALETTE[i % len(PALETTE)]
        if mask.any():
            out[mask] = (out[mask].astype(np.float32) * 0.62 + np.array(color, dtype=np.float32) * 0.38).astype(
                np.uint8
            )
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 2)
    for i, mask in enumerate(pred_masks):
        color = PALETTE[(i + 5) % len(PALETTE)]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 3)
    return out


def fit_panel(image: np.ndarray, panel_w: int = 360, panel_h: int = 300) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(panel_w / width, panel_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_h, panel_w, 3), 28, dtype=np.uint8)
    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def labeled_panel(image: np.ndarray, title: str, subtitle: str, panel_w: int = 360, panel_h: int = 300) -> np.ndarray:
    header = np.full((50, panel_w, 3), BLACK, dtype=np.uint8)
    cv2.putText(header, title[:56], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
    cv2.putText(header, subtitle[:68], (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1, cv2.LINE_AA)
    return np.concatenate([header, fit_panel(image, panel_w, panel_h)], axis=0)


def make_visual(
    image: np.ndarray,
    image_name: str,
    gt_masks: list[np.ndarray],
    gt_boxes: list[list[float]],
    model_outputs: dict[str, dict[str, Any]],
) -> np.ndarray:
    gt_colors = [PALETTE[i % len(PALETTE)] for i in range(len(gt_masks))]
    gt_view = overlay_instances(image, gt_masks, gt_colors)
    draw_boxes(gt_view, gt_boxes, gt_colors)
    panels = [labeled_panel(gt_view, f"GT OCHuman | {image_name}", f"instances={len(gt_masks)}  each color=instance")]

    for label, output in model_outputs.items():
        pred_masks = output["masks"]
        pred_boxes = output["boxes"]
        pred_scores = output["scores"]
        pred_colors = [PALETTE[(i + 3) % len(PALETTE)] for i in range(len(pred_masks))]
        pred_view = overlay_instances(image, pred_masks, pred_colors)
        draw_boxes(pred_view, pred_boxes, pred_colors)
        conf_text = ",".join(f"{x:.2f}" for x in pred_scores[:4])
        if len(pred_scores) > 4:
            conf_text += ",..."
        iou_info = best_iou_summary(gt_masks, pred_masks)
        panels.append(
            labeled_panel(
                pred_view,
                label,
                f"pred={len(pred_masks)} bestIoU={iou_info['mean_best_gt_iou']:.3f} conf={conf_text or 'none'}",
            )
        )

    return np.concatenate(panels, axis=1)


def json_safe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record in records:
        item = {
            "image": record["image"],
            "gt_instances": len(record["gt_masks"]),
            "predictions": {},
        }
        for key, pred in record["predictions"].items():
            item["predictions"][key] = {
                "pred_instances": len(pred["masks"]),
                "pred_conf": pred["scores"],
                **best_iou_summary(record["gt_masks"], pred["masks"]),
            }
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    args.imgsz = normalize_imgsz(args.imgsz)
    args.out.mkdir(parents=True, exist_ok=True)

    predictors: dict[str, PredictorBase] = {
        "pretrain": PretrainPredictor(args.pretrain, args.imgsz, args.iou, args.device),
        "last": StageDPredictor(args.last, args.imgsz, args.iou, args.max_det, args.device, args.last_decode_head),
    }
    records = load_ochuman_records(args.manifest, args.samples, args.seed)

    eval_records: list[dict[str, Any]] = []
    rows = []
    for index, record in enumerate(records):
        image_path = Path(record["image"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        gt_masks, gt_boxes = gt_masks_and_boxes(record, image.shape[:2])
        eval_record: dict[str, Any] = {"image": str(image_path), "gt_masks": gt_masks, "predictions": {}}
        vis_outputs: dict[str, dict[str, Any]] = {}
        for key, predictor in predictors.items():
            metric_masks, metric_boxes, metric_scores = predictor.predict(image, image_path, args.metric_conf)
            eval_record["predictions"][key] = {
                "masks": metric_masks,
                "boxes": metric_boxes,
                "scores": metric_scores,
            }
            vis_masks, vis_boxes, vis_scores = predictor.predict(image, image_path, args.vis_conf)
            vis_outputs[key] = {
                "masks": vis_masks,
                "boxes": vis_boxes,
                "scores": vis_scores,
            }
        visual = make_visual(image, image_path.name, gt_masks, gt_boxes, vis_outputs)
        out_file = args.out / f"{index:02d}_{image_path.stem}.jpg"
        cv2.imwrite(str(out_file), visual)
        rows.append(visual)
        eval_record["visual"] = str(out_file)
        eval_records.append(eval_record)

    contact = np.concatenate(rows, axis=0)
    contact_path = args.out / "contact_sheet.jpg"
    cv2.imwrite(str(contact_path), contact)

    metrics = {key: evaluate(eval_records, key) for key in predictors}
    summary = {
        "manifest": str(args.manifest),
        "pretrain": str(args.pretrain),
        "last": str(args.last),
        "device": args.device,
        "imgsz": args.imgsz,
        "vis_conf": args.vis_conf,
        "metric_conf": args.metric_conf,
        "iou_nms": args.iou,
        "max_det": args.max_det,
        "last_decode_head": args.last_decode_head,
        "contact_sheet": str(contact_path),
        "metrics_sample": metrics,
        "samples": json_safe_records(eval_records),
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "contact_sheet": str(contact_path), "summary": str(summary_path)}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
