#!/usr/bin/env python3
"""Visualize YOLO26-PS stage checkpoints with source-aware overlays.

The YOLO26-PS head can emit detection boxes, 2.5D pose, person masks, and scene
logits at the same time. The default detect predictor only plots boxes, so this
tool keeps the raw head outputs and chooses overlays that match each dataset:

- Objects365 and other detection guard sources: all-class boxes.
- Person mask sources: person boxes plus mask overlays.
- Pose / 3D pose sources: person boxes plus 2.5D skeletons, with limb color
  driven by relative z.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.nms import TorchNMS


DEFAULT_WEIGHTS = ROOT / "runs/detect/yolo26ps_stage_d_person_mask_maskonly_b28_valfix_continue/weights/best.pt"
DEFAULT_MANIFEST = Path(
    "/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI/manifests/stage_d_smoke_val.jsonl"
)
DEFAULT_VAL_TXT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI/stage_d_val.txt")
DEFAULT_OUT = ROOT / "examples"

SOURCE_ORDER = (
    "coco_person_mask",
    "ochuman",
    "coco_wholebody",
    "3dpw",
    "agora",
    "objects365",
    "crowdhuman",
    "wider_face",
)
POSE_SOURCES = {"coco_wholebody", "ochuman", "3dpw", "agora"}
MASK_SOURCES = {"coco_person_mask", "ochuman"}
ALL_CLASS_BOX_SOURCES = {"objects365"}
PERSON_BOX_SOURCES = {"coco_person_mask", "ochuman", "coco_wholebody", "3dpw", "agora", "crowdhuman"}
FACE_BOX_SOURCES = {"wider_face"}
PERSON_CLS = 0
FACE_CLS = 365
COCO17_SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)
BOX_PALETTE = (
    (60, 220, 255),
    (80, 255, 120),
    (255, 180, 70),
    (180, 120, 255),
    (255, 100, 180),
    (90, 180, 255),
    (220, 220, 80),
    (80, 210, 210),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO26-PS checkpoint to visualize")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Unified JSONL manifest for task sources")
    parser.add_argument("--val-txt", type=Path, default=DEFAULT_VAL_TXT, help="Stage list used to sample det guard sources")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Directory under which a timestamped run is saved")
    parser.add_argument("--name", default="", help="Optional output subdirectory name")
    parser.add_argument("--sources", default=",".join(SOURCE_ORDER), help="Comma-separated source list to sample")
    parser.add_argument("--samples-per-source", type=int, default=3, help="Images to sample per source")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[448, 768], help="Inference image size, H W or square")
    parser.add_argument("--conf", type=float, default=0.08, help="Confidence threshold for visualization")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Top-k detections kept by model head")
    parser.add_argument("--max-vis", type=int, default=30, help="Maximum boxes drawn per image")
    parser.add_argument("--pose-conf", type=float, default=0.20, help="Minimum keypoint confidence for skeleton drawing")
    parser.add_argument(
        "--face-branch",
        choices=("det", "human_det"),
        default="det",
        help="Detector branch used for WIDER_FACE/face-only visualization",
    )
    parser.add_argument(
        "--z-display",
        choices=("raw", "norm"),
        default="raw",
        help="Display raw relative z by undoing deploy bbox-height normalization, or normalized deploy z.",
    )
    parser.add_argument("--seed", type=int, default=20260529, help="Deterministic sampling seed")
    parser.add_argument("--device", default="0", help="Ultralytics device string")
    return parser.parse_args()


def normalize_imgsz(value: list[int]) -> list[int]:
    """Normalize CLI image size to [height, width]."""
    if len(value) == 1:
        return [int(value[0]), int(value[0])]
    return [int(value[0]), int(value[1])]


def canonical_source(source: str) -> str:
    """Normalize source names."""
    return str(source or "").strip().lower().replace("-", "_").replace(" ", "_")


def source_from_path(path: str) -> str:
    """Infer a source from an image path."""
    text = path.lower().replace("-", "_")
    if "objects365" in text or "objects_365" in text:
        return "objects365"
    if "crowdhuman" in text or "crowd_human" in text:
        return "crowdhuman"
    if "wider_face" in text or "widerface" in text:
        return "wider_face"
    if "ochuman" in text:
        return "ochuman"
    if "3dpw" in text:
        return "3dpw"
    if "agora" in text:
        return "agora"
    if "coco_2017" in text or "/coco/" in text:
        return "coco_wholebody"
    return "unknown"


def pick_samples(paths: list[str], n: int, rng: random.Random) -> list[str]:
    """Pick existing paths with deterministic shuffling."""
    valid = [p for p in dict.fromkeys(paths) if p and Path(p).exists()]
    if len(valid) <= n:
        return valid
    rng.shuffle(valid)
    return valid[:n]


def load_samples(args: argparse.Namespace, sources: list[str]) -> list[dict[str, str]]:
    """Sample images from manifest and det guard text lists."""
    by_source: dict[str, list[str]] = defaultdict(list)
    if args.manifest.exists():
        with args.manifest.open() as f:
            for line in f:
                record = json.loads(line)
                source = canonical_source(record.get("source"))
                image = str(record.get("image") or "")
                if source:
                    by_source[source].append(image)
    if args.val_txt.exists():
        with args.val_txt.open() as f:
            for line in f:
                image = line.strip()
                source = source_from_path(image)
                if source:
                    by_source[source].append(image)

    rng = random.Random(args.seed)
    samples: list[dict[str, str]] = []
    for source in sources:
        for image in pick_samples(by_source.get(source, []), args.samples_per_source, rng):
            samples.append({"source": source, "image": image})
    return samples


def class_name(names: Any, cls_id: int) -> str:
    """Return a display name for a class id."""
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    return str(names[cls_id]) if 0 <= cls_id < len(names) else str(cls_id)


def nms_class_aware(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, iou: float) -> torch.Tensor:
    """Class-aware NMS with local TorchNMS."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    max_coord = boxes.max()
    return TorchNMS.nms(boxes + classes.to(boxes).view(-1, 1) * (max_coord + 1), scores, iou)


def z_to_color(z_value: float, z_abs_max: float = 0.8) -> tuple[int, int, int]:
    """Map relative z to BGR: near = warm, far = cool."""
    t = float(np.clip((z_value + z_abs_max) / (2 * z_abs_max), 0.0, 1.0))
    # BGR interpolation: far blue/cyan -> near red/orange.
    far = np.array([255, 170, 40], dtype=np.float32)
    near = np.array([40, 80, 255], dtype=np.float32)
    color = (1.0 - t) * far + t * near
    return tuple(int(x) for x in color)


def put_label(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.5) -> None:
    """Draw a filled text label."""
    x, y = xy
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 6), (min(x + tw + 7, img.shape[1] - 1), y + 3), color, -1)
    cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (18, 18, 18), 1, cv2.LINE_AA)


def draw_depth_legend(img: np.ndarray) -> None:
    """Draw a small z color legend."""
    h, w = img.shape[:2]
    x0, y0 = max(8, w - 160), max(48, h - 34)
    for i in range(90):
        z = -0.8 + 1.6 * i / 89
        cv2.line(img, (x0 + i, y0), (x0 + i, y0 + 10), z_to_color(z), 1)
    cv2.putText(img, "far", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (250, 250, 250), 1, cv2.LINE_AA)
    cv2.putText(img, "near", (x0 + 58, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (250, 250, 250), 1, cv2.LINE_AA)


def decode_pose_to_image(
    pose_norm: torch.Tensor,
    boxes_orig: torch.Tensor,
    image_shape: tuple[int, int],
    boxes_model: torch.Tensor | None = None,
    z_display: str = "raw",
) -> torch.Tensor:
    """Convert bbox-normalized [x, y, z, conf] pose to image xy and diagnostic z."""
    if pose_norm.numel() == 0:
        return torch.empty((0, 17, 4), dtype=torch.float32)
    pose = pose_norm.detach().cpu().float().clone()
    boxes = boxes_orig.detach().cpu().float()
    x1, y1, x2, y2 = [x.view(-1, 1) for x in boxes.T]
    bw = (x2 - x1).clamp(min=1.0)
    bh = (y2 - y1).clamp(min=1.0)
    pose[..., 0] = x1 + pose[..., 0] * bw
    pose[..., 1] = y1 + pose[..., 1] * bh
    pose[..., 0].clamp_(0, image_shape[1] - 1)
    pose[..., 1].clamp_(0, image_shape[0] - 1)
    if z_display == "raw" and boxes_model is not None and boxes_model.numel():
        # Deployment emits z normalized by the model-input bbox height. Restore the
        # raw root-relative training target for visual diagnostics.
        box_h_model = (boxes_model.detach().cpu().float()[:, 3] - boxes_model.detach().cpu().float()[:, 1]).clamp(min=1.0)
        pose[..., 2] *= box_h_model.view(-1, 1)
    return pose


def draw_skeleton_25d(
    img: np.ndarray, pose: torch.Tensor, box: torch.Tensor, pose_conf: float, instance_index: int
) -> dict[str, Any]:
    """Draw one 2.5D skeleton and return summary stats."""
    kpts = pose.numpy()
    visible = kpts[:, 3] >= pose_conf
    z_values = kpts[visible, 2]
    for a, b in COCO17_SKELETON:
        if not (visible[a] and visible[b]):
            continue
        p1 = (int(round(kpts[a, 0])), int(round(kpts[a, 1])))
        p2 = (int(round(kpts[b, 0])), int(round(kpts[b, 1])))
        z_mid = float((kpts[a, 2] + kpts[b, 2]) * 0.5)
        cv2.line(img, p1, p2, z_to_color(z_mid), 3, cv2.LINE_AA)
    for i, (x, y, z, conf) in enumerate(kpts):
        if conf < pose_conf:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 3, z_to_color(float(z)), -1, cv2.LINE_AA)
        if i in {0, 5, 6, 11, 12}:
            cv2.putText(
                img,
                f"{z:+.2f}",
                (int(round(x)) + 4, int(round(y)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
    x1, y1, _x2, y2 = [int(round(float(v))) for v in box]
    mean_z = float(z_values.mean()) if len(z_values) else 0.0
    put_label(img, f"pose{instance_index} z={mean_z:+.2f} k={int(visible.sum())}", (x1, min(y2 + 18, img.shape[0] - 4)), (45, 45, 45), 0.45)
    return {
        "visible_keypoints": int(visible.sum()),
        "mean_z": round(mean_z, 5),
        "min_z": round(float(z_values.min()), 5) if len(z_values) else 0.0,
        "max_z": round(float(z_values.max()), 5) if len(z_values) else 0.0,
    }


def selected_classes_for_source(source: str) -> set[int] | None:
    """Return class filter for source-specific visualization."""
    if source in ALL_CLASS_BOX_SOURCES:
        return None
    if source in FACE_BOX_SOURCES:
        return {FACE_CLS}
    if source in PERSON_BOX_SOURCES:
        return {PERSON_CLS}
    return {PERSON_CLS}


def human_det_available(head: Any) -> bool:
    """Return whether this checkpoint has the human detector branch needed for human_det deploy."""
    if head is None:
        return False
    if bool(getattr(head, "end2end", False)):
        return getattr(head, "one2one_human_cv2", None) is not None and getattr(head, "one2one_human_cv3", None) is not None
    return getattr(head, "human_cv2", None) is not None and getattr(head, "human_cv3", None) is not None


def active_tasks_for_source(source: str, args: argparse.Namespace, head: Any | None = None) -> set[str]:
    """Return head branches needed for source-aware deploy outputs."""
    if source in POSE_SOURCES:
        tasks = {"human_det" if human_det_available(head) else "det", "pose"}
        if source in MASK_SOURCES:
            tasks.add("mask")
        return tasks
    if source in FACE_BOX_SOURCES and args.face_branch == "human_det" and human_det_available(head):
        # Include pose so the multi-task head returns the full deployment tuple;
        # face-class predictions still come from the human detector branch.
        return {"human_det", "pose"}
    tasks = {"det"}
    if source in MASK_SOURCES:
        tasks.add("mask")
    return tasks


def run_inference(
    model: YOLO,
    predictor: Any,
    image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[Any, Any], torch.Tensor]:
    """Run model and return raw deploy outputs plus preprocessing metadata."""
    predictor.batch_ratio_pad = None
    im = predictor.preprocess([image])
    ratio_pad = predictor.batch_ratio_pad[0] if getattr(predictor, "batch_ratio_pad", None) else None
    with torch.no_grad():
        raw = model.model(im)
    deploy = raw[0] if isinstance(raw, (tuple, list)) else raw
    if not (isinstance(deploy, (tuple, list)) and len(deploy) >= 5):
        raise RuntimeError(f"Unexpected YOLO26-PS output type: {type(raw)}")
    return deploy[:5], ratio_pad, im


def prepare_predictions(
    deploy: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    im: torch.Tensor,
    image_shape: tuple[int, int],
    ratio_pad: tuple[Any, Any],
    source: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Filter, NMS, and scale raw predictions."""
    det_out, pose25d, mask_coef, proto, _scene = deploy
    det = det_out[0]
    pose = pose25d[0]
    coef = mask_coef[0]
    classes = det[:, 5].round().long()
    keep = det[:, 4] >= args.conf
    selected = selected_classes_for_source(source)
    if selected is not None:
        cls_keep = torch.zeros_like(keep, dtype=torch.bool)
        for cls_id in selected:
            cls_keep |= classes == cls_id
        keep &= cls_keep

    boxes = det[keep, :4].clone()
    scores = det[keep, 4].clone()
    classes = classes[keep].clone()
    coef = coef[keep].clone()
    pose = pose[keep].clone()

    if boxes.numel():
        h_in, w_in = im.shape[2:]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w_in)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h_in)
        valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        boxes, scores, classes, coef, pose = boxes[valid], scores[valid], classes[valid], coef[valid], pose[valid]
    if boxes.numel():
        keep_idx = nms_class_aware(boxes, scores, classes, args.iou)
        keep_idx = keep_idx[scores[keep_idx].argsort(descending=True)][: args.max_vis]
        boxes, scores, classes, coef, pose = boxes[keep_idx], scores[keep_idx], classes[keep_idx], coef[keep_idx], pose[keep_idx]

    boxes_orig = torch.empty((0, 4))
    masks_orig = torch.empty((0, image_shape[0], image_shape[1]), dtype=torch.bool)
    pose_orig = torch.empty((0, 17, 4), dtype=torch.float32)
    if boxes.numel():
        boxes_orig = ops.scale_boxes(im.shape[2:], boxes.clone(), image_shape, ratio_pad=ratio_pad).detach().cpu()
        pose_orig = decode_pose_to_image(pose, boxes_orig, image_shape, boxes, getattr(args, "z_display", "raw"))
        masks_orig = torch.zeros((boxes.shape[0], image_shape[0], image_shape[1]), dtype=torch.bool)
        person = classes == PERSON_CLS
        if person.any() and proto.shape[-1] > 0 and proto.shape[-2] > 0:
            masks_in = ops.process_mask(proto[0], coef[person], boxes[person], im.shape[2:], upsample=True)
            masks_scaled = ops.scale_masks(masks_in[:, None].float(), image_shape, ratio_pad=ratio_pad)[:, 0]
            masks_orig[person.detach().cpu()] = masks_scaled.detach().cpu() > 0.5

    return {
        "boxes": boxes_orig,
        "scores": scores.detach().cpu(),
        "classes": classes.detach().cpu(),
        "masks": masks_orig,
        "pose": pose_orig,
    }


def draw_predictions(
    image: np.ndarray,
    preds: dict[str, Any],
    source: str,
    names: Any,
    args: argparse.Namespace,
    image_path: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Draw source-aware overlays."""
    vis = image.copy()
    detections: list[dict[str, Any]] = []
    draw_masks = source in MASK_SOURCES
    draw_pose = source in POSE_SOURCES
    draw_all = source in ALL_CLASS_BOX_SOURCES
    draw_depth_legend_flag = False

    for i, (box, score, cls_id) in enumerate(zip(preds["boxes"], preds["scores"], preds["classes"])):
        cls_i = int(cls_id)
        color = BOX_PALETTE[(cls_i if draw_all else i) % len(BOX_PALETTE)]
        mask_pixels = 0
        if draw_masks and cls_i == PERSON_CLS and i < len(preds["masks"]) and preds["masks"][i].any():
            mask_np = preds["masks"][i].numpy()
            mask_pixels = int(mask_np.sum())
            vis[mask_np] = (0.55 * vis[mask_np] + 0.45 * np.array(color, dtype=np.float32)).astype(np.uint8)

        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name(names, cls_i)} {float(score):.2f}"
        put_label(vis, label, (x1, y1 - 4), color, 0.48)

        pose_stats: dict[str, Any] | None = None
        if draw_pose and cls_i == PERSON_CLS and i < len(preds["pose"]):
            pose_stats = draw_skeleton_25d(vis, preds["pose"][i], box, args.pose_conf, i)
            draw_depth_legend_flag = True

        detections.append(
            {
                "class": cls_i,
                "name": class_name(names, cls_i),
                "conf": round(float(score), 5),
                "box_xyxy": [round(float(v), 2) for v in box.tolist()],
                "mask_pixels": mask_pixels,
                "pose25d": pose_stats,
            }
        )

    if draw_depth_legend_flag:
        draw_depth_legend(vis)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (25, 25, 25), -1)
    layers = ["boxes"]
    if draw_masks:
        layers.append("person-mask")
    if draw_pose:
        layers.append("2.5d-pose")
    title = f"{source} | {Path(image_path).name} | {','.join(layers)} | dets={len(detections)}"
    cv2.putText(vis, title[:150], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (245, 245, 245), 2, cv2.LINE_AA)
    if not detections:
        cv2.putText(vis, "NO DETECTIONS", (12, min(vis.shape[0] - 12, 76)), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (40, 40, 255), 2, cv2.LINE_AA)
    return vis, detections


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_size: tuple[int, int] = (360, 260), cols: int = 4) -> None:
    """Save a contact sheet for quick review."""
    if not paths:
        return
    thumb_w, thumb_h = thumb_size
    rows = int(np.ceil(len(paths) / cols))
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 238, dtype=np.uint8)
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = min(thumb_w / img.shape[1], thumb_h / img.shape[0])
        new_w, new_h = max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        y0 = (i // cols) * thumb_h + (thumb_h - new_h) // 2
        x0 = (i % cols) * thumb_w + (thumb_w - new_w) // 2
        sheet[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    """Run source-aware visualization."""
    args = parse_args()
    args.imgsz = normalize_imgsz(args.imgsz)
    sources = [canonical_source(x) for x in args.sources.split(",") if canonical_source(x)]
    samples = load_samples(args, sources)
    if not samples:
        raise SystemExit("No images found for requested sources.")

    out_name = args.name or f"yolo26ps_stage_visualize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    head = model.model.model[-1]
    if hasattr(head, "set_active_tasks"):
        head.set_active_tasks(active_tasks_for_source(samples[0]["source"], args, head))
    head.max_det = args.max_det
    model.predict(
        samples[0]["image"],
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
    head.max_det = args.max_det
    model.model.eval()
    names = model.model.names

    summary: dict[str, Any] = {
        "weights": str(args.weights),
        "manifest": str(args.manifest),
        "val_txt": str(args.val_txt),
        "out_dir": str(out_dir),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "pose_conf": args.pose_conf,
        "z_display": args.z_display,
        "sources": sources,
        "images": [],
    }
    saved: list[Path] = []
    for index, item in enumerate(samples):
        source, image_path = item["source"], item["image"]
        image = cv2.imread(image_path)
        if image is None:
            summary["images"].append({"source": source, "image": image_path, "error": "cv2.imread failed"})
            continue
        if hasattr(head, "set_active_tasks"):
            head.set_active_tasks(active_tasks_for_source(source, args, head))
        deploy, ratio_pad, im = run_inference(model, predictor, image, args)
        preds = prepare_predictions(deploy, im, image.shape[:2], ratio_pad, source, args)
        vis, detections = draw_predictions(image, preds, source, names, args, image_path)
        out_path = out_dir / f"{index:02d}_{source}_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        saved.append(out_path)
        summary["images"].append(
            {
                "source": source,
                "image": image_path,
                "output": str(out_path),
                "num_detections": len(detections),
                "detections": detections,
            }
        )
        cls_names = sorted({d["name"] for d in detections})
        print(f"{index + 1:02d}/{len(samples)} {source:16s} dets={len(detections):2d} classes={cls_names[:8]} -> {out_path.name}")

    contact = out_dir / "contact_sheet.jpg"
    make_contact_sheet(saved, contact)
    summary["contact_sheet"] = str(contact)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved contact sheet: {contact}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
