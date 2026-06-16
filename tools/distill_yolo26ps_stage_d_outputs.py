#!/usr/bin/env python3
"""Distill Stage D mask outputs from the official YOLO26 segmentation head."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from ultralytics import YOLO
from ultralytics.utils import LOGGER

from distill_yolo26ps_stage_d_adapter import (
    DEFAULT_DATA,
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    build_train_loader,
    feature_loss,
    save_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name", default="yolo26ps_d_p3p5only_output_distill")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[576, 768])
    parser.add_argument("--batch", type=int, default=160)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--sampling", default="weighted_random_with_replacement")
    parser.add_argument("--samples-per-epoch", type=int, default=100000)
    parser.add_argument("--sampling-weights", default="coco_person_mask=95,ochuman=5")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--train-modules", default="seg_adapt,cv4,proto")
    parser.add_argument("--lock-bn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coeff-weight", type=float, default=1.0)
    parser.add_argument("--proto-weight", type=float, default=0.1)
    parser.add_argument("--feature-weight", type=float, default=0.0)
    parser.add_argument("--feature-loss", choices=("normalized_mse", "raw_mse"), default="raw_mse")
    parser.add_argument("--teacher-person-class", type=int, default=-1)
    parser.add_argument("--person-score-threshold", type=float, default=0.0)
    parser.add_argument("--person-topk", type=int, default=0)
    parser.add_argument("--person-weight-power", type=float, default=1.0)
    parser.add_argument("--person-weight-floor", type=float, default=0.0)
    parser.add_argument("--box-weight", type=float, default=0.0)
    parser.add_argument("--score-weight", type=float, default=0.0)
    parser.add_argument("--score-background-weight", type=float, default=0.05)
    parser.add_argument("--score-loss", choices=("bce", "mse"), default="bce")
    parser.add_argument("--one2one-weight", type=float, default=1.0)
    return parser.parse_args()


def normalize_names(value: str | list[str] | tuple[str, ...] | None) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = list(value)
    aliases = {"adapter": "seg_adapt", "mask": "cv4", "coeff": "cv4", "coeff_head": "cv4"}
    return {aliases.get(str(x).strip().lower(), str(x).strip().lower()) for x in raw if str(x).strip()}


def collect_layers(model: torch.nn.Module, x: torch.Tensor, indices: set[int]) -> dict[int, torch.Tensor]:
    """Run a YOLO model only through the highest requested layer and return selected outputs."""
    y: list[Any] = []
    out: dict[int, torch.Tensor] = {}
    stop = max(indices)
    for m in model.model:
        if m.i > stop:
            break
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in model.save else None)
        if m.i in indices:
            out[m.i] = x
    return out


def set_trainable_modules(model: torch.nn.Module, names: set[str], lock_bn: bool) -> list[torch.nn.Parameter]:
    """Freeze the model except selected Stage D mask modules."""
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    head = model.model[-1]
    modules: list[torch.nn.Module] = []
    for name in sorted(names):
        module = getattr(head, name, None)
        if not isinstance(module, torch.nn.Module):
            LOGGER.warning("Requested train module %s is not present on %s", name, head.__class__.__name__)
            continue
        module.train()
        modules.append(module)
        for p in module.parameters():
            p.requires_grad = True
        if lock_bn:
            for submodule in module.modules():
                if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
                    submodule.eval()
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"No trainable parameters selected from modules={sorted(names)}")
    return params


def teacher_outputs(model: torch.nn.Module, imgs: torch.Tensor) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Return official teacher mask coefficient/proto outputs and P3-P5 features."""
    indices = [16, 19, 22]
    layers = collect_layers(model, imgs, set(indices))
    feats = [layers[i] for i in indices]
    head = model.model[-1]
    preds = head.forward_head(feats, **head.one2many)
    preds["proto"] = head.proto(feats)
    if getattr(head, "end2end", False):
        one2one = head.forward_head([x.detach() for x in feats], **head.one2one)
        one2one["proto"] = preds["proto"]
        preds["one2one"] = one2one
    return preds, feats


def student_outputs(model: torch.nn.Module, imgs: torch.Tensor) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Return Stage D adapted mask coefficient/proto outputs and adapted P3-P5 features."""
    indices = [22, 25, 28]
    layers = collect_layers(model, imgs, set(indices))
    det_feats = [layers[i] for i in indices]
    head = model.model[-1]
    adapted = head._adapt_features(det_feats)
    preds = head.forward_head(adapted, **head.one2many)
    preds["proto"] = head.proto(head._proto_features(det_feats, adapted))
    if getattr(head, "end2end", False):
        one2one = head.forward_head([x.detach() for x in adapted], **head.one2one)
        one2one["proto"] = preds["proto"]
        preds["one2one"] = one2one
    return preds, adapted


def person_anchor_weights(teacher: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor | None:
    """Return teacher-person-score weights over anchors, optionally keeping only confident person anchors."""
    cls = int(args.teacher_person_class)
    if cls < 0:
        return None
    scores = teacher.get("scores")
    if scores is None or cls >= scores.shape[1]:
        LOGGER.warning("teacher_person_class=%d is unavailable in teacher scores; using unweighted distill loss.", cls)
        return None
    weights = scores[:, cls : cls + 1].detach().float().sigmoid()
    if args.person_score_threshold > 0.0:
        weights = torch.where(weights >= float(args.person_score_threshold), weights, torch.zeros_like(weights))
    if int(args.person_topk) > 0 and weights.shape[-1] > int(args.person_topk):
        k = min(int(args.person_topk), weights.shape[-1])
        kth = weights.topk(k, dim=-1).values[..., -1:]
        weights = torch.where(weights >= kth, weights, torch.zeros_like(weights))
    if args.person_weight_power != 1.0:
        weights = weights.clamp_min(1e-6).pow(args.person_weight_power)
    if args.person_weight_floor > 0.0:
        floor = torch.full_like(weights, float(args.person_weight_floor))
        weights = torch.where(weights > 0, weights.maximum(floor), weights)
    return weights


def coefficient_loss(student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    """Return raw or teacher-person-score-weighted mask coefficient loss."""
    diff = (student["mask_coefficient"].float() - teacher["mask_coefficient"].detach().float()).square()
    weights = person_anchor_weights(teacher, args)
    if weights is None:
        return diff.mean()
    if not bool(weights.gt(0).any()):
        return diff.sum() * 0.0
    denom = weights.sum().clamp_min(1e-9) * diff.shape[1]
    return (diff * weights).sum() / denom


def box_loss(student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    """Distill raw box logits only on teacher-person anchors."""
    weights = person_anchor_weights(teacher, args)
    if weights is None or not bool(weights.gt(0).any()):
        return student["boxes"].sum() * 0.0
    diff = F.smooth_l1_loss(student["boxes"].float(), teacher["boxes"].detach().float(), reduction="none")
    denom = weights.sum().clamp_min(1e-9) * diff.shape[1]
    return (diff * weights).sum() / denom


def score_loss(student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    """Distill the teacher person-class confidence into the one-class student seg-det head."""
    cls = int(args.teacher_person_class)
    scores = teacher.get("scores")
    if cls < 0 or scores is None or cls >= scores.shape[1]:
        return student["scores"].sum() * 0.0
    target = scores[:, cls : cls + 1].detach().float().sigmoid()
    pred = student["scores"][:, :1].float()
    if args.score_loss == "mse":
        per_anchor = (pred.sigmoid() - target).square()
    else:
        per_anchor = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    positive = person_anchor_weights(teacher, args)
    if positive is None:
        weights = torch.ones_like(target)
    else:
        bg = torch.full_like(target, float(args.score_background_weight))
        weights = torch.where(positive > 0, torch.ones_like(target), bg)
    return (per_anchor * weights).sum() / weights.sum().clamp_min(1e-9)


def output_loss(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    student_feats: list[torch.Tensor],
    teacher_feats: list[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine mask coefficient, proto, and optional feature distillation losses."""
    zero = student["mask_coefficient"].new_zeros(())
    student_proto = student["proto"][0] if isinstance(student["proto"], tuple) else student["proto"]
    teacher_proto = teacher["proto"][0] if isinstance(teacher["proto"], tuple) else teacher["proto"]
    coeff = coefficient_loss(student, teacher, args)
    if args.one2one_weight and isinstance(student.get("one2one"), dict) and isinstance(teacher.get("one2one"), dict):
        coeff = (coeff + float(args.one2one_weight) * coefficient_loss(student["one2one"], teacher["one2one"], args)) / (
            1.0 + float(args.one2one_weight)
        )
    proto = F.mse_loss(student_proto.float(), teacher_proto.detach().float())
    box = box_loss(student, teacher, args) if args.box_weight else zero
    if (
        args.box_weight
        and args.one2one_weight
        and isinstance(student.get("one2one"), dict)
        and isinstance(teacher.get("one2one"), dict)
    ):
        box = (box + float(args.one2one_weight) * box_loss(student["one2one"], teacher["one2one"], args)) / (
            1.0 + float(args.one2one_weight)
        )
    score = score_loss(student, teacher, args) if args.score_weight else zero
    if (
        args.score_weight
        and args.one2one_weight
        and isinstance(student.get("one2one"), dict)
        and isinstance(teacher.get("one2one"), dict)
    ):
        score = (score + float(args.one2one_weight) * score_loss(student["one2one"], teacher["one2one"], args)) / (
            1.0 + float(args.one2one_weight)
        )
    feat = feature_loss(student_feats, teacher_feats, args.feature_loss) if args.feature_weight else zero
    total = (
        args.coeff_weight * coeff
        + args.proto_weight * proto
        + args.box_weight * box
        + args.score_weight * score
        + args.feature_weight * feat
    )
    return total, {
        "coeff": float(coeff.detach().cpu()),
        "proto": float(proto.detach().cpu()),
        "box": float(box.detach().cpu()),
        "score": float(score.detach().cpu()),
        "feature": float(feat.detach().cpu()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and str(args.device) != "cpu" else "cpu")
    save_dir = args.project / args.name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.yaml").write_text(yaml.safe_dump({k: str(v) for k, v in vars(args).items()}), encoding="utf-8")

    student_yolo = YOLO(str(args.student))
    teacher_yolo = YOLO(str(args.teacher))
    student = student_yolo.model.to(device).float()
    teacher = teacher_yolo.model.to(device).float().eval()
    for p in teacher.parameters():
        p.requires_grad = False

    train_modules = normalize_names(args.train_modules)
    params = set_trainable_modules(student, train_modules, args.lock_bn)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loader = build_train_loader(args)

    LOGGER.info(
        "Distilling Stage D mask outputs: student=%s teacher=%s modules=%s batch=%d steps=%d lr=%g "
        "teacher_person_class=%d person_score_threshold=%g person_topk=%d trainable=%d",
        args.student,
        args.teacher,
        ",".join(sorted(train_modules)),
        args.batch,
        args.steps,
        args.lr,
        args.teacher_person_class,
        args.person_score_threshold,
        args.person_topk,
        sum(p.numel() for p in params),
    )

    iterator = iter(loader)
    running = 0.0
    best_loss = math.inf
    last_loss = math.inf
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            teacher_pred, teacher_feats = teacher_outputs(teacher, imgs)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            student_pred, student_feats = student_outputs(student, imgs)
            loss, parts = output_loss(student_pred, teacher_pred, student_feats, teacher_feats, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        last_loss = float(loss.detach().cpu())
        running += last_loss
        if step == 1 or step % 20 == 0 or step == args.steps:
            denom = 20 if step % 20 == 0 else step if step < 20 else step % 20 or 20
            avg = running / denom
            running = 0.0
            LOGGER.info(
                "step %d/%d output_loss=%.6f avg=%.6f coeff=%.6f proto=%.6f box=%.6f score=%.6f feature=%.6f",
                step,
                args.steps,
                last_loss,
                avg,
                parts["coeff"],
                parts["proto"],
                parts["box"],
                parts["score"],
                parts["feature"],
            )
        if last_loss < best_loss:
            best_loss = last_loss
            save_checkpoint(student, weights_dir / "best.pt", args, step, best_loss)
            student.to(device)
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(student, weights_dir / f"step{step}.pt", args, step, last_loss)
            student.to(device)

    save_checkpoint(student, weights_dir / "last.pt", args, args.steps, last_loss)
    LOGGER.info("Saved output-distill checkpoints to %s", weights_dir)


if __name__ == "__main__":
    main()
