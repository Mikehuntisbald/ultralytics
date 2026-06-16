#!/usr/bin/env python3
"""Distill Stage D det-neck adapters from the official YOLO26 segmentation neck."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_d_person_mask_person_only.yaml"
DEFAULT_STUDENT = ROOT / "pretrains/yolo26s-ps25d-stage-d-detneck-adapter-p3p5only-init-from-e5.pt"
DEFAULT_TEACHER = ROOT / "pretrains/yolo26s-seg.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name", default="yolo26ps_d_p3p5only_adapter_distill")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[576, 768])
    parser.add_argument("--batch", type=int, default=180)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--sampling", default="weighted_random_with_replacement")
    parser.add_argument("--samples-per-epoch", type=int, default=100000)
    parser.add_argument("--sampling-weights", default="coco_person_mask=95,ochuman=5")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--loss", choices=("normalized_mse", "raw_mse"), default="normalized_mse")
    parser.add_argument("--lock-adapter-bn", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def parse_sampling_weights(value: str | dict[str, float] | None) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    out: dict[str, float] = {}
    for item in str(value).replace(" ", "").split(","):
        if not item:
            continue
        key, weight = item.split("=", 1)
        out[key] = float(weight)
    return out


def normalize_imgsz(value: list[int]) -> int | list[int]:
    return value[0] if len(value) == 1 else value[:2]


def forward_layers(model: torch.nn.Module, x: torch.Tensor, indices: set[int]) -> dict[int, torch.Tensor]:
    """Run a YOLO Sequential model once and return selected layer outputs."""
    y: list[Any] = []
    out: dict[int, torch.Tensor] = {}
    for m in model.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in model.save else None)
        if m.i in indices:
            out[m.i] = x
    return out


def set_only_adapter_trainable(model: torch.nn.Module, lock_bn: bool = False) -> torch.nn.Module:
    """Freeze the model except the Stage D segmentation adapters."""
    for p in model.parameters():
        p.requires_grad = False
    head = model.model[-1]
    for p in head.seg_adapt.parameters():
        p.requires_grad = True
    model.eval()
    head.seg_adapt.train()
    if lock_bn:
        for module in head.seg_adapt.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    return head.seg_adapt


def feature_loss(student_feats: list[torch.Tensor], teacher_feats: list[torch.Tensor], mode: str) -> torch.Tensor:
    """Scale-stable feature matching loss across P3-P5."""
    losses = []
    for student, teacher in zip(student_feats, teacher_feats):
        teacher = teacher.detach()
        if student.shape[-2:] != teacher.shape[-2:]:
            teacher = F.interpolate(teacher, student.shape[-2:], mode="bilinear", align_corners=False)
        if mode == "normalized_mse":
            student = F.normalize(student.float(), dim=1)
            teacher = F.normalize(teacher.float(), dim=1)
        else:
            student = student.float()
            teacher = teacher.float()
        losses.append(F.mse_loss(student, teacher))
    return torch.stack(losses).mean()


def build_train_loader(args: argparse.Namespace):
    data = check_det_dataset(str(args.data))
    cfg = get_cfg(
        overrides={
            "data": str(args.data),
            "imgsz": normalize_imgsz(args.imgsz),
            "rect": False,
            "cache": False,
            "single_cls": False,
            "task": "detect",
            "mode": "train",
            "batch": args.batch,
            "workers": args.workers,
            "fraction": 1.0,
            "overlap_mask": True,
            "mask_ratio": 4,
            "classes": None,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "cutmix": 0.0,
            "erasing": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "bgr": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "close_mosaic": 0,
        }
    )
    dataset = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="train", rect=False, stride=32)
    return build_dataloader(
        dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=True,
        sampling=args.sampling,
        samples_per_epoch=args.samples_per_epoch,
        sampling_weights=parse_sampling_weights(args.sampling_weights),
    )


def save_checkpoint(model: torch.nn.Module, path: Path, args: argparse.Namespace, step: int, loss: float) -> None:
    ckpt = {
        "model": model.half().cpu(),
        "ema": None,
        "updates": step,
        "optimizer": None,
        "train_args": vars(args),
        "epoch": step,
        "best_fitness": -loss,
    }
    torch.save(ckpt, path)
    model.float()


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
    adapters = set_only_adapter_trainable(student, lock_bn=args.lock_adapter_bn)
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loader = build_train_loader(args)

    student_indices = [22, 25, 28]
    teacher_indices = [16, 19, 22]
    running = 0.0
    best_loss = math.inf
    LOGGER.info(
        "Distilling Stage D adapters: student=%s teacher=%s batch=%d steps=%d loss=%s lock_adapter_bn=%s trainable=%d",
        args.student,
        args.teacher,
        args.batch,
        args.steps,
        args.loss,
        args.lock_adapter_bn,
        sum(p.numel() for p in adapters.parameters() if p.requires_grad),
    )

    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            teacher_out = forward_layers(teacher, imgs, set(teacher_indices))
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            student_out = forward_layers(student, imgs, set(student_indices))
            head = student.model[-1]
            student_feats = [head.seg_adapt[i](student_out[idx]) for i, idx in enumerate(student_indices)]
            teacher_feats = [teacher_out[idx] for idx in teacher_indices]
            loss = feature_loss(student_feats, teacher_feats, args.loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = float(loss.detach().cpu())
        running += loss_value
        if step == 1 or step % 20 == 0 or step == args.steps:
            avg = running / (20 if step % 20 == 0 else step if step < 20 else step % 20 or 20)
            running = 0.0
            LOGGER.info("step %d/%d feature_loss=%.6f avg=%.6f", step, args.steps, loss_value, avg)
        if loss_value < best_loss:
            best_loss = loss_value
            save_checkpoint(student, weights_dir / "best.pt", args, step, best_loss)
            student.to(device)
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(student, weights_dir / f"step{step}.pt", args, step, loss_value)
            student.to(device)

    save_checkpoint(student, weights_dir / "last.pt", args, args.steps, loss_value)
    LOGGER.info("Saved distill checkpoints to %s", weights_dir)


if __name__ == "__main__":
    main()
