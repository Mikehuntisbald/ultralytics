#!/usr/bin/env python3
"""Train YOLO26s-PS-2.5D Stage A detection warmup."""

from __future__ import annotations

import argparse
import subprocess
import sys
from copy import copy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


DATA_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
STAGE_A_ROOT = DATA_ROOT / "YOLO26PS_STAGE_A"
DATA_YAML = STAGE_A_ROOT / "yolo26ps_stage_a.yaml"
MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26s-ps25d.yaml"
AUX_BRANCHES = ("cv4", "one2one_cv4", "cv5", "one2one_cv5", "proto", "scene_seg", "p2_refine")
STAGE_A_LOSS_WEIGHTS = {
    "det": 1.0,
    "pose2d": 0.0,
    "pose_z": 0.0,
    "pose_vis": 0.0,
    "bone": 0.0,
    "person_mask": 0.0,
    "scene_seg": 0.0,
}


class YOLO26PSStageAValidator(DetectionValidator):
    """Detection-only validator for the multi-task deployment tuple."""

    def postprocess(self, preds):
        while isinstance(preds, (list, tuple)) and preds:
            preds = preds[0]
            if isinstance(preds, torch.Tensor):
                break
        return super().postprocess(preds)


class YOLO26PSStageATrainer(DetectionTrainer):
    """Detection warmup trainer that freezes non-detection auxiliary branches."""

    def set_stage_loss_weights(self) -> None:
        model = unwrap_model(self.model)
        model.loss_weights = STAGE_A_LOSS_WEIGHTS.copy()
        LOGGER.info(f"Stage A task loss weights: {model.loss_weights}")

    def set_model_attributes(self):
        super().set_model_attributes()
        self.set_stage_loss_weights()

    def freeze_auxiliary_branches(self) -> None:
        model = unwrap_model(self.model)
        head = model.model[-1]
        frozen = []
        for name in AUX_BRANCHES:
            module = getattr(head, name, None)
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = False
            frozen.append(name)
        if frozen:
            LOGGER.info(f"Stage A frozen auxiliary branches: {', '.join(frozen)}")

    def _setup_train(self):
        super()._setup_train()
        self.freeze_auxiliary_branches()

    def build_optimizer(self, model, *args, **kwargs):
        self.freeze_auxiliary_branches()
        optimizer = super().build_optimizer(model, *args, **kwargs)
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        optimizer.param_groups[:] = [group for group in optimizer.param_groups if group["params"]]
        trainable = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
        LOGGER.info(f"Stage A optimizer trainable parameters: {trainable:,}")
        return optimizer

    def get_validator(self):
        self.loss_names = getattr(unwrap_model(self.model).model[-1], "loss_names", ("box_loss", "cls_loss", "dfl_loss"))
        return YOLO26PSStageAValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def final_eval(self):
        """Skip final validation for short VRAM probes when val=False."""
        if not self.args.val:
            LOGGER.info("Skipping final validation because val=False.")
            return
        return super().final_eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--model", type=Path, default=MODEL_YAML)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[768],
        help=(
            "training image size. One value keeps square training; two values are parsed but current Ultralytics "
            "training coerces train/val imgsz back to one square dimension."
        ),
    )
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument(
        "--accumulate",
        type=int,
        default=4,
        help="target gradient accumulation; mapped to Ultralytics nbs=batch*accumulate",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=10, help="partial backbone freeze via Ultralytics freeze index")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name", default="yolo26ps_stage_a_detection")
    parser.add_argument("--fraction", type=float, default=1.0, help="dataset fraction, useful for VRAM probes")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="run Stage A data preparation before training")
    parser.add_argument("--no-val", action="store_true", help="disable validation, useful for short VRAM probes")
    parser.add_argument("--no-save", action="store_true", help="disable checkpoint saving, useful for short VRAM probes")
    parser.add_argument("--skip-crowdhuman-extract", action="store_true")
    return parser.parse_args()


def maybe_prepare(args: argparse.Namespace) -> None:
    if args.prepare or not args.data.exists():
        cmd = [
            sys.executable,
            str(ROOT / "tools/prepare_yolo26ps_stage_a.py"),
            "--datasets",
            str(args.data_root),
            "--out",
            str(args.data.parent),
        ]
        if args.skip_crowdhuman_extract:
            cmd.append("--skip-crowdhuman-extract")
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    maybe_prepare(args)
    model = YOLO(str(args.model))
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz[:2]
    overrides = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        nbs=args.batch * args.accumulate,
        workers=args.workers,
        freeze=args.freeze,
        project=str(args.project),
        name=args.name,
        task="detect",
        mosaic=0.7,
        mixup=0.1,
        copy_paste=0.1,
        scale=0.5,
        close_mosaic=0,
        patience=50,
        resume=args.resume,
        fraction=args.fraction,
        val=not args.no_val,
        save=not args.no_save,
    )
    if args.device is not None:
        overrides["device"] = args.device
    model.train(trainer=YOLO26PSStageATrainer, **overrides)


if __name__ == "__main__":
    main()
