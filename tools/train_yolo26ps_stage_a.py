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
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.torch_utils import unwrap_model


DATA_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
STAGE_A_ROOT = DATA_ROOT / "YOLO26PS_STAGE_A"
DATA_YAML = STAGE_A_ROOT / "yolo26ps_stage_a.yaml"
MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26s-ps25d.yaml"
PLAN_YAML = ROOT / "ultralytics/cfg/datasets/yolo26-ps25d-plan.yaml"
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


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return ivalue


def load_stage_defaults(plan: Path) -> dict:
    if not plan.exists():
        return {}
    cfg = YAML.load(plan)
    stage = cfg.get("stages", {}).get("A_detection", {})
    train_runtime = stage.get("train_runtime", {})
    defaults = {}
    for key in ("epochs", "samples_per_epoch"):
        if key in stage:
            defaults[key] = stage[key]
    if "sampling_weights" in stage:
        defaults["sampling_weights"] = stage["sampling_weights"]
    for key in ("imgsz", "batch", "accumulate", "workers", "no_val", "sampling", "sampling_weights"):
        if key in train_runtime:
            defaults[key] = train_runtime[key]
    return defaults


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

    def validate(self):
        """Skip epoch validation when val=False; upstream still validates on the final epoch by default."""
        if not self.args.val:
            return {}, 0.0
        return super().validate()

    def final_eval(self):
        """Skip final validation for short VRAM probes when val=False."""
        if not self.args.val:
            LOGGER.info("Skipping final validation because val=False.")
            return
        return super().final_eval()


def parse_args() -> argparse.Namespace:
    defaults = load_stage_defaults(PLAN_YAML)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--model", type=Path, default=MODEL_YAML)
    parser.add_argument("--plan", type=Path, default=PLAN_YAML)
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 50))
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=defaults.get("imgsz", [704]),
        help=(
            "training image size. One value keeps square training; two values are parsed but current Ultralytics "
            "training coerces train/val imgsz back to one square dimension."
        ),
    )
    parser.add_argument("--batch", type=int, default=defaults.get("batch", 8))
    parser.add_argument(
        "--accumulate",
        type=int,
        default=defaults.get("accumulate", 10),
        help="target gradient accumulation; mapped to Ultralytics nbs=batch*accumulate",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 8))
    parser.add_argument("--freeze", type=int, default=10, help="partial backbone freeze via Ultralytics freeze index")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name", default="yolo26ps_stage_a_detection")
    parser.add_argument("--fraction", type=float, default=1.0, help="dataset fraction, useful for VRAM probes")
    parser.add_argument(
        "--samples-per-epoch",
        type=positive_int,
        default=defaults.get("samples_per_epoch"),
        help="weighted random samples drawn with replacement per epoch; disables sequential dataset traversal",
    )
    parser.add_argument("--sampling", default=defaults.get("sampling", "sequential"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="run Stage A data preparation before training")
    parser.add_argument(
        "--no-val",
        action="store_true",
        default=bool(defaults.get("no_val", False)),
        help="disable validation, useful for short VRAM probes",
    )
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
    defaults = load_stage_defaults(args.plan)
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
        sampling=args.sampling,
        samples_per_epoch=args.samples_per_epoch,
        sampling_weights=defaults.get("sampling_weights", {}),
        val=not args.no_val,
        save=not args.no_save,
        plots=not args.no_val,
    )
    if args.device is not None:
        overrides["device"] = args.device
    model.train(trainer=YOLO26PSStageATrainer, **overrides)


if __name__ == "__main__":
    main()
