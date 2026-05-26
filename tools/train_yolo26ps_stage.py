#!/usr/bin/env python3
"""Train YOLO26s-PS-2.5D stages from the shared plan YAML."""

from __future__ import annotations

import argparse
import subprocess
import sys
from copy import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

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

LOSS_KEYS = ("det", "pose2d", "pose_z", "pose_vis", "bone", "person_mask", "scene_seg")
BRANCH_MODULES = {
    "det": ("cv2", "cv3", "one2one_cv2", "one2one_cv3"),
    "pose": ("cv4", "one2one_cv4"),
    "mask": ("cv5", "one2one_cv5", "proto"),
    "scene": ("scene_seg",),
    "p2_dense": ("p2_refine",),
}


def positive_int(value: str) -> int:
    """Parse positive integer CLI values."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return ivalue


def load_plan(plan: Path) -> dict[str, Any]:
    """Load the stage plan YAML."""
    return YAML.load(plan) if plan.exists() else {}


def stage_config(plan: dict[str, Any], stage_name: str) -> dict[str, Any]:
    """Return a stage config, inheriting train/loss from the base stage when a variant omits them."""
    stages = plan.get("stages", {})
    if stage_name not in stages:
        raise KeyError(f"Stage '{stage_name}' not found in {PLAN_YAML}. Available: {', '.join(stages)}")
    cfg = dict(stages[stage_name] or {})
    if "_no_" in stage_name or stage_name.endswith("_pose_heavy"):
        base_name = stage_name.split("_no_", 1)[0] if "_no_" in stage_name else "F_full_finetune"
        base = dict(stages.get(base_name, {}) or {})
        for key in ("train", "loss", "augment", "train_runtime", "sampling", "samples_per_epoch"):
            cfg.setdefault(key, base.get(key))
    return cfg


def normalize_imgsz(value: Any) -> int | list[int]:
    """Normalize YAML/CLI image size to an int or [height, width]."""
    if isinstance(value, (list, tuple)):
        value = [int(x) for x in value]
        return value[0] if len(value) == 1 else value[:2]
    return int(value)


def normalize_cache(value: Any) -> bool | str | None:
    """Normalize cache values from CLI/YAML for Ultralytics overrides."""
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"false", "0", "no", "off"}:
        return False
    if text in {"true", "1", "yes", "on"}:
        return True
    return text


def scale_gain(value: Any) -> float:
    """Convert [min, max] scale range into Ultralytics scale gain."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        return max(abs(1.0 - lo), abs(hi - 1.0))
    return float(value)


def rotate_gain(value: Any) -> float:
    """Convert [min, max] rotation range into Ultralytics degree gain."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return max(abs(float(value[0])), abs(float(value[1])))
    return float(value)


def active_tasks_from_stage(stage: dict[str, Any]) -> set[str]:
    """Resolve active head branches from stage train/loss config."""
    train = stage.get("train") or {}
    loss = stage.get("loss") or {}
    tasks = {"det"}
    if train.get("all"):
        return {"det", "pose", "mask", "scene"}
    if train.get("body25d_head") or any(float(loss.get(k, 0.0)) for k in ("pose2d", "pose_z", "pose_vis", "bone")):
        tasks.add("pose")
    if train.get("mask_head") or float(loss.get("person_mask", 0.0)):
        tasks.add("mask")
    if train.get("scene_seg_head") or float(loss.get("scene_seg", 0.0)):
        tasks.add("scene")
    return tasks


def complete_loss_weights(stage: dict[str, Any]) -> dict[str, float]:
    """Return all task loss weights, defaulting missing stage weights to zero."""
    loss = stage.get("loss") or {}
    return {key: float(loss.get(key, 0.0)) for key in LOSS_KEYS}


def stage_defaults(plan: dict[str, Any], stage_name: str) -> dict[str, Any]:
    """Extract runtime defaults from plan YAML."""
    stage = stage_config(plan, stage_name) if plan.get("stages") else {}
    runtime = stage.get("train_runtime") or {}
    model_cfg = plan.get("model", {})
    optimizer_cfg = plan.get("optimizer_scheduler") or {}
    defaults: dict[str, Any] = {}
    for key in ("epochs", "samples_per_epoch", "sampling", "sampling_weights"):
        if stage.get(key) is not None:
            defaults[key] = stage[key]
    for key in (
        "optimizer",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_bias_lr",
        "scheduler",
        "cos_lr",
        "amp",
        "cache",
    ):
        if optimizer_cfg.get(key) is not None:
            defaults[key] = optimizer_cfg[key]
    stage_lr0 = optimizer_cfg.get("stage_lr0") or {}
    if stage_lr0.get(stage_name) is not None:
        defaults["lr0"] = stage_lr0[stage_name]
    elif "_no_" in stage_name and stage_lr0.get(stage_name.split("_no_", 1)[0]) is not None:
        defaults["lr0"] = stage_lr0[stage_name.split("_no_", 1)[0]]
    elif stage_name.endswith("_pose_heavy") and stage_lr0.get("F_full_finetune") is not None:
        defaults["lr0"] = stage_lr0["F_full_finetune"]
    defaults.update(runtime)
    if "imgsz" not in defaults:
        defaults["imgsz"] = stage.get("default_imgsz") or stage.get("input") or model_cfg.get("default_imgsz") or 640
    return defaults


class YOLO26PSStageValidator(DetectionValidator):
    """Validator that unwraps det-only deployment tuples before standard detection postprocess."""

    def postprocess(self, preds):
        while isinstance(preds, (list, tuple)) and preds:
            preds = preds[0]
            if isinstance(preds, torch.Tensor):
                break
        return super().postprocess(preds)


class YOLO26PSStageTrainer(DetectionTrainer):
    """Single trainer for all YOLO26-PS stages; stages only change config, not loss code."""

    stage_cfg: dict[str, Any] = {}

    def set_model_attributes(self):
        super().set_model_attributes()
        self.apply_stage_controls(log=True, freeze=False)

    def apply_stage_controls(self, log: bool = False, freeze: bool = True) -> None:
        """Apply active tasks, loss weights, trainable branches, and eval locks."""
        model = unwrap_model(self.model)
        if not hasattr(model, "model") or not len(model.model):
            return
        head = model.model[-1]
        tasks = active_tasks_from_stage(self.stage_cfg)
        if hasattr(head, "set_active_tasks"):
            head.set_active_tasks(tasks)
        model.loss_weights = complete_loss_weights(self.stage_cfg)
        frozen = self._apply_branch_trainability(head, tasks, freeze=freeze)
        if log:
            LOGGER.info(f"Stage active tasks: {sorted(tasks)}")
            LOGGER.info(f"Stage task loss weights: {model.loss_weights}")
            if frozen:
                LOGGER.info(f"Stage frozen/eval branches: {', '.join(frozen)}")

    def _apply_branch_trainability(self, head: nn.Module, tasks: set[str], freeze: bool = True) -> list[str]:
        """Freeze inactive auxiliary modules and keep them in eval mode."""
        train = self.stage_cfg.get("train") or {}
        train_all = bool(train.get("all"))
        active_groups = set(tasks)
        if {"mask", "scene"} & tasks:
            active_groups.add("p2_dense")

        frozen: list[str] = []
        for group, names in BRANCH_MODULES.items():
            trainable = train_all or group in active_groups
            for name in names:
                module = getattr(head, name, None)
                if module is None:
                    continue
                if not trainable:
                    module.eval()
                    frozen.append(name)
                if freeze:
                    for p in module.parameters():
                        p.requires_grad = bool(trainable)
        return frozen

    def _setup_train(self):
        super()._setup_train()
        self.apply_stage_controls(log=False)

    def _model_train(self):
        super()._model_train()
        self.apply_stage_controls(log=False)

    def build_optimizer(self, model, *args, **kwargs):
        self.apply_stage_controls(log=False)
        optimizer = super().build_optimizer(model, *args, **kwargs)
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        optimizer.param_groups[:] = [group for group in optimizer.param_groups if group["params"]]
        trainable = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
        LOGGER.info(f"Stage optimizer trainable parameters: {trainable:,}")
        return optimizer

    def get_validator(self):
        self.loss_names = getattr(unwrap_model(self.model).model[-1], "loss_names", ("box_loss", "cls_loss", "dfl_loss"))
        return YOLO26PSStageValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def validate(self):
        if not self.args.val:
            return {}, 0.0
        return super().validate()

    def final_eval(self):
        if not self.args.val:
            LOGGER.info("Skipping final validation because val=False.")
            return
        return super().final_eval()


def parse_args() -> argparse.Namespace:
    """Parse CLI args. Runtime defaults are loaded in main after reading --stage/--plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="A_detection_stable", help="stage name under plan['stages']")
    parser.add_argument("--weights", type=Path, help="optional checkpoint to start from")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--model", type=Path, default=MODEL_YAML)
    parser.add_argument("--plan", type=Path, default=PLAN_YAML)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int, nargs="+", help="one square size or two values: height width")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--accumulate", type=int)
    parser.add_argument("--cache", nargs="?", const="ram", help="cache images: ram, disk, true, or false")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--freeze", type=int)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--samples-per-epoch", type=positive_int)
    parser.add_argument("--sampling")
    parser.add_argument("--optimizer")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--warmup-bias-lr", type=float)
    parser.add_argument("--cos-lr", action="store_true", help="force cosine LR scheduler")
    parser.add_argument("--no-cos-lr", action="store_true", help="force non-cosine LR scheduler")
    parser.add_argument("--amp", action="store_true", help="force AMP on")
    parser.add_argument("--no-amp", action="store_true", help="force AMP off")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="run Stage A data preparation before training")
    parser.add_argument("--no-val", action="store_true", help="disable validation")
    parser.add_argument("--no-save", action="store_true", help="disable checkpoint saving")
    parser.add_argument("--skip-crowdhuman-extract", action="store_true")
    return parser.parse_args()


def maybe_prepare(args: argparse.Namespace) -> None:
    """Prepare Stage A detection data if requested or missing."""
    should_prepare = args.prepare or not args.data.exists()
    if args.stage.startswith("A_detection") and args.data.exists() and not args.prepare:
        try:
            data_cfg = YAML.load(args.data)
            should_prepare = int(data_cfg.get("nc", 0)) != 366 or "objects365" not in str(args.data.read_text(encoding="utf-8")).lower()
        except Exception:
            should_prepare = True
    if args.stage.startswith("A_detection") and should_prepare:
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


def build_overrides(args: argparse.Namespace, plan: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    """Build Ultralytics train overrides from stage YAML and CLI."""
    if args.cos_lr and args.no_cos_lr:
        raise ValueError("Use only one of --cos-lr or --no-cos-lr.")
    if args.amp and args.no_amp:
        raise ValueError("Use only one of --amp or --no-amp.")

    defaults = stage_defaults(plan, args.stage)
    augment = stage.get("augment") or {}
    imgsz = normalize_imgsz(args.imgsz if args.imgsz is not None else defaults.get("imgsz", 640))
    batch = int(args.batch if args.batch is not None else defaults.get("batch", 8))
    accumulate = int(args.accumulate if args.accumulate is not None else defaults.get("accumulate", 1))
    no_val = bool(args.no_val or defaults.get("no_val", False))

    overrides = dict(
        data=str(args.data),
        epochs=int(args.epochs if args.epochs is not None else defaults.get("epochs", stage.get("epochs", 50))),
        imgsz=imgsz,
        batch=batch,
        nbs=batch * accumulate,
        workers=int(args.workers if args.workers is not None else defaults.get("workers", 8)),
        freeze=int(args.freeze if args.freeze is not None else defaults.get("freeze", 10 if args.stage.startswith("A_detection") else 0)),
        project=str(args.project),
        name=args.name or f"yolo26ps_{args.stage.lower()}",
        task="detect",
        close_mosaic=0,
        patience=50,
        resume=args.resume,
        fraction=args.fraction,
        sampling=args.sampling or defaults.get("sampling") or stage.get("sampling") or "sequential",
        samples_per_epoch=args.samples_per_epoch or defaults.get("samples_per_epoch") or stage.get("samples_per_epoch"),
        sampling_weights=defaults.get("sampling_weights") or stage.get("sampling_weights") or {},
        val=not no_val,
        save=not args.no_save,
        plots=not no_val,
        multi_scale=0.0 if stage.get("multi_scale") is False else float(defaults.get("multi_scale", 0.0)),
    )
    cache = normalize_cache(args.cache if args.cache is not None else defaults.get("cache"))
    if cache is not None:
        overrides["cache"] = cache
    if "mosaic" in augment:
        overrides["mosaic"] = float(augment["mosaic"])
    if "mixup" in augment:
        overrides["mixup"] = float(augment["mixup"])
    if "copy_paste" in augment:
        overrides["copy_paste"] = float(augment["copy_paste"])
    if "scale" in augment:
        overrides["scale"] = scale_gain(augment["scale"])
    if "rotate" in augment:
        overrides["degrees"] = rotate_gain(augment["rotate"])
    if "flip" in augment:
        overrides["fliplr"] = float(augment["flip"])
    for key, cli_key in (
        ("optimizer", "optimizer"),
        ("lr0", "lr0"),
        ("lrf", "lrf"),
        ("momentum", "momentum"),
        ("weight_decay", "weight_decay"),
        ("warmup_epochs", "warmup_epochs"),
        ("warmup_bias_lr", "warmup_bias_lr"),
    ):
        cli_value = getattr(args, cli_key)
        value = cli_value if cli_value is not None else defaults.get(key)
        if value is not None:
            overrides[key] = value
    if args.cos_lr:
        overrides["cos_lr"] = True
    elif args.no_cos_lr:
        overrides["cos_lr"] = False
    elif defaults.get("cos_lr") is not None:
        overrides["cos_lr"] = bool(defaults["cos_lr"])
    elif str(defaults.get("scheduler", "")).lower() == "cosine":
        overrides["cos_lr"] = True
    if args.amp:
        overrides["amp"] = True
    elif args.no_amp:
        overrides["amp"] = False
    elif defaults.get("amp") is not None:
        overrides["amp"] = bool(defaults["amp"])
    if args.device is not None:
        overrides["device"] = args.device
    return overrides


def main() -> None:
    """Train the requested stage."""
    args = parse_args()
    plan = load_plan(args.plan)
    stage = stage_config(plan, args.stage)
    YOLO26PSStageTrainer.stage_cfg = stage
    maybe_prepare(args)
    model = YOLO(str(args.weights or args.model))
    model.train(trainer=YOLO26PSStageTrainer, **build_overrides(args, plan, stage))


if __name__ == "__main__":
    main()
