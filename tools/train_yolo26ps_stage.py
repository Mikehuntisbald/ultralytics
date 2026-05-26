#!/usr/bin/env python3
"""Train YOLO26s-PS-2.5D stages from the shared plan YAML."""

from __future__ import annotations

import argparse
import numpy as np
import subprocess
import sys
from copy import copy
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER, RANK, YAML
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.torch_utils import unwrap_model


DATA_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
STAGE_A_ROOT = DATA_ROOT / "YOLO26PS_STAGE_A"
DATA_YAML = STAGE_A_ROOT / "yolo26ps_stage_a.yaml"
MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26s-ps25d.yaml"
PLAN_YAML = ROOT / "ultralytics/cfg/datasets/yolo26-ps25d-plan.yaml"
STAGE_DATA_YAMLS = {
    "A_detection": DATA_YAML,
    "A_detection_stable": DATA_YAML,
    "B_pose2d": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d.yaml",
    "C_pose25d": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "C_pose25d_no_H3WB": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "D_person_mask": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_d_person_mask.yaml",
    "E_scene_seg": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_e_scene_seg.yaml",
    "F_full_finetune": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_f_full_finetune.yaml",
    "F_full_finetune_pose_heavy": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_f_full_finetune.yaml",
}

LOSS_KEYS = ("det", "pose2d", "pose_z", "pose_vis", "bone", "person_mask", "scene_seg")
MIXED_AUG_KEYS = ("mosaic", "mixup", "copy_paste", "cutmix")
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
    """Validator that unwraps det-only outputs and records source-level stage metrics."""

    def postprocess(self, preds):
        while isinstance(preds, (list, tuple)) and preds:
            preds = preds[0]
            if isinstance(preds, torch.Tensor):
                break
        return super().postprocess(preds)

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.person_cls = int(self.data.get("person_cls", 0))
        self.face_cls = int(self.data.get("face_cls", max(self.nc - 1, 0)))
        self.stage_metric_buckets = {name: DetMetrics(names=self.names) for name in self._stage_bucket_names()}
        self.stage_task_counts = {"pose2d": 0, "pose3d": 0, "person_mask": 0, "scene_seg": 0}

    @staticmethod
    def _stage_bucket_names() -> tuple[str, ...]:
        return (
            "objects365",
            "objects365_person",
            "crowdhuman_person",
            "wider_face",
            "small",
        )

    @staticmethod
    def _source_name(im_file: str | Path) -> str:
        parts = {p.lower() for p in Path(im_file).parts}
        if "objects365" in parts:
            return "objects365"
        if "crowdhuman" in parts:
            return "crowdhuman"
        if "wider_face" in parts or "wider" in parts:
            return "wider_face"
        return "unknown"

    def _bucket_update(
        self,
        name: str,
        predn: dict[str, torch.Tensor],
        pbatch: dict[str, Any],
        class_id: int | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> None:
        metric = self.stage_metric_buckets.get(name)
        if metric is None:
            return
        bucket_batch = dict(pbatch)
        if gt_mask is not None:
            bucket_batch["cls"] = bucket_batch["cls"][gt_mask]
            bucket_batch["bboxes"] = bucket_batch["bboxes"][gt_mask]
        if class_id is not None:
            class_mask = bucket_batch["cls"] == class_id
            bucket_batch["cls"] = bucket_batch["cls"][class_mask]
            bucket_batch["bboxes"] = bucket_batch["bboxes"][class_mask]
            pred_mask = predn["cls"] == class_id
            pred_eval = {**predn, "bboxes": predn["bboxes"][pred_mask], "conf": predn["conf"][pred_mask], "cls": predn["cls"][pred_mask]}
        else:
            pred_eval = predn

        cls = bucket_batch["cls"].cpu().numpy()
        no_pred = pred_eval["cls"].shape[0] == 0
        metric.update_stats(
            {
                **self._process_batch(pred_eval, bucket_batch),
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": np.zeros(0) if no_pred else pred_eval["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred else pred_eval["cls"].cpu().numpy(),
                "im_name": Path(bucket_batch["im_file"]).name,
            }
        )

    def _update_stage_counts(self, batch: dict[str, Any], si: int) -> None:
        for key, out_key in (
            ("has_pose2d", "pose2d"),
            ("has_pose3d", "pose3d"),
            ("has_person_mask", "person_mask"),
            ("has_scene_seg", "scene_seg"),
        ):
            value = batch.get(key)
            if torch.is_tensor(value) and si < value.numel() and bool(value[si]):
                self.stage_task_counts[out_key] += 1

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        for si, pred in enumerate(preds):
            self.seen += 1
            self._update_stage_counts(batch, si)
            if torch.is_tensor(batch.get("has_det")) and si < batch["has_det"].numel() and not bool(batch["has_det"][si]):
                continue

            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            stat = {
                **self._process_batch(predn, pbatch),
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                "im_name": Path(pbatch["im_file"]).name,
            }
            self.metrics.update_stats(stat)

            source = self._source_name(pbatch["im_file"])
            if source == "objects365":
                self._bucket_update("objects365", predn, pbatch)
                self._bucket_update("objects365_person", predn, pbatch, class_id=self.person_cls)
            elif source == "crowdhuman":
                self._bucket_update("crowdhuman_person", predn, pbatch, class_id=self.person_cls)
            elif source == "wider_face":
                self._bucket_update("wider_face", predn, pbatch, class_id=self.face_cls)
            if pbatch["bboxes"].numel():
                boxes_scaled = self.scale_preds({"bboxes": pbatch["bboxes"].clone()}, pbatch)["bboxes"]
                small = (boxes_scaled[:, 2] - boxes_scaled[:, 0]) * (boxes_scaled[:, 3] - boxes_scaled[:, 1]) < 32**2
                if bool(small.any()):
                    self._bucket_update("small", predn, pbatch, gt_mask=small)

            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(
                        batch["img"][si],
                        pbatch["im_file"],
                        self.save_dir,
                        self.args.show_labels,
                        self.args.show_conf,
                    )

            if not no_pred and (self.args.save_json or self.args.save_txt):
                predn_scaled = self.scale_preds(predn, pbatch)
                if self.args.save_json:
                    self.pred_to_json(predn_scaled, pbatch)
                if self.args.save_txt:
                    self.save_one_txt(
                        predn_scaled,
                        self.args.save_conf,
                        pbatch["ori_shape"],
                        self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                    )

    def gather_stats(self) -> None:
        super().gather_stats()
        if RANK == 0:
            for metric in self.stage_metric_buckets.values():
                self._gather_stage_metric(metric)
            self._gather_stage_counts()
        elif RANK > 0:
            for metric in self.stage_metric_buckets.values():
                self._gather_stage_metric(metric)
            self._gather_stage_counts()

    @staticmethod
    def _gather_stage_metric(metric: DetMetrics) -> None:
        """Gather a stage metric bucket across DDP ranks."""
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(metric.stats, gathered, dst=0)
            merged = {key: [] for key in metric.stats.keys()}
            for stats in gathered:
                if not stats:
                    continue
                for key in merged:
                    merged[key].extend(stats[key])
            metric.stats = merged
            metric.clear_image_metrics()
        elif RANK > 0:
            dist.gather_object(metric.stats, None, dst=0)
            metric.clear_stats()
            metric.clear_image_metrics()

    def _gather_stage_counts(self) -> None:
        """Gather per-task image counters across DDP ranks."""
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(self.stage_task_counts, gathered, dst=0)
            totals = {key: 0 for key in self.stage_task_counts}
            for counts in gathered:
                if not counts:
                    continue
                for key in totals:
                    totals[key] += int(counts.get(key, 0))
            self.stage_task_counts = totals
        elif RANK > 0:
            dist.gather_object(self.stage_task_counts, None, dst=0)

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats.update(self._stage_results())
        for name, count in self.stage_task_counts.items():
            stats[f"metrics/stage/{name}_images"] = count
        return stats

    def _stage_results(self) -> dict[str, float]:
        results = {}
        prefix_map = {
            "objects365": "metrics/stage_a/objects365",
            "objects365_person": "metrics/stage_a/objects365/person",
            "crowdhuman_person": "metrics/stage_a/crowdhuman/person",
            "wider_face": "metrics/stage_a/wider_face/face",
            "small": "metrics/stage_a/small",
        }
        for name, metric in self.stage_metric_buckets.items():
            prefix = prefix_map[name]
            has_stats = bool(metric.stats.get("target_cls")) and any(len(x) for x in metric.stats["target_cls"])
            if has_stats:
                metric.process(save_dir=self.save_dir / "stage_metrics" / name, plot=False, on_plot=self.on_plot)
                results[f"{prefix}/mAP50(B)"] = float(metric.box.map50)
                results[f"{prefix}/mAP50-95(B)"] = float(metric.box.map)
                metric.clear_stats()
                metric.clear_image_metrics()
            else:
                results[f"{prefix}/mAP50(B)"] = 0.0
                results[f"{prefix}/mAP50-95(B)"] = 0.0
        return results


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
    parser.add_argument("--data", type=Path, help="dataset YAML; defaults to the selected stage YAML")
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
            should_prepare = not (
                int(data_cfg.get("nc", 0)) == 366
                and int(data_cfg.get("det_base_nc", 0)) == 365
                and int(data_cfg.get("person_cls", -1)) == 0
                and int(data_cfg.get("face_cls", -1)) == 365
                and (args.data.parent / str(data_cfg.get("train", "train.txt"))).exists()
            )
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
    for key in MIXED_AUG_KEYS:
        overrides[key] = float(augment.get(key, 0.0))
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
    if args.data is None:
        data_yaml = stage.get("data_yaml")
        args.data = (ROOT / data_yaml) if data_yaml else STAGE_DATA_YAMLS.get(args.stage, DATA_YAML)
    YOLO26PSStageTrainer.stage_cfg = stage
    maybe_prepare(args)
    model = YOLO(str(args.weights or args.model))
    model.train(trainer=YOLO26PSStageTrainer, **build_overrides(args, plan, stage))


if __name__ == "__main__":
    main()
