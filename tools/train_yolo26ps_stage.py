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
    "A_det_escape_bridge": DATA_YAML,
    "A_det_escape_main": DATA_YAML,
    "A_det_escape_rare_small": DATA_YAML,
    "A_det_escape_probe": STAGE_A_ROOT / "yolo26ps_stage_a_probe_escape.yaml",
    "B_pose2d_probe": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d.yaml",
    "B_pose2d_det_probe": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d.yaml",
    "B_pose2d": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d.yaml",
    "C_pose25d": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "C_det_reanchor": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_det_reanchor.yaml",
    "D_person_mask": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_d_person_mask.yaml",
    "D_det_recover_objects365": DATA_YAML,
    "D_det_recover_objects365_unfreeze": DATA_YAML,
    "D_det_recover_objects365_shock": DATA_YAML,
    "D_det_recover_objects365_prodigy_fast": DATA_YAML,
    "D_det_recover_objects365_prodigy_unfreeze_fast": DATA_YAML,
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
BRANCH_TRAIN_FLAGS = {
    "det": "det_head",
    "pose": "body25d_head",
    "mask": "mask_head",
    "scene": "scene_seg_head",
}
MODEL_GROUP_RANGES = {
    "backbone": (0, 11),
    "neck": (11, -1),
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
        for key in ("train", "loss", "augment", "train_runtime", "sampling", "samples_per_epoch", "val_samples"):
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


def parse_sampling_weights(value: Any) -> dict[str, float] | None:
    """Parse sampling weights from YAML dicts or CLI strings like source=weight,source2=weight."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    weights: dict[str, float] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"sampling weight '{item}' must be formatted as name=weight")
        name, weight = item.split("=", 1)
        weights[name.strip()] = float(weight)
    return weights


def train_flag_enabled(train: dict[str, Any], key: str, default: bool = True) -> bool:
    """Read train/freeze booleans while allowing legacy string values such as partial."""
    if train.get("all"):
        return True
    if key not in train:
        return default
    value = train[key]
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off", "freeze", "frozen"}
    return bool(value)


def truthy(value: Any) -> bool:
    """Read boolean-like YAML/CLI values."""
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "no", "off", "none", "null"}
    return bool(value)


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
    runtime_tasks = (stage.get("train_runtime") or {}).get("active_tasks")
    if runtime_tasks:
        if isinstance(runtime_tasks, str):
            runtime_tasks = [item.strip() for item in runtime_tasks.split(",") if item.strip()]
        tasks = {str(task).strip().lower() for task in runtime_tasks}
        tasks.add("det")
        return tasks
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
    for key in ("epochs", "samples_per_epoch", "sampling", "sampling_weights", "val_samples", "plots"):
        if stage.get(key) is not None:
            defaults[key] = stage[key]
    for key in (
        "optimizer",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_bias_lr",
        "prodigy_d0",
        "prodigy_d_coef",
        "prodigy_growth_rate",
        "prodigy_slice_p",
        "prodigy_decouple",
        "prodigy_use_bias_correction",
        "prodigy_safeguard_warmup",
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
            "coco_person",
            "3dpw_person",
            "agora_person",
            "small",
        )

    @staticmethod
    def _source_name(im_file: str | Path) -> str:
        parts = {p.lower() for p in Path(im_file).parts}
        normalized = str(im_file).replace("\\", "/").lower()
        if "objects365" in parts:
            return "objects365"
        if "crowdhuman" in parts:
            return "crowdhuman"
        if "wider_face" in parts or "wider" in parts:
            return "wider_face"
        if "coco_wholebody" in parts or "coco_2017" in parts or "coco2017" in normalized:
            return "coco_wholebody"
        if "3dpw" in parts or "3dpw" in normalized:
            return "3dpw"
        if "agora" in parts or "agora" in normalized:
            return "agora"
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
            elif source == "coco_wholebody":
                self._bucket_update("coco_person", predn, pbatch, class_id=self.person_cls)
            elif source == "3dpw":
                self._bucket_update("3dpw_person", predn, pbatch, class_id=self.person_cls)
            elif source == "agora":
                self._bucket_update("agora_person", predn, pbatch, class_id=self.person_cls)
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
            "coco_person": "metrics/stage_c/coco/person",
            "3dpw_person": "metrics/stage_c/3dpw/person",
            "agora_person": "metrics/stage_c/agora/person",
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
        frozen_groups = self._apply_model_trainability(model, freeze=freeze)
        frozen = self._apply_branch_trainability(head, tasks, freeze=freeze)
        if truthy((self.stage_cfg.get("train_runtime") or {}).get("freeze_trainable_bn", False)):
            self._freeze_trainable_norm_stats(model)
        if log:
            LOGGER.info(f"Stage active tasks: {sorted(tasks)}")
            LOGGER.info(f"Stage task loss weights: {model.loss_weights}")
            if frozen_groups:
                LOGGER.info(f"Stage frozen/eval model groups: {', '.join(frozen_groups)}")
            if frozen:
                LOGGER.info(f"Stage frozen/eval branches: {', '.join(frozen)}")

    def _apply_model_trainability(self, model: nn.Module, freeze: bool = True) -> list[str]:
        """Freeze backbone/neck groups from the stage train config and keep their BN layers in eval mode."""
        train = self.stage_cfg.get("train") or {}
        layers = getattr(model, "model", [])
        frozen: list[str] = []
        for group, (start, end) in MODEL_GROUP_RANGES.items():
            trainable = train_flag_enabled(train, group, default=True)
            stop = len(layers) if end is None else end if end >= 0 else len(layers) + end
            modules = layers[start:stop]
            if not trainable:
                frozen.append(group)
            for module in modules:
                if not trainable:
                    module.eval()
                if freeze:
                    for p in module.parameters():
                        p.requires_grad = bool(trainable)
        return frozen

    def _apply_branch_trainability(self, head: nn.Module, tasks: set[str], freeze: bool = True) -> list[str]:
        """Freeze inactive or explicitly frozen head modules and keep them in eval mode."""
        train = self.stage_cfg.get("train") or {}
        active_groups = set(tasks)
        if {"mask", "scene"} & tasks:
            active_groups.add("p2_dense")

        frozen: list[str] = []
        for group, names in BRANCH_MODULES.items():
            if group == "p2_dense":
                default = group in active_groups
                trainable = train_flag_enabled(train, "mask_head", default=False) or train_flag_enabled(
                    train, "scene_seg_head", default=False
                )
                trainable = bool(train.get("all")) or (default and trainable)
            else:
                trainable = train_flag_enabled(train, BRANCH_TRAIN_FLAGS[group], default=group in active_groups)
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

    @staticmethod
    def _freeze_trainable_norm_stats(model: nn.Module) -> None:
        """Keep normalization statistics fixed while allowing affine weights to train."""
        norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        for module in model.modules():
            if isinstance(module, norm_types):
                module.eval()

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
    parser.add_argument("--weights", type=Path, help="optional same-architecture checkpoint to start from")
    parser.add_argument("--pretrain", type=Path, help="optional detection pretrain to partially load into --model")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data", type=Path, help="dataset YAML; defaults to the selected stage YAML")
    parser.add_argument("--model", type=Path, default=MODEL_YAML)
    parser.add_argument("--plan", type=Path, default=PLAN_YAML)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int, nargs="+", help="one square size or two values: height width")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--val-batch", type=int, help="validation batch size; defaults to train batch x2")
    parser.add_argument("--val-workers", type=int, help="validation dataloader workers; defaults to workers x2")
    parser.add_argument("--accumulate", type=int)
    parser.add_argument("--cache", nargs="?", const="ram", help="cache images: ram, disk, true, or false")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--freeze", type=int)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--val-samples", type=positive_int, help="limit validation to N evenly spaced images")
    parser.add_argument("--samples-per-epoch", type=positive_int)
    parser.add_argument("--sampling")
    parser.add_argument("--sampling-weights", type=parse_sampling_weights, help="override sampler weights: name=weight,...")
    parser.add_argument("--class-aware-sampling", action="store_true", help="enable rare-class reweighting inside one source")
    parser.add_argument("--no-class-aware-sampling", action="store_true", help="disable rare-class reweighting")
    parser.add_argument("--class-aware-source", help="source name for class-aware reweighting")
    parser.add_argument("--class-aware-power", type=float)
    parser.add_argument("--class-aware-min-multiplier", type=float)
    parser.add_argument("--class-aware-max-multiplier", type=float)
    parser.add_argument("--small-object-sampling", action="store_true", help="boost images with small boxes in sampler")
    parser.add_argument("--no-small-object-sampling", action="store_true", help="disable small-object sampler boost")
    parser.add_argument("--small-object-source")
    parser.add_argument("--small-object-area", type=float)
    parser.add_argument("--small-object-boost", type=float)
    parser.add_argument("--small-object-crop", type=float)
    parser.add_argument("--small-object-crop-source")
    parser.add_argument("--small-object-crop-area", type=float)
    parser.add_argument("--small-object-crop-scale", type=float)
    parser.add_argument("--small-object-crop-min-keep", type=positive_int)
    parser.add_argument("--optimizer")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--warmup-bias-lr", type=float)
    parser.add_argument("--prodigy-d0", type=float)
    parser.add_argument("--prodigy-d-coef", type=float)
    parser.add_argument("--prodigy-growth-rate", type=float)
    parser.add_argument("--prodigy-slice-p", type=positive_int)
    parser.add_argument("--prodigy-decouple", action="store_true")
    parser.add_argument("--no-prodigy-decouple", action="store_true")
    parser.add_argument("--prodigy-use-bias-correction", action="store_true")
    parser.add_argument("--no-prodigy-use-bias-correction", action="store_true")
    parser.add_argument("--prodigy-safeguard-warmup", action="store_true")
    parser.add_argument("--no-prodigy-safeguard-warmup", action="store_true")
    parser.add_argument("--tal-topk-one2many", type=positive_int)
    parser.add_argument("--tal-topk-one2one", type=positive_int)
    parser.add_argument("--tal-topk2-one2one", type=positive_int)
    parser.add_argument("--tal-high-gt-threshold", type=positive_int)
    parser.add_argument("--tal-high-gt-topk-one2many", type=positive_int)
    parser.add_argument("--tal-high-gt-topk-one2one", type=positive_int)
    parser.add_argument("--tal-high-gt-topk2-one2one", type=positive_int)
    parser.add_argument("--tal-metric-chunk-gt", type=positive_int)
    parser.add_argument("--det-class-mask-normalization", choices=("sqrt", "linear", "none", "off"))
    parser.add_argument("--det-partial-cls-positive-only", action="store_true")
    parser.add_argument("--no-det-partial-cls-positive-only", action="store_true")
    parser.add_argument("--det-area-loss-weight", action="store_true")
    parser.add_argument("--no-det-area-loss-weight", action="store_true")
    parser.add_argument("--det-area-loss-weight-max", type=float)
    parser.add_argument("--det-nwd-ratio", type=float)
    parser.add_argument("--det-nwd-constant", type=float)
    parser.add_argument("--cos-lr", action="store_true", help="force cosine LR scheduler")
    parser.add_argument("--no-cos-lr", action="store_true", help="force non-cosine LR scheduler")
    parser.add_argument("--amp", action="store_true", help="force AMP on")
    parser.add_argument("--no-amp", action="store_true", help="force AMP off")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="run Stage A data preparation before training")
    parser.add_argument("--no-val", action="store_true", help="disable validation")
    parser.add_argument("--plots", action="store_true", help="force train/val plots on")
    parser.add_argument("--no-plots", action="store_true", help="force train/val plots off")
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
    val_batch = args.val_batch if args.val_batch is not None else defaults.get("val_batch")
    val_workers = args.val_workers if args.val_workers is not None else defaults.get("val_workers")
    accumulate = int(args.accumulate if args.accumulate is not None else defaults.get("accumulate", 1))
    no_val = bool(args.no_val or defaults.get("no_val", False))
    val_samples = args.val_samples if args.val_samples is not None else defaults.get("val_samples")
    plots_default = defaults.get("plots")
    plots = False if no_val else (truthy(plots_default) if plots_default is not None else True)
    if args.plots:
        plots = True
    if args.no_plots:
        plots = False

    overrides = dict(
        data=str(args.data),
        epochs=int(args.epochs if args.epochs is not None else defaults.get("epochs", stage.get("epochs", 50))),
        imgsz=imgsz,
        batch=batch,
        val_batch=int(val_batch) if val_batch is not None else None,
        val_workers=int(val_workers) if val_workers is not None else None,
        nbs=batch * accumulate,
        workers=int(args.workers if args.workers is not None else defaults.get("workers", 8)),
        freeze=int(args.freeze if args.freeze is not None else defaults.get("freeze", 0)),
        project=str(args.project),
        name=args.name or f"yolo26ps_{args.stage.lower()}",
        task="detect",
        close_mosaic=0,
        patience=50,
        resume=args.resume,
        fraction=args.fraction,
        sampling=args.sampling or defaults.get("sampling") or stage.get("sampling") or "sequential",
        samples_per_epoch=args.samples_per_epoch or defaults.get("samples_per_epoch") or stage.get("samples_per_epoch"),
        sampling_weights=args.sampling_weights or defaults.get("sampling_weights") or stage.get("sampling_weights") or {},
        val=not no_val,
        val_samples=val_samples,
        save=not args.no_save,
        plots=plots,
        multi_scale=0.0 if stage.get("multi_scale") is False else float(defaults.get("multi_scale", 0.0)),
    )
    if args.class_aware_sampling and args.no_class_aware_sampling:
        raise ValueError("Use only one of --class-aware-sampling or --no-class-aware-sampling.")
    if args.small_object_sampling and args.no_small_object_sampling:
        raise ValueError("Use only one of --small-object-sampling or --no-small-object-sampling.")
    if args.det_area_loss_weight and args.no_det_area_loss_weight:
        raise ValueError("Use only one of --det-area-loss-weight or --no-det-area-loss-weight.")
    if args.class_aware_sampling:
        overrides["class_aware_sampling"] = True
    elif args.no_class_aware_sampling:
        overrides["class_aware_sampling"] = False
    elif defaults.get("class_aware_sampling") is not None:
        overrides["class_aware_sampling"] = bool(defaults["class_aware_sampling"])
    for key, cli_key in (
        ("class_aware_source", "class_aware_source"),
        ("class_aware_power", "class_aware_power"),
        ("class_aware_min_multiplier", "class_aware_min_multiplier"),
        ("class_aware_max_multiplier", "class_aware_max_multiplier"),
        ("small_object_source", "small_object_source"),
        ("small_object_area", "small_object_area"),
        ("small_object_boost", "small_object_boost"),
        ("small_object_crop", "small_object_crop"),
        ("small_object_crop_source", "small_object_crop_source"),
        ("small_object_crop_area", "small_object_crop_area"),
        ("small_object_crop_scale", "small_object_crop_scale"),
        ("small_object_crop_min_keep", "small_object_crop_min_keep"),
        ("det_area_loss_weight_max", "det_area_loss_weight_max"),
        ("det_nwd_ratio", "det_nwd_ratio"),
        ("det_nwd_constant", "det_nwd_constant"),
    ):
        cli_value = getattr(args, cli_key)
        value = cli_value if cli_value is not None else defaults.get(key)
        if value is not None:
            overrides[key] = value
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
        ("prodigy_d0", "prodigy_d0"),
        ("prodigy_d_coef", "prodigy_d_coef"),
        ("prodigy_growth_rate", "prodigy_growth_rate"),
        ("prodigy_slice_p", "prodigy_slice_p"),
    ):
        cli_value = getattr(args, cli_key)
        value = cli_value if cli_value is not None else defaults.get(key)
        if value is not None:
            overrides[key] = value
    for key, cli_on, cli_off in (
        ("prodigy_decouple", "prodigy_decouple", "no_prodigy_decouple"),
        ("prodigy_use_bias_correction", "prodigy_use_bias_correction", "no_prodigy_use_bias_correction"),
        ("prodigy_safeguard_warmup", "prodigy_safeguard_warmup", "no_prodigy_safeguard_warmup"),
    ):
        if getattr(args, cli_on) and getattr(args, cli_off):
            raise ValueError(f"Use only one of --{cli_on.replace('_', '-')} or --{cli_off.replace('_', '-')}.")
        if getattr(args, cli_on):
            overrides[key] = True
        elif getattr(args, cli_off):
            overrides[key] = False
        elif defaults.get(key) is not None:
            overrides[key] = bool(defaults[key])
    for key, cli_key in (
        ("tal_topk_one2many", "tal_topk_one2many"),
        ("tal_topk_one2one", "tal_topk_one2one"),
        ("tal_topk2_one2one", "tal_topk2_one2one"),
        ("tal_high_gt_threshold", "tal_high_gt_threshold"),
        ("tal_high_gt_topk_one2many", "tal_high_gt_topk_one2many"),
        ("tal_high_gt_topk_one2one", "tal_high_gt_topk_one2one"),
        ("tal_high_gt_topk2_one2one", "tal_high_gt_topk2_one2one"),
        ("tal_metric_chunk_gt", "tal_metric_chunk_gt"),
    ):
        cli_value = getattr(args, cli_key, None)
        value = cli_value if cli_value is not None else defaults.get(key)
        if value is not None:
            overrides[key] = int(value)
    det_mask_norm = args.det_class_mask_normalization or defaults.get("det_class_mask_normalization")
    if det_mask_norm is not None:
        overrides["det_class_mask_normalization"] = str(det_mask_norm)
    if args.det_partial_cls_positive_only:
        overrides["det_partial_cls_positive_only"] = True
    elif args.no_det_partial_cls_positive_only:
        overrides["det_partial_cls_positive_only"] = False
    elif defaults.get("det_partial_cls_positive_only") is not None:
        overrides["det_partial_cls_positive_only"] = bool(defaults.get("det_partial_cls_positive_only"))
    if args.small_object_sampling:
        overrides["small_object_sampling"] = True
    elif args.no_small_object_sampling:
        overrides["small_object_sampling"] = False
    elif defaults.get("small_object_sampling") is not None:
        overrides["small_object_sampling"] = bool(defaults.get("small_object_sampling"))
    if args.det_area_loss_weight:
        overrides["det_area_loss_weight"] = True
    elif args.no_det_area_loss_weight:
        overrides["det_area_loss_weight"] = False
    elif defaults.get("det_area_loss_weight") is not None:
        overrides["det_area_loss_weight"] = bool(defaults.get("det_area_loss_weight"))
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
        if data_yaml:
            data_yaml = Path(data_yaml)
            args.data = data_yaml if data_yaml.is_absolute() else ROOT / data_yaml
        else:
            args.data = STAGE_DATA_YAMLS.get(args.stage, DATA_YAML)
    YOLO26PSStageTrainer.stage_cfg = stage
    maybe_prepare(args)
    model = YOLO(str(args.weights or args.model))
    if args.pretrain:
        if args.weights:
            LOGGER.warning("--pretrain is ignored when --weights is provided; same-architecture checkpoint already loaded.")
        else:
            LOGGER.info(f"Loading partial pretrained weights into {args.model}: {args.pretrain}")
            model.load(args.pretrain)
    model.train(trainer=YOLO26PSStageTrainer, **build_overrides(args, plan, stage))


if __name__ == "__main__":
    main()
