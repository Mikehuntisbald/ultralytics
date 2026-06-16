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
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils import LOGGER, RANK, YAML, nms, ops
from ultralytics.utils.metrics import DetMetrics, OKS_SIGMA, PoseMetrics, kpt_iou
from ultralytics.utils.torch_utils import unwrap_model


DATA_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
STAGE_A_ROOT = DATA_ROOT / "YOLO26PS_STAGE_A"
DATA_YAML = STAGE_A_ROOT / "yolo26ps_stage_a.yaml"
MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26s-ps25d.yaml"
STAGE_D_MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26s-ps25d-stage-d-seg.yaml"
STAGE_D_SEGBRANCH_MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26s-ps25d-stage-d-segbranch.yaml"
PLAN_YAML = ROOT / "ultralytics/cfg/datasets/yolo26-ps25d-plan.yaml"
STAGE_DATA_YAMLS = {
    "A_detection": DATA_YAML,
    "A_person_only": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_a_person_only.yaml",
    "A_detection_stable": DATA_YAML,
    "A_det_escape_bridge": DATA_YAML,
    "A_det_escape_main": DATA_YAML,
    "A_det_escape_rare_small": DATA_YAML,
    "A_det_escape_probe": STAGE_A_ROOT / "yolo26ps_stage_a_probe_escape.yaml",
    "B_pose2d_probe": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d.yaml",
    "B_pose2d_det_probe": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d.yaml",
    "B_pose2d": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_b_pose2d.yaml",
    "C_pose25d": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d_detreplay.yaml",
    "C_pose25d_poseonly": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d_poseonly_ochuman.yaml",
    "C_pose25d_refine2d": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "C_pose25d_refine2d_detbox": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "C_pose25d_refine2d_balanced": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "C_pose25d_refine2d_clean": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "C_pose25d_refine2d_targetfit": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "C_pose25d_refine2d_denseanchors": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "C_pose25d_refine2d_posewide": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml",
    "C_pose25d_refine2d_targetclean": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d_targetclean.yaml",
    "C_pose25d_refine2d_targetstrict": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d_targetstrict.yaml",
    "C_pose25d_refine2d_targeteasy": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d_targeteasy.yaml",
    "C_pose25d_refine2d_targeteasy_neckprobe": ROOT
    / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d_targeteasy.yaml",
    "C_det_reanchor": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_det_reanchor.yaml",
    "D_person_mask": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_d_person_mask_person_only.yaml",
    "D_det_recover_objects365": DATA_YAML,
    "D_det_recover_objects365_unfreeze": DATA_YAML,
    "D_det_recover_objects365_shock": DATA_YAML,
    "D_det_recover_objects365_prodigy_fast": DATA_YAML,
    "D_det_recover_objects365_prodigy_unfreeze_fast": DATA_YAML,
    "E_scene_seg": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_e_scene_seg.yaml",
    "F_full_finetune": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_f_full_finetune.yaml",
    "F_full_finetune_pose_heavy": ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_f_full_finetune.yaml",
}

LOSS_KEYS = ("det", "human_det", "pose2d", "pose_z", "pose_vis", "pose_rle", "bone", "person_mask", "scene_seg")
MIXED_AUG_KEYS = ("mosaic", "mixup", "copy_paste", "cutmix")
BRANCH_MODULES = {
    "det": ("cv2", "cv3", "one2one_cv2", "one2one_cv3"),
    "human_det": ("human_cv2", "human_cv3", "one2one_human_cv2", "one2one_human_cv3"),
    "pose": (
        "pose_cv2",
        "pose_cv3",
        "one2one_pose_cv2",
        "one2one_pose_cv3",
        "cv4",
        "one2one_cv4",
        "cv4_kpts",
        "one2one_cv4_kpts",
        "cv4_z",
        "one2one_cv4_z",
        "cv4_sigma",
        "one2one_cv4_sigma",
        "flow_model",
    ),
    "pose_adapter": ("pose_adapter",),
    "seg": ("cv5", "one2one_cv5", "proto"),
    "sem": ("scene_seg",),
    "p2_dense": ("p2_refine",),
}
BRANCH_TRAIN_FLAGS = {
    "det": "det_head",
    "human_det": "human_det_head",
    "pose": "body25d_head",
    "pose_adapter": "pose_adapter",
    "seg": "seg_head",
    "seg_neck": "seg_neck",
    "sem": "sem_head",
}
MODEL_GROUP_RANGES = {
    "backbone": (0, 11),
    "det_neck": (11, 29),
    "seg_neck": (0, 0),
    "pose_neck": (29, -1),
}
NORM_TYPES = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
HEAD_BN_ALIASES = {
    "a": "det",
    "stage_a": "det",
    "detection": "det",
    "det_head": "det",
    "human": "human_det",
    "human_det": "human_det",
    "human_detection": "human_det",
    "human_det_head": "human_det",
    "person_only": "human_det",
    "person_face": "human_det",
    "b": "pose",
    "stage_b": "pose",
    "c": "pose",
    "stage_c": "pose",
    "body25d": "pose",
    "body25d_head": "pose",
    "pose2d": "pose",
    "pose25d": "pose",
    "pose_head": "pose",
    "pose_adapter": "pose_adapter",
    "pose_only_adapter": "pose_adapter",
    "d": "seg",
    "stage_d": "seg",
    "person_mask": "seg",
    "person_seg": "seg",
    "instance_seg": "seg",
    "mask": "seg",
    "mask_head": "seg",
    "seg": "seg",
    "seg_head": "seg",
    "seg_neck": "seg",
    "e": "sem",
    "stage_e": "sem",
    "scene_seg": "sem",
    "scene_seg_head": "sem",
    "scene": "sem",
    "sem": "sem",
    "sem_head": "sem",
    "semantic_seg": "sem",
    "semantic_seg_head": "sem",
    "p2": "p2_dense",
    "p2_refine": "p2_dense",
    "adapter": "p2_dense",
    "seg_adapt": "p2_dense",
    "seg_adapter": "p2_dense",
}

TASK_ALIASES = {
    "mask": "seg",
    "person_mask": "seg",
    "person_seg": "seg",
    "instance_seg": "seg",
    "scene": "sem",
    "scene_seg": "sem",
    "semantic_seg": "sem",
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


def normalize_name_list(value: Any) -> set[str]:
    """Normalize a YAML/CLI module-name list to lowercase names."""
    if value in (None, False):
        return set()
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = [value]
    return {str(x).strip().lower() for x in raw if str(x).strip()}


def normalize_decode_head(value: Any) -> str:
    """Normalize staged person-mask assignment/decode head names."""
    head = str(value or "seg").strip().lower()
    if head in {"seg", "segment", "mask", "det", "default"}:
        return "seg"
    if head in {"human", "human_det", "person", "person_face"}:
        return "human"
    return "seg"


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


def train_group_enabled(train: dict[str, Any], key: str, default: bool = True) -> bool:
    """Read model-group trainability with a legacy fallback for the old combined neck key."""
    if key in {"det_neck", "seg_neck", "pose_neck"} and key not in train and "neck" in train:
        return train_flag_enabled(train, "neck", default=default)
    return train_flag_enabled(train, key, default=default)


def truthy(value: Any) -> bool:
    """Read boolean-like YAML/CLI values."""
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "no", "off", "none", "null"}
    return bool(value)


def normalize_task_name(task: Any) -> str:
    """Normalize staged task/branch aliases to canonical training branch names."""
    name = str(task).strip().lower()
    return TASK_ALIASES.get(name, name)


def branch_flag_enabled(train: dict[str, Any], canonical_key: str, legacy_key: str, default: bool = True) -> bool:
    """Read a canonical train flag with support for the previous legacy key."""
    if canonical_key in train:
        return train_flag_enabled(train, canonical_key, default=default)
    return train_flag_enabled(train, legacy_key, default=default)


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
        tasks = {normalize_task_name(task) for task in runtime_tasks}
        return tasks
    train = stage.get("train") or {}
    loss = stage.get("loss") or {}
    tasks = {"det"}
    if train.get("all"):
        return {"det", "human_det", "pose", "seg", "sem"}
    if train.get("human_det_head") or float(loss.get("human_det", 0.0)):
        tasks.add("human_det")
    if train.get("body25d_head") or any(
        float(loss.get(k, 0.0)) for k in ("pose2d", "pose_z", "pose_vis", "pose_rle", "bone")
    ):
        tasks.add("pose")
    if train.get("seg_head", train.get("mask_head")) or float(loss.get("person_mask", 0.0)):
        tasks.add("seg")
    if train.get("sem_head", train.get("scene_seg_head")) or float(loss.get("scene_seg", 0.0)):
        tasks.add("sem")
    return tasks


def branch_trainable_from_stage(stage: dict[str, Any], group: str, active: bool) -> bool:
    """Return whether a head branch should update weights for the current stage."""
    train = stage.get("train") or {}
    loss = stage.get("loss") or {}
    if train.get("all"):
        return True
    if group == "det":
        return train_flag_enabled(train, "det_head", default=float(loss.get("det", 0.0)) > 0 or active)
    if group == "human_det":
        return train_flag_enabled(train, "human_det_head", default=False) or float(loss.get("human_det", 0.0)) > 0
    if group == "seg":
        return branch_flag_enabled(train, "seg_head", "mask_head", default=active)
    if group == "sem":
        return branch_flag_enabled(train, "sem_head", "scene_seg_head", default=active)
    return train_flag_enabled(train, BRANCH_TRAIN_FLAGS[group], default=active)


def complete_loss_weights(stage: dict[str, Any]) -> dict[str, float]:
    """Return all task loss weights, defaulting missing stage weights to zero."""
    loss = stage.get("loss") or {}
    return {key: float(loss.get(key, 0.0)) for key in LOSS_KEYS}


def load_stage_pretrain(model: YOLO, weights: Path) -> None:
    """Load a stage pretrain checkpoint, including old final-head weights after architecture edits."""
    yaml_file = getattr(model.model, "yaml_file", None)
    if yaml_file is None:
        yaml_file = "model"
    LOGGER.info(f"Loading partial pretrained weights into {yaml_file}: {weights}")
    ckpt, _ = torch_safe_load(weights)
    source_model = (ckpt.get("ema") or ckpt["model"]).float()
    source = source_model.state_dict()
    target = model.model.state_dict()
    transfer = {k: v for k, v in source.items() if k in target and v.shape == target[k].shape}
    final_old, final_new = _final_layer_prefix(source), _final_layer_prefix(target)
    remapped = 0
    if final_old and final_new and final_old != final_new:
        old_prefix, new_prefix = f"model.{final_old}.", f"model.{final_new}."
        for key, value in source.items():
            if not key.startswith(old_prefix):
                continue
            new_key = new_prefix + key[len(old_prefix) :]
            if new_key in target and new_key not in transfer and value.shape == target[new_key].shape:
                transfer[new_key] = value
                remapped += 1
    person_rows = _transfer_person_only_cls_rows(source, target, transfer, final_old, final_new)
    tail_human = _transfer_tail_aligned_human_head(source, target, transfer, final_old, final_new)
    model.model.load_state_dict(transfer, strict=False)
    LOGGER.info(
        f"Transferred {len(transfer)}/{len(target)} items from stage pretrain weights "
        f"({remapped} final-head remapped, {person_rows} person-only cls rows, "
        f"{tail_human} tail-aligned human-det items)"
    )
    model.ckpt = {"model": model.model}
    model.ckpt_path = str(weights)


def _final_layer_prefix(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Return the largest numeric model layer prefix in a YOLO state dict."""
    layers = []
    for key in state_dict:
        parts = key.split(".", 2)
        if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
            layers.append(int(parts[1]))
    return max(layers) if layers else None


def _transfer_person_only_cls_rows(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    transfer: dict[str, torch.Tensor],
    final_old: int | None,
    final_new: int | None,
    person_row: int = 0,
) -> int:
    """Slice COCO person logits from an 80-class YOLO seg head into a 1-class Stage D head."""
    if final_old is None or final_new is None:
        return 0
    old_prefix, new_prefix = f"model.{final_old}.", f"model.{final_new}."
    copied = 0
    for key, value in source.items():
        if not key.startswith(old_prefix) or not any(x in key for x in (".cv3.", ".one2one_cv3.", ".proto.semseg.")):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        if new_key not in target or new_key in transfer:
            continue
        dst = target[new_key]
        if value.ndim >= 1 and dst.shape[0] == 1 and value.shape[0] > person_row and value.shape[1:] == dst.shape[1:]:
            transfer[new_key] = value[person_row : person_row + 1].clone()
            copied += 1
    return copied


def _head_branch_layers(state_dict: dict[str, torch.Tensor], final_layer: int | None, branch: str) -> list[int]:
    """Return sorted per-scale layer indices for a final-head branch."""
    if final_layer is None:
        return []
    prefix = f"model.{final_layer}.{branch}."
    layers = set()
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        part = key[len(prefix) :].split(".", 1)[0]
        if part.isdigit():
            layers.add(int(part))
    return sorted(layers)


def _transfer_tail_aligned_human_head(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    transfer: dict[str, torch.Tensor],
    final_old: int | None,
    final_new: int | None,
) -> int:
    """Map 4-scale PS human-det heads to 3-scale Stage D P3-P5 adapter heads.

    Stage C/A human_det is P2-P5, while the det-neck adapter segment head consumes only
    PAN P3-P5. Same-name partial loading would silently copy P2/P3/P4 into P3/P4/P5,
    so align the target's three scales with the source tail instead.
    """
    copied = 0
    if final_old is None or final_new is None:
        return copied
    for branch in ("human_cv2", "human_cv3", "one2one_human_cv2", "one2one_human_cv3"):
        src_layers = _head_branch_layers(source, final_old, branch)
        dst_layers = _head_branch_layers(target, final_new, branch)
        if not src_layers or not dst_layers or len(src_layers) <= len(dst_layers):
            continue
        layer_map = dict(zip(dst_layers, src_layers[-len(dst_layers) :]))
        src_prefix = f"model.{final_old}.{branch}."
        dst_prefix = f"model.{final_new}.{branch}."
        for dst_layer, src_layer in layer_map.items():
            src_layer_prefix = f"{src_prefix}{src_layer}."
            dst_layer_prefix = f"{dst_prefix}{dst_layer}."
            for key, value in source.items():
                if not key.startswith(src_layer_prefix):
                    continue
                new_key = dst_layer_prefix + key[len(src_layer_prefix) :]
                if new_key in target and value.shape == target[new_key].shape:
                    transfer[new_key] = value
                    copied += 1
    return copied


def _copy_seg_pretrain_tensor(
    target: dict[str, torch.Tensor],
    transfer: dict[str, torch.Tensor],
    new_key: str,
    value: torch.Tensor,
    person_row: int,
) -> bool:
    """Copy a source tensor, slicing COCO person rows when the Stage D tensor is person-only."""
    if new_key not in target:
        return False
    dst = target[new_key]
    if value.shape == dst.shape:
        transfer[new_key] = value
        return True
    if value.ndim >= 1 and dst.shape[0] == 1 and value.shape[0] > person_row and value.shape[1:] == dst.shape[1:]:
        transfer[new_key] = value[person_row : person_row + 1].clone()
        return True
    return False


def load_seg_head_pretrain(
    model: YOLO, weights: Path, person_row: int = 0, scope: str = "head", neck_offset: int | None = None
) -> None:
    """Load YOLO26 segmentation pretrained weights into a Stage D segmentation branch."""
    yaml_file = getattr(model.model, "yaml_file", None) or "model"
    scope = str(scope or "head").strip().lower()
    LOGGER.info(f"Loading YOLO26 seg {scope} weights into {yaml_file}: {weights}")
    ckpt, _ = torch_safe_load(weights)
    source_model = (ckpt.get("ema") or ckpt["model"]).float()
    source = source_model.state_dict()
    target = model.model.state_dict()
    final_old, final_new = _final_layer_prefix(source), _final_layer_prefix(target)
    transfer: dict[str, torch.Tensor] = {}
    copied = 0
    skipped = 0
    if final_old is None or final_new is None:
        LOGGER.warning("Could not identify final head layer while loading seg head pretrain.")
        return
    if scope in {"all", "backbone_neck_head", "full"}:
        for key, value in source.items():
            if key.startswith(f"model.{final_old}."):
                continue
            if key in target and value.shape == target[key].shape:
                transfer[key] = value
                copied += 1
    if scope in {"neck_head", "segbranch", "branch", "all", "backbone_neck_head", "full"}:
        # Map the official yolo26-seg neck layers 11..22 onto the Stage D segbranch neck.
        # For yolo26-ps25d-stage-d-segbranch.yaml the seg neck starts at layer 20 because
        # layers 11..19 are the frozen PS detector neck used only for P2 fusion.
        offset = int(9 if neck_offset is None else neck_offset)
        for key, value in source.items():
            parts = key.split(".", 2)
            if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
                continue
            layer = int(parts[1])
            if not (11 <= layer < final_old):
                continue
            new_key = f"model.{layer + offset}.{parts[2]}"
            if _copy_seg_pretrain_tensor(target, transfer, new_key, value, person_row):
                copied += 1
    old_prefix, new_prefix = f"model.{final_old}.", f"model.{final_new}."
    for key, value in source.items():
        if not key.startswith(old_prefix):
            continue
        rel = key[len(old_prefix) :]
        if rel.startswith(("seg_adapt.", "p2_refine.", "human_")) or rel.startswith("one2one_human_"):
            continue
        new_key = new_prefix + rel
        if _copy_seg_pretrain_tensor(target, transfer, new_key, value, person_row):
            copied += 1
        else:
            skipped += 1
    model.model.load_state_dict(transfer, strict=False)
    LOGGER.info(f"Transferred {len(transfer)} YOLO26 seg {scope} items from {weights} ({skipped} head items skipped).")


def apply_loss_overrides(stage: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply optional CLI loss-weight overrides to a stage config."""
    overrides = {
        key: float(value)
        for key in LOSS_KEYS
        if (value := getattr(args, f"loss_{key}", None)) is not None
    }
    if not overrides:
        return stage
    stage = dict(stage)
    loss = dict(stage.get("loss") or {})
    loss.update(overrides)
    stage["loss"] = loss
    return stage


def pose_fitness_sources(value: Any) -> dict[str, float]:
    """Parse optional source weights for pose MPJPE fitness."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k).strip().lower(): float(v) for k, v in value.items() if float(v)}
    if isinstance(value, (list, tuple, set)):
        return {str(k).strip().lower(): 1.0 for k in value if str(k).strip()}
    sources = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, weight = item.split("=", 1)
            sources[name.strip().lower()] = float(weight)
        else:
            sources[item.lower()] = 1.0
    return {k: v for k, v in sources.items() if v}


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
        "warmup_momentum",
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
        "nbs",
        "close_mosaic",
        "o2m",
        "final_o2m",
        "o2m_decay_updates",
        "o2m_mix_mode",
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
        pose = None
        if isinstance(preds, (list, tuple)) and preds:
            deploy = preds[0]
            if isinstance(deploy, (list, tuple)) and len(deploy) == 5:
                det, pose = deploy[0], deploy[1]
            else:
                det = deploy
        else:
            det = preds
        if pose is None:
            return super().postprocess(det)

        if det.shape[-1] == 6 or self.end2end:
            preds_out = []
            classes = None if self.args.classes is None else torch.tensor(self.args.classes, device=det.device)
            for i, pred in enumerate(det):
                keep = pred[:, 4] > self.args.conf
                if classes is not None:
                    keep &= (pred[:, 5:6] == classes).any(1)
                # End-to-end deploy outputs are already top-k, but class filtering can leave sparse indices.
                # Keep pose indices aligned while preserving the highest-confidence boxes for source metrics.
                idx = torch.where(keep)[0]
                if idx.numel():
                    idx = idx[pred[idx, 4].argsort(descending=True)[: self.args.max_det]]
                x = pred[idx]
                preds_out.append(
                    {
                        "bboxes": x[:, :4],
                        "conf": x[:, 4],
                        "cls": x[:, 5],
                        "extra": x[:, 6:],
                        "pose25d": pose[i, idx] if len(idx) else pose.new_zeros((0, pose.shape[2], pose.shape[3])),
                    }
                )
            return preds_out

        outputs, keep_idxs = nms.non_max_suppression(
            det,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
            return_idxs=True,
        )
        preds_out = []
        for i, x in enumerate(outputs):
            pred = {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]}
            if len(x):
                pred["pose25d"] = pose[i, keep_idxs[i].long()]
            else:
                pred["pose25d"] = pose.new_zeros((0, pose.shape[2], pose.shape[3]))
            preds_out.append(pred)
        return preds_out

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.person_cls = int(self.data.get("person_cls", 0))
        self.face_cls = int(self.data.get("face_cls", max(self.nc - 1, 0)))
        self.stage_metric_buckets = {name: DetMetrics(names=self.names) for name in self._stage_bucket_names()}
        self.pose_metrics = PoseMetrics(names=self.names)
        self.kpt_shape = self.data.get("kpt_shape", [17, 3])
        nkpt = int(self.kpt_shape[0]) if self.kpt_shape else 17
        self.sigma = OKS_SIGMA if self.kpt_shape == [17, 3] else np.ones(nkpt) / nkpt
        self.stage_task_counts = {"pose2d": 0, "pose3d": 0, "person_mask": 0, "scene_seg": 0}
        self.pose2d_mpjpes: list[float] = []
        self.pose2d_input_mpjpes: list[float] = []
        self.pose2d_box_h_norms: list[float] = []
        self.pose2d_source_stats = {
            "coco_keypoints": {"mpjpes": [], "input_mpjpes": [], "box_h_norms": [], "matched": 0, "total": 0},
            "coco_wholebody": {"mpjpes": [], "input_mpjpes": [], "box_h_norms": [], "matched": 0, "total": 0},
            "ochuman": {"mpjpes": [], "input_mpjpes": [], "box_h_norms": [], "matched": 0, "total": 0},
            "3dpw": {"mpjpes": [], "input_mpjpes": [], "box_h_norms": [], "matched": 0, "total": 0},
            "agora": {"mpjpes": [], "input_mpjpes": [], "box_h_norms": [], "matched": 0, "total": 0},
        }
        self.pose2d_matched = 0
        self.pose2d_total = 0

    @staticmethod
    def _stage_bucket_names() -> tuple[str, ...]:
        return (
            "objects365",
            "objects365_person",
            "crowdhuman_person",
            "wider_face",
            "coco_person",
            "ochuman_person",
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
        if "ochuman" in parts or "ochuman" in normalized:
            return "ochuman"
        if "coco_keypoints" in parts or "coco_keypoints" in normalized:
            return "coco_keypoints"
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
            pose_stat = self._process_pose_batch(predn, pbatch, batch, si)
            self._update_pose2d_mpjpe(predn, pbatch, batch, si)
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
            if pose_stat is not None:
                self.pose_metrics.update_stats(pose_stat)

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
            elif source == "ochuman":
                self._bucket_update("ochuman_person", predn, pbatch, class_id=self.person_cls)
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

    def _process_pose_batch(
        self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any], batch: dict[str, Any], si: int
    ) -> dict[str, np.ndarray] | None:
        """Return COCO OKS pose TP stats for pose-capable person instances."""
        if "pose25d" not in predn or predn["pose25d"].numel() == 0:
            return None
        keypoints = batch.get("keypoints")
        if keypoints is None or not torch.is_tensor(keypoints) or keypoints.numel() == 0:
            return None
        idx = batch["batch_idx"] == si
        if not bool(idx.any()):
            return None
        gt_cls_all = batch["cls"][idx].view(-1).to(self.device)
        gt_kpts = keypoints[idx].to(self.device).float()
        instance_flags = batch.get("instance_flags")
        flags = instance_flags[idx].to(self.device).bool() if torch.is_tensor(instance_flags) else None
        gt_pose_mask = flags[:, 1] if flags is not None else gt_kpts[..., 2].gt(0).any(-1)
        gt_pose_mask &= gt_cls_all.long().eq(self.person_cls)
        gt_pose_mask &= gt_kpts[..., 2].gt(0).sum(-1).ge(1)
        pred_mask = predn["cls"].long().eq(self.person_cls)
        pred_boxes = predn["bboxes"][pred_mask]
        pred_pose = predn["pose25d"][pred_mask]
        pred_conf = predn["conf"][pred_mask]
        pred_cls = predn["cls"][pred_mask]
        if pred_boxes.numel() == 0 and not bool(gt_pose_mask.any()):
            return None

        gt_cls = gt_cls_all[gt_pose_mask]
        gt_boxes = pbatch["bboxes"][gt_pose_mask]
        gt_kpts = gt_kpts[gt_pose_mask].clone()
        if gt_kpts.numel():
            gt_kpts[..., 0] *= pbatch["imgsz"][1]
            gt_kpts[..., 1] *= pbatch["imgsz"][0]
        pred_xy = self._pose_norm_to_input_xy(pred_pose, pred_boxes)
        pred_kpts = pred_pose.new_zeros((pred_pose.shape[0], pred_pose.shape[1], 3))
        if pred_pose.numel():
            pred_kpts[..., :2] = pred_xy
            pred_kpts[..., 2] = pred_pose[..., 3]

        if gt_cls.shape[0] == 0 or pred_cls.shape[0] == 0:
            tp_p = np.zeros((pred_cls.shape[0], self.niou), dtype=bool)
        else:
            area = ops.xyxy2xywh(gt_boxes)[:, 2:].prod(1) * 0.53
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
            tp_p = self.match_predictions(pred_cls, gt_cls, iou).cpu().numpy()
        target_cls = gt_cls.cpu().numpy()
        pred_cls_np = pred_cls.cpu().numpy()
        return {
            "tp": np.zeros((pred_cls.shape[0], self.niou), dtype=bool),
            "tp_p": tp_p,
            "conf": pred_conf.cpu().numpy(),
            "pred_cls": pred_cls_np,
            "target_cls": target_cls,
            "target_img": np.unique(target_cls),
            "im_name": Path(pbatch["im_file"]).name,
        }

    def _update_pose2d_mpjpe(
        self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any], batch: dict[str, Any], si: int
    ) -> None:
        """Accumulate bbox-matched 2D pose MPJPE in original-image pixels."""
        if "pose25d" not in predn or predn["pose25d"].numel() == 0:
            return
        keypoints = batch.get("keypoints")
        instance_flags = batch.get("instance_flags")
        if keypoints is None or not torch.is_tensor(keypoints) or keypoints.numel() == 0:
            return

        idx = batch["batch_idx"] == si
        if not bool(idx.any()):
            return
        gt_cls = batch["cls"][idx].view(-1).to(self.device)
        gt_kpts = keypoints[idx].to(self.device).float()
        gt_boxes = pbatch["bboxes"]
        flags = instance_flags[idx].to(self.device).bool() if torch.is_tensor(instance_flags) else None
        gt_pose_mask = flags[:, 1] if flags is not None else gt_kpts[..., 2].gt(0).any(-1)
        gt_pose_mask &= gt_cls.long().eq(self.person_cls)
        gt_pose_mask &= gt_kpts[..., 2].gt(0).sum(-1).ge(1)
        if not bool(gt_pose_mask.any()):
            return

        gt_boxes = gt_boxes[gt_pose_mask]
        gt_kpts = gt_kpts[gt_pose_mask].clone()
        self.pose2d_total += int(gt_boxes.shape[0])
        source = self._source_name(pbatch["im_file"])
        source_stats = self.pose2d_source_stats.get(source)
        if source_stats is not None:
            source_stats["total"] += int(gt_boxes.shape[0])

        pred_mask = predn["cls"].long().eq(self.person_cls)
        pred_boxes = predn["bboxes"][pred_mask]
        pred_pose = predn["pose25d"][pred_mask]
        if pred_boxes.numel() == 0:
            return

        pred_boxes_orig = self.scale_preds({"bboxes": pred_boxes.clone()}, pbatch)["bboxes"]
        pred_xy_input = self._pose_norm_to_input_xy(pred_pose, pred_boxes)
        pred_xy = ops.scale_coords(
            pbatch["imgsz"],
            pred_xy_input.clone(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )
        gt_boxes_orig = self.scale_preds({"bboxes": gt_boxes.clone()}, pbatch)["bboxes"]
        gt_kpts[..., 0] *= pbatch["imgsz"][1]
        gt_kpts[..., 1] *= pbatch["imgsz"][0]
        gt_xy_input = gt_kpts[..., :2].clone()
        gt_xy = ops.scale_coords(
            pbatch["imgsz"],
            gt_xy_input.clone(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )

        iou = self._box_iou_original(gt_boxes_orig, pred_boxes_orig)
        used: set[int] = set()
        for gt_i in range(gt_boxes.shape[0]):
            candidates = [(float(iou[gt_i, pred_i]), pred_i) for pred_i in range(pred_boxes.shape[0]) if pred_i not in used]
            if not candidates:
                continue
            best_iou, pred_i = max(candidates, key=lambda item: item[0])
            if best_iou < 0.50:
                continue
            visible = gt_kpts[gt_i, :, 2].gt(0)
            if not bool(visible.any()):
                continue
            dist = (pred_xy[pred_i, visible] - gt_xy[gt_i, visible]).norm(dim=-1)
            dist_input = (pred_xy_input[pred_i, visible] - gt_xy_input[gt_i, visible]).norm(dim=-1)
            box_h = (gt_boxes_orig[gt_i, 3] - gt_boxes_orig[gt_i, 1]).clamp(min=1.0)
            mpjpe = float(dist.mean().detach().cpu())
            input_mpjpe = float(dist_input.mean().detach().cpu())
            box_h_norm = float((dist.mean() / box_h).detach().cpu())
            self.pose2d_mpjpes.append(mpjpe)
            self.pose2d_input_mpjpes.append(input_mpjpe)
            self.pose2d_box_h_norms.append(box_h_norm)
            self.pose2d_matched += 1
            if source_stats is not None:
                source_stats["mpjpes"].append(mpjpe)
                source_stats["input_mpjpes"].append(input_mpjpe)
                source_stats["box_h_norms"].append(box_h_norm)
                source_stats["matched"] += 1
            used.add(pred_i)

    @staticmethod
    def _pose_norm_to_input_xy(pose_norm: torch.Tensor, boxes_input: torch.Tensor) -> torch.Tensor:
        """Decode bbox-normalized pose xy to validator input-image coordinates."""
        if pose_norm.numel() == 0:
            return pose_norm.new_zeros((0, 0, 2))
        boxes = boxes_input.to(pose_norm.dtype)
        x1, y1, x2, y2 = [x.view(-1, 1) for x in boxes.T]
        xy = pose_norm[..., :2].clone()
        xy[..., 0] = x1 + xy[..., 0] * (x2 - x1).clamp(min=1.0)
        xy[..., 1] = y1 + xy[..., 1] * (y2 - y1).clamp(min=1.0)
        return xy

    @staticmethod
    def _box_iou_original(gt_boxes_orig: torch.Tensor, pred_boxes_orig: torch.Tensor) -> torch.Tensor:
        """Return IoU between original-image GT and prediction boxes."""
        if gt_boxes_orig.numel() == 0 or pred_boxes_orig.numel() == 0:
            return gt_boxes_orig.new_zeros((gt_boxes_orig.shape[0], pred_boxes_orig.shape[0]))
        x1 = torch.maximum(gt_boxes_orig[:, None, 0], pred_boxes_orig[None, :, 0])
        y1 = torch.maximum(gt_boxes_orig[:, None, 1], pred_boxes_orig[None, :, 1])
        x2 = torch.minimum(gt_boxes_orig[:, None, 2], pred_boxes_orig[None, :, 2])
        y2 = torch.minimum(gt_boxes_orig[:, None, 3], pred_boxes_orig[None, :, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_a = (gt_boxes_orig[:, 2] - gt_boxes_orig[:, 0]).clamp(min=0) * (
            gt_boxes_orig[:, 3] - gt_boxes_orig[:, 1]
        ).clamp(min=0)
        area_b = (pred_boxes_orig[:, 2] - pred_boxes_orig[:, 0]).clamp(min=0) * (
            pred_boxes_orig[:, 3] - pred_boxes_orig[:, 1]
        ).clamp(min=0)
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

    def gather_stats(self) -> None:
        super().gather_stats()
        if RANK == 0:
            for metric in self.stage_metric_buckets.values():
                self._gather_stage_metric(metric)
            self._gather_stage_metric(self.pose_metrics)
            self._gather_stage_counts()
            self._gather_pose2d_metrics()
        elif RANK > 0:
            for metric in self.stage_metric_buckets.values():
                self._gather_stage_metric(metric)
            self._gather_stage_metric(self.pose_metrics)
            self._gather_stage_counts()
            self._gather_pose2d_metrics()

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

    def _gather_pose2d_metrics(self) -> None:
        """Gather pose MPJPE lists and counters across DDP ranks."""
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(
                {
                    "mpjpes": self.pose2d_mpjpes,
                    "input_mpjpes": self.pose2d_input_mpjpes,
                    "box_h_norms": self.pose2d_box_h_norms,
                    "source_stats": self.pose2d_source_stats,
                    "matched": self.pose2d_matched,
                    "total": self.pose2d_total,
                },
                gathered,
                dst=0,
            )
            self.pose2d_mpjpes = []
            self.pose2d_input_mpjpes = []
            self.pose2d_box_h_norms = []
            self.pose2d_source_stats = {
                "coco_keypoints": {"mpjpes": [], "input_mpjpes": [], "box_h_norms": [], "matched": 0, "total": 0},
                "coco_wholebody": {"mpjpes": [], "input_mpjpes": [], "box_h_norms": [], "matched": 0, "total": 0},
                "3dpw": {"mpjpes": [], "input_mpjpes": [], "box_h_norms": [], "matched": 0, "total": 0},
                "agora": {"mpjpes": [], "input_mpjpes": [], "box_h_norms": [], "matched": 0, "total": 0},
            }
            self.pose2d_matched = 0
            self.pose2d_total = 0
            for item in gathered:
                if not item:
                    continue
                self.pose2d_mpjpes.extend(float(x) for x in item.get("mpjpes", []))
                self.pose2d_input_mpjpes.extend(float(x) for x in item.get("input_mpjpes", []))
                self.pose2d_box_h_norms.extend(float(x) for x in item.get("box_h_norms", []))
                self.pose2d_matched += int(item.get("matched", 0))
                self.pose2d_total += int(item.get("total", 0))
                for source, stats in (item.get("source_stats") or {}).items():
                    if source not in self.pose2d_source_stats:
                        self.pose2d_source_stats[source] = {
                            "mpjpes": [],
                            "input_mpjpes": [],
                            "box_h_norms": [],
                            "matched": 0,
                            "total": 0,
                        }
                    self.pose2d_source_stats[source]["mpjpes"].extend(float(x) for x in stats.get("mpjpes", []))
                    self.pose2d_source_stats[source]["input_mpjpes"].extend(
                        float(x) for x in stats.get("input_mpjpes", [])
                    )
                    self.pose2d_source_stats[source]["box_h_norms"].extend(
                        float(x) for x in stats.get("box_h_norms", [])
                    )
                    self.pose2d_source_stats[source]["matched"] += int(stats.get("matched", 0))
                    self.pose2d_source_stats[source]["total"] += int(stats.get("total", 0))
        elif RANK > 0:
            dist.gather_object(
                {
                    "mpjpes": self.pose2d_mpjpes,
                    "input_mpjpes": self.pose2d_input_mpjpes,
                    "box_h_norms": self.pose2d_box_h_norms,
                    "source_stats": self.pose2d_source_stats,
                    "matched": self.pose2d_matched,
                    "total": self.pose2d_total,
                },
                None,
                dst=0,
            )

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats.update(self._stage_results())
        stats.update(self._pose_ap_results())
        stats.update(self._pose2d_results())
        for name, count in self.stage_task_counts.items():
            stats[f"metrics/stage/{name}_images"] = count
        return stats

    def _pose_ap_results(self) -> dict[str, float]:
        """Return OKS pose AP metrics for the active validation split."""
        results = {
            "metrics/pose/precision(P)": 0.0,
            "metrics/pose/recall(P)": 0.0,
            "metrics/pose/mAP50(P)": 0.0,
            "metrics/pose/mAP75(P)": 0.0,
            "metrics/pose/mAP50-95(P)": 0.0,
        }
        has_stats = bool(self.pose_metrics.stats.get("target_cls")) and any(
            len(x) for x in self.pose_metrics.stats["target_cls"]
        )
        if not has_stats:
            return results
        self.pose_metrics.process(save_dir=self.save_dir / "pose_metrics", plot=False, on_plot=self.on_plot)
        pose_p = np.asarray(self.pose_metrics.pose.p)
        pose_r = np.asarray(self.pose_metrics.pose.r)
        pose_ap75 = float(self.pose_metrics.pose.map75)
        results["metrics/pose/precision(P)"] = float(pose_p.mean()) if pose_p.size else 0.0
        results["metrics/pose/recall(P)"] = float(pose_r.mean()) if pose_r.size else 0.0
        results["metrics/pose/mAP50(P)"] = float(self.pose_metrics.pose.map50)
        results["metrics/pose/mAP75(P)"] = pose_ap75
        results["metrics/pose/mAP50-95(P)"] = float(self.pose_metrics.pose.map)
        self.pose_metrics.clear_stats()
        self.pose_metrics.clear_image_metrics()
        return results

    def _pose2d_results(self) -> dict[str, float]:
        mpjpes = np.asarray(self.pose2d_mpjpes, dtype=np.float32)
        input_mpjpes = np.asarray(self.pose2d_input_mpjpes, dtype=np.float32)
        box_h_norms = np.asarray(self.pose2d_box_h_norms, dtype=np.float32)
        out = {
            "metrics/pose2d/mpjpe_matched": float(self.pose2d_matched),
            "metrics/pose2d/mpjpe_total": float(self.pose2d_total),
            "metrics/pose2d/mpjpe_match_ratio": float(self.pose2d_matched / max(self.pose2d_total, 1)),
        }
        out.update(self._pose2d_array_results("metrics/pose2d/mpjpe", mpjpes, suffix="px"))
        out.update(self._pose2d_array_results("metrics/pose2d/mpjpe_input", input_mpjpes, suffix="px"))
        out.update(self._pose2d_array_results("metrics/pose2d/mpjpe_box_h_norm", box_h_norms))
        source_prefix = {
            "coco_keypoints": "metrics/pose2d/source/coco",
            "coco_wholebody": "metrics/pose2d/source/coco",
            "ochuman": "metrics/pose2d/source/ochuman",
            "3dpw": "metrics/pose2d/source/3dpw",
            "agora": "metrics/pose2d/source/agora",
        }
        for source, prefix in source_prefix.items():
            stats = self.pose2d_source_stats.get(source) or {}
            source_mpjpes = np.asarray(stats.get("mpjpes", []), dtype=np.float32)
            source_input = np.asarray(stats.get("input_mpjpes", []), dtype=np.float32)
            source_norm = np.asarray(stats.get("box_h_norms", []), dtype=np.float32)
            matched = int(stats.get("matched", 0))
            total = int(stats.get("total", 0))
            out[f"{prefix}/matched"] = float(matched)
            out[f"{prefix}/total"] = float(total)
            out[f"{prefix}/match_ratio"] = float(matched / max(total, 1))
            out.update(self._pose2d_array_results(f"{prefix}/mpjpe", source_mpjpes, suffix="px"))
            out.update(self._pose2d_array_results(f"{prefix}/mpjpe_input", source_input, suffix="px"))
            out.update(self._pose2d_array_results(f"{prefix}/mpjpe_box_h_norm", source_norm))
        return out

    @staticmethod
    def _pose2d_array_results(prefix: str, values: np.ndarray, suffix: str | None = None) -> dict[str, float]:
        """Summarize a pose metric array for CSV output."""
        names = ("mean", "median", "p90")
        if suffix:
            keys = (f"{prefix}_{name}_{suffix}" for name in names)
        else:
            keys = (f"{prefix}_{name}" for name in names)
        if not values.size:
            return {key: 0.0 for key in keys}
        return {
            key: float(value)
            for key, value in zip(keys, (values.mean(), np.median(values), np.quantile(values, 0.9)))
        }

    def _stage_results(self) -> dict[str, float]:
        results = {}
        prefix_map = {
            "objects365": "metrics/stage_a/objects365",
            "objects365_person": "metrics/stage_a/objects365/person",
            "crowdhuman_person": "metrics/stage_a/crowdhuman/person",
            "wider_face": "metrics/stage_a/wider_face/face",
            "coco_person": "metrics/stage_c/coco/person",
            "ochuman_person": "metrics/stage_c/ochuman/person",
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


class YOLO26PSStageDMaskValidator(SegmentationValidator):
    """Stage D validator that evaluates adapter masks from the configured detection branch."""

    def __init__(self, *args, decode_head: str = "seg", **kwargs):
        """Initialize the Stage D mask validator with an explicit decode branch."""
        super().__init__(*args, **kwargs)
        self.decode_head = normalize_decode_head(decode_head)

    def init_metrics(self, model):
        """Initialize segmentation metrics and keep a handle to the adapter head for raw human-det decoding."""
        super().init_metrics(model)
        layers = getattr(model, "model", None)
        if hasattr(layers, "model"):
            layers = layers.model
        self.head = layers[-1]
        self.nc = 1
        self.names = {0: "Person"}
        self.metrics.names = self.names

    def _raw_one2many(self, preds):
        """Return the raw one-to-many dict for branch-specific mask decoding."""
        raw = preds[1] if isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[1], dict) else None
        if raw is None and isinstance(preds, dict):
            raw = preds
        if isinstance(raw, dict) and "one2many" in raw:
            raw = raw["one2many"]
        return raw

    def _raw_prediction_tensor(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build an NMS input tensor from raw boxes/scores and mask coefficients."""
        use_human = self.decode_head == "human" and "human_boxes" in raw and "human_scores" in raw
        decode_raw = raw
        if use_human:
            decode_raw = {
                **raw,
                "boxes": raw["human_boxes"],
                "scores": raw["human_scores"][:, :1],
                "feats": raw.get("human_feats", raw.get("feats")),
            }
        self.head._get_decode_boxes(decode_raw)
        boxes = self.head.decode_bboxes(
            self.head.dfl(decode_raw["boxes"]), self.head.anchors.unsqueeze(0), xywh=False
        )
        boxes = (boxes * self.head.strides).permute(0, 2, 1)
        scores = decode_raw["scores"][:, :1].sigmoid().permute(0, 2, 1)
        mask_coef = raw["mask_coefficient"].permute(0, 2, 1)
        return torch.cat((ops.xyxy2xywh(boxes), scores, mask_coef), dim=2).permute(0, 2, 1)

    def postprocess(self, preds):
        """Decode boxes and aligned mask coefficients from raw one-to-many Stage D outputs."""
        raw = self._raw_one2many(preds)
        if not isinstance(raw, dict) or "mask_coefficient" not in raw:
            return super().postprocess(preds[0] if isinstance(preds, tuple) else preds)
        proto = raw.get("proto")
        if isinstance(proto, tuple):
            proto = proto[0]
        if proto is None:
            raise RuntimeError("Stage D mask validator requires proto outputs.")

        pred = self._raw_prediction_tensor(raw)
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
        imgsz = [4 * x for x in proto.shape[2:]]
        processed = []
        for i, x in enumerate(outputs):
            pred_i = {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5] * 0, "extra": x[:, 6:]}
            coefficient = pred_i["extra"]
            pred_i["masks"] = (
                self.process(proto[i], coefficient, pred_i["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred_i["bboxes"].device,
                )
            )
            processed.append(pred_i)
        return processed

    def _mask_image_enabled(self, batch: dict[str, Any], si: int) -> bool:
        """Return whether this validation image has person-mask supervision."""
        has_mask = batch.get("has_person_mask")
        return bool(torch.is_tensor(has_mask) and si < has_mask.numel() and bool(has_mask[si]))

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare only person instances with mask supervision for Stage D mask metrics."""
        prepared = super()._prepare_batch(si, batch)
        if prepared["cls"].numel() == 0:
            return prepared

        keep = prepared["cls"].long().eq(int(self.data.get("person_cls", 0)))
        flags = batch.get("instance_flags")
        idx = batch["batch_idx"] == si
        if torch.is_tensor(flags) and flags.numel():
            flags_i = flags[idx].to(keep.device).bool()
            if flags_i.ndim == 2 and flags_i.shape[0] == keep.shape[0] and flags_i.shape[1] > 3:
                keep &= flags_i[:, 3]
        prepared["cls"] = prepared["cls"][keep] * 0
        prepared["bboxes"] = prepared["bboxes"][keep]
        prepared["masks"] = prepared["masks"][keep]
        return prepared

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics only on images that carry person-mask labels."""
        mask_preds = []
        mask_indices = []
        for si, pred in enumerate(preds):
            if not self._mask_image_enabled(batch, si):
                continue
            keep = pred["cls"].long().eq(0)
            pred = {
                **pred,
                "bboxes": pred["bboxes"][keep],
                "conf": pred["conf"][keep],
                "cls": pred["cls"][keep] * 0,
                "extra": pred["extra"][keep] if "extra" in pred else pred.get("extra", pred["bboxes"].new_zeros((0, 0))),
                "masks": pred["masks"][keep],
            }
            mask_indices.append(si)
            mask_preds.append(pred)

        if not mask_preds:
            return
        local_batch = dict(batch)
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[:1] == (len(preds),):
                local_batch[key] = value[mask_indices]
            elif key in {"im_file", "ori_shape", "ratio_pad"} and isinstance(value, (list, tuple)):
                local_batch[key] = [value[i] for i in mask_indices]
        old_batch_idx = batch["batch_idx"]
        selected = torch.zeros_like(old_batch_idx, dtype=torch.bool)
        for new_i, old_i in enumerate(mask_indices):
            m = old_batch_idx == old_i
            selected |= m
        local_batch["batch_idx"] = old_batch_idx[selected].clone()
        for new_i, old_i in enumerate(mask_indices):
            local_batch["batch_idx"][local_batch["batch_idx"] == old_i] = new_i
        inst_count = old_batch_idx.shape[0]
        for key in ("cls", "bboxes", "segments", "keypoints", "body_kpts_3d", "instance_flags"):
            value = batch.get(key)
            if torch.is_tensor(value) and value.shape[:1] == (inst_count,):
                local_batch[key] = value[selected]
        super().update_metrics(mask_preds, local_batch)


class YOLO26PSStageTrainer(DetectionTrainer):
    """Single trainer for all YOLO26-PS stages; stages only change config, not loss code."""

    stage_cfg: dict[str, Any] = {}

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Build the training model and attach optional stage-only modules after loading weights."""
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self._apply_optional_pose_adapter(model, log=verbose and RANK in {-1, 0})
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.apply_stage_controls(log=True, freeze=False)

    def _apply_optional_pose_adapter(self, model: nn.Module, log: bool = False) -> None:
        """Attach a pose-only feature adapter requested by train_runtime."""
        runtime = self.stage_cfg.get("train_runtime") or {}
        if not truthy(runtime.get("pose_adapter", False)):
            return
        layers = getattr(model, "model", [])
        if not layers:
            return
        head = layers[-1]
        if not hasattr(head, "enable_pose_adapter"):
            LOGGER.warning("pose_adapter requested, but the model head does not support it.")
            return
        hidden_ratio = float(runtime.get("pose_adapter_hidden_ratio", 0.5))
        scale = float(runtime.get("pose_adapter_scale", 1.0))
        module = head.enable_pose_adapter(hidden_ratio=hidden_ratio, scale=scale)
        module.to(next(model.parameters()).device)
        if log:
            params = sum(p.numel() for p in module.parameters())
            LOGGER.info(
                f"Stage pose adapter enabled: hidden_ratio={hidden_ratio:g}, scale={scale:g}, params={params:,}"
            )

    def apply_stage_controls(self, log: bool = False, freeze: bool = True) -> None:
        """Apply active tasks, loss weights, trainable branches, and eval locks."""
        model = unwrap_model(self.model)
        if not hasattr(model, "model") or not len(model.model):
            return
        head = model.model[-1]
        tasks = active_tasks_from_stage(self.stage_cfg)
        if hasattr(head, "set_active_tasks"):
            head.set_active_tasks(tasks)
        self._apply_optional_pose_adapter(model, log=False)
        runtime = self.stage_cfg.get("train_runtime") or {}
        loss_cfg = self.stage_cfg.get("loss") or {}
        assignment_head = normalize_decode_head(
            runtime.get("person_mask_assignment_head", loss_cfg.get("person_mask_assignment_head", "seg"))
        )
        decode_head = normalize_decode_head(runtime.get("person_mask_decode_head", assignment_head))
        model.person_mask_assignment_head = assignment_head
        model.person_mask_decode_head = decode_head
        setattr(head, "person_mask_decode_head", decode_head)
        setattr(head, "use_human_det_branch", assignment_head == "human" or decode_head == "human")
        if hasattr(head, "use_p2_refine"):
            head.use_p2_refine = truthy(runtime.get("use_p2_refine", getattr(head, "use_p2_refine", False)))
        model.loss_weights = complete_loss_weights(self.stage_cfg)
        frozen_groups = self._apply_model_trainability(model, freeze=freeze)
        frozen = self._apply_branch_trainability(head, tasks, freeze=freeze)
        bn_locks = self._apply_bn_policy(model, head, tasks)
        if log:
            LOGGER.info(f"Stage active tasks: {sorted(tasks)}")
            if "seg" in tasks:
                LOGGER.info(f"Stage person-mask assignment/decode heads: {assignment_head}/{decode_head}")
            LOGGER.info(f"Stage task loss weights: {model.loss_weights}")
            if frozen_groups:
                LOGGER.info(f"Stage frozen/eval model groups: {', '.join(frozen_groups)}")
            if frozen:
                LOGGER.info(f"Stage frozen/eval branches: {', '.join(frozen)}")
            if bn_locks:
                LOGGER.info(f"Stage BN eval locks: {', '.join(bn_locks)}")

    @staticmethod
    def _model_group_range(model: nn.Module, group: str) -> range:
        """Return model layer indices for a named trainability group."""
        layers = getattr(model, "model", [])
        yaml_file = str(getattr(model, "yaml_file", "") or "")
        head_name = layers[-1].__class__.__name__ if layers else ""
        is_stage_d_segbranch = "stage-d-segbranch" in yaml_file or (head_name == "YOLO26PSSegment" and len(layers) >= 33)
        if is_stage_d_segbranch:
            ranges = {
                "backbone": (0, 11),
                "det_neck": (11, 20),
                "seg_neck": (20, -1),
                "pose_neck": (len(layers), len(layers)),
            }
            start, end = ranges[group]
            stop = len(layers) + end if end < 0 else end
            start = max(0, min(start, len(layers)))
            stop = max(start, min(stop, len(layers) - 1 if group == "seg_neck" else len(layers)))
            return range(start, stop)
        start, end = MODEL_GROUP_RANGES[group]
        stop = len(layers) + end if end < 0 else end
        start = max(0, min(start, len(layers)))
        stop = max(start, min(stop, len(layers)))
        return range(start, stop)

    def _apply_model_trainability(self, model: nn.Module, freeze: bool = True) -> list[str]:
        """Freeze named model groups from the stage train config."""
        train = self.stage_cfg.get("train") or {}
        layers = getattr(model, "model", [])
        frozen: list[str] = []
        for group in MODEL_GROUP_RANGES:
            indices = self._model_group_range(model, group)
            if not indices:
                continue
            trainable = train_group_enabled(train, group, default=True)
            if not trainable:
                frozen.append(group)
            for layer_idx in indices:
                module = layers[layer_idx]
                if not trainable:
                    module.eval()
                if freeze:
                    for p in module.parameters():
                        p.requires_grad = bool(trainable)
        return frozen

    def _apply_branch_trainability(self, head: nn.Module, tasks: set[str], freeze: bool = True) -> list[str]:
        """Freeze inactive or explicitly frozen head modules and keep them in eval mode."""
        train = self.stage_cfg.get("train") or {}
        runtime = self.stage_cfg.get("train_runtime") or {}
        allowed_modules = normalize_name_list(runtime.get("train_head_modules") or runtime.get("train_head_module_names"))
        blocked_modules = normalize_name_list(runtime.get("freeze_head_modules") or runtime.get("freeze_head_module_names"))
        active_groups = set(tasks)
        if {"seg", "sem"} & tasks:
            active_groups.add("p2_dense")
        branch_modules = self._branch_modules_for_head(head)

        frozen: list[str] = []
        for group, names in branch_modules.items():
            if group == "p2_dense":
                default = group in active_groups
                trainable = branch_flag_enabled(train, "seg_head", "mask_head", default=False) or branch_flag_enabled(
                    train, "sem_head", "scene_seg_head", default=False
                )
                trainable = bool(train.get("all")) or (default and trainable)
            elif group == "pose_adapter":
                default = truthy(runtime.get("pose_adapter", False)) and "pose" in active_groups
                trainable = train_flag_enabled(train, BRANCH_TRAIN_FLAGS[group], default=default)
                trainable = bool(train.get("all")) or (default and trainable)
            else:
                trainable = branch_trainable_from_stage(self.stage_cfg, group, group in active_groups)
            for name in names:
                module = getattr(head, name, None)
                if module is None:
                    continue
                module_name = name.lower()
                module_trainable = bool(trainable)
                if allowed_modules and module_name not in allowed_modules:
                    module_trainable = False
                if module_name in blocked_modules:
                    module_trainable = False
                if not module_trainable and isinstance(module, nn.Module):
                    module.eval()
                if not module_trainable:
                    frozen.append(name)
                if freeze:
                    if isinstance(module, nn.Parameter):
                        module.requires_grad = module_trainable
                    elif isinstance(module, nn.Module):
                        for p in module.parameters():
                            p.requires_grad = module_trainable
        return frozen

    @staticmethod
    def _branch_modules_for_head(head: nn.Module) -> dict[str, tuple[str, ...]]:
        """Return logical branch modules, accounting for Segment-style Stage D heads."""
        modules = dict(BRANCH_MODULES)
        if head.__class__.__name__ in {"YOLO26PSSegment", "YOLO26PSAdapterSegment"}:
            modules["det"] = ()
            modules["human_det"] = ("human_cv2", "human_cv3", "one2one_human_cv2", "one2one_human_cv3")
            modules["pose"] = ()
            modules["seg"] = ("cv2", "cv3", "one2one_cv2", "one2one_cv3", "cv4", "one2one_cv4", "proto")
            modules["sem"] = ()
            modules["p2_dense"] = ("seg_adapt", "p2_refine", "p2_gate")
            if head.__class__.__name__ == "YOLO26PSSegment":
                modules["p2_dense"] = ("p2_refine", "p2_gate")
        return modules

    def _apply_bn_policy(self, model: nn.Module, head: nn.Module, tasks: set[str]) -> list[str]:
        """Lock BatchNorm/Norm stats by model group or head branch while allowing selected head BN to update."""
        runtime = self.stage_cfg.get("train_runtime") or {}
        train = self.stage_cfg.get("train") or {}
        if truthy(runtime.get("freeze_trainable_bn", False)):
            self._set_norm_training(model, train=False)
            return ["all"]

        locked: list[str] = []
        layers = getattr(model, "model", [])
        model_bn_policy = self._resolve_model_bn_policy(runtime)
        for group, freeze_bn in model_bn_policy.items():
            indices = self._model_group_range(model, group)
            if not indices:
                continue
            self._set_norm_training(nn.Sequential(*(layers[i] for i in indices)), train=not freeze_bn)
            if freeze_bn:
                locked.append(group)

        head_bn_cfg = runtime.get("freeze_head_bn")
        if head_bn_cfg is None:
            return locked

        branch_policy = self._resolve_head_bn_policy(head, head_bn_cfg, tasks)
        for group, freeze_bn in branch_policy.items():
            modules = self._head_branch_modules(head, group, train)
            if not modules:
                continue
            for module in modules:
                self._set_norm_training(module, train=not freeze_bn)
            if freeze_bn:
                locked.append(f"head:{group}")
        return locked

    @staticmethod
    def _set_norm_training(module: nn.Module, train: bool) -> None:
        """Set only Norm layers to train/eval, leaving affine requires_grad untouched."""
        for submodule in module.modules():
            if isinstance(submodule, NORM_TYPES):
                submodule.train(mode=train)

    def _resolve_model_bn_policy(self, runtime: dict[str, Any]) -> dict[str, bool]:
        """Resolve model-group BN locks for backbone, det neck, and pose neck."""
        groups = set(MODEL_GROUP_RANGES)
        value = runtime.get("freeze_model_bn")
        if isinstance(value, dict):
            policy: dict[str, bool] = {}
            for key, freeze_bn in value.items():
                name = str(key).strip().lower()
                if name == "neck":
                    policy["det_neck"] = truthy(freeze_bn)
                    policy["seg_neck"] = truthy(freeze_bn)
                    policy["pose_neck"] = truthy(freeze_bn)
                elif name in groups:
                    policy[name] = truthy(freeze_bn)
            return policy
        if value is not None:
            freeze_all = truthy(value)
            return {group: freeze_all for group in groups}
        if truthy(runtime.get("freeze_backbone_neck_bn", False)):
            return {group: True for group in groups}
        return {}

    def _resolve_head_bn_policy(self, head: nn.Module, value: Any, tasks: set[str]) -> dict[str, bool]:
        """Resolve freeze_head_bn config to branch -> freeze_bn."""
        groups = set(self._branch_modules_for_head(head))
        if isinstance(value, dict):
            policy: dict[str, bool] = {}
            for key, freeze_bn in value.items():
                name = HEAD_BN_ALIASES.get(str(key).strip().lower(), str(key).strip().lower())
                if name in groups:
                    policy[name] = truthy(freeze_bn)
            return policy
        if isinstance(value, str) and value.strip().lower() not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            names = {HEAD_BN_ALIASES.get(x.strip().lower(), x.strip().lower()) for x in value.split(",") if x.strip()}
            return {group: group in names for group in groups}
        freeze_all = truthy(value)
        return {group: freeze_all for group in groups}

    def _head_branch_modules(self, head: nn.Module, group: str, train: dict[str, Any]) -> list[nn.Module]:
        """Return modules that belong to a logical head branch."""
        names = self._branch_modules_for_head(head).get(group, ())
        if group == "p2_dense":
            names = names if branch_flag_enabled(train, "seg_head", "mask_head", default=False) or branch_flag_enabled(
                train, "sem_head", "scene_seg_head", default=False
            ) else ()
        return [module for name in names if isinstance((module := getattr(head, name, None)), nn.Module)]

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

    def _optimizer_metrics(self) -> dict[str, float]:
        groups = getattr(getattr(self, "optimizer", None), "param_groups", None) or []
        prodigy = next((group for group in groups if "d" in group), None)
        if prodigy is None:
            return {}
        lr = max(float(group.get("lr", 0.0)) for group in groups)
        d = float(prodigy.get("d", 0.0))
        metrics = {
            "prodigy/d": d,
            "prodigy/effective_lr": d * lr,
            "prodigy/k": float(prodigy.get("k", 0.0)),
        }
        if "d_max" in prodigy:
            metrics["prodigy/d_max"] = float(prodigy["d_max"])
        return metrics

    @staticmethod
    def _pose_fitness_metric_name(pose_fitness: str) -> str | None:
        """Map a runtime pose_fitness mode to the corresponding MPJPE metric stem."""
        key = pose_fitness.strip().lower()
        aliases = {
            "mpjpe": "mpjpe",
            "mpjpe_raw": "mpjpe",
            "raw_mpjpe": "mpjpe",
            "mpjpe_input": "mpjpe_input",
            "input_mpjpe": "mpjpe_input",
            "input_px": "mpjpe_input",
            "mpjpe_box_h_norm": "mpjpe_box_h_norm",
            "box_h_norm": "mpjpe_box_h_norm",
            "box_norm": "mpjpe_box_h_norm",
        }
        return aliases.get(key)

    @staticmethod
    def _pose_metric_value(metrics: dict[str, Any], prefix: str, metric_name: str, stat: str) -> float | None:
        suffix = "_px" if metric_name in {"mpjpe", "mpjpe_input"} else ""
        key = f"{prefix}/{metric_name}_{stat}{suffix}"
        value = metrics.get(key)
        return None if value is None else float(value)

    def _pose_mpjpe_fitness(self, metrics: dict[str, Any], runtime: dict[str, Any], pose_fitness: str) -> float | None:
        """Return negative MPJPE fitness using optional source-aware metric selection."""
        metric_name = self._pose_fitness_metric_name(pose_fitness)
        if metric_name is None:
            return None
        p90_weight = float(runtime.get("pose_fitness_p90_weight", 0.25))
        source_weights = pose_fitness_sources(runtime.get("pose_fitness_sources"))
        if not source_weights:
            mean = self._pose_metric_value(metrics, "metrics/pose2d", metric_name, "mean")
            p90 = self._pose_metric_value(metrics, "metrics/pose2d", metric_name, "p90")
            if mean is None or p90 is None:
                return None
            error = mean + p90_weight * p90
        else:
            total_error = 0.0
            total_weight = 0.0
            for source, weight in source_weights.items():
                prefix = f"metrics/pose2d/source/{source}"
                matched = float(metrics.get(f"{prefix}/matched", 0.0) or 0.0)
                if matched <= 0:
                    continue
                mean = self._pose_metric_value(metrics, prefix, metric_name, "mean")
                p90 = self._pose_metric_value(metrics, prefix, metric_name, "p90")
                if mean is None or p90 is None:
                    continue
                total_error += float(weight) * (mean + p90_weight * p90)
                total_weight += float(weight)
            if total_weight <= 0:
                return None
            error = total_error / total_weight
        metrics["metrics/pose2d/fitness_error"] = float(error)
        return -float(error)

    def _update_pose_best_fitness(self, fitness: float) -> None:
        """Track best pose fitness separately from detector-oriented defaults."""
        if not hasattr(self, "_stage_pose_best_fitness"):
            self._stage_pose_best_fitness = None
        if self._stage_pose_best_fitness is None or self._stage_pose_best_fitness < fitness:
            self._stage_pose_best_fitness = fitness
            self.best_fitness = fitness
        else:
            self.best_fitness = self._stage_pose_best_fitness

    def save_metrics(self, metrics):
        return super().save_metrics({**metrics, **self._optimizer_metrics()})

    def get_validator(self):
        self.loss_names = getattr(unwrap_model(self.model).model[-1], "loss_names", ("box_loss", "cls_loss", "dfl_loss"))
        tasks = active_tasks_from_stage(self.stage_cfg)
        if "seg" in tasks and not ({"pose", "sem"} & tasks):
            runtime = self.stage_cfg.get("train_runtime") or {}
            loss_cfg = self.stage_cfg.get("loss") or {}
            assignment_head = runtime.get("person_mask_assignment_head", loss_cfg.get("person_mask_assignment_head", "seg"))
            decode_head = runtime.get("person_mask_decode_head", assignment_head)
            return YOLO26PSStageDMaskValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
                decode_head=decode_head,
            )
        return YOLO26PSStageValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def validate(self):
        if not self.args.val:
            return {}, 0.0
        metrics, fitness = super().validate()
        stage_loss = self.stage_cfg.get("loss") or {}
        pose_weight = float(stage_loss.get("pose2d", 0.0)) + float(stage_loss.get("pose_vis", 0.0))
        runtime = self.stage_cfg.get("train_runtime") or {}
        pose_fitness = str(runtime.get("pose_fitness", "")).strip().lower()
        if pose_weight and pose_fitness:
            pose_metric_fitness = self._pose_mpjpe_fitness(metrics, runtime, pose_fitness)
            if pose_metric_fitness is None:
                return metrics, fitness
            fitness = pose_metric_fitness
            self._update_pose_best_fitness(fitness)
        elif pose_weight and not float(stage_loss.get("det", 0.0)):
            pose2d_loss = metrics.get("val/pose2d_loss")
            pose_vis_loss = metrics.get("val/pose_vis_loss", 0.0)
            if pose2d_loss is None:
                return metrics, fitness
            fitness = -(float(pose2d_loss) + float(pose_vis_loss) * float(stage_loss.get("pose_vis", 0.0)))
            self._update_pose_best_fitness(fitness)
        return metrics, fitness

    def final_eval(self):
        if not self.args.val:
            LOGGER.info("Skipping final validation because val=False.")
            return
        if "seg" in active_tasks_from_stage(self.stage_cfg):
            LOGGER.info("Skipping final fused validation for Stage D; per-epoch mask validation uses raw adapter outputs.")
            return
        return super().final_eval()


def parse_args() -> argparse.Namespace:
    """Parse CLI args. Runtime defaults are loaded in main after reading --stage/--plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="A_detection_stable", help="stage name under plan['stages']")
    parser.add_argument("--weights", type=Path, help="optional same-architecture checkpoint to start from")
    parser.add_argument("--pretrain", type=Path, help="optional detection pretrain to partially load into --model")
    parser.add_argument("--seg-pretrain", type=Path, help="optional YOLO26 segmentation checkpoint to load final head from")
    parser.add_argument(
        "--seg-pretrain-scope",
        choices=("head", "neck_head", "all"),
        default="head",
        help="which YOLO26 segmentation weights to import from --seg-pretrain",
    )
    parser.add_argument("--seg-pretrain-neck-offset", type=int, help="target layer offset for yolo26-seg neck import")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data", type=Path, help="dataset YAML; defaults to the selected stage YAML")
    parser.add_argument("--model", type=Path)
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
    for key in LOSS_KEYS:
        parser.add_argument(f"--loss-{key.replace('_', '-')}", dest=f"loss_{key}", type=float)
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
    parser.add_argument("--hard-image-list", type=Path, help="file with image stems/paths to boost in the sampler")
    parser.add_argument("--hard-image-boost", type=float)
    parser.add_argument("--optimizer")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--warmup-momentum", type=float)
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
    parser.add_argument("--pose-anchor-topk", type=int)
    parser.add_argument("--pose-anchor-radius", type=float)
    parser.add_argument("--pose-xy-loss", choices=("bbox", "grid", "pixel", "mpjpe", "oks", "yolo", "keypoint"))
    parser.add_argument("--pose-xy-beta", type=float)
    parser.add_argument("--pose-oks-conf-weight", action="store_true")
    parser.add_argument("--no-pose-oks-conf-weight", dest="pose_oks_conf_weight", action="store_false")
    parser.set_defaults(pose_oks_conf_weight=None)
    parser.add_argument("--pose-oks-max-px", type=float)
    parser.add_argument("--pose-oks-max-box-frac", type=float)
    parser.add_argument("--pose-oks-loss-clip", type=float)
    parser.add_argument("--pose-mpjpe-hard-px", type=float)
    parser.add_argument("--pose-mpjpe-hard-gain", type=float)
    parser.add_argument("--pose-mpjpe-hard-power", type=float)
    parser.add_argument("--pose-mpjpe-hard-max", type=float)
    parser.add_argument("--pose-mpjpe-kpt-weights", help="comma/space separated per-keypoint MPJPE weights")
    parser.add_argument("--pose-adapter", action="store_true", help="enable a pose-only residual feature adapter")
    parser.add_argument("--no-pose-adapter", action="store_true", help="disable the pose-only residual feature adapter")
    parser.add_argument("--pose-adapter-hidden-ratio", type=float)
    parser.add_argument("--pose-adapter-scale", type=float)
    parser.add_argument("--train-head-modules", help="comma/space separated head module names to keep trainable")
    parser.add_argument("--freeze-head-modules", help="comma/space separated head module names to force frozen")
    parser.add_argument(
        "--freeze-head-bn",
        help="head BN policy, e.g. seg=true,p2_dense=false or comma names to freeze",
    )
    parser.add_argument("--person-mask-assignment-head", choices=("seg", "human"))
    parser.add_argument("--person-mask-decode-head", choices=("seg", "human"))
    parser.add_argument("--person-mask-dice-weight", type=float, help="optional cropped Dice gain for person masks")
    parser.add_argument("--mosaic", type=float)
    parser.add_argument("--mixup", type=float)
    parser.add_argument("--copy-paste", type=float)
    parser.add_argument("--cutmix", type=float)
    parser.add_argument("--translate", type=float)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--degrees", type=float)
    parser.add_argument("--fliplr", type=float)
    parser.add_argument("--hsv-h", type=float)
    parser.add_argument("--hsv-s", type=float)
    parser.add_argument("--hsv-v", type=float)
    parser.add_argument("--erasing", type=float)
    parser.add_argument("--det-class-mask-normalization", choices=("sqrt", "linear", "none", "off"))
    parser.add_argument("--det-partial-cls-positive-only", action="store_true")
    parser.add_argument("--no-det-partial-cls-positive-only", action="store_true")
    parser.add_argument("--det-area-loss-weight", action="store_true")
    parser.add_argument("--no-det-area-loss-weight", action="store_true")
    parser.add_argument("--det-area-loss-weight-max", type=float)
    parser.add_argument("--det-nwd-ratio", type=float)
    parser.add_argument("--det-nwd-constant", type=float)
    parser.add_argument("--det-focal-gamma", type=float)
    parser.add_argument("--det-focal-alpha", type=float)
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
    parser.add_argument("--train-profile-steps", type=positive_int, help="log average train batch stage timings")
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
        nbs=int(defaults.get("nbs", batch * accumulate)),
        workers=int(args.workers if args.workers is not None else defaults.get("workers", 8)),
        freeze=int(args.freeze if args.freeze is not None else defaults.get("freeze", 0)),
        project=str(args.project),
        name=args.name or f"yolo26ps_{args.stage.lower()}",
        task="detect",
        close_mosaic=int(defaults.get("close_mosaic", 0)),
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
    train_profile_steps = args.train_profile_steps if args.train_profile_steps is not None else defaults.get("train_profile_steps")
    if train_profile_steps is not None:
        overrides["train_profile_steps"] = int(train_profile_steps)
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
        ("hard_image_list", "hard_image_list"),
        ("hard_image_boost", "hard_image_boost"),
        ("det_area_loss_weight_max", "det_area_loss_weight_max"),
        ("det_nwd_ratio", "det_nwd_ratio"),
        ("det_nwd_constant", "det_nwd_constant"),
        ("det_focal_gamma", "det_focal_gamma"),
        ("det_focal_alpha", "det_focal_alpha"),
    ):
        cli_value = getattr(args, cli_key)
        value = cli_value if cli_value is not None else defaults.get(key)
        if value is not None:
            overrides[key] = value
    cache = normalize_cache(args.cache if args.cache is not None else defaults.get("cache"))
    if cache is not None:
        overrides["cache"] = cache
    for key in MIXED_AUG_KEYS:
        cli_value = getattr(args, key)
        overrides[key] = float(cli_value if cli_value is not None else augment.get(key, 0.0))
    for key in ("translate", "hsv_h", "hsv_s", "hsv_v", "erasing"):
        cli_value = getattr(args, key)
        value = cli_value if cli_value is not None else augment.get(key)
        if value is not None:
            overrides[key] = float(value)
    if args.scale is not None:
        overrides["scale"] = float(args.scale)
    elif "scale" in augment:
        overrides["scale"] = scale_gain(augment["scale"])
    if args.degrees is not None:
        overrides["degrees"] = float(args.degrees)
    elif "rotate" in augment:
        overrides["degrees"] = rotate_gain(augment["rotate"])
    if args.fliplr is not None:
        overrides["fliplr"] = float(args.fliplr)
    elif "flip" in augment:
        overrides["fliplr"] = float(augment["flip"])
    for key, cli_key in (
        ("optimizer", "optimizer"),
        ("lr0", "lr0"),
        ("lrf", "lrf"),
        ("momentum", "momentum"),
        ("weight_decay", "weight_decay"),
        ("warmup_epochs", "warmup_epochs"),
        ("warmup_momentum", "warmup_momentum"),
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
    for key in ("o2m", "final_o2m", "o2m_decay_updates", "o2m_mix_mode"):
        value = defaults.get(key)
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
    pose_anchor_topk = args.pose_anchor_topk if args.pose_anchor_topk is not None else defaults.get("pose_anchor_topk")
    if pose_anchor_topk is not None:
        overrides["pose_anchor_topk"] = int(pose_anchor_topk)
    pose_anchor_radius = args.pose_anchor_radius if args.pose_anchor_radius is not None else defaults.get("pose_anchor_radius")
    if pose_anchor_radius is not None:
        overrides["pose_anchor_radius"] = float(pose_anchor_radius)
    pose_xy_loss = args.pose_xy_loss or defaults.get("pose_xy_loss")
    if pose_xy_loss is not None:
        overrides["pose_xy_loss"] = str(pose_xy_loss)
    pose_xy_beta = args.pose_xy_beta if args.pose_xy_beta is not None else defaults.get("pose_xy_beta")
    if pose_xy_beta is not None:
        overrides["pose_xy_beta"] = float(pose_xy_beta)
    for key, cli_key in (
        ("pose_oks_conf_weight", "pose_oks_conf_weight"),
        ("pose_oks_max_px", "pose_oks_max_px"),
        ("pose_oks_max_box_frac", "pose_oks_max_box_frac"),
        ("pose_oks_loss_clip", "pose_oks_loss_clip"),
    ):
        cli_value = getattr(args, cli_key, None)
        value = cli_value if cli_value is not None else defaults.get(key)
        if value is not None:
            overrides[key] = bool(value) if key == "pose_oks_conf_weight" else float(value)
    for key, cli_key in (
        ("pose_mpjpe_hard_px", "pose_mpjpe_hard_px"),
        ("pose_mpjpe_hard_gain", "pose_mpjpe_hard_gain"),
        ("pose_mpjpe_hard_power", "pose_mpjpe_hard_power"),
        ("pose_mpjpe_hard_max", "pose_mpjpe_hard_max"),
    ):
        cli_value = getattr(args, cli_key)
        value = cli_value if cli_value is not None else defaults.get(key)
        if value is not None:
            overrides[key] = float(value)
    pose_mpjpe_kpt_weights = args.pose_mpjpe_kpt_weights or defaults.get("pose_mpjpe_kpt_weights")
    if pose_mpjpe_kpt_weights:
        overrides["pose_mpjpe_kpt_weights"] = str(pose_mpjpe_kpt_weights)
    if args.pose_adapter and args.no_pose_adapter:
        raise ValueError("Use only one of --pose-adapter or --no-pose-adapter.")
    if args.pose_adapter:
        overrides["pose_adapter"] = True
    elif args.no_pose_adapter:
        overrides["pose_adapter"] = False
    elif defaults.get("pose_adapter") is not None:
        overrides["pose_adapter"] = bool(defaults.get("pose_adapter"))
    for key, cli_key in (
        ("pose_adapter_hidden_ratio", "pose_adapter_hidden_ratio"),
        ("pose_adapter_scale", "pose_adapter_scale"),
        ("person_mask_dice_weight", "person_mask_dice_weight"),
    ):
        cli_value = getattr(args, cli_key)
        value = cli_value if cli_value is not None else defaults.get(key)
        if value is not None:
            overrides[key] = float(value)
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


def resolve_stage_init_paths(args: argparse.Namespace, stage: dict[str, Any]) -> None:
    """Apply optional checkpoint defaults from stage['init'] when the CLI did not provide them."""
    init = stage.get("init") or {}

    def _stage_path(value: Any) -> Path | None:
        if value in (None, ""):
            return None
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    if args.weights is None:
        args.weights = _stage_path(init.get("resume_from") or init.get("weights"))
    if args.weights is None and args.pretrain is None:
        args.pretrain = _stage_path(init.get("base_checkpoint") or init.get("pretrain"))
    if args.weights is None and args.seg_pretrain is None:
        args.seg_pretrain = _stage_path(init.get("seg_pretrain"))
    if init.get("seg_pretrain_scope") and args.seg_pretrain_scope == "head":
        args.seg_pretrain_scope = str(init["seg_pretrain_scope"])
    if args.seg_pretrain_neck_offset is None and init.get("seg_pretrain_neck_offset") is not None:
        args.seg_pretrain_neck_offset = int(init["seg_pretrain_neck_offset"])


def main() -> None:
    """Train the requested stage."""
    args = parse_args()
    plan = load_plan(args.plan)
    stage = stage_config(plan, args.stage)
    resolve_stage_init_paths(args, stage)
    if args.pose_adapter or args.no_pose_adapter:
        stage = dict(stage)
        runtime = dict(stage.get("train_runtime") or {})
        runtime["pose_adapter"] = bool(args.pose_adapter)
        stage["train_runtime"] = runtime
    for key in ("pose_adapter_hidden_ratio", "pose_adapter_scale"):
        value = getattr(args, key)
        if value is not None:
            stage = dict(stage)
            runtime = dict(stage.get("train_runtime") or {})
            runtime[key] = float(value)
            stage["train_runtime"] = runtime
    for arg_name, runtime_key in (
        ("train_head_modules", "train_head_modules"),
        ("freeze_head_modules", "freeze_head_modules"),
        ("person_mask_assignment_head", "person_mask_assignment_head"),
        ("person_mask_decode_head", "person_mask_decode_head"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            stage = dict(stage)
            runtime = dict(stage.get("train_runtime") or {})
            runtime[runtime_key] = (
                sorted(normalize_name_list(value)) if runtime_key.endswith("modules") else normalize_decode_head(value)
            )
            stage["train_runtime"] = runtime
    if args.freeze_head_bn is not None:
        stage = dict(stage)
        runtime = dict(stage.get("train_runtime") or {})
        text = args.freeze_head_bn.strip()
        if "=" in text:
            policy = {}
            for item in text.replace(",", " ").split():
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                policy[key.strip()] = truthy(value)
            runtime["freeze_head_bn"] = policy
        else:
            runtime["freeze_head_bn"] = text
        stage["train_runtime"] = runtime
    stage = apply_loss_overrides(stage, args)
    if args.data is None:
        data_yaml = stage.get("data_yaml")
        if data_yaml:
            data_yaml = Path(data_yaml)
            args.data = data_yaml if data_yaml.is_absolute() else ROOT / data_yaml
        else:
            args.data = STAGE_DATA_YAMLS.get(args.stage, DATA_YAML)
    if args.model is None:
        model_yaml = stage.get("model_yaml")
        if model_yaml:
            model_yaml = Path(model_yaml)
            args.model = model_yaml if model_yaml.is_absolute() else ROOT / model_yaml
        elif args.stage == "D_person_mask":
            args.model = STAGE_D_MODEL_YAML
        else:
            args.model = MODEL_YAML
    YOLO26PSStageTrainer.stage_cfg = stage
    maybe_prepare(args)
    model = YOLO(str(args.weights or args.model))
    if args.pretrain:
        if args.weights:
            LOGGER.warning("--pretrain is ignored when --weights is provided; same-architecture checkpoint already loaded.")
        else:
            load_stage_pretrain(model, args.pretrain)
    if args.seg_pretrain:
        load_seg_head_pretrain(
            model,
            args.seg_pretrain,
            scope=args.seg_pretrain_scope,
            neck_offset=args.seg_pretrain_neck_offset,
        )
    model.train(trainer=YOLO26PSStageTrainer, **build_overrides(args, plan, stage))


if __name__ == "__main__":
    main()
