#!/usr/bin/env python3
"""Run a real Ultralytics Dataset mask check for YOLO26-PS Stage D."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG, YAML


DATA_YAML = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_d_person_mask.yaml"
OUT_DIR = ROOT / "examples/stage_d_mask_batch_check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--per-source", type=int, default=8, help="number of transformed samples to inspect per source")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[576, 768], help="Ultralytics image size, e.g. 576 768")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--mask-ratio", type=int, default=4)
    parser.add_argument("--no-smoke", action="store_true", help="use full Stage D list/manifest instead of smoke files")
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return bool(value.item()) if value.shape == () else bool(value.any())
    if torch.is_tensor(value):
        return bool(value.item()) if value.ndim == 0 else bool(value.any().item())
    return bool(value)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_data_cfg(path: Path, split: str, use_smoke: bool) -> tuple[dict, Path]:
    data = YAML.load(path)
    root = Path(data.get("path", ".")).resolve()
    if use_smoke:
        data["train"] = "stage_d_smoke_train.txt"
        data["val"] = "stage_d_smoke_val.txt"
        data["unified_manifest"] = {
            "train": "manifests/stage_d_smoke_train.jsonl",
            "val": "manifests/stage_d_smoke_val.jsonl",
        }
    img_path = Path(data[split])
    if not img_path.is_absolute():
        img_path = root / img_path
    return data, img_path


def build_dataset(data: dict, img_path: Path, split: str, imgsz: list[int], batch: int, mask_ratio: int) -> YOLODataset:
    cfg = get_cfg(
        DEFAULT_CFG,
        {
            "task": "detect",
            "imgsz": imgsz if len(imgsz) > 1 else imgsz[0],
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "cutmix": 0.0,
            "mask_ratio": mask_ratio,
            "overlap_mask": True,
            "bgr": 0.0,
        },
    )
    return YOLODataset(
        img_path=str(img_path),
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=False,
        hyp=cfg,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix=f"{split}: ",
        task="detect",
        classes=None,
        data=data,
        split=split,
    )


def select_indices(dataset: YOLODataset, per_source: int) -> list[int]:
    wanted = ("coco_person_mask", "ochuman")
    selected: dict[str, list[int]] = {source: [] for source in wanted}
    for i, label in enumerate(dataset.labels):
        source = str(label.get("source", ""))
        if source in selected and len(selected[source]) < per_source:
            selected[source].append(i)
        if all(len(v) >= per_source for v in selected.values()):
            break
    missing = {k: per_source - len(v) for k, v in selected.items() if len(v) < per_source}
    if missing:
        raise RuntimeError(f"Not enough source samples in dataset: {missing}")
    return [i for source in wanted for i in selected[source]]


def sample_mask_pixels(sample: dict) -> int:
    masks = tensor_to_numpy(sample.get("masks", np.zeros((0, 0, 0))))
    if masks.size == 0:
        return 0
    return int((masks > 0).sum())


def label_source_summary(dataset: YOLODataset) -> dict[str, Any]:
    source_counts = Counter(str(label.get("source", "")) for label in dataset.labels)
    sampling_counts: Counter[str] = Counter()
    collisions = []
    for label in dataset.labels:
        source = str(label.get("source", ""))
        sampling_sources = list(label.get("sampling_sources") or [])
        sampling_counts.update(sampling_sources)
        if source == "ochuman" and "coco_person_mask" in sampling_sources:
            collisions.append(label.get("im_file"))
    return {
        "source_counts": dict(sorted(source_counts.items())),
        "sampling_source_counts": dict(sorted(sampling_counts.items())),
        "ochuman_coco_person_mask_collisions": collisions[:20],
        "collision_count": len(collisions),
    }


def inspect_sample(dataset: YOLODataset, index: int) -> tuple[dict[str, Any], dict]:
    label = dataset.labels[index]
    sample = dataset[index]
    flags = sample.get("instance_flags")
    flags_np = tensor_to_numpy(flags).astype(bool) if flags is not None else np.zeros((0, 4), dtype=bool)
    masks = sample.get("masks")
    masks_shape = list(masks.shape) if masks is not None else []
    sem_shape = list(sample["sem_masks"].shape) if sample.get("sem_masks") is not None else []
    info = {
        "index": index,
        "im_file": sample.get("im_file"),
        "source": str(sample.get("source", label.get("source", ""))),
        "sampling_sources": list(sample.get("sampling_sources", label.get("sampling_sources", [])) or []),
        "instances": int(len(sample.get("cls", []))),
        "mask_instances": int(flags_np[:, 3].sum()) if flags_np.size else 0,
        "mask_pixels": sample_mask_pixels(sample),
        "masks_shape": masks_shape,
        "sem_masks_shape": sem_shape,
        "has_person_mask": as_bool(sample.get("has_person_mask", label.get("has_person_mask", False))),
        "instance_flags_shape": list(flags_np.shape),
    }
    return info, sample


def collated_batch_summary(dataset: YOLODataset, samples: list[dict]) -> dict[str, Any]:
    batch = dataset.collate_fn(samples)
    masks = batch.get("masks")
    flags = batch.get("instance_flags")
    batch_idx = batch.get("batch_idx")
    sources = [str(x) for x in batch.get("source", [])]
    sampling_sources = [list(x) for x in batch.get("sampling_sources", [])]
    return {
        "sample_count": len(samples),
        "img_shape": list(batch["img"].shape),
        "masks_shape": list(masks.shape) if masks is not None else [],
        "instance_flags_shape": list(flags.shape) if flags is not None else [],
        "batch_idx_shape": list(batch_idx.shape) if batch_idx is not None else [],
        "mask_pixels_total": int((masks > 0).sum().item()) if masks is not None else 0,
        "mask_instances_total": int(flags[:, 3].sum().item()) if flags is not None and flags.numel() else 0,
        "sources": sources,
        "sampling_sources": sampling_sources,
    }


def make_tile(sample: dict, info: dict[str, Any], tile_w: int = 360, tile_h: int = 300) -> np.ndarray:
    img = tensor_to_numpy(sample["img"]).transpose(1, 2, 0)
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    mask = tensor_to_numpy(sample.get("masks", np.zeros((1, 1, 1))))
    if mask.ndim == 3:
        mask = mask[0]
    mask = (mask > 0).astype(np.uint8)
    if mask.size:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = img.copy()
    if mask.size and mask.any():
        color = np.array([255, 72, 48], dtype=np.uint8)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    tile = cv2.resize(overlay, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    header = np.zeros((46, tile_w, 3), dtype=np.uint8)
    source = info["source"]
    sources = ",".join(info["sampling_sources"])
    line1 = f"{source}  inst={info['instances']} mask_inst={info['mask_instances']}"
    line2 = f"px={info['mask_pixels']}  sampling={sources}"
    cv2.putText(header, line1[:58], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(header, line2[:70], (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    return np.concatenate([header, tile], axis=0)


def save_contact_sheet(samples: list[dict], infos: list[dict[str, Any]], out_file: Path) -> None:
    tiles = [make_tile(sample, info) for sample, info in zip(samples, infos)]
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 28, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    data, img_path = load_data_cfg(args.data, args.split, use_smoke=not args.no_smoke)
    dataset = build_dataset(data, img_path, args.split, args.imgsz, args.batch, args.mask_ratio)
    indices = select_indices(dataset, args.per_source)

    infos, samples = [], []
    for index in indices:
        info, sample = inspect_sample(dataset, index)
        infos.append(info)
        samples.append(sample)

    by_source: dict[str, dict[str, Any]] = defaultdict(dict)
    for source in sorted({x["source"] for x in infos}):
        rows = [x for x in infos if x["source"] == source]
        by_source[source] = {
            "samples": len(rows),
            "mask_pixels_min": min(x["mask_pixels"] for x in rows),
            "mask_pixels_max": max(x["mask_pixels"] for x in rows),
            "mask_pixels_mean": float(np.mean([x["mask_pixels"] for x in rows])),
            "mask_instances_total": int(sum(x["mask_instances"] for x in rows)),
            "sampling_sources": sorted({tuple(x["sampling_sources"]) for x in rows}),
            "has_person_mask_all": all(x["has_person_mask"] for x in rows),
        }

    summary = {
        "data": str(args.data),
        "split": args.split,
        "img_path": str(img_path),
        "dataset_len": len(dataset),
        "cache_version": getattr(__import__("ultralytics.data.dataset", fromlist=["UNIFIED_CACHE_VERSION"]), "UNIFIED_CACHE_VERSION"),
        "label_summary": label_source_summary(dataset),
        "batch_check": by_source,
        "collated_batch": collated_batch_summary(dataset, samples),
        "samples": infos,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    contact_sheet = args.out / "contact_sheet.jpg"
    save_contact_sheet(samples, infos, contact_sheet)

    print(json.dumps({**summary, "contact_sheet": str(contact_sheet), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
