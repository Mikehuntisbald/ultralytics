# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
DATASET_CACHE_VERSION = "1.0.3"
UNIFIED_TASK_FLAG_KEYS = ("has_det", "has_pose2d", "has_pose3d", "has_person_mask", "has_scene_seg")
UNIFIED_INSTANCE_FLAG_KEYS = ("has_bbox", "has_body2d", "has_body3d", "has_person_mask")


class YOLODataset(BaseDataset):
    """Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB) tasks using the YOLO format.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Return list of label dictionaries for YOLO training.
        build_transforms: Build and append transforms to the list.
        close_mosaic: Disable mosaic, copy_paste, mixup and cutmix augmentations and build transformations.
        update_labels_info: Update label format for different tasks.
        collate_fn: Collate data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", split: str = "train", **kwargs):
        """Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        self.split = split
        self.use_unified_labels = self._has_unified_labels()
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)

    def _has_unified_labels(self) -> bool:
        """Return whether this dataset should read the YOLO26-PS unified label schema."""
        return bool(
            self.data.get("unified_schema")
            or self.data.get("label_schema") == "unified"
            or self.data.get("unified_labels")
            or self.data.get("unified_manifest")
        )

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        if x["labels"]:
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """Return list of label dictionaries for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (list[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        if self.use_unified_labels:
            return self.get_unified_labels()

        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        labels = cache["labels"]
        if not labels:
            issues = "\n  ".join(sorted(set(cache["msgs"]))) or "no error details"
            raise RuntimeError(f"No valid images found in {cache_path}.\n  {issues}\n{HELP_URL}")
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def get_unified_labels(self) -> list[dict]:
        """Return labels parsed from the YOLO26-PS unified JSON schema."""
        records = self._load_unified_manifest()
        labels = []
        missing = 0
        for im_file in self.im_files:
            record = records.get(str(Path(im_file).resolve())) or records.get(Path(im_file).name)
            if record is None:
                label_file = self._unified_label_file(im_file)
                if label_file and label_file.exists():
                    record = json.loads(label_file.read_text(encoding="utf-8"))
            if record is None:
                missing += 1
                record = {"image": im_file, "instances": [], "task_flags": {}}
            labels.append(self._parse_unified_record(record, im_file))
        if missing:
            LOGGER.warning(f"{self.prefix}{missing} images have no unified JSON label; using empty partial labels.")
        self.im_files = [lb["im_file"] for lb in labels]
        return labels

    def _split_value(self, key: str) -> Any:
        value = self.data.get(key)
        if isinstance(value, dict):
            value = value.get(self.split) or value.get("default")
        return value

    def _resolve_data_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = Path(self.data.get("path", ".")).resolve() / path
        return path

    def _resolve_record_path(self, value: str | Path | None, im_file: str | Path | None = None) -> str | None:
        """Resolve a path from a unified record relative to the image file or data root."""
        if value is None or value == "":
            return None
        path = Path(value)
        if path.is_absolute():
            return str(path)
        if im_file is not None:
            sibling = Path(im_file).resolve().parent / path
            if sibling.exists():
                return str(sibling)
        return str((Path(self.data.get("path", ".")).resolve() / path).resolve())

    def _load_unified_manifest(self) -> dict[str, dict]:
        """Load optional split manifest and index records by absolute image path and basename."""
        manifest = self._resolve_data_path(self._split_value("unified_manifest"))
        if manifest is None or not manifest.exists():
            return {}
        if manifest.suffix == ".jsonl":
            records = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                records = data.get("samples") or data.get("images") or data.get("annotations") or data
            else:
                records = data
        if isinstance(records, dict):
            records = [{**v, "image": k} if isinstance(v, dict) else {"image": k, "instances": v} for k, v in records.items()]
        out = {}
        root = Path(self.data.get("path", ".")).resolve()
        for rec in records:
            image = rec.get("image") or rec.get("im_file") or rec.get("file_name")
            if not image:
                continue
            path = Path(image)
            abs_path = path if path.is_absolute() else root / path
            out[str(abs_path.resolve())] = rec
            out[path.name] = rec
        return out

    def _unified_label_file(self, im_file: str | Path) -> Path | None:
        label_root = self._resolve_data_path(self._split_value("unified_labels"))
        im_path = Path(im_file)
        if label_root is None:
            return Path(img2label_paths([str(im_path)])[0]).with_suffix(".json")
        parts = im_path.parts
        rel = None
        marker = ("images", self.split)
        for i in range(len(parts) - 1):
            if parts[i : i + 2] == marker:
                rel = Path(*parts[i + 2 :])
                break
        if rel is None and "images" in parts:
            rel = Path(*parts[parts.index("images") + 1 :])
        if rel is None:
            rel = Path(im_path.name)
        return label_root / rel.with_suffix(".json")

    def _parse_unified_record(self, record: dict, im_file: str | Path) -> dict:
        im_file = str(Path(im_file).resolve())
        width, height = self._record_shape(record, im_file)
        names = self.data.get("names", {})
        classes, bboxes, segments, kpts2d, kpts3d, person_masks, instance_flags = [], [], [], [], [], [], []
        task_flags = {k: bool(record.get("task_flags", {}).get(k, False)) for k in UNIFIED_TASK_FLAG_KEYS}
        nkpt = int((self.data.get("kpt_shape") or [17, 3])[0])

        for inst in record.get("instances", []):
            flags = inst.get("flags", {})
            bbox = inst.get("bbox")
            has_bbox = bool(flags.get("has_bbox", bbox is not None))
            cls = self._category_to_id(inst, names)
            if not has_bbox or bbox is None or cls is None:
                continue
            box = self._normalize_bbox(bbox, width, height, inst.get("bbox_format", record.get("bbox_format", "xyxy")))
            if box is None:
                continue
            body2d = self._normalize_kpts2d(inst.get("body_kpts_2d"), width, height, nkpt)
            body3d = self._normalize_kpts3d(inst.get("body_kpts_3d"), body2d, width, height, nkpt)
            has_body2d = bool(flags.get("has_body2d", np.any(body2d[..., 2] > 0)))
            has_body3d = bool(flags.get("has_body3d", np.any(body3d[..., 3] > 0)))
            person_mask = inst.get("person_mask")
            if isinstance(person_mask, str):
                person_mask = self._resolve_record_path(person_mask, im_file)
            has_person_mask = bool(flags.get("has_person_mask", person_mask is not None))

            classes.append([float(cls)])
            bboxes.append(box)
            kpts2d.append(body2d)
            kpts3d.append(body3d)
            person_masks.append(person_mask)
            instance_flags.append([has_bbox, has_body2d, has_body3d, has_person_mask])
            if isinstance(person_mask, list):
                seg = self._normalize_segment(person_mask, width, height)
                segments.append(seg if seg is not None else np.zeros((0, 2), dtype=np.float32))
            elif isinstance(person_mask, str):
                seg = self._mask_file_to_segment(person_mask, width, height)
                segments.append(seg if seg is not None else self._dummy_segment_from_box(box))
            else:
                segments.append(self._dummy_segment_from_box(box))

        if classes:
            cls = np.array(classes, dtype=np.float32)
            bboxes = np.array(bboxes, dtype=np.float32)
            keypoints = np.stack(kpts2d, axis=0).astype(np.float32)
            body_kpts_3d = np.stack(kpts3d, axis=0).astype(np.float32)
            instance_flags = np.array(instance_flags, dtype=bool)
        else:
            cls = np.zeros((0, 1), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
            keypoints = np.zeros((0, nkpt, 3), dtype=np.float32)
            body_kpts_3d = np.zeros((0, nkpt, 4), dtype=np.float32)
            instance_flags = np.zeros((0, len(UNIFIED_INSTANCE_FLAG_KEYS)), dtype=bool)

        task_flags["has_det"] = bool(task_flags["has_det"] or len(cls))
        task_flags["has_pose2d"] = bool(task_flags["has_pose2d"] or instance_flags[:, 1].any() if len(instance_flags) else task_flags["has_pose2d"])
        task_flags["has_pose3d"] = bool(task_flags["has_pose3d"] or instance_flags[:, 2].any() if len(instance_flags) else task_flags["has_pose3d"])
        task_flags["has_person_mask"] = bool(
            task_flags["has_person_mask"] or instance_flags[:, 3].any() if len(instance_flags) else task_flags["has_person_mask"]
        )
        scene_seg = self._resolve_record_path(record.get("scene_seg"), im_file)
        task_flags["has_scene_seg"] = bool(task_flags["has_scene_seg"] or scene_seg)

        return {
            "im_file": im_file,
            "shape": (height, width),
            "cls": cls,
            "bboxes": bboxes,
            "segments": segments,
            "keypoints": keypoints,
            "body_kpts_3d": body_kpts_3d,
            "person_mask": person_masks,
            "instance_flags": instance_flags,
            "scene_seg": scene_seg,
            **{k: np.array(task_flags[k], dtype=bool) for k in UNIFIED_TASK_FLAG_KEYS},
            "normalized": True,
            "bbox_format": "xywh",
        }

    def _record_shape(self, record: dict, im_file: str) -> tuple[int, int]:
        width, height = record.get("width"), record.get("height")
        if width and height:
            return int(width), int(height)
        with Image.open(im_file) as im:
            return im.size

    def _category_to_id(self, inst: dict, names: dict) -> int | None:
        category = inst.get("cls_id", inst.get("category_id", inst.get("category")))
        if category is None:
            return None
        if isinstance(category, (int, float)):
            return int(category)
        if isinstance(category, str) and category.isdigit():
            return int(category)
        mapping = self.data.get("category_mapping") or self.data.get("class_map") or {}
        if category in mapping:
            return int(mapping[category])
        reverse = {str(v): int(k) for k, v in names.items()}
        for k, v in names.items():
            for alias in str(v).split("/"):
                reverse.setdefault(alias.strip(), int(k))
        return reverse.get(str(category))

    @staticmethod
    def _normalize_bbox(bbox: list[float], width: int, height: int, bbox_format: str) -> list[float] | None:
        if len(bbox) < 4:
            return None
        x1, y1, x2, y2 = map(float, bbox[:4])
        if bbox_format == "xywh":
            cx, cy, bw, bh = x1, y1, x2, y2
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) > 2:
                cx, bw = cx / width, bw / width
                cy, bh = cy / height, bh / height
        else:
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 2:
                x1, x2 = x1 / width, x2 / width
                y1, y2 = y1 / height, y2 / height
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw * 0.5, y1 + bh * 0.5
        if bw <= 0 or bh <= 0:
            return None
        return [cx, cy, bw, bh]

    @staticmethod
    def _normalize_kpts2d(kpts: list | None, width: int, height: int, nkpt: int) -> np.ndarray:
        arr = np.zeros((nkpt, 3), dtype=np.float32)
        if not kpts:
            return arr
        src = np.asarray(kpts, dtype=np.float32).reshape(-1, 3)
        n = min(nkpt, len(src))
        arr[:n] = src[:n]
        if arr[:, 0].max(initial=0) > 2 or arr[:, 1].max(initial=0) > 2:
            arr[:, 0] /= width
            arr[:, 1] /= height
        return arr

    @staticmethod
    def _normalize_kpts3d(kpts: list | None, kpts2d: np.ndarray, width: int, height: int, nkpt: int) -> np.ndarray:
        arr = np.zeros((nkpt, 4), dtype=np.float32)
        arr[:, :2] = kpts2d[:, :2]
        arr[:, 3] = kpts2d[:, 2]
        if not kpts:
            return arr
        src = np.asarray(kpts, dtype=np.float32).reshape(-1, 4)
        n = min(nkpt, len(src))
        arr[:n] = src[:n]
        if arr[:, 0].max(initial=0) > 2 or arr[:, 1].max(initial=0) > 2:
            arr[:, 0] /= width
            arr[:, 1] /= height
        return arr

    @staticmethod
    def _normalize_segment(mask: list, width: int, height: int) -> np.ndarray | None:
        arr = np.asarray(mask, dtype=np.float32).reshape(-1, 2)
        if len(arr) < 3:
            return None
        if arr[:, 0].max(initial=0) > 2 or arr[:, 1].max(initial=0) > 2:
            arr[:, 0] /= width
            arr[:, 1] /= height
        return arr

    @staticmethod
    def _mask_file_to_segment(mask_file: str, width: int, height: int) -> np.ndarray | None:
        """Approximate a binary instance-mask file as its largest contour polygon."""
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        if len(contour) < 3:
            return None
        contour[:, 0] /= width
        contour[:, 1] /= height
        return contour

    @staticmethod
    def _dummy_segment_from_box(box: list[float]) -> np.ndarray:
        """Return a zero-area polygon inside a bbox for non-mask instances."""
        cx, cy, bw, bh = box
        x1, y1 = cx - bw * 0.5, cy - bh * 0.5
        x2, y2 = cx + bw * 0.5, cy + bh * 0.5
        return np.array([[x1, y1], [x2, y1], [x2, y1]], dtype=np.float32)

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Build and append transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments or self.use_unified_labels,
                return_keypoint=self.use_keypoints or self.use_unified_labels,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp: dict) -> None:
        """Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: dict) -> dict:
        """Update label format for different tasks.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Notes:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        if label.get("scene_seg") and label.get("scene_mask") is None:
            scene_mask = cv2.imread(str(label["scene_seg"]), cv2.IMREAD_UNCHANGED)
            if scene_mask is not None:
                if scene_mask.ndim == 3:
                    scene_mask = scene_mask[..., 0]
                img_shape = label["img"].shape[:2]
                if scene_mask.shape[:2] != img_shape:
                    scene_mask = cv2.resize(scene_mask, img_shape[::-1], interpolation=cv2.INTER_NEAREST)
                label["scene_mask"] = scene_mask

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        if label.get("body_kpts_3d") is not None:
            label["body_kpts_3d"] = np.asarray(label["body_kpts_3d"], dtype=np.float32)
        if label.get("instance_flags") is not None:
            label["instance_flags"] = np.asarray(label["instance_flags"], dtype=bool)
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate data samples into batches.

        Args:
            batch (list[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats", "sem_masks", "scene_seg"} and torch.is_tensor(value[0]):
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb", "body_kpts_3d", "instance_flags"}:
                value = [torch.as_tensor(v) for v in value]
                value = torch.cat(value, 0)
            elif k in UNIFIED_TASK_FLAG_KEYS:
                value = torch.tensor([bool(v) for v in value], dtype=torch.bool)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """Dataset class for loading object detection and/or segmentation labels in YOLO format with multi-modal support.

    This class extends YOLODataset to add text information for multi-modal model training, enabling models to process
    both image and text data.

    Methods:
        update_labels_info: Add text information for multi-modal model training.
        build_transforms: Enhance data transformations with text augmentation.

    Examples:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        """Initialize a YOLOMultiModalDataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: dict) -> dict:
        """Add text information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts.
        """
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # NOTE: and `RandomLoadText` would randomly select one of them if there are multiple words.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Enhance data transformations with optional text augmentation for multi-modal training.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return category names for the dataset.

        Returns:
            (set[str]): Set of class names.
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # category names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # to check
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(YOLODataset):
    """Dataset class for object detection tasks using annotations from a JSON file in grounding format.

    This dataset is designed for grounding tasks where annotations are provided in a JSON file rather than the standard
    YOLO format text files.

    Attributes:
        json_file (str): Path to the JSON file containing annotations.

    Methods:
        get_img_files: Return empty list as image files are read in get_labels.
        get_labels: Load annotations from a JSON file and prepare them for training.
        build_transforms: Configure augmentations for training with optional text loading.

    Examples:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # Number of valid images with annotations
    """

    def __init__(self, *args, task: str = "detect", json_file: str = "", max_samples: int = 80, **kwargs):
        """Initialize a GroundingDataset for object detection.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            task (str): Must be 'detect' or 'segment' for GroundingDataset.
            max_samples (int): Maximum number of samples to load for text augmentation.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> list:
        """The image files would be read in `get_labels` function, return empty list here.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            (list): Empty list as image files are read in get_labels.
        """
        return []

    def verify_labels(self, labels: list[dict[str, Any]]) -> None:
        """Verify the number of instances in the dataset matches expected counts.

        This method checks if the total number of bounding box instances in the provided labels matches the expected
        count for known datasets. It performs validation against a predefined set of datasets with known instance
        counts.

        Args:
            labels (list[dict[str, Any]]): List of label dictionaries, where each dictionary contains dataset
                annotations. Each label dict must have a 'bboxes' key with a numpy array or tensor containing bounding
                box coordinates.

        Raises:
            AssertionError: If the actual instance count doesn't match the expected count for a recognized dataset.

        Notes:
            For unrecognized datasets (those not in the predefined expected_counts),
            a warning is logged and verification is skipped.
        """
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        """Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict[str, Any]): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls, *s]
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """Load labels from cache or generate them from JSON file.

        Returns:
            (list[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Configure augmentations for training with optional text loading.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return unique category names from the dataset."""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same collation
    function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate data samples into batches.

        Args:
            batch (list[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: dict) -> None:
        """Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """Dataset class for image classification tasks wrapping torchvision ImageFolder functionality.

    This class offers functionalities like image augmentation, caching, and verification. It's designed to efficiently
    handle large datasets for training deep learning models, with optional image transformations and caching mechanisms
    to speed up training.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of lists, each containing the path to an image, its class index, path to its .npy cache
            file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        root (str): Root directory of the dataset.
        prefix (str): Prefix for logging and cache filenames.

    Methods:
        __getitem__: Return transformed image and class index for the given sample index.
        __len__: Return the total number of samples in the dataset.
        verify_images: Verify all images in dataset.
    """

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """Initialize YOLO classification dataset with root directory, arguments, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings.
            augment (bool, optional): Whether to apply augmentations to the dataset.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [[*list(x), Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i: int) -> dict:
        """Return transformed image and class index for the given sample index.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self) -> list[tuple]:
        """Verify all images in dataset.

        Returns:
            (list[tuple]): List of valid samples after verification.
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # check image read speeds
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        # NOTE: ModuleNotFoundError to prevent numpy version conflicts when loading cache files created with different numpy versions
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
