# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import TORCH_2_0


class InfiniteDataLoader(dataloader.DataLoader):
    """DataLoader that reuses workers for infinite iteration.

    This dataloader extends the PyTorch DataLoader to provide infinite recycling of workers, which improves efficiency
    for training loops that need to iterate through the dataset multiple times without recreating workers.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.
        iterator (Iterator): The iterator from the parent DataLoader.

    Methods:
        __len__: Return the length of the batch sampler's sampler.
        __iter__: Yield batches from the underlying iterator.
        __del__: Ensure workers are properly terminated.
        reset: Reset the iterator, useful when modifying dataset settings during training.

    Examples:
        Create an infinite DataLoader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # Infinite iteration
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)  # not supported by earlier versions
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """Create an iterator that yields indefinitely from the underlying iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the DataLoader is deleted."""
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:  # force terminate
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # cleanup
        except Exception:
            pass

    def reset(self):
        """Reset the iterator to allow modifications to the dataset during training."""
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """Sampler that repeats forever for infinite iteration.

    This sampler wraps another sampler and yields its contents indefinitely, allowing for infinite iteration over a
    dataset without recreating the sampler.

    Attributes:
        sampler (torch.utils.data.Sampler): The sampler to repeat.
    """

    def __init__(self, sampler: Any):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


def _path_source(path: str, weights: dict[str, float]) -> str:
    normalized = str(path).replace("\\", "/").lower()
    segments = normalized.split("/")
    for source in weights:
        if source in segments or any(source in segment for segment in segments):
            return source
    return ""


def _label_source(label: dict[str, Any] | None, weights: dict[str, float]) -> str:
    """Return a sampler source from cached labels, falling back to empty when unavailable."""
    sources = _label_sources(label, weights)
    return max(sources, key=lambda source: weights[source]) if sources else ""


def _label_sources(label: dict[str, Any] | None, weights: dict[str, float]) -> list[str]:
    """Return all weighted sampler sources represented by a cached label."""
    if not isinstance(label, dict):
        return []
    sources = label.get("sampling_sources") or []
    if isinstance(sources, str):
        sources = [sources]
    normalized_sources = []
    for source in sources:
        source = _normalize_source_name(source)
        if source in weights:
            normalized_sources.append(source)
    if normalized_sources:
        return list(dict.fromkeys(normalized_sources))
    source = str(label.get("source") or "").strip().lower().replace("-", "_").replace(" ", "_")
    source = _normalize_source_name(source)
    if source in weights:
        return [source]
    return []


def _normalize_source_name(source: Any) -> str:
    """Normalize dataset/source names used by weighted replacement sampling."""
    source = str(source or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "object365": "objects365",
        "wider": "wider_face",
        "widerface": "wider_face",
        "coco": "coco_wholebody",
        "coco_wholebody": "coco_wholebody",
        "human36m": "h3wb",
        "human3.6m": "h3wb",
        "adechallenge": "ade20k",
        "ade_challenge": "ade20k",
        "adechallengedata2016": "ade20k",
        "ade_challenge_data_2016": "ade20k",
    }
    return aliases.get(source, source)


def _label_class_ids(label: dict[str, Any] | None) -> list[int]:
    """Return unique non-negative class ids from a cached YOLO label."""
    if not isinstance(label, dict) or "cls" not in label:
        return []
    cls = label.get("cls")
    if torch.is_tensor(cls):
        cls = cls.detach().cpu().numpy()
    try:
        cls = np.asarray(cls).reshape(-1)
    except Exception:
        return []
    if cls.size == 0:
        return []
    cls = cls[np.isfinite(cls)]
    cls = cls[cls >= 0]
    return sorted({int(c) for c in cls.tolist()})


def _label_has_small_object(label: dict[str, Any] | None, area_threshold: float = 32.0**2) -> bool:
    """Return whether a cached label has at least one bbox below the pixel-area threshold."""
    if not isinstance(label, dict) or "bboxes" not in label:
        return False
    bboxes = label.get("bboxes")
    try:
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    except Exception:
        return False
    if not bboxes.size:
        return False
    shape = label.get("shape") or (1, 1)
    try:
        h, w = float(shape[0]), float(shape[1])
    except Exception:
        h, w = 1.0, 1.0
    normalized = bool(label.get("normalized", True))
    fmt = str(label.get("bbox_format", "xywh")).lower()
    if fmt == "xyxy":
        bw = np.clip(bboxes[:, 2] - bboxes[:, 0], 0.0, None)
        bh = np.clip(bboxes[:, 3] - bboxes[:, 1], 0.0, None)
    else:
        bw = np.clip(bboxes[:, 2], 0.0, None)
        bh = np.clip(bboxes[:, 3], 0.0, None)
    if normalized:
        bw *= max(w, 1.0)
        bh *= max(h, 1.0)
    return bool(np.any((bw * bh) <= float(area_threshold)))


def weighted_replacement_sampler(
    dataset: Dataset,
    weights: dict[str, float],
    samples_per_epoch: int,
    class_aware_sampling: bool = False,
    class_aware_source: str = "objects365",
    class_aware_power: float = 0.5,
    class_aware_min_multiplier: float = 0.5,
    class_aware_max_multiplier: float = 8.0,
    small_object_sampling: bool = False,
    small_object_source: str = "objects365",
    small_object_area: float = 32.0**2,
    small_object_boost: float = 1.0,
) -> WeightedRandomSampler:
    """Create a weighted replacement sampler from cached source names or image path segments."""
    if samples_per_epoch <= 0:
        raise ValueError(f"samples_per_epoch must be positive, got {samples_per_epoch}.")
    if not weights:
        raise ValueError("sampling_weights must be set for weighted_random_with_replacement sampling.")
    im_files = getattr(dataset, "im_files", None)
    if not im_files:
        raise ValueError("weighted_random_with_replacement requires a dataset with im_files.")
    weights = {_normalize_source_name(k): float(v) for k, v in weights.items()}
    counts = {source: 0 for source in weights}
    labels = getattr(dataset, "labels", None) or []
    sources_per_image = []
    classes_per_image: list[list[int]] = []
    small_object_per_image: list[bool] = []
    class_counts: dict[int, int] = {}
    class_aware_source = _normalize_source_name(class_aware_source)
    small_object_source = _normalize_source_name(small_object_source)
    for i, path in enumerate(im_files):
        label = labels[i] if i < len(labels) else None
        sources = _label_sources(label, weights)
        if not sources:
            source = _path_source(path, weights)
            sources = [source] if source else []
        sources_per_image.append(sources)
        for source in sources:
            if source in counts:
                counts[source] += 1
        class_ids = _label_class_ids(label) if class_aware_source in sources else []
        classes_per_image.append(class_ids)
        for class_id in class_ids:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        small_object_per_image.append(
            _label_has_small_object(label, area_threshold=small_object_area) if small_object_source in sources else False
        )
    sample_weights = []
    for sources in sources_per_image:
        weight = 0.0
        for source in sources:
            count = counts.get(source, 0)
            weight += weights.get(source, 0.0) / count if count else 0.0
        sample_weights.append(weight)
    if not any(sample_weights):
        raise ValueError("Could not match sampling_weights to any dataset image path segments.")
    if class_aware_sampling:
        if class_aware_source not in weights:
            raise ValueError(f"class_aware_source='{class_aware_source}' is not present in sampling_weights.")
        if not class_counts:
            LOGGER.warning(f"Class-aware sampler requested for source='{class_aware_source}', but no class labels matched.")
        else:
            class_aware_power = max(float(class_aware_power), 0.0)
            class_aware_min_multiplier = max(float(class_aware_min_multiplier), 0.0)
            class_aware_max_multiplier = max(float(class_aware_max_multiplier), class_aware_min_multiplier)
            median_count = float(np.median(list(class_counts.values())))
            raw_multipliers = [1.0] * len(sample_weights)
            target_indices = []
            for i, (sources, class_ids) in enumerate(zip(sources_per_image, classes_per_image)):
                if class_aware_source not in sources:
                    continue
                target_indices.append(i)
                if not class_ids:
                    continue
                multiplier = max((median_count / max(class_counts[class_id], 1)) ** class_aware_power for class_id in class_ids)
                raw_multipliers[i] = min(max(multiplier, class_aware_min_multiplier), class_aware_max_multiplier)
            mean_multiplier = float(np.mean([raw_multipliers[i] for i in target_indices])) if target_indices else 1.0
            mean_multiplier = mean_multiplier if mean_multiplier > 0.0 else 1.0
            effective = []
            for i in target_indices:
                multiplier = raw_multipliers[i] / mean_multiplier
                sample_weights[i] *= multiplier
                effective.append(multiplier)
            LOGGER.info(
                "Class-aware sampler: "
                f"source={class_aware_source}, classes={len(class_counts)}, images={len(target_indices)}, "
                f"power={class_aware_power:g}, raw_median_count={median_count:g}, "
                f"multiplier={min(effective):.3g}-{max(effective):.3g} mean={np.mean(effective):.3g}"
            )
    if small_object_sampling and float(small_object_boost) > 1.0:
        if small_object_source not in weights:
            raise ValueError(f"small_object_source='{small_object_source}' is not present in sampling_weights.")
        target_indices = [i for i, sources in enumerate(sources_per_image) if small_object_source in sources]
        if not target_indices:
            LOGGER.warning(f"Small-object sampler requested for source='{small_object_source}', but no images matched.")
        else:
            boost = float(small_object_boost)
            raw_multipliers = [boost if small_object_per_image[i] else 1.0 for i in target_indices]
            mean_multiplier = float(np.mean(raw_multipliers)) if raw_multipliers else 1.0
            mean_multiplier = mean_multiplier if mean_multiplier > 0.0 else 1.0
            effective = []
            for i, raw in zip(target_indices, raw_multipliers):
                multiplier = raw / mean_multiplier
                sample_weights[i] *= multiplier
                effective.append(multiplier)
            small_images = sum(bool(small_object_per_image[i]) for i in target_indices)
            LOGGER.info(
                "Small-object sampler: "
                f"source={small_object_source}, small_images={small_images}/{len(target_indices)}, "
                f"area<={float(small_object_area):g}px^2, raw_boost={boost:g}, "
                f"multiplier={min(effective):.3g}-{max(effective):.3g} mean={np.mean(effective):.3g}"
            )
    LOGGER.info(
        "Weighted replacement sampler: "
        + ", ".join(f"{source}={counts[source]} images, weight={weights[source]:g}" for source in weights)
        + f", samples_per_epoch={samples_per_epoch}"
    )
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=samples_per_epoch,
        replacement=True,
    )


class ContiguousDistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler that assigns contiguous batch-aligned chunks of the dataset to each GPU.

    Unlike PyTorch's DistributedSampler which distributes samples in a round-robin fashion (GPU 0 gets indices
    [0,2,4,...], GPU 1 gets [1,3,5,...]), this sampler gives each GPU contiguous batches of the dataset (GPU 0 gets
    batches [0,1,2,...], GPU 1 gets batches [k,k+1,...], etc.). This preserves any ordering or grouping in the original
    dataset, which is critical when samples are organized by similarity (e.g., images sorted by size to enable efficient
    batching without padding when using rect=True).

    The sampler handles uneven batch counts by distributing remainder batches to the first few ranks, ensuring all
    samples are covered exactly once across all GPUs.

    Args:
        dataset (Dataset): Dataset to sample from. Must implement __len__.
        num_replicas (int, optional): Number of distributed processes. Defaults to world size.
        batch_size (int, optional): Batch size used by dataloader. Defaults to dataset.batch_size or 1.
        rank (int, optional): Rank of current process. Defaults to current rank.
        shuffle (bool, optional): Whether to shuffle indices within each rank's chunk. Defaults to False. When True,
            shuffling is deterministic and controlled by set_epoch() for reproducibility.

    Examples:
        >>> # For validation with size-grouped images
        >>> sampler = ContiguousDistributedSampler(val_dataset, batch_size=32, shuffle=False)
        >>> loader = DataLoader(val_dataset, batch_size=32, sampler=sampler)
        >>> # For training with shuffling
        >>> sampler = ContiguousDistributedSampler(train_dataset, batch_size=32, shuffle=True)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in loader:
        ...         ...
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        batch_size: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
    ) -> None:
        """Initialize the sampler with dataset and distributed training parameters."""
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if batch_size is None:
            batch_size = getattr(dataset, "batch_size", 1)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.total_size = len(dataset)
        # ensure all ranks have a sample if batch size >= total size; degenerates to round-robin sampler
        self.batch_size = 1 if batch_size >= self.total_size else batch_size
        self.num_batches = math.ceil(self.total_size / self.batch_size)

    def _get_rank_indices(self) -> tuple[int, int]:
        """Calculate the start and end sample indices for this rank."""
        # Calculate which batches this rank handles
        batches_per_rank_base = self.num_batches // self.num_replicas
        remainder = self.num_batches % self.num_replicas

        # This rank gets an extra batch if rank < remainder
        batches_for_this_rank = batches_per_rank_base + (1 if self.rank < remainder else 0)

        # Calculate starting batch: base position + number of extra batches given to earlier ranks
        start_batch = self.rank * batches_per_rank_base + min(self.rank, remainder)
        end_batch = start_batch + batches_for_this_rank

        # Convert batch indices to sample indices
        start_idx = start_batch * self.batch_size
        end_idx = min(end_batch * self.batch_size, self.total_size)

        return start_idx, end_idx

    def __iter__(self) -> Iterator:
        """Generate indices for this rank's contiguous chunk of the dataset."""
        start_idx, end_idx = self._get_rank_indices()
        indices = list(range(start_idx, end_idx))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]

        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples in this rank's chunk."""
        start_idx, end_idx = self._get_rank_indices()
        return end_idx - start_idx

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler to ensure different shuffling patterns across epochs.

        Args:
            epoch (int): Epoch number to use as the random seed for shuffling.
        """
        self.epoch = epoch


def seed_worker(worker_id: int) -> None:
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
) -> Dataset:
    """Build and return a YOLO dataset based on configuration parameters."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        split=mode,
        fraction=cfg.fraction if mode == "train" else 1.0,
        max_samples=getattr(cfg, "val_samples", None) if mode == "val" else None,
    )


def build_grounding(
    cfg: IterableSimpleNamespace,
    img_path: str,
    json_file: str,
    batch: int,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    max_samples: int = 80,
) -> Dataset:
    """Build and return a GroundingDataset based on configuration parameters."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        max_samples=max_samples,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
    sampling: str | None = None,
    samples_per_epoch: int | None = None,
    sampling_weights: dict[str, float] | None = None,
    class_aware_sampling: bool = False,
    class_aware_source: str = "objects365",
    class_aware_power: float = 0.5,
    class_aware_min_multiplier: float = 0.5,
    class_aware_max_multiplier: float = 8.0,
    small_object_sampling: bool = False,
    small_object_source: str = "objects365",
    small_object_area: float = 32.0**2,
    small_object_boost: float = 1.0,
) -> InfiniteDataLoader:
    """Create and return an InfiniteDataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker processes for data loading.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
        pin_memory (bool, optional): Whether to use pinned memory for dataloader.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    if sampling == "weighted_random_with_replacement":
        if rank != -1:
            raise NotImplementedError("weighted_random_with_replacement is currently implemented for single-GPU only.")
        sampler = weighted_replacement_sampler(
            dataset,
            sampling_weights or {},
            int(samples_per_epoch or 0),
            class_aware_sampling=class_aware_sampling,
            class_aware_source=class_aware_source,
            class_aware_power=class_aware_power,
            class_aware_min_multiplier=class_aware_min_multiplier,
            class_aware_max_multiplier=class_aware_max_multiplier,
            small_object_sampling=small_object_sampling,
            small_object_source=small_object_source,
            small_object_area=small_object_area,
            small_object_boost=small_object_boost,
        )
        shuffle = False
    else:
        sampler = (
            None
            if rank == -1
            else distributed.DistributedSampler(dataset, shuffle=shuffle)
            if shuffle
            else ContiguousDistributedSampler(dataset)
        )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,  # increase over default 2
        pin_memory=nd > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )


def check_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
) -> tuple[Any, bool, bool, bool, bool, bool]:
    """Check the type of input source and return corresponding flag values.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source to check.

    Returns:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The processed source.
        webcam (bool): Whether the source is a webcam.
        screenshot (bool): Whether the source is a screenshot.
        from_img (bool): Whether the source is an image or list of images.
        in_memory (bool): Whether the source is an in-memory object.
        tensor (bool): Whether the source is a torch.Tensor.

    Examples:
        Check a file path source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source("image.jpg")

        Check a webcam source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source(0)
    """
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        source_lower = source.lower()
        is_url = source_lower.startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        is_file = (urlsplit(source_lower).path if is_url else source_lower).rpartition(".")[-1] in (
            IMG_FORMATS | VID_FORMATS
        )
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source_lower == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
    batch: int = 1,
    vid_stride: int = 1,
    buffer: bool = False,
    channels: int = 3,
):
    """Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source for
            inference.
        batch (int, optional): Batch size for dataloaders.
        vid_stride (int, optional): The frame interval for video sources.
        buffer (bool, optional): Whether stream frames will be buffered.
        channels (int, optional): The number of input channels for the model.

    Returns:
        (Dataset): A dataset object for the specified input source with attached source_type attribute.

    Examples:
        Load an image source for inference
        >>> dataset = load_inference_source("image.jpg", batch=1)

        Load a video stream source
        >>> dataset = load_inference_source("rtsp://example.com/stream", vid_stride=2)
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # DataLoader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer, channels=channels)
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = LoadPilAndNumpy(source, channels=channels)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
