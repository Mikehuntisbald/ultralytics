# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ultralytics.data.augment import MixUp, Mosaic


class DummyDataset:
    """Minimal dataset exposing labels and a sample buffer for mix-transform tests."""

    cache = None
    buffer = [0, 1, 2, 3, 4]

    def __init__(self):
        self.labels = [
            {"det_class_mask": np.array([1, 1, 0, 0], dtype=bool)},
            {"det_class_mask": np.array([1, 1, 0, 0], dtype=bool)},
            {"det_class_mask": np.array([0, 0, 1, 0], dtype=bool)},
            {"det_class_mask": np.array([0, 0, 1, 0], dtype=bool)},
            {"det_class_mask": np.array([0, 0, 0, 1], dtype=bool)},
        ]

    def __len__(self):
        return len(self.labels)


def test_source_aware_mosaic_samples_same_detection_domain():
    dataset = DummyDataset()
    mosaic = Mosaic(dataset, imgsz=(64, 96), p=1.0, n=4)

    indexes = mosaic.get_indexes(dataset.labels[2])

    assert len(indexes) == 3
    assert set(indexes) <= {2, 3}


def test_source_aware_mixup_samples_same_detection_domain():
    dataset = DummyDataset()
    mixup = MixUp(dataset, p=1.0)

    index = mixup.get_indexes(dataset.labels[4])

    assert index == 4

