#!/usr/bin/env python3
"""Compatibility wrapper for Stage A detection warmup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_yolo26ps_stage import main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv[1:1] = ["--stage", "A_detection_stable"]
    main()
