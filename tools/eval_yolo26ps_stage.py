#!/usr/bin/env python3
"""Evaluate YOLO26s-PS-2.5D checkpoints with stage-aware detection metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import YAML

from tools.train_yolo26ps_stage import DATA_YAML, PLAN_YAML, YOLO26PSStageValidator, normalize_imgsz, stage_config


def parse_imgsz(value: str) -> int | list[int]:
    """Parse square or H,W image size CLI values."""
    parts = [x for x in value.replace("x", ",").split(",") if x]
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        return [int(parts[0]), int(parts[1])]
    raise argparse.ArgumentTypeError("imgsz must be SIZE or H,W")


def parse_bool(value: str) -> bool:
    """Parse boolean CLI values."""
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on", "e2e"}:
        return True
    if value in {"0", "false", "no", "off", "one2many", "o2m"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def load_plan(plan: Path) -> dict[str, Any]:
    """Load YAML plan."""
    return YAML.load(plan) if plan.exists() else {}


def run_eval(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    """Run one eval case and return flattened metrics."""
    model = YOLO(str(args.weights))
    if hasattr(model.model.model[-1], "set_active_tasks"):
        model.model.model[-1].set_active_tasks({"det"})
    model.model.end2end = bool(case["end2end"])
    model.model.set_head_attr(max_det=int(case["max_det"]))

    save_name = f"{args.name}_{case['name']}" if args.name else f"eval_yolo26ps_{case['name']}"
    val_args = {
        **model.overrides,
        "mode": "val",
        "task": "detect",
        "model": str(args.weights),
        "data": str(args.data),
        "imgsz": case["imgsz"],
        "batch": int(case["batch"]),
        "val_batch": int(case["batch"]),
        "workers": int(args.workers),
        "val_workers": int(args.workers),
        "val_samples": args.val_samples,
        "max_det": int(case["max_det"]),
        "end2end": bool(case["end2end"]),
        "device": args.device,
        "plots": False,
        "save_json": False,
        "save_txt": False,
        "project": str(args.project),
        "name": save_name,
        "split": args.split,
        "rect": True,
        "verbose": False,
    }
    validator = YOLO26PSStageValidator(args=val_args, _callbacks=model.callbacks)
    stats = validator(model=model.model)
    row = dict(stats)
    row.update(
        {
            "case": case["name"],
            "weights": str(args.weights),
            "data": str(args.data),
            "imgsz": "x".join(str(x) for x in (case["imgsz"] if isinstance(case["imgsz"], list) else [case["imgsz"]])),
            "end2end": bool(case["end2end"]),
            "max_det": int(case["max_det"]),
            "batch": int(case["batch"]),
            "save_dir": str(args.project / save_name),
        }
    )
    return row


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build requested eval cases."""
    base_imgsz = normalize_imgsz(args.imgsz)
    high_imgsz = normalize_imgsz(args.high_res_imgsz)
    return [
        {"name": "e2e", "end2end": True, "imgsz": base_imgsz, "max_det": args.max_det, "batch": args.batch},
        {"name": "one2many", "end2end": False, "imgsz": base_imgsz, "max_det": args.max_det, "batch": args.batch},
        {"name": "high_res", "end2end": True, "imgsz": high_imgsz, "max_det": args.max_det, "batch": args.high_res_batch},
        {
            "name": "high_max_det",
            "end2end": True,
            "imgsz": base_imgsz,
            "max_det": args.high_max_det,
            "batch": args.batch,
        },
    ]


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Write JSON and CSV summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = [
        "case",
        "imgsz",
        "end2end",
        "max_det",
        "batch",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/stage_a/objects365/mAP50(B)",
        "metrics/stage_a/objects365/mAP50-95(B)",
        "metrics/stage_a/small/mAP50(B)",
        "metrics/stage_a/small/mAP50-95(B)",
        "metrics/stage_a/objects365/person/mAP50(B)",
        "metrics/stage_a/crowdhuman/person/mAP50(B)",
        "metrics/stage_a/wider_face/face/mAP50(B)",
        "save_dir",
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN_YAML)
    parser.add_argument("--stage", default="D_det_recover_objects365_prodigy_unfreeze_fast")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect")
    parser.add_argument("--name", default="yolo26ps_eval_compare")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "runs/detect/yolo26ps_eval_compare")
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=22)
    parser.add_argument("--high-res-batch", type=int, default=10)
    parser.add_argument("--val-samples", type=int, default=10000)
    parser.add_argument("--imgsz", type=parse_imgsz, default=[576, 768])
    parser.add_argument("--high-res-imgsz", type=parse_imgsz, default=[768, 1024])
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--high-max-det", type=int, default=1000)
    parser.add_argument("--case", choices=("all", "e2e", "one2many", "high_res", "high_max_det"), default="all")
    parser.add_argument("--end2end", type=parse_bool, help="override end2end for a single custom case")
    return parser.parse_args()


def main() -> None:
    """Run eval comparison."""
    args = parse_args()
    plan = load_plan(args.plan)
    if args.data is None:
        stage = stage_config(plan, args.stage) if plan else {}
        data_yaml = stage.get("data_yaml")
        if data_yaml:
            data_yaml = Path(data_yaml)
            args.data = data_yaml if data_yaml.is_absolute() else ROOT / data_yaml
        else:
            args.data = DATA_YAML
    cases = build_cases(args)
    if args.case != "all":
        cases = [case for case in cases if case["name"] == args.case]
    if args.end2end is not None and len(cases) == 1:
        cases[0]["end2end"] = args.end2end

    rows = []
    for case in cases:
        print(
            f"Running {case['name']}: imgsz={case['imgsz']} end2end={case['end2end']} "
            f"max_det={case['max_det']} batch={case['batch']}"
        )
        rows.append(run_eval(args, case))
        write_summary(rows, args.summary_dir)
    write_summary(rows, args.summary_dir)
    print(f"Wrote {args.summary_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
