#!/usr/bin/env python3
"""Build the YOLO26-PS Stage A person/face-only split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML


VISION_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
STAGE_A = VISION_ROOT / "YOLO26PS_STAGE_A"
STAGE_MULTI = VISION_ROOT / "YOLO26PS_STAGE_MULTI"
OUT_YAML = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_a_person_only.yaml"
TEMPLATE_YAML = ROOT / "ultralytics/cfg/datasets/yolo26ps_stage_c_pose25d.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", type=Path, default=STAGE_A)
    parser.add_argument("--stage-multi", type=Path, default=STAGE_MULTI)
    parser.add_argument("--template-yaml", type=Path, default=TEMPLATE_YAML)
    parser.add_argument("--out-yaml", type=Path, default=OUT_YAML)
    parser.add_argument("--train-list-name", default="stage_a_person_only_train.txt")
    parser.add_argument("--val-list-name", default="stage_a_person_only_val.txt")
    parser.add_argument("--train-manifest-name", default="stage_a_person_only_train.jsonl")
    parser.add_argument("--val-manifest-name", default="stage_a_person_only_val.jsonl")
    parser.add_argument("--summary-name", default="stage_a_person_only_summary.json")
    parser.add_argument("--include-ochuman", action="store_true", help="append OCHuman person-box records")
    parser.add_argument(
        "--ochuman-train-manifest",
        type=Path,
        default=STAGE_MULTI / "manifests" / "stage_c_train_ochuman.jsonl",
        help="manifest containing OCHuman train records",
    )
    parser.add_argument(
        "--ochuman-val-manifest",
        type=Path,
        default=STAGE_MULTI / "manifests" / "stage_c_val_ochuman.jsonl",
        help="manifest containing OCHuman val records",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def source_from_line(line: str) -> str:
    text = str(line).replace("\\", "/").lower()
    if "crowdhuman" in text:
        return "crowdhuman"
    if "wider_face" in text or "wider" in text:
        return "wider_face"
    if "ochuman" in text:
        return "ochuman"
    if "3dpw" in text:
        return "3dpw"
    if "agora" in text:
        return "agora"
    return ""


def source_from_record(record: dict[str, Any]) -> str:
    source = str(record.get("source", "")).strip().lower().replace("-", "_").replace(" ", "_")
    return source or source_from_line(record.get("image", ""))


def yolo_label_path_for_image(line: str) -> Path:
    path = str(Path(line))
    if "/images/" in path:
        return Path(path.replace("/images/", "/labels/")).with_suffix(".txt")
    return Path(path).with_suffix(".txt")


def stage_a_lines(stage_a: Path, split: str) -> list[str]:
    lines = []
    for line in read_lines(stage_a / f"{split}.txt"):
        if source_from_line(line) not in {"crowdhuman", "wider_face"}:
            continue
        if yolo_label_path_for_image(line).exists():
            lines.append(line)
    return lines


def pose_records(stage_multi: Path, split: str) -> list[dict[str, Any]]:
    records = read_jsonl(stage_multi / "manifests" / f"stage_c_{split}.jsonl")
    return [record for record in records if source_from_record(record) in {"3dpw", "agora"}]


def ochuman_records(path: Path) -> list[dict[str, Any]]:
    """Read OCHuman records from a mixed manifest."""
    records = read_jsonl(path)
    return [record for record in records if source_from_record(record) == "ochuman"]


def dedupe_keep_order(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for line in lines:
        key = str(Path(line).resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def counts(records: list[dict[str, Any]], lines: list[str]) -> dict[str, int]:
    counter = Counter(source_from_record(record) for record in records)
    counter.update(source_from_line(line) for line in lines)
    counter.pop("", None)
    return dict(sorted(counter.items()))


def write_dataset_yaml(args: argparse.Namespace) -> None:
    data = YAML.load(args.template_yaml)
    data["path"] = str(args.stage_multi)
    data["train"] = args.train_list_name
    data["val"] = args.val_list_name
    data["unified_schema"] = True
    data.pop("unified_labels", None)
    data["unified_manifest"] = {
        "train": f"manifests/{args.train_manifest_name}",
        "val": f"manifests/{args.val_manifest_name}",
    }
    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(args.out_yaml, data)


def main() -> None:
    args = parse_args()
    train_records = pose_records(args.stage_multi, "train")
    val_records = pose_records(args.stage_multi, "val")
    if args.include_ochuman:
        train_records.extend(ochuman_records(args.ochuman_train_manifest))
        val_records.extend(ochuman_records(args.ochuman_val_manifest))
    train_det = stage_a_lines(args.stage_a, "train")
    val_det = stage_a_lines(args.stage_a, "val")

    train_lines = dedupe_keep_order([record["image"] for record in train_records] + train_det)
    val_lines = dedupe_keep_order([record["image"] for record in val_records] + val_det)

    train_manifest = args.stage_multi / "manifests" / args.train_manifest_name
    val_manifest = args.stage_multi / "manifests" / args.val_manifest_name
    train_list = args.stage_multi / args.train_list_name
    val_list = args.stage_multi / args.val_list_name
    summary_path = args.stage_multi / args.summary_name

    write_jsonl(train_manifest, train_records)
    write_jsonl(val_manifest, val_records)
    write_lines(train_list, train_lines)
    write_lines(val_list, val_lines)
    write_dataset_yaml(args)

    summary = {
        "train_list": str(train_list),
        "val_list": str(val_list),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "dataset_yaml": str(args.out_yaml),
        "train_sources": counts(train_records, train_det),
        "val_sources": counts(val_records, val_det),
        "target_sampling_weights": {
            "crowdhuman": 15,
            "wider_face": 50,
            "3dpw": 10,
            "agora": 17,
            **({"ochuman": 8} if args.include_ochuman else {}),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
