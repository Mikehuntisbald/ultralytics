#!/usr/bin/env python3
"""Prepare YOLO26s-PS-2.5D Stage A detection warmup data.

The script converts LVIS, CrowdHuman, and WIDER FACE annotations into a single YOLO detection dataset:

    LVIS categories -> 0..1202
    person -> 1203
    face -> 1204

It uses symlinks for images so the prepared dataset is mostly labels and file lists.
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image


DATASETS = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
OUT = DATASETS / "YOLO26PS_STAGE_A"
COCO = DATASETS / "COCO_2017"
LVIS = DATASETS / "LVIS" / "data"
WIDER = DATASETS / "WIDER_FACE" / "extracted" / "WIDER_Face_Detection"
CROWD = DATASETS / "CrowdHuman" / "CrowdHuman"
CROWD_TAR = CROWD / "crowdhuman.tar.00"

NC = 1205
PERSON_CLS = 1203
FACE_CLS = 1204


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=DATASETS, help="vision_benchmarks root")
    parser.add_argument("--out", type=Path, default=OUT, help="prepared dataset root")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--lvis", type=int, default=55, help="Stage A train sampling weight")
    parser.add_argument("--crowdhuman", type=int, default=25, help="Stage A train sampling weight")
    parser.add_argument("--wider-face", type=int, default=20, help="Stage A train sampling weight")
    parser.add_argument("--skip-crowdhuman-extract", action="store_true", help="do not extract CrowdHuman tar")
    return parser.parse_args()


def mkdirs(out: Path) -> None:
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)


def symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def yolo_line(cls: int, box: list[float], width: int, height: int) -> str | None:
    x, y, w, h = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    cx, cy = (x1 + x2) * 0.5 / width, (y1 + y2) * 0.5 / height
    return f"{cls} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}"


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def lvis_json(split: str) -> dict:
    zpath = LVIS / f"lvis_v1_{split}.json.zip"
    with zipfile.ZipFile(zpath) as zf:
        with zf.open(f"lvis_v1_{split}.json") as f:
            return json.load(f)


def prepare_lvis(out: Path, split: str) -> list[str]:
    data = lvis_json(split)
    image_dir = COCO / ("train2017" if split == "train" else "val2017")
    person_category_ids = {int(c["id"]) for c in data.get("categories", []) if c.get("name") == "person"}
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        category_id = int(ann["category_id"])
        cls = category_id - 1
        if category_id in person_category_ids:
            # Reserve class 1203 as unified person. LVIS person is remapped there.
            cls = PERSON_CLS
        by_image[int(ann["image_id"])].append((cls, ann["bbox"]))

    list_paths = []
    for img in data["images"]:
        image_id = int(img["id"])
        file_name = Path(img.get("coco_url", "")).name or f"{image_id:012d}.jpg"
        src = image_dir / file_name
        if not src.exists():
            continue
        rel = Path("lvis") / file_name
        dst_img = out / "images" / split / rel
        dst_lb = out / "labels" / split / rel.with_suffix(".txt")
        lines = [line for cls, box in by_image.get(image_id, []) if (line := yolo_line(cls, box, img["width"], img["height"]))]
        symlink(src, dst_img)
        write_label(dst_lb, lines)
        list_paths.append(str(dst_img))
    return list_paths


def parse_wider(txt: Path) -> dict[str, list[list[float]]]:
    lines = txt.read_text(encoding="utf-8").splitlines()
    records = {}
    i = 0
    while i < len(lines):
        rel = lines[i].strip()
        i += 1
        if not rel:
            continue
        n = int(lines[i].strip())
        i += 1
        boxes = []
        rows = max(n, 1)
        for row in range(rows):
            parts = [float(x) for x in lines[i].split()]
            i += 1
            if row >= n:
                continue
            x, y, w, h = parts[:4]
            invalid = int(parts[7]) if len(parts) > 7 else 0
            if invalid == 0 and w > 1 and h > 1:
                boxes.append([x, y, w, h])
        records[rel] = boxes
    return records


def prepare_wider(out: Path, split: str) -> list[str]:
    txt = WIDER / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    records = parse_wider(txt)
    image_dir = WIDER / f"WIDER_{split}" / "images"
    list_paths = []
    for rel_str, boxes in records.items():
        src = image_dir / rel_str
        if not src.exists():
            continue
        with Image.open(src) as im:
            width, height = im.size
        rel = Path("wider_face") / rel_str
        dst_img = out / "images" / split / rel
        dst_lb = out / "labels" / split / rel.with_suffix(".txt")
        lines = [line for box in boxes if (line := yolo_line(FACE_CLS, box, width, height))]
        symlink(src, dst_img)
        write_label(dst_lb, lines)
        list_paths.append(str(dst_img))
    return list_paths


def extract_crowdhuman() -> None:
    marker = CROWD / "crowdhuman" / "annotation_train.odgt"
    if marker.exists():
        return
    with tarfile.open(CROWD_TAR) as tf:
        tf.extractall(CROWD)


def read_crowdhuman(split: str) -> list[dict]:
    odgt = CROWD / "crowdhuman" / f"annotation_{split}.odgt"
    return [json.loads(line) for line in odgt.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_crowdhuman(out: Path, split: str) -> list[str]:
    records = read_crowdhuman(split)
    image_dir = CROWD / "crowdhuman" / split / "Images"
    list_paths = []
    for rec in records:
        src = image_dir / f"{rec['ID']}.jpg"
        if not src.exists():
            continue
        with Image.open(src) as im:
            width, height = im.size
        lines = []
        for gt in rec.get("gtboxes", []):
            if gt.get("tag") != "person":
                continue
            extra = gt.get("extra", {})
            if extra.get("ignore", 0):
                continue
            line = yolo_line(PERSON_CLS, gt["fbox"], width, height)
            if line:
                lines.append(line)
        rel = Path("crowdhuman") / f"{rec['ID']}.jpg"
        dst_img = out / "images" / split / rel
        dst_lb = out / "labels" / split / rel.with_suffix(".txt")
        symlink(src, dst_img)
        write_label(dst_lb, lines)
        list_paths.append(str(dst_img))
    return list_paths


def weighted_repeat(paths: dict[str, list[str]], weights: dict[str, int], seed: int) -> list[str]:
    rng = random.Random(seed)
    total = sum(weights.values())
    base = min(len(paths[k]) / (weights[k] / total) for k in weights if paths[k])
    mixed = []
    for key, weight in weights.items():
        target = max(1, int(round(base * weight / total)))
        source = paths[key]
        if not source:
            continue
        reps, rem = divmod(target, len(source))
        sampled = source * reps + rng.sample(source, rem)
        rng.shuffle(sampled)
        mixed.extend(sampled)
    rng.shuffle(mixed)
    return mixed


def write_lists(out: Path, train_paths: dict[str, list[str]], val_paths: dict[str, list[str]], weights: dict[str, int], seed: int) -> None:
    train = weighted_repeat(train_paths, weights, seed)
    val = []
    for key in ("lvis", "crowdhuman", "wider_face"):
        val.extend(val_paths.get(key, []))
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    summary = {f"train_{k}": len(v) for k, v in train_paths.items()} | {f"val_{k}": len(v) for k, v in val_paths.items()}
    summary["train_weighted_total"] = len(train)
    summary["val_total"] = len(val)
    (out / "stage_a_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_smoke_lists(out: Path, train: list[str], val: list[str], train_n: int = 8, val_n: int = 4) -> None:
    (out / "smoke_train.txt").write_text("\n".join(train[:train_n]) + "\n", encoding="utf-8")
    (out / "smoke_val.txt").write_text("\n".join(val[:val_n]) + "\n", encoding="utf-8")


def yaml_text(out: Path, train: str, val: str) -> str:
    names = [f"lvis_{i}" for i in range(1203)] + ["person", "face"]
    text = [
        "path: " + str(out.resolve()),
        f"train: {train}",
        f"val: {val}",
        "nc: 1205",
        "names:",
    ]
    text += [f"  {i}: {name}" for i, name in enumerate(names)]
    return "\n".join(text) + "\n"


def write_yaml(out: Path) -> None:
    (out / "yolo26ps_stage_a.yaml").write_text(yaml_text(out, "train.txt", "val.txt"), encoding="utf-8")
    (out / "yolo26ps_stage_a_smoke.yaml").write_text(
        yaml_text(out, "smoke_train.txt", "smoke_val.txt"), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    global DATASETS, OUT, COCO, LVIS, WIDER, CROWD, CROWD_TAR
    DATASETS = args.datasets
    OUT = args.out
    COCO = DATASETS / "COCO_2017"
    LVIS = DATASETS / "LVIS" / "data"
    WIDER = DATASETS / "WIDER_FACE" / "extracted" / "WIDER_Face_Detection"
    CROWD = DATASETS / "CrowdHuman" / "CrowdHuman"
    CROWD_TAR = CROWD / "crowdhuman.tar.00"

    mkdirs(OUT)
    if not args.skip_crowdhuman_extract:
        extract_crowdhuman()

    train_paths = {
        "lvis": prepare_lvis(OUT, "train"),
        "crowdhuman": prepare_crowdhuman(OUT, "train"),
        "wider_face": prepare_wider(OUT, "train"),
    }
    val_paths = {
        "lvis": prepare_lvis(OUT, "val"),
        "crowdhuman": prepare_crowdhuman(OUT, "val"),
        "wider_face": prepare_wider(OUT, "val"),
    }
    weights = {"lvis": args.lvis, "crowdhuman": args.crowdhuman, "wider_face": args.wider_face}
    write_lists(OUT, train_paths, val_paths, weights, args.seed)
    train = (OUT / "train.txt").read_text(encoding="utf-8").splitlines()
    val = (OUT / "val.txt").read_text(encoding="utf-8").splitlines()
    write_smoke_lists(OUT, train, val)
    write_yaml(OUT)
    print((OUT / "yolo26ps_stage_a.yaml").resolve())


if __name__ == "__main__":
    main()
