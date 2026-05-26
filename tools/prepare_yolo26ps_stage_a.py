#!/usr/bin/env python3
"""Prepare YOLO26s-PS-2.5D Stage A detection warmup data.

The script converts Objects365, CrowdHuman, and WIDER FACE annotations into a single YOLO detection dataset:

    Objects365 categories -> 0..364, with Objects365 Person at class 0
    WIDER FACE -> 365

CrowdHuman person boxes are also mapped to class 0. Objects365 is the only source that should supervise the full
Objects365 class set; the dataloader/loss infer per-image class supervision scope from source path segments.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
import zipfile
from pathlib import Path

import yaml
from PIL import Image


DATASETS = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
OUT = DATASETS / "YOLO26PS_STAGE_A"
OBJECTS365 = DATASETS / "Objects365"
OBJECTS365_DSDL = OBJECTS365 / "OpenDataLab___Objects365" / "dsdl" / "dsdl_Det_full.zip"
OBJECTS365_IMAGES = OBJECTS365 / "extracted" / "Objects365" / "data"
WIDER = DATASETS / "WIDER_FACE" / "extracted" / "WIDER_Face_Detection"
CROWD = DATASETS / "CrowdHuman" / "CrowdHuman"
CROWD_TAR = CROWD / "crowdhuman.tar.00"

OBJECTS365_NC = 365
NC = OBJECTS365_NC + 1
PERSON_CLS = 0
FACE_CLS = 365


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=DATASETS, help="vision_benchmarks root")
    parser.add_argument("--out", type=Path, default=OUT, help="prepared dataset root")
    parser.add_argument("--seed", type=int, default=42, help="legacy weighted train-list sampling seed")
    parser.add_argument("--objects365", type=int, default=45, help="legacy Stage A train sampling weight")
    parser.add_argument("--crowdhuman", type=int, default=35, help="legacy Stage A train sampling weight")
    parser.add_argument("--wider-face", type=int, default=20, help="legacy Stage A train sampling weight")
    parser.add_argument("--skip-crowdhuman-extract", action="store_true", help="do not extract CrowdHuman tar")
    parser.add_argument("--max-objects365-train", type=int, help="optional cap for quick local preparation")
    parser.add_argument("--max-objects365-val", type=int, help="optional cap for quick local preparation")
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


def objects365_class_names() -> list[str]:
    with zipfile.ZipFile(OBJECTS365_DSDL) as zf:
        text = zf.read("dsdl_Det_full/defs/class-domain.yaml").decode("utf-8")
    data = yaml.safe_load(text)
    names = data["Object365ClassDomain"]["classes"]
    return [str(name).split(".")[-1].strip().replace(" ", "_") for name in names]


def objects365_image_roots(split: str) -> list[Path]:
    candidates = [
        OBJECTS365_IMAGES / split,
        OBJECTS365_IMAGES / ("val" if split == "val" else "train"),
        OBJECTS365 / "extracted" / "Objects365" / split,
        OBJECTS365 / split,
    ]
    roots = [p for p in candidates if p.exists() and p.is_dir()]
    return roots or [OBJECTS365_IMAGES]


def resolve_objects365_image(media_path: str, split: str) -> Path | None:
    rel = Path(media_path)
    for root in objects365_image_roots(split):
        candidates = [
            root / rel,
            root / Path(*rel.parts[1:]) if len(rel.parts) > 1 else root / rel.name,
            root / rel.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def iter_dsdl_samples(split: str):
    """Yield DSDL sample dicts without materializing multi-GB train JSON into memory."""
    member = f"dsdl_Det_full/set-{split}/{split}_samples.json"
    with zipfile.ZipFile(OBJECTS365_DSDL) as zf:
        with zf.open(member) as f:
            yield from iter_samples_array(f)


def iter_samples_array(raw) -> dict:
    """Stream objects from the top-level ``{"samples": [...]}`` DSDL JSON payload."""
    text = io.TextIOWrapper(raw, encoding="utf-8")
    decoder = json.JSONDecoder()
    buf = ""
    in_samples = False
    eof = False

    while True:
        if not eof and len(buf) < 1 << 20:
            chunk = text.read(1 << 20)
            eof = not chunk
            buf += chunk
        if not in_samples:
            marker = '"samples"'
            idx = buf.find(marker)
            if idx < 0:
                if eof:
                    return
                buf = buf[-len(marker) :]
                continue
            arr = buf.find("[", idx + len(marker))
            if arr < 0:
                if eof:
                    return
                buf = buf[idx:]
                continue
            buf = buf[arr + 1 :]
            in_samples = True

        buf = buf.lstrip()
        if buf.startswith("]"):
            return
        try:
            obj, end = decoder.raw_decode(buf)
        except json.JSONDecodeError:
            if eof:
                raise
            chunk = text.read(1 << 20)
            eof = not chunk
            buf += chunk
            continue
        yield obj
        buf = buf[end:].lstrip()
        if buf.startswith(","):
            buf = buf[1:]


def prepare_objects365(out: Path, split: str, max_items: int | None = None) -> list[str]:
    list_paths = []
    for sample in iter_dsdl_samples(split):
        media = sample.get("media", {})
        media_path = media.get("media_path") or media.get("path") or media.get("file_name")
        if not media_path:
            continue
        src = resolve_objects365_image(media_path, split)
        if src is None:
            continue
        shape = media.get("media_shape") or media.get("shape")
        if shape and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
        else:
            with Image.open(src) as im:
                width, height = im.size
        lines = []
        for ann in sample.get("annotations", []):
            if ann.get("isfake", 0) or ann.get("ignore", 0):
                continue
            category_id = int(ann["category_id"])
            cls = category_id - 1
            if not (0 <= cls < OBJECTS365_NC):
                continue
            line = yolo_line(cls, ann["bbox"], width, height)
            if line:
                lines.append(line)
        rel = Path("objects365") / Path(media_path)
        dst_img = out / "images" / split / rel
        dst_lb = out / "labels" / split / rel.with_suffix(".txt")
        symlink(src, dst_img)
        write_label(dst_lb, lines)
        list_paths.append(str(dst_img))
        if max_items and len(list_paths) >= max_items:
            break
    if not list_paths:
        raise RuntimeError(
            f"No Objects365 {split} images were prepared. Extract Objects365 images under {OBJECTS365_IMAGES} first."
        )
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
    active = {k: v for k, v in paths.items() if v and weights.get(k, 0) > 0}
    if not active:
        return []
    total = sum(weights[k] for k in active)
    base = min(len(paths[k]) / (weights[k] / total) for k in active)
    mixed = []
    for key in active:
        target = max(1, int(round(base * weights[key] / total)))
        source = paths[key]
        reps, rem = divmod(target, len(source))
        sampled = source * reps + rng.sample(source, rem)
        rng.shuffle(sampled)
        mixed.extend(sampled)
    rng.shuffle(mixed)
    return mixed


def source_counts(paths: list[str]) -> dict[str, int]:
    counts = {"objects365": 0, "crowdhuman": 0, "wider_face": 0}
    for path in paths:
        parts = set(Path(path).parts)
        for key in counts:
            if key in parts:
                counts[key] += 1
                break
    return counts


def write_lists(
    out: Path, train_paths: dict[str, list[str]], val_paths: dict[str, list[str]], weights: dict[str, int], seed: int
) -> None:
    train = []
    for key in ("objects365", "crowdhuman", "wider_face"):
        train.extend(train_paths.get(key, []))
    train_weighted_legacy = weighted_repeat(train_paths, weights, seed)
    val = []
    for key in ("objects365", "crowdhuman", "wider_face"):
        val.extend(val_paths.get(key, []))
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "train_all.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "train_weighted_legacy.txt").write_text("\n".join(train_weighted_legacy) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    summary = {f"train_{k}": len(v) for k, v in train_paths.items()} | {f"val_{k}": len(v) for k, v in val_paths.items()}
    summary["train_total"] = len(train)
    summary["train_all_total"] = len(train)
    summary["train_weighted_legacy_total"] = len(train_weighted_legacy)
    summary |= {f"train_weighted_legacy_{k}": v for k, v in source_counts(train_weighted_legacy).items()}
    summary["val_total"] = len(val)
    (out / "stage_a_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_smoke_lists(out: Path, train: list[str], val: list[str], train_n: int = 8, val_n: int = 4) -> None:
    (out / "smoke_train.txt").write_text("\n".join(train[:train_n]) + "\n", encoding="utf-8")
    (out / "smoke_val.txt").write_text("\n".join(val[:val_n]) + "\n", encoding="utf-8")


def yaml_text(out: Path, train: str, val: str) -> str:
    names = objects365_class_names() + ["face"]
    text = [
        "path: " + str(out.resolve()),
        f"train: {train}",
        f"val: {val}",
        f"nc: {NC}",
        f"person_cls: {PERSON_CLS}",
        f"face_cls: {FACE_CLS}",
        f"det_base_nc: {OBJECTS365_NC}",
        "det_extra_classes: [face]",
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
    global DATASETS, OUT, OBJECTS365, OBJECTS365_DSDL, OBJECTS365_IMAGES, WIDER, CROWD, CROWD_TAR
    DATASETS = args.datasets
    OUT = args.out
    OBJECTS365 = DATASETS / "Objects365"
    OBJECTS365_DSDL = OBJECTS365 / "OpenDataLab___Objects365" / "dsdl" / "dsdl_Det_full.zip"
    OBJECTS365_IMAGES = OBJECTS365 / "extracted" / "Objects365" / "data"
    WIDER = DATASETS / "WIDER_FACE" / "extracted" / "WIDER_Face_Detection"
    CROWD = DATASETS / "CrowdHuman" / "CrowdHuman"
    CROWD_TAR = CROWD / "crowdhuman.tar.00"

    mkdirs(OUT)
    if not args.skip_crowdhuman_extract:
        extract_crowdhuman()

    train_paths = {
        "objects365": prepare_objects365(OUT, "train", args.max_objects365_train),
        "crowdhuman": prepare_crowdhuman(OUT, "train"),
        "wider_face": prepare_wider(OUT, "train"),
    }
    val_paths = {
        "objects365": prepare_objects365(OUT, "val", args.max_objects365_val),
        "crowdhuman": prepare_crowdhuman(OUT, "val"),
        "wider_face": prepare_wider(OUT, "val"),
    }
    weights = {"objects365": args.objects365, "crowdhuman": args.crowdhuman, "wider_face": args.wider_face}
    write_lists(OUT, train_paths, val_paths, weights, args.seed)
    train = (OUT / "train.txt").read_text(encoding="utf-8").splitlines()
    val = (OUT / "val.txt").read_text(encoding="utf-8").splitlines()
    write_smoke_lists(OUT, train, val)
    write_yaml(OUT)
    print((OUT / "yolo26ps_stage_a.yaml").resolve())


if __name__ == "__main__":
    main()
