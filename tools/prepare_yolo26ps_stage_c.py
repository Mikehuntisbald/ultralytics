#!/usr/bin/env python3
"""Prepare YOLO26s-PS-2.5D Stage C 2.5D pose data.

The Stage C warmup needs real `has_pose3d` samples. This script builds the
unified-schema train/val lists expected by `yolo26ps_stage_c_pose25d.yaml` from:

- COCO-WholeBody records already prepared for Stage B (2D only)
- 3DPW sequence pkl files (2D + root-relative z)
- AGORA 1280x720 images and SMPL dataframe pkl files (2D + root-relative z)
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


VISION_ROOT = Path("/home/haoyi/Downloads/datasets/vision_benchmarks")
HUMAN_ROOT = Path("/home/haoyi/Downloads/datasets/human_benchmarks")
OUT = VISION_ROOT / "YOLO26PS_STAGE_MULTI"

COCO17_FROM_OPENPOSE18 = {
    0: 0,  # nose
    1: 15,  # left eye
    2: 14,  # right eye
    3: 17,  # left ear
    4: 16,  # right ear
    5: 5,  # left shoulder
    6: 2,  # right shoulder
    7: 6,  # left elbow
    8: 3,  # right elbow
    9: 7,  # left wrist
    10: 4,  # right wrist
    11: 11,  # left hip
    12: 8,  # right hip
    13: 12,  # left knee
    14: 9,  # right knee
    15: 13,  # left ankle
    16: 10,  # right ankle
}

COCO17_FROM_SMPL24 = {
    0: 15,  # head proxy for nose
    5: 16,
    6: 17,
    7: 18,
    8: 19,
    9: 20,
    10: 21,
    11: 1,
    12: 2,
    13: 4,
    14: 5,
    15: 7,
    16: 8,
}

COCO17_FROM_AGORA45 = {
    **COCO17_FROM_SMPL24,
    0: 24,  # AGORA face proxy for nose
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-root", type=Path, default=VISION_ROOT)
    parser.add_argument("--human-root", type=Path, default=HUMAN_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-coco-train", type=int, default=0, help="optional COCO-WholeBody train cap")
    parser.add_argument("--max-coco-val", type=int, default=0, help="optional COCO-WholeBody val cap")
    parser.add_argument("--max-3dpw-train", type=int, default=0, help="optional 3DPW train image cap")
    parser.add_argument("--max-3dpw-val", type=int, default=0, help="optional 3DPW val image cap")
    parser.add_argument("--max-agora-train", type=int, default=0, help="optional AGORA train image cap")
    parser.add_argument("--max-agora-val", type=int, default=0, help="optional AGORA val image cap")
    parser.add_argument("--agora-occlusion-thr", type=float, default=95.0)
    parser.add_argument("--kpt-conf-thr", type=float, default=0.05)
    parser.add_argument(
        "--3dpw-kpt-conf-thr",
        type=float,
        default=None,
        help="OpenPose confidence threshold used only for 3DPW keypoints; defaults to max(kpt-conf-thr, 0.20)",
    )
    parser.add_argument("--no-3dpw-reproj-filter", action="store_true", help="disable 3DPW SMPL-vs-OpenPose sanity filter")
    parser.add_argument("--3dpw-reproj-min-kpts", type=int, default=6)
    parser.add_argument("--3dpw-reproj-median-thr", type=float, default=25.0)
    parser.add_argument("--3dpw-reproj-p80-thr", type=float, default=60.0)
    parser.add_argument("--3dpw-reproj-max-thr", type=float, default=120.0)
    parser.add_argument(
        "--3dpw-joint-reproj-thr",
        type=float,
        default=40.0,
        help="Drop individual 3DPW joints whose OpenPose-vs-SMPL projection error exceeds this threshold; <=0 disables",
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


def cap_records(records: list[dict], limit: int) -> list[dict]:
    if limit and len(records) > limit:
        return records[:limit]
    return records


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def bbox_from_keypoints(kpts: np.ndarray, width: int, height: int) -> list[float] | None:
    valid = kpts[:, 2] > 0
    if not bool(valid.any()):
        return None
    pts = kpts[valid, :2].copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    pad = max(bw, bh) * 0.08
    x1 = max(0.0, float(x1 - pad))
    y1 = max(0.0, float(y1 - pad))
    x2 = min(float(width - 1), float(x2 + pad))
    y2 = min(float(height - 1), float(y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def rel_z(kpts3d: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros((17, 4), dtype=np.float32)
    if valid[11] and valid[12]:
        root_z = float((kpts3d[11, 2] + kpts3d[12, 2]) * 0.5)
    else:
        root_z = 0.0
    out[:, 2] = kpts3d[:, 2] - root_z
    out[:, 3] = valid.astype(np.float32)
    return out


def make_instance(kpts2d: np.ndarray, kpts3d: np.ndarray, valid3d: np.ndarray, width: int, height: int) -> dict | None:
    bbox = bbox_from_keypoints(kpts2d, width, height)
    if bbox is None:
        return None
    body3d = rel_z(kpts3d, valid3d)
    body3d[:, :2] = kpts2d[:, :2]
    return {
        "category": "person",
        "bbox": [round(float(v), 3) for v in bbox],
        "body_kpts_2d": np.round(kpts2d, 3).tolist(),
        "body_kpts_3d": np.round(body3d, 5).tolist(),
        "flags": {
            "has_bbox": True,
            "has_body2d": bool((kpts2d[:, 2] > 0).any()),
            "has_body3d": bool(valid3d.any()),
            "has_person_mask": False,
        },
    }


def make_record(image_path: Path, width: int, height: int, source: str, instances: list[dict]) -> dict:
    return {
        "image": str(image_path.resolve()),
        "source": source,
        "width": int(width),
        "height": int(height),
        "instances": instances,
        "task_flags": {
            "has_det": bool(instances),
            "has_pose2d": any(inst["flags"]["has_body2d"] for inst in instances),
            "has_pose3d": any(inst["flags"]["has_body3d"] for inst in instances),
            "has_person_mask": False,
            "has_scene_seg": False,
        },
    }


def openpose18_to_coco17(pose: np.ndarray, conf_thr: float) -> np.ndarray:
    out = np.zeros((17, 3), dtype=np.float32)
    pts = np.asarray(pose, dtype=np.float32).reshape(18, 3)
    for coco_i, op_i in COCO17_FROM_OPENPOSE18.items():
        x, y, conf = pts[op_i]
        if conf > conf_thr and x > 0 and y > 0:
            out[coco_i] = [x, y, 2.0]
    return out


def smpl24_to_coco17(points3d: np.ndarray, kpts2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((17, 3), dtype=np.float32)
    valid = np.zeros(17, dtype=bool)
    pts = np.asarray(points3d, dtype=np.float32).reshape(-1, 3)
    for coco_i, smpl_i in COCO17_FROM_SMPL24.items():
        if smpl_i < len(pts) and kpts2d[coco_i, 2] > 0:
            out[coco_i] = pts[smpl_i]
            valid[coco_i] = True
    return out, valid


def project_3dpw_coco17(kpts3d: np.ndarray, cam_intrinsics: np.ndarray, cam_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3DPW SMPL joints into image coordinates using the frame camera pose."""
    points = np.asarray(kpts3d, dtype=np.float32).reshape(-1, 3)
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    cam = (np.asarray(cam_pose, dtype=np.float32) @ hom.T).T[:, :3]
    z = cam[:, 2]
    xy = np.full((len(points), 2), np.nan, dtype=np.float32)
    valid = np.abs(z) > 1e-6
    k = np.asarray(cam_intrinsics, dtype=np.float32)
    xy[valid, 0] = k[0, 0] * cam[valid, 0] / z[valid] + k[0, 2]
    xy[valid, 1] = k[1, 1] * cam[valid, 1] / z[valid] + k[1, 2]
    return xy, z


def good_3dpw_reprojection(
    kpts2d: np.ndarray,
    kpts3d: np.ndarray,
    valid3d: np.ndarray,
    cam_intrinsics: np.ndarray,
    cam_pose: np.ndarray,
    min_kpts: int,
    median_thr: float,
    p80_thr: float,
    max_thr: float,
) -> bool:
    """Return whether OpenPose 2D is consistent with 3DPW SMPL camera projection."""
    projected, depth = project_3dpw_coco17(kpts3d, cam_intrinsics, cam_pose)
    mask = (kpts2d[:, 2] > 0) & valid3d & np.isfinite(projected[:, 0]) & (depth > 0.05)
    if int(mask.sum()) < min_kpts:
        return False
    error = np.linalg.norm(projected[mask] - kpts2d[mask, :2], axis=1)
    return bool(np.median(error) <= median_thr and np.percentile(error, 80) <= p80_thr and error.max() <= max_thr)


def clean_3dpw_reprojected_joints(
    kpts2d: np.ndarray,
    valid3d: np.ndarray,
    kpts3d: np.ndarray,
    cam_intrinsics: np.ndarray,
    cam_pose: np.ndarray,
    joint_thr: float,
) -> int:
    """Hide individual 3DPW joints with large OpenPose-vs-SMPL projection disagreement."""
    if joint_thr <= 0:
        return 0
    projected, depth = project_3dpw_coco17(kpts3d, cam_intrinsics, cam_pose)
    mask = (kpts2d[:, 2] > 0) & valid3d & np.isfinite(projected[:, 0]) & (depth > 0.05)
    if not bool(mask.any()):
        return 0
    error = np.zeros(len(kpts2d), dtype=np.float32)
    error[mask] = np.linalg.norm(projected[mask] - kpts2d[mask, :2], axis=1)
    bad = mask & (error > joint_thr)
    if not bool(bad.any()):
        return 0
    kpts2d[bad] = 0.0
    kpts3d[bad] = 0.0
    valid3d[bad] = False
    return int(bad.sum())


def agora45_to_coco17(points2d: np.ndarray, points3d: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kpts2d = np.zeros((17, 3), dtype=np.float32)
    kpts3d = np.zeros((17, 3), dtype=np.float32)
    valid3d = np.zeros(17, dtype=bool)
    pts2d = np.asarray(points2d, dtype=np.float32).reshape(-1, 2) / 3.0
    pts3d = np.asarray(points3d, dtype=np.float32).reshape(-1, 3)
    for coco_i, src_i in COCO17_FROM_AGORA45.items():
        if src_i >= len(pts2d) or src_i >= len(pts3d):
            continue
        x, y = pts2d[src_i]
        if -width * 0.25 <= x <= width * 1.25 and -height * 0.25 <= y <= height * 1.25:
            kpts2d[coco_i] = [x, y, 2.0]
            kpts3d[coco_i] = pts3d[src_i]
            valid3d[coco_i] = True
    return kpts2d, kpts3d, valid3d


def prepare_3dpw(
    root: Path,
    split: str,
    limit: int,
    conf_thr: float,
    use_reproj_filter: bool,
    reproj_min_kpts: int,
    reproj_median_thr: float,
    reproj_p80_thr: float,
    reproj_max_thr: float,
    joint_reproj_thr: float,
) -> tuple[list[dict], dict[str, int]]:
    seq_dir = root / "sequenceFiles" / split
    image_root = root / "imageFiles"
    records: list[dict] = []
    by_image: dict[Path, list[dict]] = defaultdict(list)
    stats = {
        "instances_seen": 0,
        "instances_kept": 0,
        "instances_low_kpt_rejected": 0,
        "instances_reproj_rejected": 0,
        "instances_joint_cleaned_rejected": 0,
        "joints_reproj_cleaned": 0,
    }

    for pkl_path in sorted(seq_dir.glob("*.pkl")):
        with pkl_path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        sequence = str(data.get("sequence", pkl_path.stem))
        frame_ids = [int(x) for x in data["img_frame_ids"]]
        cam_intrinsics = np.asarray(data.get("cam_intrinsics", np.eye(3)), dtype=np.float32)
        cam_poses = np.asarray(data.get("cam_poses", []), dtype=np.float32)
        image_dir = image_root / sequence
        for frame_i, frame_id in enumerate(frame_ids):
            # 3DPW pkl arrays are indexed by extracted-frame order, while img_frame_ids
            # keep the original video frame numbers. The extracted image files are
            # named by the compact array index, not by the original video frame id.
            image_path = image_dir / f"image_{frame_i:05d}.jpg"
            if not image_path.exists():
                continue
            try:
                width, height = image_size(image_path)
            except OSError:
                continue
            instances = []
            for person_i, pose2d in enumerate(data.get("poses2d", [])):
                pose2d = np.asarray(pose2d)
                if frame_i >= pose2d.shape[0]:
                    continue
                stats["instances_seen"] += 1
                kpts2d = openpose18_to_coco17(pose2d[frame_i].T, conf_thr)
                if int((kpts2d[:, 2] > 0).sum()) < 6:
                    stats["instances_low_kpt_rejected"] += 1
                    continue
                joints = np.asarray(data.get("jointPositions", [])[person_i], dtype=np.float32)
                if joints.ndim == 2 and frame_i < joints.shape[0]:
                    kpts3d, valid3d = smpl24_to_coco17(joints[frame_i].reshape(-1, 3), kpts2d)
                else:
                    kpts3d, valid3d = np.zeros((17, 3), dtype=np.float32), np.zeros(17, dtype=bool)
                if (
                    use_reproj_filter
                    and valid3d.any()
                    and frame_i < len(cam_poses)
                    and not good_3dpw_reprojection(
                        kpts2d,
                        kpts3d,
                        valid3d,
                        cam_intrinsics,
                        cam_poses[frame_i],
                        reproj_min_kpts,
                        reproj_median_thr,
                        reproj_p80_thr,
                        reproj_max_thr,
                    )
                ):
                    stats["instances_reproj_rejected"] += 1
                    continue
                if use_reproj_filter and valid3d.any() and frame_i < len(cam_poses):
                    cleaned = clean_3dpw_reprojected_joints(
                        kpts2d,
                        valid3d,
                        kpts3d,
                        cam_intrinsics,
                        cam_poses[frame_i],
                        joint_reproj_thr,
                    )
                    stats["joints_reproj_cleaned"] += cleaned
                    if cleaned and int((kpts2d[:, 2] > 0).sum()) < 6:
                        stats["instances_joint_cleaned_rejected"] += 1
                        continue
                inst = make_instance(kpts2d, kpts3d, valid3d, width, height)
                if inst is not None:
                    instances.append(inst)
                    stats["instances_kept"] += 1
            if instances:
                by_image[image_path].extend(instances)

    for image_path, instances in sorted(by_image.items()):
        width, height = image_size(image_path)
        records.append(make_record(image_path, width, height, "3dpw", instances))
        if limit and len(records) >= limit:
            break
    return records, stats


def agora_image_path(root: Path, split: str, shard: int, img_path: str) -> Path:
    src = Path(str(img_path))
    name = f"{src.stem}_1280x720{src.suffix}"
    return (root / ("validation" if split == "val" else f"train_{shard}") / name).resolve()


def iterable_value(values: object, index: int, default: object) -> object:
    try:
        return values[index]  # type: ignore[index]
    except Exception:
        return default


def prepare_agora(root: Path, split: str, limit: int, occlusion_thr: float) -> list[dict]:
    if split == "train":
        pkl_paths = sorted((root / "SMPL").glob("train_*_withjv.pkl"))
    else:
        pkl_paths = sorted((root / "validation_SMPL" / "SMPL").glob("validation_*_withjv.pkl"))
    records: list[dict] = []

    for pkl_path in pkl_paths:
        shard = int(pkl_path.stem.split("_")[1])
        df = pd.read_pickle(pkl_path)
        for _, row in df.iterrows():
            image_path = agora_image_path(root, split, shard, str(row["imgPath"]))
            if not image_path.exists():
                continue
            width, height = image_size(image_path)
            instances = []
            joints2d = row.get("gt_joints_2d") or []
            joints3d = row.get("gt_joints_3d") or []
            valids = row.get("isValid") or []
            occlusions = row.get("occlusion") or []
            for person_i, pts2d in enumerate(joints2d):
                if person_i >= len(joints3d):
                    continue
                is_valid = bool(iterable_value(valids, person_i, True))
                occlusion = float(iterable_value(occlusions, person_i, 0.0))
                if not is_valid or occlusion >= occlusion_thr:
                    continue
                kpts2d, kpts3d, valid3d = agora45_to_coco17(pts2d, joints3d[person_i], width, height)
                if int((kpts2d[:, 2] > 0).sum()) < 6:
                    continue
                inst = make_instance(kpts2d, kpts3d, valid3d, width, height)
                if inst is not None:
                    instances.append(inst)
            if instances:
                records.append(make_record(image_path, width, height, "agora", instances))
                if limit and len(records) >= limit:
                    return records
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def write_list(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(Path(rec["image"]).resolve()) for rec in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def source_counts(records: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[str(rec.get("source", ""))] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels" / "stage_c" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "stage_c" / "val").mkdir(parents=True, exist_ok=True)

    stage_b_manifest = out / "manifests"
    coco_train = cap_records(read_jsonl(stage_b_manifest / "stage_b_train.jsonl"), args.max_coco_train)
    coco_val = cap_records(read_jsonl(stage_b_manifest / "stage_b_val.jsonl"), args.max_coco_val)
    for rec in coco_train + coco_val:
        rec.setdefault("source", "coco_wholebody")

    threedpw_root = args.human_root / "3DPW" / "extracted"
    agora_root = args.human_root / "AGORA" / "extracted" / "AGORA"
    threedpw_kpt_conf_thr = (
        max(args.kpt_conf_thr, 0.20)
        if args.__dict__["3dpw_kpt_conf_thr"] is None
        else float(args.__dict__["3dpw_kpt_conf_thr"])
    )
    threedpw_train, threedpw_train_stats = prepare_3dpw(
        threedpw_root,
        "train",
        args.max_3dpw_train,
        threedpw_kpt_conf_thr,
        not args.no_3dpw_reproj_filter,
        args.__dict__["3dpw_reproj_min_kpts"],
        args.__dict__["3dpw_reproj_median_thr"],
        args.__dict__["3dpw_reproj_p80_thr"],
        args.__dict__["3dpw_reproj_max_thr"],
        args.__dict__["3dpw_joint_reproj_thr"],
    )
    threedpw_val, threedpw_val_stats = prepare_3dpw(
        threedpw_root,
        "validation",
        args.max_3dpw_val,
        threedpw_kpt_conf_thr,
        not args.no_3dpw_reproj_filter,
        args.__dict__["3dpw_reproj_min_kpts"],
        args.__dict__["3dpw_reproj_median_thr"],
        args.__dict__["3dpw_reproj_p80_thr"],
        args.__dict__["3dpw_reproj_max_thr"],
        args.__dict__["3dpw_joint_reproj_thr"],
    )
    agora_train = prepare_agora(agora_root, "train", args.max_agora_train, args.agora_occlusion_thr)
    agora_val = prepare_agora(agora_root, "val", args.max_agora_val, args.agora_occlusion_thr)

    train_records = coco_train + threedpw_train + agora_train
    val_records = coco_val + threedpw_val + agora_val

    write_list(out / "stage_c_train.txt", train_records)
    write_list(out / "stage_c_val.txt", val_records)
    write_jsonl(out / "manifests" / "stage_c_train.jsonl", train_records)
    write_jsonl(out / "manifests" / "stage_c_val.jsonl", val_records)

    summary = {
        "train_total": len(train_records),
        "val_total": len(val_records),
        "train_sources": source_counts(train_records),
        "val_sources": source_counts(val_records),
        "3dpw_reprojection_filter": {
            "enabled": not args.no_3dpw_reproj_filter,
            "kpt_conf_thr": threedpw_kpt_conf_thr,
            "min_kpts": args.__dict__["3dpw_reproj_min_kpts"],
            "median_thr": args.__dict__["3dpw_reproj_median_thr"],
            "p80_thr": args.__dict__["3dpw_reproj_p80_thr"],
            "max_thr": args.__dict__["3dpw_reproj_max_thr"],
            "joint_thr": args.__dict__["3dpw_joint_reproj_thr"],
            "train": threedpw_train_stats,
            "val": threedpw_val_stats,
        },
        "note": "Stage C warmup lists include pose/3D sources. Detection-only sources are intentionally left to detection-enabled stages.",
    }
    (out / "stage_c_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
