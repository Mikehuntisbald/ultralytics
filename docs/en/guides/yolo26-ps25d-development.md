---
comments: true
description: Developer notes and Stage A training recipe for the YOLO26s-PS-2.5D multi-task model.
keywords: YOLO26, PS-2.5D, multi-task, Stage A, LVIS, CrowdHuman, WIDER FACE, dynamic input
---

# YOLO26s-PS-2.5D Development Notes

This page records the implementation contract for `YOLO26s-PS-2.5D` and the current Stage A detection warmup workflow.
It is a developer-facing note for this fork, not an upstream Ultralytics product page.

## Model Contract

`YOLO26s-PS-2.5D` keeps the YOLO26 backbone, P2-P5 PAN/FPN neck, NMS-free detection path, and no-DFL design. The final
head is `YOLO26PSDetect25D` and exports:

| Output | Shape | Notes |
| --- | --- | --- |
| `det_out` | `[B, 300, 6]` | `[x1, y1, x2, y2, score, cls_id]` after topK decode. |
| `body25d_out` | `[B, 300, 17, 4]` | `[x_norm, y_norm, z_norm, conf]`, valid only for `person` detections. |
| `mask_coef` | `[B, 300, 32]` | Person-bound instance mask coefficients. |
| `mask_proto` | `[B, 32, padded_H/4, padded_W/4]` | Dense stride-4 prototype map on padded input. |
| `scene_seg` | `[B, 150, padded_H/4, padded_W/4]` | ADE20K-style semantic logits on padded input. |

Training targets remain raw image/depth values:

```yaml
body25d:
  output: [x_norm, y_norm, z_norm, conf]
  z_normalization: torso_length_3d_or_bbox_height
  training_target:
    raw: [x, y, z_rel, conf]
    deploy_output: [x_norm, y_norm, z_norm, conf]
```

The `mask_proto` and `scene_seg` dense maps intentionally use padded input size. Postprocess must inverse-letterbox to
the original image when presenting masks or semantic maps.

## Dynamic Input Status

The model head, predictor, validator, exporter, and inverse-letterbox postprocess paths are dynamic. Detection decode no
longer assumes a fixed feature map size, and the dense stride-4 outputs follow the padded input shape.

Current caveat: the stock Ultralytics training loop calls `check_imgsz(..., max_dim=1)` and coerces train/val `imgsz` to
a single square integer. Passing `--imgsz 448 768` is parsed by the Stage A script, but the trainer currently converts it
to `imgsz=768`. Dynamic rectangular train sizes need a separate trainer/dataloader change. Prediction/export can still
use rectangular inputs.

## Unified Label Schema

The loader accepts both legacy YOLO labels and unified JSON records. Task flags are independent:

```json
{
  "task_flags": {
    "has_det": true,
    "has_pose2d": false,
    "has_pose3d": false,
    "has_person_mask": false,
    "has_scene_seg": false
  }
}
```

The unified multi-task loss gates every term by flags. Images with `has_det=false` are skipped by detector loss and are
not treated as background negatives.

## Stage A Data

Stage A uses:

- LVIS: 55%
- CrowdHuman: 25%
- WIDER FACE: 20%

Prepared dataset path:

```bash
/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_A
```

Preparation command:

```bash
python tools/prepare_yolo26ps_stage_a.py \
  --datasets /home/haoyi/Downloads/datasets/vision_benchmarks \
  --out /home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_A
```

Current prepared summary:

```json
{
  "train_lvis": 100170,
  "train_crowdhuman": 15000,
  "train_wider_face": 12880,
  "val_lvis": 4809,
  "val_crowdhuman": 4370,
  "val_wider_face": 3226,
  "train_weighted_total": 60000,
  "val_total": 12405
}
```

## Stage A Training

Stage A freezes pose/mask/scene auxiliary branches and uses the unified loss with detection weight only:

```yaml
loss:
  det: 1.0
  pose2d: 0.0
  pose_z: 0.0
  pose_vis: 0.0
  bone: 0.0
  person_mask: 0.0
  scene_seg: 0.0
```

Stable RTX 5090 32GB parameters found by probe:

```bash
python tools/train_yolo26ps_stage_a.py \
  --imgsz 704 \
  --batch 8 \
  --accumulate 10 \
  --workers 8 \
  --epochs 50 \
  --no-val \
  --device 0 \
  --name yolo26ps_stage_a_detection_704_b8_acc10
```

This gives an effective batch of 80. `--no-val` is recommended for the long Stage A run; run a separate validation pass
from the saved checkpoint when the 50-epoch warmup finishes. Probe results:

- `imgsz=768, batch=8` reached about 27.8 GB but repeatedly fell back to CPU in `TaskAlignedAssigner`.
- `imgsz=768, batch=7` reached about 27.5 GB in PyTorch and about 31.3 GB in `nvidia-smi` without fallback, but ran at about 1.0 it/s.
- `imgsz=704, batch=8` reached about 26.2 GB in PyTorch and about 29.8 GB in `nvidia-smi` without fallback, with about 2.4 it/s in the 1% probe.
- `imgsz=704, batch=9` repeatedly fell back to CPU in `TaskAlignedAssigner`.
- `imgsz=640, batch=10` completed without fallback, but did not improve throughput enough to offset the lower small-object resolution.
- `imgsz=768, batch=8`, `imgsz=768, batch=16`, and `imgsz=768, batch=32` OOM or fell back in the assigner.

For quick smoke or VRAM probes:

```bash
python tools/train_yolo26ps_stage_a.py \
  --imgsz 704 \
  --batch 8 \
  --accumulate 10 \
  --epochs 1 \
  --fraction 0.01 \
  --no-val \
  --no-save \
  --device 0 \
  --name yolo26ps_stage_a_probe
```
