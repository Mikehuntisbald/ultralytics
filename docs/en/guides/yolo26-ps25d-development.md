---
comments: true
description: Developer notes and Stage A training recipe for the YOLO26s-PS-2.5D multi-task model.
keywords: YOLO26, PS-2.5D, multi-task, Stage A, Objects365, CrowdHuman, WIDER FACE, dynamic input
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

The model head, trainer, validator, predictor, exporter, and inverse-letterbox postprocess paths are dynamic. Detection
decode no longer assumes a fixed feature map size, and dense stride-4 outputs follow the padded input shape. Training and
validation accept rectangular `imgsz=[height, width]`; the default deployment-oriented size is `[448, 768]` with
`pad_stride=32`.

Training accepts fixed rectangular `imgsz=[height, width]` without forcing a square canvas, so Stage A can still use
Mosaic/MixUp/CopyPaste at the deployment-oriented aspect ratio. True `rect=True` aspect-ratio batch grouping remains
validation-oriented and disables mosaic-style mixing, as in the base Ultralytics pipeline.

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

Detection class supervision is also scoped per image. Objects365 images supervise the full remapped Objects365 class
set. CrowdHuman, pose, and mask datasets supervise only `person`, while WIDER FACE supervises only `face`; they do not
turn unannotated Objects365 classes into negative labels.

Mosaic, MixUp, and CopyPaste in `mixup` mode are source-aware for this model family. Mixed samples are drawn from the
same `det_class_mask` supervision domain as the anchor image, so an Objects365 mosaic stays Objects365-supervised, a
person-only mosaic stays person-only, and a WIDER mosaic stays face-only. This avoids the old failure mode where
cross-source mosaic intersections produced an empty or overly narrow class supervision mask.

Stage YAML owns mixed augmentation. If a stage omits `augment`, the stage trainer forces `mosaic`, `mixup`,
`copy_paste`, and `cutmix` to `0.0` instead of inheriting Ultralytics defaults. Pose, mask, scene, and final finetune
stages should keep those mixed augmentations explicitly closed unless the stage has been designed for them.

## Stage Sampling

Every stage uses the same sampler contract:

```yaml
epochs: 15
samples_per_epoch: 40000
sampling: weighted_random_with_replacement
sampling_weights: {objects365: 45, crowdhuman: 35, wider_face: 20}
```

`samples_per_epoch` controls the epoch length. With weighted replacement sampling, each step first samples a source by
`sampling_weights`, then draws a random image from that source. The implementation maps this onto PyTorch
`WeightedRandomSampler` by giving each image weight `source_weight / source_image_count`, matched from source names in
image path segments.

Stages should only change three things: data sampling weights, trainable branches, and task loss weights. The dataloader
and unified loss stay shared across Stage A-F.

## Stage A Data

Stage A `train.txt` is a full source list:

- all Objects365 train images
- all CrowdHuman train images
- all WIDER FACE train images

`train_all.txt` is kept as an explicit full-list alias, and `train_weighted_legacy.txt` preserves the old
`weighted_repeat` output for traceability. The training runtime can still use weighted replacement sampling over the
full list:

- Objects365: 45%
- CrowdHuman: 35%
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

`stage_a_summary.json` is regenerated by the preparation script and records per-source counts for the current local
Objects365 extraction.

## Stage B-F Data YAMLs

Stages B-F use unified-schema dataset YAMLs committed under `ultralytics/cfg/datasets/`:

- `yolo26ps_stage_b_pose2d.yaml`
- `yolo26ps_stage_c_pose25d.yaml`
- `yolo26ps_stage_d_person_mask.yaml`
- `yolo26ps_stage_e_scene_seg.yaml`
- `yolo26ps_stage_f_full_finetune.yaml`

They point to `/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI` and expect each converter to write
`stage_<letter>_train.txt`, `stage_<letter>_val.txt`, `manifests/stage_<letter>_*.jsonl`, and matching
`labels/stage_<letter>/...` JSON labels. The generic stage trainer reads the selected stage's `data_yaml` from the plan,
so `--stage B_pose2d` automatically uses the Stage B YAML unless `--data` is explicitly passed.

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

The shared stage entrypoint reads the plan YAML and applies the three stage controls: sampling weights, trainable
branches, and task loss weights. Stage A also sets `head.active_tasks={"det"}` so `p2_refine`, `proto`, `scene_seg`,
`cv4`, and `cv5` are not computed; inactive branches are frozen and forced back to `eval()` after every trainer
`model.train()` call to avoid BN stat pollution.

```bash
python tools/train_yolo26ps_stage.py \
  --stage A_detection_stable \
  --plan ultralytics/cfg/datasets/yolo26-ps25d-plan.yaml \
  --pretrain pretrains/yolo26s-det.pt \
  --device 0
```

Stage A should start from the official YOLO26s detection pretrain when available. Use `--pretrain`, not `--weights`,
for this case: the trainer first builds `yolo26s-ps25d.yaml`, then partially transfers matching backbone/neck weights
from `pretrains/yolo26s-det.pt`. `--weights` is reserved for same-architecture YOLO26s-PS-2.5D checkpoints when
resuming or chaining stages. Confirm the log prints `Transferred ... items from pretrained weights`; otherwise the run
is effectively random initialization.

Current plan defaults for Stage A:

- `epochs=15`
- `samples_per_epoch=40000`
- `val_samples=2000`
- `imgsz=[448, 768]`
- `batch=16`
- `accumulate=4`
- `freeze=0`
- `val=true` with simple validation capped by `val_samples`
- `optimizer=SGD`
- `lr0=0.006`
- `lrf=0.01`
- `momentum=0.937`
- `weight_decay=0.0005`
- `warmup_epochs=3.0`
- `cos_lr=true`
- `amp=true`

The shared optimizer schedule is cosine for every stage. Stage A uses SGD for the detection warmup; later stages keep
the plan default optimizer unless overridden. The plan keeps the stage learning rates explicit:

| Stage | `lr0` |
| --- | ---: |
| A detection | 0.006 |
| B pose2d | 0.003 |
| C pose25d | 0.002 |
| D person mask | 0.002 |
| E scene seg | 0.0015 |
| F finetune | 0.001 |

The old `tools/train_yolo26ps_stage_a.py` remains as a compatibility wrapper around the shared entrypoint.

Do not enable `freeze` by default for Stage A. Freezing early modules only makes sense after a compatible pretrain is
confirmed; if partial transfer fails, freezing random backbone layers prevents the detection warmup from recovering.
The current plan keeps `freeze=0` and trains the backbone, neck, and detection head while auxiliary task heads stay
inactive.

Validation is enabled for the normal Stage A run, but it is intentionally capped by `val_samples=2000`. The full Stage A
validation split is large enough to dominate every epoch, so routine training uses a deterministic evenly spaced subset
for quick signal. Run a full validation explicitly at stage boundaries or before selecting a release checkpoint.

For quick smoke or VRAM probes:

```bash
python tools/train_yolo26ps_stage.py \
  --stage A_detection_stable \
  --imgsz 448 768 \
  --batch 2 \
  --accumulate 1 \
  --epochs 1 \
  --samples-per-epoch 8 \
  --no-val \
  --no-save \
  --device 0 \
  --name yolo26ps_stage_a_probe
```

The previous square probe found `704, batch=7, accumulate=12` stable on the RTX 5090 32GB path. Rectangular `[448,768]`
uses less pixel area than square `704`. The current stable Stage A runtime uses rectangular `[448,768]`, `batch=16`,
`accumulate=4`, simple validation, and official YOLO26s detection pretrain. Batch 20 repeatedly triggered
TaskAlignedAssigner CPU fallback near the 32 GB limit, and batch 18 still produced intermittent fallback on crowd-heavy
batches. A slightly smaller batch that keeps assignment on GPU is usually faster than a larger batch that falls back to
CPU.

Stage A disables mosaic by default. If the deployment target should be closer to 16:10, prefer testing `[480,768]`
before lowering resolution; it is stride-32 aligned and exactly 16:10, but costs about 7% more pixels than `[448,768]`.
Only try `[480,768]` after `[448,768]` is free of TaskAlignedAssigner fallback.

## End-to-End Detection Loss

YOLO26-PS keeps the YOLO26 end-to-end detection design: training has two detection branches and deployment uses the
one-to-one branch for NMS-free output.

| Branch | Purpose | Current TAL settings |
| --- | --- | --- |
| `one2many` | Dense supervision. One GT may match several anchors, which improves early recall and stabilizes training. | default `tal_topk=10`; Stage A stable uses `8` |
| `one2one` | End-to-end/NMS-free supervision. It encourages each GT to settle onto one final prediction. | default `tal_topk=7`, `tal_topk2=1`; Stage A stable uses `5`, `1` |

The wrapper is implemented in `YOLO26PS25DE2ELoss` and reads the values from `model.args`:

```python
tal_topk_one2many: 10
tal_topk_one2one: 7
tal_topk2_one2one: 1
self.o2m = 0.8
self.o2o = 0.2
self.final_o2m = 0.1
```

`criterion.update()` decays the one-to-many contribution during training. At the beginning, one-to-many provides most of
the signal. Later, the loss shifts toward one-to-one so the exported branch is the one being optimized most directly.

`tal_topk` is not deployment `topK=300`. Deployment topK controls how many decoded detections are returned. TAL topK is
training-only and controls how many candidate anchors a GT can use during assignment.

## TaskAlignedAssigner Notes

`TaskAlignedAssigner` chooses positive anchors for each GT box. For every batch it builds intermediate tensors with a
shape close to:

```text
[batch, max_gt_in_batch, total_anchors]
```

With `[448,768]` and P2-P5, the model has about 28.5k candidate points per image:

```text
P2: 112x192 = 21504
P3:  56x96  = 5376
P4:  28x48  = 1344
P5:  14x24  = 336
total ~= 28560
```

Objects365 and CrowdHuman can produce batches with many GT boxes, so the assignment peak can be the real memory limit
even when the forward/backward pass itself fits. When assignment OOMs, Ultralytics catches the exception and reruns TAL
on CPU. That keeps training alive but lowers GPU utilization and increases epoch time.

Operational rule:

- If there is no TAL fallback, fill remaining memory by testing larger batch or `[480,768]`.
- If TAL fallback appears, reduce batch first. Do not raise resolution.
- If fallback persists at a reasonable batch, lower the plan TAL topK first. Stage A stable currently uses
  `tal_topk_one2many=8`, `tal_topk_one2one=5`, and `tal_topk2_one2one=1`.
- Keep `workers=20` unless CPU image loading clearly becomes the bottleneck; the current low-util symptom is assignment
  fallback, not dataloader starvation.

Useful checks during training:

```bash
rg -n "TaskAlignedAssigner|CUDA OutOfMemoryError|using CPU" runs/detect/<run>_train.log
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv
tail -n 8 runs/detect/<run>/results.csv
```

## GPU Utilization Tuning

The desired state is not simply "highest memory used." The desired state is steady GPU-side training without repeated
CPU fallbacks or long validation stalls.

Preferred tuning order for Stage A:

1. Keep `mosaic=0.0`, `mixup=0.0`, `copy_paste=0.0`, and `cutmix=0.0`.
2. Use `[448,768]`, `batch=16`, `accumulate=4` as the stable high-util setting.
3. If TAL fallback still appears, keep the same image size and reduce TAL topK before raising resolution.
4. After a clean epoch with no fallback, test `[480,768]` only if 16:10 fidelity matters more than throughput.
5. Keep simple validation capped with `val_samples=2000`; full validation belongs at stage boundaries.

Low GPU utilization can be healthy during validation, checkpoint save, plot generation, or the first few dataloader warmup
steps. It is unhealthy when it coincides with repeated `TaskAlignedAssigner` CPU fallback during the training progress
bar.

## Stage Chaining

Use `--pretrain` only when starting from an official YOLO26 detection checkpoint and building the PS-2.5D model from
YAML. Use `--weights` when the checkpoint already has the YOLO26s-PS-2.5D architecture.

Examples:

```bash
# Stage A from official detection pretrain.
python tools/train_yolo26ps_stage.py \
  --stage A_detection_stable \
  --pretrain pretrains/yolo26s-det.pt \
  --device 0

# Stage B from a finished Stage A checkpoint.
python tools/train_yolo26ps_stage.py \
  --stage B_pose2d \
  --weights runs/detect/<stage_a_run>/weights/best.pt \
  --device 0
```

Avoid `resume=True` when intentionally changing augmentation, image size, batch size, or stage behavior. A plain
`--weights runs/.../last.pt` load starts a new run with the current plan settings. True resume is only for continuing the
same run configuration after interruption.

## Validation Contract

Stage validation is meant to catch source-specific regressions, not just total loss movement. The Stage A validator
records lightweight buckets:

- `stage_a/objects365/*`
- `stage_a/objects365/person/*`
- `stage_a/crowdhuman/person/*`
- `stage_a/wider_face/face/*`
- `stage_a/small/*`

Early Stage A absolute mAP can be low because the run has hundreds of classes, a short warmup, and a capped validation
subset. A healthy Stage A should still show:

- train box/cls losses trending down
- CrowdHuman person AP improving
- WIDER FACE face AP improving
- Objects365 mAP improving, even if slowly
- no large detector regression after introducing a new source or augmentation setting

For B-F, add or inspect task-specific metrics at stage boundaries:

| Stage | Required signal |
| --- | --- |
| B pose2d | pose AP/PCK or normalized xy error on COCO/OCHuman-style sources |
| C pose25d | normalized z error and 3D consistency on 3DPW/AGORA |
| D person mask | person mask AP/Dice on COCO person mask and OCHuman |
| E scene seg | ADE20K mIoU and detection AP retention |
| F finetune | all task metrics with ADE20K weight kept small |

## Known Work Items

These are deliberate engineering items, not model-contract changes:

- Cache person-positive assignment for pose and mask losses in B-F so the same matching result is reused across task
  heads.
- Expand validation dashboards so each stage reports its own primary task metrics by source.
- Consider a strict "resume with plan overrides" path if interrupted runs need to preserve optimizer state while changing
  augmentation controls.
