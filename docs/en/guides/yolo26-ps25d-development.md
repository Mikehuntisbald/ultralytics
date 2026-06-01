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
| `scene_seg` | `[B, 150, padded_H/4, padded_W/4]` | ADEChallengeData2016 150-class semantic logits on padded input. |

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

Scene segmentation uses the MIT Scene Parsing Benchmark `ADEChallengeData2016` layout, not the full ADE20K object
vocabulary. Its masks store `0` as ignore/background and `1..150` as source class IDs, so the unified loader maps them to
train IDs `0..149` with ignore `255`.

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

## Dataset Preparation Debug Ledger

This section records the data-preparation reasoning as observable engineering decisions: symptoms, causes, fixes, and
verification steps. It intentionally avoids hidden mental notes; keep future entries reproducible from files, commands,
and logs.

Canonical roots:

```bash
repo=/home/haoyi/Downloads/ultralytics
vision=/home/haoyi/Downloads/datasets/vision_benchmarks
human=/home/haoyi/Downloads/datasets/human_benchmarks
stage_a=$vision/YOLO26PS_STAGE_A
stage_multi=$vision/YOLO26PS_STAGE_MULTI
```

Stage data dependencies:

| Stage | Prepared from | Important output files |
| --- | --- | --- |
| A detection | Objects365, CrowdHuman, WIDER FACE | `$stage_a/train.txt`, `$stage_a/val.txt`, `$stage_a/yolo26ps_stage_a.yaml` |
| B pose2d | Stage A detection lists plus COCO-WholeBody records | `stage_b_train.txt`, `stage_b_val.txt`, `manifests/stage_b_*.jsonl` |
| C pose25d | Stage B manifests plus 3DPW and AGORA | `stage_c_train.txt`, `stage_c_val.txt`, `manifests/stage_c_*.jsonl` |
| D person mask | Stage C manifests plus COCO person masks, OCHuman, detection guard lists | `stage_d_train.txt`, `stage_d_val.txt`, `manifests/stage_d_*.jsonl` |
| E scene seg | Stage D manifests plus ADEChallengeData2016 scene records | `stage_e_train.txt`, `stage_e_val.txt`, `manifests/stage_e_*.jsonl` |
| F finetune | Same prepared multi-stage pool, with final sampling weights | `stage_f_train.txt`, `stage_f_val.txt` when prepared |

Core invariants discovered during preparation:

- List files are source inventories, not pre-weighted downsampled lists. Sampling ratios live in the stage plan through
  `sampling_weights` and `samples_per_epoch`.
- Unified records must keep `has_det`, `has_pose2d`, `has_pose3d`, `has_person_mask`, and `has_scene_seg` independent.
- ADEChallengeData2016 records are scene-only: `has_det=false` and `has_scene_seg=true`.
- Partial detection sources must carry `det_class_mask`; they must not create Objects365 negative labels.
- Multi-supervision records need `sampling_sources`, not just one `source`, because a merged COCO record may carry both
  pose and mask supervision.
- After changing schema, source inference, scene mapping, or mask representation, bump `UNIFIED_CACHE_VERSION` and
  delete stale `*.cache` files for the affected stage.

Current local summary checks:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/home/haoyi/Downloads/datasets/vision_benchmarks/YOLO26PS_STAGE_MULTI")
for name in ("stage_b_summary.json", "stage_c_summary.json", "stage_d_summary.json", "stage_e_summary.json"):
    p = root / name
    print("\n##", name, p.exists())
    if p.exists():
        d = json.loads(p.read_text())
        for k in ("train_total", "val_total", "manifest_train", "manifest_val", "train_sources", "val_sources", "note"):
            if k in d:
                print(k, d[k])
PY
```

At the last verification, Stage E contained `165220` training images and `20383` validation images. ADEChallengeData2016
contributed `20210` train and `2000` validation images; 3DPW and AGORA were also present in the merged source counts.

ADEChallengeData2016 verification:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np

root = Path("/home/haoyi/Downloads/datasets/vision_benchmarks/ADEChallengeData2016")
print("train jpg", len(list((root / "images/training").glob("*.jpg"))))
print("val jpg", len(list((root / "images/validation").glob("*.jpg"))))
for rel in ("annotations/training/ADE_train_00000001.png", "annotations/validation/ADE_val_00000001.png"):
    arr = np.array(Image.open(root / rel))
    print(rel, arr.dtype, arr.shape, int(arr.min()), int(arr.max()))
PY
```

Expected result: masks are `uint8` with values in `0..150`. The data YAML maps source `0` to ignore `255` and source
`1..150` to train IDs `0..149`.

Stage E smoke requirements before a long run:

- `stage_e_val.txt` exists and includes ADE validation images.
- `ultralytics/cfg/datasets/yolo26ps_stage_e_scene_seg.yaml` has `scene_nc: 150` and
  `scene_label_mapping: ade20k_150`.
- ADE samples in the unified cache have `has_det=false`, `has_scene_seg=true`, and `scene_seg` pointing at
  `ADEChallengeData2016/annotations/...`.
- A mixed validation batch has finite `scene_seg_loss`; if not, inspect dtype and invalid labels before changing data.
- Live training columns `box_loss`, `cls_loss`, and `dfl_loss` are the detection loss. A tiny `dfl_loss` is expected for
  the no-DFL/reg-max-1 design and is not the same as detector loss being zero.

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

## Detection Local-Best Escape Plan

The current Stage D recovery basin is useful but too narrow for the `Objects365 mAP50 > 0.30` target. The best stable
Stage D detection-recovery checkpoint reached about:

- overall `mAP50=0.1559`
- Objects365 `mAP50=0.1682`
- small-object `mAP50=0.0465`
- Objects365 person `mAP50=0.6858`
- CrowdHuman person `mAP50=0.7632`
- WIDER face `mAP50=0.7547`

Several high-resolution continuations around `lr0=1e-4..3e-4` either stayed flat or raised train loss without improving
Objects365. Treat that as a local best for the Stage D checkpoint, not as a reason to keep lowering LR indefinitely.

The escape route is a detection-mainline bootstrap from the official YOLO26s detector:

1. Build the PS-2.5D architecture from YAML and partial-load the official detector with `--pretrain`.
2. Freeze the transferred backbone for a short bridge while the random PS-2.5D P2-P5 neck and 366-class det heads learn.
3. Unfreeze the full detection path and train mostly on Objects365.
4. Use a final rare/small pass only after Objects365 mAP is clearly rising.

Run order:

```bash
python tools/train_yolo26ps_stage.py \
  --stage A_det_escape_bridge \
  --model ultralytics/cfg/models/26/yolo26s-ps25d.yaml \
  --pretrain pretrains/yolo26s-det.pt \
  --device 0 \
  --name yolo26ps_a_det_escape_bridge

python tools/train_yolo26ps_stage.py \
  --stage A_det_escape_main \
  --weights runs/detect/yolo26ps_a_det_escape_bridge/weights/best.pt \
  --device 0 \
  --name yolo26ps_a_det_escape_main

python tools/train_yolo26ps_stage.py \
  --stage A_det_escape_rare_small \
  --weights runs/detect/yolo26ps_a_det_escape_main/weights/best.pt \
  --device 0 \
  --name yolo26ps_a_det_escape_rare_small
```

Gate the run by real trends, not one noisy validation:

- continue bridge if train box/cls loss drops and person/face anchors remain healthy
- promote main if Objects365 mAP rises for at least 3 validations
- start rare/small only after general Objects365 mAP is rising
- reject a run if train cls loss rises for 2 epochs and Objects365 mAP is flat or down
- use `val_samples>=8000` for decisions near the target; `val_samples=2000` is fast but noisy

This path intentionally does not preserve a Stage D mask/pose optimum. Once detection is strong enough, either continue
through B/C/D again from the new detector or merge the strong detection path back into a multitask checkpoint and run a
short pose/mask repair stage.

## Detection Recovery After Stage D

`D_det_recover_objects365` is a narrow recovery stage for the case where Stage D protects or improves mask quality but
Objects365 detection quality needs to come back up. It is meant to start from a Stage D PS-2.5D checkpoint and update
only the detection head:

```bash
python tools/train_yolo26ps_stage.py \
  --stage D_det_recover_objects365 \
  --weights runs/detect/<stage_d_run>/weights/best.pt \
  --device 0 \
  --name yolo26ps_stage_d_det_recover_objects365
```

The stage uses the Stage A detection dataset YAML, with weighted replacement sampling biased toward Objects365:

```yaml
sampling_weights: {objects365: 80, crowdhuman: 15, wider_face: 5}
active_tasks: [det]
train: {backbone: false, neck: false, det_head: true, body25d_head: false, mask_head: false, scene_seg_head: false}
loss: {det: 1.0, pose2d: 0.0, pose_z: 0.0, pose_vis: 0.0, person_mask: 0.0, scene_seg: 0.0}
```

This intentionally freezes backbone, neck, pose, mask, and scene heads. The goal is to improve the Objects365 classifier
and detector regressors without moving the features that the person pose and mask branches depend on. Use a conservative
LR and watch `Objects365/person`, CrowdHuman person, WIDER face, and small-object buckets; if person detection drops, cut
the LR or increase the CrowdHuman share before continuing.

`D_det_recover_objects365_prodigy_unfreeze_fast` is the aggressive variant used when the goal is to move Objects365 fast
and temporary pose/mask preservation is not the priority. It starts from the current Stage D detector-recovery checkpoint,
activates only `det`, but unfreezes backbone, neck, and the detection head:

```yaml
sampling_weights: {objects365: 92, crowdhuman: 2, wider_face: 6}
imgsz: [576, 768]
batch: 18
accumulate: 4
optimizer: Prodigy
lr0: 1.0
lrf: 1.0
prodigy_d0: 0.000001
prodigy_d_coef: 1.0
prodigy_growth_rate: 1.02
prodigy_slice_p: 11
prodigy_decouple: true
prodigy_use_bias_correction: false
prodigy_safeguard_warmup: true
mosaic: 0.1
tal_topk: {one2many: 7, one2one: 5, topk2_one2one: 1}
tal_high_gt_threshold: 900
tal_metric_chunk_gt: 256
det_class_mask_normalization: none
```

The first 400k-sample epoch of the `576x768_mosaic01_talfix_b18` run improved the paused baseline:

| Metric | New epoch 1 | Paused baseline | Delta |
| --- | ---: | ---: | ---: |
| overall mAP50 | 0.23050 | 0.22108 | +0.00942 |
| overall mAP50-95 | 0.15695 | 0.14936 | +0.00759 |
| Objects365 mAP50 | 0.24652 | 0.23529 | +0.01123 |
| Objects365 mAP50-95 | 0.16726 | 0.15838 | +0.00888 |
| small mAP50 | 0.05084 | 0.04286 | +0.00798 |
| CrowdHuman person mAP50 | 0.75273 | 0.76802 | -0.01529 |
| WIDER face mAP50 | 0.68932 | 0.65123 | +0.03809 |

Guardrail: this profile intentionally uses nearly all available 32 GB GPU memory. After the TAL allocation fix, high-GT
batches with about 2000 instances stayed on GPU, but `batch=18` is already near the safe limit. Do not raise batch or
resolution unless a full epoch finishes without repeated OOM fallback.

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
| E scene seg | ADEChallengeData2016 150-class mIoU and detection AP retention |
| F finetune | all task metrics with ADEChallengeData2016 weight kept small |

## Training Issue Log

This section is the debugging ledger for the current YOLO26s-PS-2.5D branch. Keep new failures here in the same shape:
symptom, cause, fix, and the guardrail that prevents the same failure from coming back.

### 1. Development Workspace Drift

Symptom: changes were sometimes made in `/home/haoyi/Downloads/ultralytics-gitee` while the real development branch was
`/home/haoyi/Downloads/ultralytics`.

Cause: both repositories have the same project layout, so shell commands and IDE tabs can look valid in either tree.

Fix: all implementation, training, validation, and documentation work for this model should happen in:

```bash
/home/haoyi/Downloads/ultralytics
```

Guardrail: before editing, committing, or launching a long run, check `pwd` and `git status --short`. Only sync to other
remotes or mirrors after the real branch is clean.

### 2. Dynamic Input And Stride Padding

Symptom: early design notes mixed fixed logical sizes, dynamic resolution, and padded dense outputs.

Cause: detection can decode dynamically from feature-map shapes, but mask prototypes and semantic logits are dense maps
and therefore inherit the padded tensor shape.

Fix: remove the separate `logical_size` concept. The contract is:

- `dynamic_input: true`
- `pad_stride: 32`, because the model uses P5
- default deployment/training size `imgsz=[448,768]`
- detection decode derives anchors and strides from current feature maps
- mask and scene outputs are `[padded_H/4, padded_W/4]`
- postprocess inverse-letterboxes boxes, keypoints, masks, and scene maps back to the original image

Guardrail: never hard-code `768x432`, `448x768`, or a fixed feature-map size inside decode or postprocess. Only defaults
and stage recipes may mention a fixed size.

### 3. Dense Stride-4 Outputs Versus Original Detection

Symptom: `mask_proto` and `scene_seg` output shapes looked different from original YOLO26 detection outputs.

Cause: original detection does not expose a dense image-aligned output. Person masks and scene segmentation need dense
stride-4 maps, so they naturally scale with padded input size.

Fix: keep `mask_proto=[B,32,padded_H/4,padded_W/4]` and `scene_seg=[B,150,padded_H/4,padded_W/4]`. This is expected and
is not a regression from the original detector.

Guardrail: compare detection AP using decoded boxes, and compare mask/scene quality after inverse letterbox. Do not
compare dense map shapes to original detection head outputs.

### 4. Stage A Auxiliary Branch Waste And BN Pollution

Symptom: Stage A spent memory and time computing pose, mask, P2 refine, and scene branches even though only detection was
being trained.

Cause: freezing `requires_grad=False` alone does not prevent forward compute, and `trainer.model.train()` can re-enable
BatchNorm updates on frozen branches.

Fix: set `head.active_tasks={"det"}` for Stage A. The head skips `p2_refine`, `proto`, `scene_seg`, `cv4`, and `cv5`
when the corresponding task is inactive. The stage trainer also forces inactive modules to `eval()` after every
`model.train()` call.

Guardrail: Stage A must show active tasks as detection only. Frozen auxiliary branches should have no train-time BN stat
updates and no forward memory cost.

### 5. Weighted Downsampled `train.txt`

Symptom: early Stage A preparation generated a weighted/downsampled training list, which permanently discarded many
Objects365 images before the sampler ever ran.

Cause: sampling ratio was encoded into the file list instead of the sampler.

Fix: `train.txt` and `train_all.txt` are full source lists. Every stage uses weighted random sampling with replacement:

```python
dataset_name = sample_by_weight(stage_cfg["sampling_weights"])
sample = dataset[dataset_name].random_sample()
```

Guardrail: stage YAML controls `epochs`, `samples_per_epoch`, and `sampling_weights`. The list files should stay full
source inventories unless a file name explicitly says it is legacy or diagnostic.

### 6. Rectangular Training Was Being Treated Like Square Training

Symptom: using square `704` or `768` made training heavier than the intended deployment aspect ratio, and validation or
multi-scale paths could accidentally squash `[h,w]` into a square.

Cause: the original training path assumes a scalar `imgsz` in many places.

Fix: the stage trainer, dataloader, validator, and LetterBox path accept `[height,width]`. Stage defaults use
`[448,768]`, and `[480,768]` is the preferred exact 16:10 probe if the stable recipe has memory headroom.

Guardrail: do not enable multi-scale while debugging rectangular training. Keep `pad_stride=32` and use `val_batch` to
control validation memory separately from train batch.

### 7. LVIS To Objects365 Migration

Symptom: the original plan used LVIS, then the training source changed to Objects365.

Cause: local detection data availability and the target category base changed.

Fix: the plan now uses:

```yaml
det_classes:
  base: Objects365_remapped
  extra: [face]
```

`Objects365 Person` maps to the shared `person` class. `C_det = len(remapped_Objects365_classes) + 1`, where `+1` is
`face`.

Guardrail: do not leave `LVIS`, `remapped_LVIS`, or MPI-INF-3DHP/Human3.6M/H3WB references in active stage recipes.
Only add optional datasets after the converter, labels, cache, and stage weights exist.

### 8. Partial Detection Labels As Full-Class Negatives

Symptom: person-only and face-only datasets could silently train all unannotated Objects365 classes as background.

Cause: standard detection BCE assumes every class not annotated in the image is a negative class. That is valid for
complete detection sources, but false for CrowdHuman, WIDER FACE, COCO-WholeBody, 3DPW, and AGORA.

Fix: carry `det_class_mask` in each label and batch. Objects365 supervises the full base class set. Person-only sources
supervise only `person`. WIDER FACE supervises only `face`.

Guardrail: `det_class_mask` must survive caching, collation, mosaic, and loss. If a source is partial-label, it must not
produce full Objects365 negative BCE.

### 9. Single-Class Partial Images Still Suppressed Person Confidence

Symptom: C_det_reanchor looked numerically stable but person AP collapsed after longer runs. The bad run had
`mAP50=0.01425`, `Objects365/person=0.00240`, `COCO/person=0.00129`, `3DPW/person=0.00001`, and
`AGORA/person=0.00044`.

Cause: class masks prevented unannotated Objects365 classes from becoming negatives, but single-class partial images
still treated unmatched anchors as negative for the supervised class itself. In person-only images, an unmatched anchor
is not proof that no person exists there when the dataset has partial boxes or detector/pose matching is imperfect.

Fix: add `det_partial_cls_positive_only`. For images with only one supervised detection class, class BCE is applied to
assigned positives only. Complete Objects365 images still train full background negatives normally.

Guardrail: C_det_reanchor keeps:

```yaml
det_class_mask_normalization: sqrt
det_partial_cls_positive_only: true
loss.det: 0.04
samples_per_epoch: 1024
```

The accepted short reanchor run is
`runs/detect/yolo26ps_stage_c_det_reanchor_short_official/weights/best.pt`:

| Run | mAP50 | Objects365/person | Small | COCO/person | 3DPW/person | AGORA/person | `val/pose_z_loss` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline no update | 0.03133 | 0.50864 | 0.01261 | 0.56541 | 0.27409 | 0.76238 | 0.01113 |
| fixed short reanchor | 0.03039 | 0.50741 | 0.01272 | 0.55943 | 0.26483 | 0.76312 | 0.01101 |

Do not promote the longer `partial_pos_official` run without another stability pass; it was better than the failed run
but still pulled person AP down.

### 10. Stage B All-Zero Detection Validation

Symptom: some Stage B runs produced all-zero or near-zero detection AP even though pose losses were moving.

Cause: Stage B was allowing detection updates while most samples were partial person labels. This amplified the partial
negative-label problem and made detector confidence collapse. A separate same-checkpoint validation proved the validator
itself was not the reason for all zeros.

Fix: Stage B is pose-first:

- `det_head: false`
- `loss.det: 0.0`
- backbone and neck frozen for the stable pose head warmup
- run small smoke tests from the Stage A checkpoint before official Stage B

Guardrail: if Stage B detection AP suddenly goes to zero, stop and run a no-update validation from the input checkpoint.
If no-update AP is healthy, the training recipe is damaging the detector; do not continue that run.

### 11. Stage C Depth Loss And NaN Diagnosis

Symptom: Stage C debugging showed NaN-looking `pose_z` values in some logs and unstable 2.5D behavior.

Cause: two different cases were mixed together:

- Stage B has no 3D supervision, so `val/pose_z_loss=nan` can mean "no valid 3D samples in this validation subset", not
  a training NaN.
- Stage C depth loss must compare raw `z_rel` targets to raw z predictions. Only bone consistency should use normalized
  z.

Fix: `pose_z_loss = SmoothL1(pred_z_raw, gt_z_rel_raw)`. Bone loss builds pseudo-3D coordinates with normalized
`x`, `y`, and `z`. COCO-WholeBody and OCHuman train `x/y/conf` only; 3DPW and AGORA train `x/y/z/conf`.

Guardrail: when checking NaNs, inspect both tensor finiteness and source flags. A missing task subset is not the same as
an invalid loss. Keep `pose_z` gated by `has_pose3d`.

### 12. AGORA Data Availability

Symptom: Stage C needed AGORA for 2.5D stability, but not every AGORA archive was initially extracted.

Cause: Stage C depends on the prepared unified manifests and image paths; missing AGORA files reduce or break 3D
coverage.

Fix: extract only the AGORA splits referenced by the stage converter and verify the generated `stage_c_*` manifests
resolve to existing images and labels.

Guardrail: before Stage C official training, sample the manifest and check source counts for `3dpw` and `agora`.
Human3.6M/H3WB is not part of the current active recipe.

### 13. ADE20K Class Count Confusion

Symptom: scene segmentation was described as ADE20K, which can be confused with the full ADE20K object vocabulary.

Cause: the intended training set is the MIT Scene Parsing Benchmark layout, not every ADE20K object annotation.

Fix: use `ADEChallengeData2016` with 150 semantic classes. Source masks use `0` as ignore/background and `1..150` as
class IDs; the loader remaps to train IDs `0..149` and ignore `255`.

Guardrail: scene head stays `classes: 150`. Do not train scene segmentation with a 3000-class ADE20K vocabulary.

### 14. Stage E Manifest And Validation List Missing

Symptom: the first Stage E launch failed before training because `stage_e_val.txt` did not exist.

Cause: Stage E had a plan entry and data YAML, but no converter had materialized the Stage E list files and manifests
from Stage D plus ADEChallengeData2016.

Fix: add `tools/prepare_yolo26ps_stage_e.py`. It reads Stage D train/val lists and manifests, appends ADE scene-only
records, and writes:

```text
stage_e_train.txt
stage_e_val.txt
manifests/stage_e_train.jsonl
manifests/stage_e_val.jsonl
stage_e_summary.json
```

Guardrail: before starting any stage, check that the stage YAML's `train`, `val`, and `unified_manifest` targets all
exist. Do not rely on the generic trainer to discover missing stage preparation late.

### 15. Scene Segmentation Labels Collated As Mixed Types

Symptom: after preparing Stage E, dataloader collation failed when a batch mixed ADE images with non-ADE images.

Cause: ADE samples carried a `scene_seg` path/string before formatting, while non-scene samples had no dense scene
tensor. The default collate path tried to stack or carry incompatible values.

Fix:

- `Format.__call__` loads scene masks into a tensor field and clears non-tensor `scene_seg` values.
- `YOLODataset.collate_fn` special-cases `scene_seg`: if any sample has a tensor scene target, non-scene samples are
  filled with a same-shaped `255` ignore tensor.
- `_scene_seg_loss` returns a safe zero if no valid scene pixels exist in the selected image subset.

Guardrail: a mixed Stage E batch should contain `scene_seg` as a tensor of shape `[B,H/4,W/4]` or equivalent resized
target shape, with non-ADE samples entirely `255`.

### 16. ADE Scene Loss Produced `inf` Or `nan` Under AMP

Symptom: Stage E validation initially wrote `val/scene_seg_loss=nan` even though ADE masks existed and model logits were
finite.

Cause: dense semantic cross entropy over half-precision logits could overflow to `inf` during validation. Invalid or
out-of-range labels also needed to be forced to ignore before flattening.

Fix: compute scene CE in float32 and only over valid pixels:

```python
pred = preds["scene_seg"][image_mask].float()
target = target.long()
invalid = (target != 255) & ((target < 0) | (target >= num_classes))
target[invalid] = 255
valid = target.reshape(-1) != 255
loss = F.cross_entropy(flat_pred[valid], flat_target[valid])
```

Guardrail: before a long Stage E run, run one small ADE validation batch and confirm `_scene_seg_loss` is finite. If it
is not finite, inspect logits, target min/max, ignore index, and dtype before changing sampling weights.

### 17. Multi-Supervision Records Were Invisible To Weighted Sampling

Symptom: Stage E sampler counts showed `coco_wholebody=0` even though the plan gave COCO-WholeBody a nonzero sampling
weight.

Cause: Stage D merges COCO-WholeBody pose records with COCO person-mask records by image. A single `source` string was
not enough to represent every supervision domain on a merged record, so the sampler only saw `coco_person_mask`.

Fix: add `sampling_sources` to unified cached labels and teach the weighted replacement sampler to count every listed
domain. A merged COCO record can now contribute to both `coco_wholebody` and `coco_person_mask` sampling buckets.

Guardrail: after rebuilding a unified cache, print source counts from labels and make sure every nonzero stage sampling
weight has available samples. If a source count is unexpectedly zero, inspect `sampling_sources` before altering data
ratios.

### 18. Unified Cache Version Must Move With Data Semantics

Symptom: source counts, task flags, or scene labels could stay wrong after code fixes because old cache files were still
being reused.

Cause: unified cache files outlive converter and loader changes. A matching image-path hash is not sufficient when the
meaning of `source`, `sampling_sources`, `scene_seg`, or `det_class_mask` changes.

Fix: bump `UNIFIED_CACHE_VERSION` whenever unified label semantics change and delete affected stage caches. The current
cache version that includes multi-source sampling is `1.0.8`.

Guardrail: if a training run contradicts the manifest or the plan, inspect cache metadata first. Rebuild cache before
debugging model code.

### 19. OCHuman Split And Archive Assumptions

Symptom: Stage D needed OCHuman masks, but the local OCHuman layout did not provide a conventional train split.

Cause: the available local OCHuman package exposes val/test annotations for this workflow. Treating missing train files
as a converter failure would incorrectly block mask training.

Fix: Stage D uses OCHuman val as training data and OCHuman test as validation data, and records this in
`stage_d_summary.json`:

```text
OCHuman provides val/test annotations only here; val is used as Stage D train and test as Stage D validation.
```

Guardrail: keep this split policy documented and visible in the summary. If a full OCHuman train split is later added,
update the converter and summary together.

### 20. AGORA Extraction Must Match Converter Paths

Symptom: Stage C could not fully use AGORA until the required AGORA archives were extracted.

Cause: `tools/prepare_yolo26ps_stage_c.py` expects AGORA under:

```text
/home/haoyi/Downloads/datasets/human_benchmarks/AGORA/extracted/AGORA
```

with the image files and SMPL dataframe pickle files referenced by the converter. Partial extraction silently lowers the
3D data pool or produces missing-image records.

Fix: extract only the AGORA splits referenced by the Stage C converter, then regenerate Stage C manifests and check
`stage_c_summary.json` source counts. The verified local Stage C train counts include `agora: 14411` and `3dpw: 8832`.

Guardrail: before official Stage C, sample both `3dpw` and `agora` records from the manifest, assert image paths exist,
and assert at least one instance has `has_body3d=true`.

### 21. Detection Loss "0" Can Mean Gating Or Log Misread

Symptom: during Stage E, `det loss 0` was suspected while training output still showed nonzero `box_loss` and
`cls_loss`.

Cause: there are three different cases:

- ADE-only images correctly gate detection loss off because `has_det=false`.
- Mixed Stage E batches still compute detection loss on the images that have detection labels.
- The live training columns name detection as `box_loss`, `cls_loss`, and `dfl_loss`; there is no single `det_loss`
  column. The small `dfl_loss` value is expected because this branch keeps the no-DFL/reg-max-1 design.

Fix: interpret detection training from the first three loss columns, and inspect `has_det` if a whole batch appears to
skip detector supervision.

Guardrail: if `box_loss` and `cls_loss` are exactly zero in a mixed stage where `loss.det>0`, check `has_det` collation,
`sampling_sources`, and whether the current sampled batch is ADE-only before changing the loss.

### 22. Cross-Source Mosaic And Inherited Mixed Augmentation

Symptom: mixed augmentations could combine Objects365, person-only, and face-only samples into one mosaic, causing class
supervision masks to intersect incorrectly or become too narrow. Stages without an `augment` block could also inherit
Ultralytics default mosaic unexpectedly.

Cause: standard mosaic does not know about partial-label supervision domains, and default augmentation inheritance is
unsafe for multi-task partial labels.

Fix: source-aware mixed augmentation samples companions from the same `det_class_mask` supervision domain as the anchor
image. The stage trainer also closes `mosaic`, `mixup`, `copy_paste`, and `cutmix` when a stage does not explicitly set
augmentation.

Guardrail: C/D/E/F keep mixed augmentation at `0.0`. Stage A also currently keeps mosaic off for stability. Re-enable
mixed augmentation only with source-aware sampling and a small smoke run.

### 23. TaskAlignedAssigner CPU Fallback And GPU Utilization

Symptom: larger batches filled memory but reduced speed, with intermittent low GPU utilization.

Cause: P2-P5 at `[448,768]` creates about 28.5k anchor points per image. Dense sources with many GT boxes can make TAL
assignment tensors peak above available GPU memory. Ultralytics catches the OOM and retries assignment on CPU, which
keeps training alive but slows the run.

Fix: prefer a smaller batch that keeps TAL on GPU over a larger batch that falls back to CPU. Stage A settled on
`batch=16`, `accumulate=4`, `workers=20`, and TAL `8/5/1`. The plan exposes:

```yaml
tal_topk_one2many
tal_topk_one2one
tal_topk2_one2one
```

Guardrail: when utilization drops, first search logs for `TaskAlignedAssigner`, `CUDA OutOfMemoryError`, or `using CPU`.
Do not reduce workers unless dataloader starvation is visible. Do not raise resolution while TAL is falling back.

### 23b. High-GT TAL Overlap Resolution Can OOM

Symptom: the aggressive Objects365 recovery profile fit the forward and backward pass, then failed inside
`TaskAlignedAssigner.select_highest_overlaps()` on a crowd-heavy/high-GT batch. The failing allocation was a dense
`[B, max_gt, anchors]` helper tensor of roughly 1 GB; the fallback path could also fail because it rebuilt the same
temporary tensors on CPU.

Cause: the original overlap resolution allocated full-size helper tensors for two narrow operations:

- resolving anchors assigned to multiple GT boxes with `is_max_overlaps`
- applying the one-to-one/secondary `topk2` mask with `topk_idx`

At `[576,768]`, P2-P5 anchors and large Objects365 instance counts make those helpers dominate peak assignment memory.

Fix: keep the existing assignment semantics but update `mask_pos` in place:

- for multi-GT anchors, iterate per batch over only the conflicting anchor indices and zero/set those entries
- for `topk2`, multiply `align_metric` in place, compute top-k, zero `mask_pos`, and scatter valid top-k entries back

Guardrail: after changing TAL memory code, run a small synthetic assigner smoke test and then one real high-instance
training probe. The fix should remove OOM fallback without changing output shapes.

### 24. E2E Loss Decay Was Not Updating At The Right Time

Symptom: the intended one-to-many decay could be ineffective or misleading if `criterion.update()` was not tied to
optimizer steps.

Cause: YOLO26-PS E2E training has one-to-many and one-to-one branches. The decay should track optimizer updates, not
only epoch boundaries.

Fix: call `criterion.update()` immediately after `optimizer_step()`. Loss logging keeps the weighted task items so
stage changes are visible in `results.csv`.

Guardrail: after changing gradient accumulation or resuming with a different batch, confirm that update-driven schedules
still move per optimizer step.

### 25. Repeated Assignment For Pose And Mask Losses

Symptom: pose and mask losses repeatedly recomputed detector assignment for the same person positives.

Cause: every task loss originally called its own assignment path.

Fix: cache assignment results inside the unified loss by image subset. Pose and person mask losses reuse the detection
assignment when the subset matches, or slice from a cached superset when possible.

Guardrail: any new person-bound task should request assignment through the shared cache, not call TAL directly.

### 26. Unified Label Cache Staleness

Symptom: changing source mapping, manifest paths, or task flags could leave stale cache records that made later runs look
broken.

Cause: unified labels are cached separately from legacy YOLO labels, and old cache files may still match image paths if
the schema version is not bumped.

Fix: unified cache includes a dedicated `UNIFIED_CACHE_VERSION`, manifest-aware hashes, source names, task flags,
`det_class_mask`, 2D/3D keypoints, masks, and scene paths. Bump the unified cache version after schema or source-mask
changes and rebuild caches.

Guardrail: if a training result contradicts expectations, inspect the cache version and source counts before changing
model code.

### 26b. Probe Lists Should Not Reuse Full Train Caches

Symptom: small probe or escape-list runs could reuse a full `labels.cache` built from a different image list, making a
short smoke test look like it had the wrong sample population. Large duplicate-label warning dumps also made training
logs hard to read.

Cause: the legacy YOLO label cache path was based on the label directory, not on the specific list file. Any list under
the same label directory could collide with the full train/val cache.

Fix: when the dataset is created from a single ad-hoc list file whose stem is not `train`, `val`, `test`, or `train_all`,
write the cache beside that list as `<list>.cache`. Cache warning logs are also capped to the first 50 messages with an
omitted-count summary.

Guardrail: full stage train/val lists continue to use the shared directory cache. Probe lists get isolated caches so
they can be deleted or regenerated without touching the main stage cache.

### 27. Validation Was Too Coarse

Symptom: total mAP or total loss hid source-specific regressions. In particular, detector person confidence could drop
while the aggregate mAP looked only slightly worse.

Cause: the validation mix includes many classes and sources; a regression in `person` can be diluted by the rest of the
Objects365 class set.

Fix: stage validation reports source buckets such as Objects365, Objects365/person, small objects, COCO person, 3DPW
person, and AGORA person. Routine training uses `val_samples=2000` for speed; full validation is reserved for stage
boundaries.

Guardrail: monitor at least `mAP50`, `small mAP50`, `Objects365/person`, and the relevant stage source metrics before
accepting a checkpoint.

### 28. Pretrain, Freeze, And Resume Semantics

Symptom: `freeze=10`, `--weights`, and `--pretrain` were easy to mix up.

Cause: official YOLO26 detection weights are not the same architecture as YOLO26s-PS-2.5D, while stage checkpoints are
same-architecture PS-2.5D checkpoints.

Fix:

- use `--pretrain pretrains/yolo26s-det.pt` only when building PS-2.5D from YAML and partially transferring compatible
  weights
- use `--weights runs/.../weights/last.pt` or `best.pt` when chaining PS-2.5D stages
- keep `freeze=0` for Stage A unless transfer is confirmed
- prefer active task gating over freezing as the first way to save memory
- use a new `--weights last.pt` run, not `resume=True`, when changing plan settings such as batch, augmentation, or
  stage losses

Guardrail: the log must show transferred items when using official pretrain. If transfer fails, freezing early layers is
actively harmful.

### 29. Stage C Reanchor Should Preserve 3D

Symptom: after 2.5D became stable, det reanchor was needed, but det-only reanchor could lose the 3D improvements.

Cause: detection and pose share person positives; changing detection confidence without any pose losses can drift away
from the 2.5D objective.

Fix: `C_det_reanchor` keeps both active tasks and small pose weights:

```yaml
active_tasks: [det, pose]
loss: {det: 0.04, pose2d: 0.3, pose_z: 0.1, pose_vis: 0.05, bone: 0.01}
```

Guardrail: accept a reanchor checkpoint only if source person AP is near baseline and `val/pose_z_loss` remains stable.

### 30. Stage D Detection Recovery Should Not Move Pose Or Mask Features

Symptom: after Stage D, Objects365 detection can lag behind the person mask branch, but full finetuning risks moving
backbone/neck features used by pose and mask.

Cause: pose, mask, and detection share the same feature pyramid. Updating the backbone or neck for general object
detection can disturb person-bound heads that are already stable.

Fix: use `D_det_recover_objects365` from the Stage D checkpoint. It uses Stage A detection data, activates only `det`,
and freezes backbone, neck, body25d, mask, and scene branches. Objects365 gets most of the samples, with a small
CrowdHuman/WIDER share to keep person and face detection anchored.

Guardrail: this stage should not change pose or mask weights. Validate detection by source, and only promote the
checkpoint if Objects365 improves without person or face regression.

### 31. Stage D Detection Recovery Can Hit A Local Best

Symptom: after several Stage D recovery runs, Objects365 detection stopped improving around `mAP50=0.16..0.17` while
person and face stayed strong. Higher LR runs caused loss or validation regressions; smaller LR runs mostly produced
flat metrics.

Cause: Stage D is a multitask checkpoint whose shared features were shaped by person pose and mask objectives. Updating
only the detection head preserves those heads but does not give the 366-class Objects365 detector enough freedom. Full
unfreeze from that checkpoint still stays near the same basin because the starting classifier and P2-P5 detection neck
are already specialized.

Fix: stop using Stage D recovery as the path to `Objects365 mAP50 > 0.30`. Use the `A_det_escape_*` stages instead:
bootstrap the PS-2.5D detector from official YOLO26s detection pretrain, train a detection-only mainline, then re-enter
B/C/D or merge the strong detector back into the multitask checkpoint.

Guardrail: if the goal is general Objects365 ability, compare against the escape mainline metrics, not just Stage D
recovery metrics. Keep Stage D recovery for preserving person pose/mask while making small detector repairs.

### 32. Official Pretrain Transfers Only Part Of PS-2.5D

Symptom: Stage A from `pretrains/yolo26s-det.pt` improves person and face quickly, but Objects365 all-class mAP starts
very low and climbs slowly.

Cause: official YOLO26s detection pretrain is an 80-class P3-P5 `Detect` model. YOLO26s-PS-2.5D is a 366-class P2-P5
head with 192-channel neck outputs and auxiliary branches. A direct partial transfer loaded `270/1172` state-dict items
in the current local check, leaving much of the PS-2.5D neck and detection head randomly initialized.

Fix: use a bridge stage before the full detection mainline:

```yaml
A_det_escape_bridge:
  train: {backbone: false, neck: true, det_head: true}
  sampling_weights: {objects365: 95, crowdhuman: 3, wider_face: 2}
```

Then continue with `A_det_escape_main` and `A_det_escape_rare_small`.

Guardrail: `--pretrain` should print a transferred-items log. Do not assume a high transfer count from the official
checkpoint; the bridge stage exists because the compatible subset is intentionally limited.

## Known Work Items

These are deliberate engineering items, not model-contract changes:

- Extend assignment-cache reuse and source metrics as new person-bound heads are added; pose and person mask already use
  the shared cache path.
- Expand validation dashboards so each stage reports its own primary task metrics by source, especially mask Dice/AP and
  scene mIoU.
- Consider a strict "resume with plan overrides" path if interrupted runs need to preserve optimizer state while changing
  augmentation controls.
