# YOLO26s-PS-2.5D 数据集选型说明

## 1. 当前目标

模型目标：在尽量保持 YOLO26 结构的前提下，训练一个多任务模型：

| 模块 | 输出 | 训练目标 |
|---|---|---|
| Detection Head | `Objects365 objects + person + face` | 通用目标、人体、人脸检测 |
| Body 2.5D Pose Head | `person → 17 × [x, y, z_rel, conf]` | 人体 2D 关键点 + root-relative depth |
| Person Instance Mask Head | `person → mask coeff + proto` | 人体实例分割 |
| Scene Semantic Seg Head | `ADE20K semantic logits` | 场景语义分割 |

最终采用的数据集组合：

```yaml
detection:
  - Objects365
  - CrowdHuman
  - WIDER_FACE

pose2d:
  - COCO_WholeBody
  - OCHuman
  - 3DPW_projected_2d
  - AGORA_projected_2d
  - H3WB_projected_2d_optional

pose3d:
  - 3DPW
  - AGORA
  - H3WB_optional

person_mask:
  - COCO_person_mask
  - OCHuman
  - AGORA_optional

scene_seg:
  - ADE20K
```

---

## 2. 主数据集选型表

| 数据集 | 对应任务 | 使用方式 | 为什么选它 | 解决的问题 | 主要注意点 |
|---|---|---|---|---|---|
| **Objects365** | Detection | 主检测数据；`Objects365 Person → person`；额外加 `face` 类 | 类别数比 COCO 多，覆盖更现代/更丰富的日常物体；相比 LVIS 更适合做普通 closed-set YOLO 检测训练 | 替代 LVIS，降低 federated/partial-negative 训练复杂度；提升通用物体检测覆盖 | 类别仍需 remap；如果存在 face-like 类，第一版建议 ignore，由 WIDER FACE 统一监督 face |
| **CrowdHuman** | Detection / person | `full body → person`；`visible body → optional ignore/person`；`head → ignore` | 专门面向拥挤人体检测，包含大量遮挡和密集人群场景 | 稳定 person detector，补 COCO/Objects365 在拥挤遮挡人体上的不足 | 只把 person 当有效类别；其他 Objects365 类和 face 不应当作负样本 |
| **WIDER FACE** | Detection / face | `face bbox → face` | 人脸检测经典数据，覆盖尺度、姿态、遮挡变化 | 稳定小目标 face 检测，补 Objects365/CrowdHuman 的人脸监督不足 | 只把 face 当有效类别；person 和 Objects365 类不要当负样本 |
| **COCO-WholeBody** | Body 2D keypoints | 取 COCO-17 body keypoints；训练 `[x, y, conf]`，不训练 `z` | 和 COCO 2017 split 对齐，提供 whole-body 关键点，body 17 点可直接作为主 2D pose 监督 | 稳定 2D body keypoints；和 COCO 风格评估/预处理兼容 | 只监督 2D，不提供真实 3D depth；z 分支必须 mask 掉 |
| **3DPW** | Body 2.5D / 3D pose | 训练 `[x, y, z_rel, conf]`；也可投影成 2D 监督 | in-the-wild 3D human pose 数据，包含移动相机、真实场景、SMPL/3D pose 信息 | 提高真实场景 2.5D pose 泛化；作为 3D pose 评估集也有价值 | 数据量不大，不适合单独作为 3D 主数据；建议中等比例混入 |
| **AGORA** | Body 2.5D / 3D pose / optional mask | 训练 `[x, y, z_rel, conf]`；可补人体框、遮挡、多人体 | 高真实感合成数据，含多人、遮挡、自然衣着、SMPL-X 标注 | 补 3DPW 数据量不足；增强多人、遮挡、衣着变化下的 2.5D pose | synthetic/render bias 明显，不宜占比过高；需要和真实数据混训 |
| **H3WB optional** | Body 2.5D / 3D pose | 可选 3D pose 补充；映射到 COCO-17 body | 使用 COCO-WholeBody layout，含 133 whole-body keypoints，其中 body 17 与当前 schema 对齐 | 解决 2D COCO-WholeBody 与 3D 数据 skeleton 不一致问题 | 基于 Human3.6M 扩展，场景偏受控；建议 optional，不作为唯一 3D 数据 |
| **COCO person mask** | Person instance mask | 只取 person instance mask | COCO 的 person instance mask 是人体分割最稳的主数据之一 | 训练 person mask proto/coeff 主力数据 | 只用于 person instance mask；不要用 COCO 其他类扩展当前 mask head |
| **OCHuman** | Person mask / occluded pose | 训练遮挡人体 mask；可补 2D pose | 面向 heavily occluded human，带 bbox、pose、instance mask | 补重遮挡人体分割和遮挡下 pose 稳定性 | 不是所有实例同时具备 keypoint+mask；训练时必须按 flag 做 partial-label |
| **ADE20K** | Scene semantic segmentation | 训练 150 类 semantic logits | 场景解析标准数据，覆盖 indoor/outdoor、stuff/object 类别 | 给机器人环境理解补充 wall/floor/table/chair/road/sky 等场景语义 | 只训练 scene seg，不训练 det/pose/mask；占比过高可能拉低 detection/pose |

---

## 3. 为什么不用或不作为主数据

| 数据集 | 当前决策 | 原因 | 备注 |
|---|---|---|---|
| **LVIS** | 替换为 Objects365 | LVIS 有 federated / partial annotation 机制，不能把未标注类别直接当负样本；YOLO 普通闭集训练需要额外 `valid_classes_mask` 和 category-level ignore | 如后续需要长尾类别，可低比例加入，但必须实现 LVIS pos/neg/not-exhaustive 逻辑 |
| **MPI-INF-3DHP** | 不使用 | 当前约束明确不要用；同时结构上可以用 3DPW + AGORA + optional H3WB 替代 3D 监督 | 3D 泛化压力主要由 3DPW/AGORA 承担 |
| **Open Images** | 不作为主检测集 | 有 image-level positive/negative verification 机制，不能简单当普通 fully-labeled closed-set 检测数据 | 如果使用，同样需要 verified label mask，不比 LVIS 简单太多 |
| **V3Det** | 第一版不用全量 | 类别极多，直接接 YOLO26s 检测头会导致分类头过重、类别长尾复杂、训练不稳定 | 后续可筛 300~1000 个业务类做子集 |
| **Visual Genome** | 不作为 closed-set detection 主数据 | 类别/短语粒度噪声大，同义词多，标注风格更偏 region graph / relationship | 更适合 open-vocabulary 或文本区域预训练 |
| **COCO detection 80 类** | 不作为 detection 主力 | 类别数太少，现代物品覆盖不足；但 COCO person mask 和 COCO-WholeBody 仍然保留 | COCO 的价值主要在 person mask / keypoint，而不是当前通用检测主类表 |

---

## 4. 各任务的最终数据职责

| 任务 | 主数据 | 辅助数据 | 不参与该任务的数据 |
|---|---|---|---|
| 通用目标检测 | Objects365 | CrowdHuman、WIDER FACE | ADE20K 不参与 det；pose/mask 数据只提供 person bbox |
| person 检测 | CrowdHuman、Objects365 Person | COCO-WholeBody、COCO person mask、OCHuman、3DPW、AGORA | WIDER FACE 不作为 person 负样本 |
| face 检测 | WIDER FACE | Objects365 中 face-like 类第一版建议 ignore | CrowdHuman / COCO / 3DPW / AGORA 不作为 face 负样本 |
| 2D body keypoints | COCO-WholeBody | OCHuman、3DPW projected 2D、AGORA projected 2D、H3WB optional | Objects365、CrowdHuman、WIDER FACE、ADE20K |
| 2.5D body pose | 3DPW、AGORA | H3WB optional | COCO-WholeBody / OCHuman 只训 2D，不训 z |
| person instance mask | COCO person mask | OCHuman、AGORA optional | ADE20K person semantic 不作为 instance mask |
| scene semantic segmentation | ADE20K | 无 | 其他数据不训练 scene seg |

---

## 5. partial-label 策略

即使去掉 LVIS，当前混合数据仍然必须做 partial-label。原因是每个数据集只标注自己负责的对象类别，不能把未标注类别都当负样本。

| 来源数据 | 有效检测类别 | 其他检测类别处理 |
|---|---|---|
| Objects365 | Objects365 remapped classes，包括 `person` | `face` ignore |
| CrowdHuman | `person` | Objects365 其他类、face ignore |
| WIDER FACE | `face` | person、Objects365 类 ignore |
| COCO-WholeBody | `person` | Objects365 其他类、face ignore |
| COCO person mask | `person` | Objects365 其他类、face ignore |
| OCHuman | `person` | Objects365 其他类、face ignore |
| 3DPW | `person` | Objects365 其他类、face ignore |
| AGORA | `person` | Objects365 其他类、face ignore |
| ADE20K | 无 detection | detection loss 全部 ignore |

推荐每张图带：

```python
valid_classes_mask: [C_det]
```

分类 loss 只在 `valid_classes_mask == 1` 的类别上计算。

---

## 6. 推荐阶段采样配方

### Stage A: Detection stable

```yaml
stage_A_detection_stable:
  epochs: 15
  samples_per_epoch: 40000
  data:
    Objects365: 45
    CrowdHuman: 35
    WIDER_FACE: 20
```

### Stage B: 2D Body Pose

```yaml
stage_B_pose2d:
  epochs: 60
  samples_per_epoch: 35000
  data:
    COCO_WholeBody: 60
    OCHuman: 10
    CrowdHuman: 10
    Objects365: 10
    WIDER_FACE: 10
```

### Stage C: 3D / 2.5D Pose

```yaml
stage_C_pose25d:
  epochs: 70
  samples_per_epoch: 35000
  data:
    COCO_WholeBody: 35
    3DPW: 25
    AGORA: 28
    H3WB_optional: 6
    OCHuman: 4
    CrowdHuman: 1
    WIDER_FACE: 1
```

### Stage D: Person Mask

```yaml
stage_D_person_mask:
  epochs: 60
  samples_per_epoch: 35000
  data:
    COCO_person_mask: 40
    OCHuman: 20
    COCO_WholeBody: 14
    3DPW: 7
    AGORA: 10
    H3WB_optional: 3
    Objects365: 4
    CrowdHuman: 2
```

### Stage E: Scene Segmentation

```yaml
stage_E_scene_seg:
  epochs: 50
  samples_per_epoch: 30000
  data:
    ADE20K: 10
    COCO_WholeBody: 22
    COCO_person_mask: 15
    OCHuman: 8
    3DPW: 10
    AGORA: 10
    H3WB_optional: 3
    Objects365: 12
    CrowdHuman: 6
    WIDER_FACE: 4
```

### Stage F: Full Finetune

```yaml
stage_F_full_finetune:
  epochs: 40
  samples_per_epoch: 30000
  data:
    Objects365: 12
    CrowdHuman: 9
    WIDER_FACE: 8
    COCO_WholeBody: 23
    3DPW: 14
    AGORA: 14
    H3WB_optional: 5
    COCO_person_mask: 8
    OCHuman: 5
    ADE20K: 2
```

---

## 7. 总结

这套数据集的核心逻辑是：

| 目标 | 选型逻辑 |
|---|---|
| 通用检测稳定 | 用 Objects365 替代 LVIS，降低 federated annotation 复杂度 |
| person 稳定 | CrowdHuman 负责拥挤、遮挡、密集人体 |
| face 稳定 | WIDER FACE 负责小脸、遮挡、姿态变化 |
| 2D pose 稳定 | COCO-WholeBody 作为 body keypoint 主数据 |
| 2.5D pose 可训 | 3DPW + AGORA 提供真实/合成互补的 3D 监督 |
| skeleton 对齐 | H3WB optional，用于 COCO-WholeBody layout 的 3D 补充 |
| mask 稳定 | COCO person mask 为主，OCHuman 补重遮挡 |
| scene seg 独立 | ADE20K 只训练语义分割，不干扰 instance mask |

最终选择：

```yaml
keep:
  - Objects365
  - CrowdHuman
  - WIDER_FACE
  - COCO_WholeBody
  - 3DPW
  - AGORA
  - H3WB_optional
  - COCO_person_mask
  - OCHuman
  - ADE20K

remove:
  - LVIS
  - MPI-INF-3DHP
```

---

## 8. 参考来源

| 数据集 | 来源 |
|---|---|
| Objects365 | Objects365 official / paper |
| CrowdHuman | CrowdHuman official |
| WIDER FACE | WIDER FACE official / TensorFlow Datasets |
| COCO-WholeBody | COCO-WholeBody GitHub / ECCV paper |
| COCO | COCO official / Ultralytics COCO docs |
| OCHuman | OCHuman API repository |
| 3DPW | 3DPW official |
| AGORA | AGORA official / CVPR paper material |
| H3WB | H3WB official repository / ICCV paper |
| ADE20K | ADE20K official / MIT Scene Parsing Benchmark |
