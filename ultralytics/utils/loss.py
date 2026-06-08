# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import CITYSCAPES_WEIGHT, OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16, hyp=None):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.hyp = hyp
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_pos = pred_bboxes[fg_mask]
        target_pos = target_bboxes[fg_mask]
        iou = bbox_iou(pred_pos, target_pos, xywh=False, CIoU=True)
        box_error = 1.0 - iou
        nwd_ratio = float(getattr(self.hyp, "det_nwd_ratio", 0.0) or 0.0) if self.hyp is not None else 0.0
        if nwd_ratio > 0:
            nwd = self._nwd_similarity(
                pred_pos,
                target_pos,
                constant=float(getattr(self.hyp, "det_nwd_constant", 12.8) or 12.8),
            )
            box_error = box_error * (1.0 - nwd_ratio) + (1.0 - nwd) * nwd_ratio
        loss_iou = (box_error * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    @staticmethod
    def _nwd_similarity(pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, constant: float = 12.8) -> torch.Tensor:
        """Return Normalized Wasserstein Distance similarity for xyxy boxes."""
        p_xy = (pred_bboxes[..., 0:2] + pred_bboxes[..., 2:4]) * 0.5
        t_xy = (target_bboxes[..., 0:2] + target_bboxes[..., 2:4]) * 0.5
        p_wh = (pred_bboxes[..., 2:4] - pred_bboxes[..., 0:2]).clamp(min=0)
        t_wh = (target_bboxes[..., 2:4] - target_bboxes[..., 0:2]).clamp(min=0)
        wasserstein = (p_xy - t_xy).pow(2).sum(-1, keepdim=True) + ((p_wh - t_wh) * 0.5).pow(2).sum(
            -1, keepdim=True
        )
        return torch.exp(-torch.sqrt(wasserstein.clamp(min=0) + 1e-9) / max(float(constant), 1e-6))


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        tal_high_gt_threshold: int = 0,
        tal_high_gt_topk: int | None = None,
        tal_high_gt_topk2: int | None = None,
        tal_metric_chunk_gt: int = 0,
    ):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
            high_gt_threshold=tal_high_gt_threshold,
            high_gt_topk=tal_high_gt_topk,
            high_gt_topk2=tal_high_gt_topk2,
            metric_chunk_gt=tal_metric_chunk_gt,
        )
        self.bbox_loss = BboxLoss(m.reg_max, h).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp(min=1)
        if str(getattr(self.hyp, "det_area_loss_weight", False)).lower() in {"1", "true", "yes", "on"}:
            area_weight = self._target_area_weight(gt_bboxes, mask_gt, target_gt_idx, fg_mask, pred_scores.dtype)
            if area_weight is not None:
                target_scores = target_scores * area_weight.unsqueeze(-1)
                target_scores_sum = target_scores.sum().clamp(min=1)

        # Cls loss with optional per-image class supervision masks.
        # Partial-label datasets may supervise only person or face. Normalize by
        # the number of supervised classes so full Objects365 images do not
        # dominate mixed-domain reanchor stages with hundreds of extra negatives.
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        focal_gamma = float(getattr(self.hyp, "det_focal_gamma", 0.0) or 0.0)
        if focal_gamma > 0:
            target_scores_f = target_scores.to(dtype)
            pred_prob = pred_scores.sigmoid()
            p_t = target_scores_f * pred_prob + (1.0 - target_scores_f) * (1.0 - pred_prob)
            bce_loss *= (1.0 - p_t).clamp_(0.0, 1.0).pow(focal_gamma)
            focal_alpha = float(getattr(self.hyp, "det_focal_alpha", -1.0))
            if focal_alpha > 0:
                positive = target_scores_f.gt(0).to(dtype)
                bce_loss *= positive * focal_alpha + (1.0 - positive) * (1.0 - focal_alpha)
        det_class_mask = batch.get("det_class_mask")
        det_class_count = None
        if torch.is_tensor(det_class_mask):
            det_class_mask = det_class_mask.to(pred_scores.device).bool()
            if det_class_mask.ndim == 1:
                det_class_mask = det_class_mask.view(1, -1).expand(batch_size, -1)
            if det_class_mask.shape[-1] == pred_scores.shape[-1]:
                bce_loss *= det_class_mask[:, None, :].to(dtype)
                det_class_count = det_class_mask.sum(-1).clamp(min=1).to(dtype)
        if det_class_count is not None:
            mode = str(getattr(self.hyp, "det_class_mask_normalization", "sqrt")).lower()
            if mode not in {"0", "false", "none", "off"}:
                norm = det_class_count if mode in {"linear", "true", "1"} else det_class_count.sqrt()
                bce_loss = bce_loss / norm.view(batch_size, 1, 1)
        partial_pos_only = str(getattr(self.hyp, "det_partial_cls_positive_only", False)).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if partial_pos_only and det_class_count is not None:
            # Person/face-only images do not prove that every unmatched anchor is background for that class. In
            # mixed-domain pose stages this otherwise pushes person confidence down, so keep full background BCE
            # for complete detection sources and use positive cls supervision only for single-class partial labels.
            partial_images = det_class_count <= 1
            if partial_images.any():
                positive_anchors = target_scores.detach().amax(dim=-1).gt(0)
                keep_anchors = (~partial_images[:, None]) | positive_anchors
                bce_loss *= keep_anchors[:, :, None].to(dtype)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)

    def get_assignment(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Return assignment tensors without materializing detection losses or per-class target scores."""
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        target_bboxes, fg_mask, target_gt_idx = self.assigner.assign_bboxes(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        return fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor

    def _target_area_weight(
        self,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return per-anchor positive weights from sqrt(mean_area / area), clamped to [1, max_w]."""
        wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp(min=0)
        areas = (wh[..., 0] * wh[..., 1]).to(dtype)
        valid = mask_gt.squeeze(-1).bool() & torch.isfinite(areas) & areas.gt(0)
        if not bool(valid.any() and fg_mask.any()):
            return None
        mean_area = areas[valid].mean().clamp(min=1.0)
        max_w = max(float(getattr(self.hyp, "det_area_loss_weight_max", 2.0)), 1.0)
        gt_weights = torch.sqrt(mean_area / areas.clamp(min=1.0)).clamp(min=1.0, max=max_w)
        gt_weights = torch.nan_to_num(gt_weights, nan=1.0, posinf=max_w, neginf=1.0)
        assigned = torch.gather(gt_weights, 1, target_gt_idx.clamp(min=0))
        assigned = torch.nan_to_num(assigned, nan=1.0, posinf=max_w, neginf=1.0)
        return torch.where(fg_mask, assigned, torch.ones_like(assigned))

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semseg
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW

                if self.overlap:
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl, semseg)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Vectorized fill: compute within-batch position for each keypoint using cumulative offsets
        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx_long]
        batched_keypoints[batch_idx_long, within_idx] = keypoints

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        if not kpt_mask.any():
            return pred_kpt[..., :0].sum()

        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
        if not error.numel():
            return pred_kpt[..., :0].sum()

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return pred_kpt[..., :0].sum()

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class YOLO26PS25DLoss:
    """Unified partial-label loss wrapper for YOLO26 person-scene 2.5D models.

    Stages only change data sampling, trainable modules, and scalar task weights. The criterion itself is shared across
    stages and gates every task by the unified label flags, so images without ``has_det`` are never treated as detector
    background negatives.
    """

    DEFAULT_WEIGHTS = {
        "det": 1.0,
        "human_det": 0.0,
        "pose2d": 1.0,
        "pose_z": 0.5,
        "pose_vis": 0.3,
        "bone": 0.1,
        "person_mask": 0.7,
        "scene_seg": 0.2,
    }

    INSTANCE_KEYS = {"batch_idx", "cls", "bboxes", "keypoints", "body_kpts_3d", "instance_flags", "segments", "obb"}

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        tal_high_gt_threshold: int = 0,
        tal_high_gt_topk: int | None = None,
        tal_high_gt_topk2: int | None = None,
        tal_metric_chunk_gt: int = 0,
    ):
        """Initialize detection loss, auxiliary criteria, and multi-task scalar weights."""
        self.det = v8DetectionLoss(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            tal_high_gt_threshold=tal_high_gt_threshold,
            tal_high_gt_topk=tal_high_gt_topk,
            tal_high_gt_topk2=tal_high_gt_topk2,
            tal_metric_chunk_gt=tal_metric_chunk_gt,
        )
        self.device = self.det.device
        self.model = model
        head = model.model[-1]
        self.human_det = v8DetectionLoss(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            tal_high_gt_threshold=tal_high_gt_threshold,
            tal_high_gt_topk=tal_high_gt_topk,
            tal_high_gt_topk2=tal_high_gt_topk2,
            tal_metric_chunk_gt=tal_metric_chunk_gt,
        )
        self.human_det.nc = int(getattr(head, "human_nc", 2))
        self.human_det.no = self.human_det.nc + self.human_det.reg_max * 4
        self.human_det.assigner.num_classes = self.human_det.nc
        self.human_det.class_weights = None
        self.kpt_shape = list(getattr(head, "kpt_shape", [17, 4]))
        self.person_cls = int(getattr(head, "person_cls", -1))
        self.face_cls = int(getattr(head, "face_cls", -1))
        self.human_global_classes = tuple(
            int(c) for c in getattr(head, "human_global_classes", (self.person_cls, self.face_cls))
        )
        self.overlap = bool(getattr(model.args, "overlap_mask", True))
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.weights.update(self._weight_overrides(model))

        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if nkpt == 17 else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
        self.bce_pose = nn.BCEWithLogitsLoss(reduction="none")
        self.pose_anchor_topk = int(getattr(model.args, "pose_anchor_topk", 0) or 0)
        self.pose_anchor_radius = float(getattr(model.args, "pose_anchor_radius", 0.0) or 0.0)
        self.pose_xy_loss = str(getattr(model.args, "pose_xy_loss", "bbox") or "bbox").lower()
        self.pose_xy_beta = float(getattr(model.args, "pose_xy_beta", 0.05) or 0.05)
        self.pose_mpjpe_hard_px = float(getattr(model.args, "pose_mpjpe_hard_px", 0.0) or 0.0)
        self.pose_mpjpe_hard_gain = float(getattr(model.args, "pose_mpjpe_hard_gain", 0.0) or 0.0)
        self.pose_mpjpe_hard_power = float(getattr(model.args, "pose_mpjpe_hard_power", 1.0) or 1.0)
        self.pose_mpjpe_hard_max = float(getattr(model.args, "pose_mpjpe_hard_max", 4.0) or 4.0)
        self.pose_mpjpe_kpt_weights = self._parse_pose_kpt_weights(
            getattr(model.args, "pose_mpjpe_kpt_weights", None), nkpt
        )
        self.bone_pairs = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [5, 6],
                [5, 7],
                [7, 9],
                [6, 8],
                [8, 10],
                [5, 11],
                [6, 12],
                [11, 12],
                [11, 13],
                [13, 15],
                [12, 14],
                [14, 16],
            ],
            device=self.device,
            dtype=torch.long,
        )

    def parse_output(self, preds: dict[str, torch.Tensor] | tuple) -> dict[str, torch.Tensor]:
        """Parse model predictions to training dictionaries."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(self, preds: dict[str, torch.Tensor] | tuple, batch: dict[str, torch.Tensor]):
        """Compute partial-label multi-task losses."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        """Return total gated loss and detached loss items.

        Loss item order follows ``YOLO26PSDetect25D.loss_names``:
        box, cls, dfl, pose2d, pose_z, pose_vis, bone, person_mask, scene_seg, human_box, human_cls, human_dfl.
        """
        batch_size = preds["boxes"].shape[0]
        weights = self.task_weights()
        loss_items = torch.zeros(12, device=self.device)
        total_loss = self._zero_aux(preds) * batch_size
        assignment_cache: dict[tuple[bool, ...], tuple[dict[str, torch.Tensor], dict[str, Any], tuple]] = {}

        if weights["det"]:
            has_det = self._task_image_mask(batch, "has_det", batch_size, default=True)
            if has_det.any():
                det_preds = self._select_preds_by_images(preds, has_det)
                det_batch = self._select_batch_by_images(batch, has_det)
                assignment, det_loss, det_items = self.det.get_assigned_targets_and_loss(det_preds, det_batch)
                assignment_cache[self._assignment_key(has_det)] = (det_preds, det_batch, assignment)
                total_loss = total_loss + det_loss.sum() * det_preds["boxes"].shape[0] * weights["det"]
                loss_items[:3] = det_items * weights["det"]

        if weights["human_det"] and "human_boxes" in preds and "human_scores" in preds:
            human_batch = self._human_det_batch(batch, batch_size)
            has_human_det = self._task_image_mask(human_batch, "has_det", batch_size)
            if has_human_det.any():
                human_preds = {**preds, "boxes": preds["human_boxes"], "scores": preds["human_scores"]}
                human_preds = self._select_preds_by_images(human_preds, has_human_det)
                human_task_batch = self._select_batch_by_images(human_batch, has_human_det)
                _, human_loss, human_items = self.human_det.get_assigned_targets_and_loss(
                    human_preds, human_task_batch
                )
                total_loss = total_loss + human_loss.sum() * human_preds["boxes"].shape[0] * weights["human_det"]
                loss_items[9:12] = human_items * weights["human_det"]

        wants_pose2d = bool(weights["pose2d"] or weights["pose_vis"])
        wants_pose3d = bool(weights["pose_z"] or weights["bone"])
        if "pose25d" in preds and (wants_pose2d or wants_pose3d):
            pose_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            if wants_pose2d:
                pose_mask |= self._task_image_mask(batch, "has_pose2d", batch_size)
            if wants_pose3d:
                pose_mask |= self._task_image_mask(batch, "has_pose3d", batch_size)
            if pose_mask.any():
                pose2d, pose_z, pose_vis, bone = self._pose25d_loss_terms(preds, batch, pose_mask, assignment_cache)
                total_loss = total_loss + pose2d * weights["pose2d"]
                total_loss = total_loss + pose_z * weights["pose_z"]
                total_loss = total_loss + pose_vis * weights["pose_vis"]
                total_loss = total_loss + bone * weights["bone"]
                loss_items[3] = pose2d.detach() * weights["pose2d"]
                loss_items[4] = pose_z.detach() * weights["pose_z"]
                loss_items[5] = pose_vis.detach() * weights["pose_vis"]
                loss_items[6] = bone.detach() * weights["bone"]

        if weights["person_mask"] and "mask_coefficient" in preds and "proto" in preds:
            has_mask = self._task_image_mask(batch, "has_person_mask", batch_size)
            if has_mask.any():
                person_mask_loss = self._person_mask_loss(preds, batch, has_mask, assignment_cache)
                total_loss = total_loss + person_mask_loss * weights["person_mask"]
                loss_items[7] = person_mask_loss.detach() * weights["person_mask"]

        if weights["scene_seg"] and "scene_seg" in preds:
            has_scene = self._task_image_mask(batch, "has_scene_seg", batch_size)
            if has_scene.any():
                scene_loss = self._scene_seg_loss(preds, batch, has_scene)
                total_loss = total_loss + scene_loss * weights["scene_seg"]
                loss_items[8] = scene_loss.detach() * weights["scene_seg"]

        return total_loss, loss_items

    def _human_det_batch(self, batch: dict[str, torch.Tensor], batch_size: int) -> dict[str, Any]:
        """Return a two-class person/face detection batch for the human-centric head."""
        out = dict(batch)
        cls = batch.get("cls")
        batch_idx = batch.get("batch_idx")
        if not (torch.is_tensor(cls) and torch.is_tensor(batch_idx)):
            out["batch_idx"] = torch.zeros(0, device=self.device, dtype=torch.long)
            out["cls"] = torch.zeros(0, 1, device=self.device, dtype=torch.float32)
            out["bboxes"] = torch.zeros(0, 4, device=self.device)
            out["has_det"] = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            out["det_class_mask"] = torch.zeros(batch_size, self.human_det.nc, device=self.device, dtype=torch.bool)
            return out

        cls_flat = cls.to(self.device).view(-1).long()
        batch_idx = batch_idx.to(self.device).long().view(-1)
        person_cls, face_cls = self.human_global_classes[:2]
        keep = cls_flat.eq(person_cls) | cls_flat.eq(face_cls)
        inst_count = int(cls_flat.numel())

        for key, value in batch.items():
            if torch.is_tensor(value) and key in self.INSTANCE_KEYS and value.shape[:1] == (inst_count,):
                out[key] = value[keep.to(value.device)]

        kept_cls = cls_flat[keep]
        out["cls"] = torch.where(kept_cls.eq(face_cls), torch.ones_like(kept_cls), torch.zeros_like(kept_cls)).view(-1, 1)
        out["cls"] = out["cls"].to(device=cls.device, dtype=cls.dtype)
        out["batch_idx"] = batch_idx[keep].to(device=batch["batch_idx"].device, dtype=batch["batch_idx"].dtype)

        has_source_det = self._task_image_mask(batch, "has_det", batch_size, default=True)
        has_instances = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        if out["batch_idx"].numel():
            has_instances[out["batch_idx"].to(self.device).long()] = True
        out["has_det"] = has_source_det & has_instances
        out["det_class_mask"] = self._human_det_class_mask(batch, batch_size)
        return out

    def _human_det_class_mask(self, batch: dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """Map global per-image detection class masks to person/face supervision masks."""
        mask = batch.get("det_class_mask")
        out = torch.zeros(batch_size, self.human_det.nc, device=self.device, dtype=torch.bool)
        person_cls, face_cls = self.human_global_classes[:2]
        if torch.is_tensor(mask):
            mask = mask.to(self.device).bool()
            if mask.ndim == 1:
                mask = mask.view(1, -1).expand(batch_size, -1)
            mask = mask[:batch_size]
            if mask.shape[-1] == self.human_det.nc:
                out[:, : mask.shape[-1]] = mask
                return out
            if 0 <= person_cls < mask.shape[-1]:
                out[:, 0] = mask[:, person_cls]
            if self.human_det.nc > 1 and 0 <= face_cls < mask.shape[-1]:
                out[:, 1] = mask[:, face_cls]
            return out

        cls = batch.get("cls")
        batch_idx = batch.get("batch_idx")
        if torch.is_tensor(cls) and torch.is_tensor(batch_idx):
            cls = cls.to(self.device).view(-1).long()
            batch_idx = batch_idx.to(self.device).view(-1).long()
            for b in range(batch_size):
                image_cls = cls[batch_idx == b]
                out[b, 0] = bool(image_cls.eq(person_cls).any())
                if self.human_det.nc > 1:
                    out[b, 1] = bool(image_cls.eq(face_cls).any())
        return out

    def task_weights(self) -> dict[str, float]:
        """Return current task weights, allowing trainers to override ``model.loss_weights`` per stage."""
        weights = self.weights.copy()
        weights.update(self._weight_overrides(self.model))
        return weights

    @classmethod
    def _weight_overrides(cls, model) -> dict[str, float]:
        """Collect optional task-loss weights from model attributes or trainer args."""
        out = {}
        args = getattr(model, "args", None)
        for source in (
            getattr(model, "loss_weights", None),
            getattr(model, "task_loss_weight", None),
            getattr(args, "loss_weights", None) if args is not None else None,
            getattr(args, "task_loss_weight", None) if args is not None else None,
        ):
            if isinstance(source, dict):
                out.update({k: float(v) for k, v in source.items() if k in cls.DEFAULT_WEIGHTS})
        for source in (model, args):
            if source is None:
                continue
            for key in cls.DEFAULT_WEIGHTS:
                attr = f"loss_{key}"
                if hasattr(source, attr):
                    out[key] = float(getattr(source, attr))
        return out

    def _parse_pose_kpt_weights(self, value, nkpt: int) -> torch.Tensor | None:
        """Parse optional per-keypoint MPJPE weights."""
        if value in (None, "", False):
            return None
        if isinstance(value, torch.Tensor):
            values = value.detach().flatten().float().tolist()
        elif isinstance(value, (list, tuple)):
            values = [float(x) for x in value]
        else:
            text = str(value).replace(";", ",").replace(" ", ",")
            values = [float(x) for x in text.split(",") if x.strip()]
        if len(values) != nkpt:
            raise ValueError(f"pose_mpjpe_kpt_weights expects {nkpt} values, got {len(values)}")
        weights = torch.tensor(values, device=self.device, dtype=torch.float32).clamp(min=0.0)
        if float(weights.sum()) <= 0:
            return None
        return weights / weights.mean().clamp(min=1e-6)

    def _pose25d_loss_terms(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        image_mask: torch.Tensor,
        assignment_cache: dict[tuple[bool, ...], tuple[dict[str, torch.Tensor], dict[str, Any], tuple]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gated 2D, depth, visibility, and bone losses for matched person positives."""
        zero = self._safe_zero(preds["pose25d"])
        task_preds, task_batch, assignment = self._get_assignment(preds, batch, image_mask, assignment_cache)
        if task_batch["batch_idx"].numel() == 0:
            return zero, zero, zero, zero

        fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assignment
        if not fg_mask.any():
            return zero, zero, zero, zero

        pred_pose = self._decode_pose25d(task_preds, anchor_points, stride_tensor)
        pred_pose_grid = self._decode_pose25d_grid(task_preds, anchor_points)
        imgsz = torch.tensor(task_preds["feats"][0].shape[2:], device=self.device, dtype=pred_pose.dtype) * self.det.stride[0]
        instance_bboxes = self._target_boxes_by_instance(task_batch, imgsz, pred_pose.dtype)
        pose_mask, pose_target_gt_idx = self._pose_anchor_targets(
            task_batch, fg_mask, target_gt_idx, instance_bboxes, anchor_points, stride_tensor
        )
        selected_cls = self._select_target_instances(
            task_batch["cls"].to(self.device), task_batch["batch_idx"], target_gt_idx, fg_mask
        )
        pose_selected_cls = self._select_target_instances(
            task_batch["cls"].to(self.device), task_batch["batch_idx"], pose_target_gt_idx, pose_mask
        )
        is_person = selected_cls.squeeze(-1).long() == self.person_cls if self.person_cls >= 0 else torch.ones_like(fg_mask)
        pose_is_person = (
            pose_selected_cls.squeeze(-1).long() == self.person_cls if self.person_cls >= 0 else torch.ones_like(pose_mask)
        )

        flags = task_batch.get("instance_flags")
        if flags is not None and flags.numel():
            selected_flags = self._select_target_instances(
                flags.to(self.device).bool(), task_batch["batch_idx"], target_gt_idx, fg_mask
            )
            inst_has_body2d = selected_flags[..., 1]
            inst_has_body3d = selected_flags[..., 2]
            pose_selected_flags = self._select_target_instances(
                flags.to(self.device).bool(), task_batch["batch_idx"], pose_target_gt_idx, pose_mask
            )
            pose_inst_has_body2d = pose_selected_flags[..., 1]
        else:
            inst_has_body2d = torch.ones_like(fg_mask)
            inst_has_body3d = torch.ones_like(fg_mask)
            pose_inst_has_body2d = torch.ones_like(pose_mask)

        bs = task_preds["boxes"].shape[0]
        has_pose2d = self._task_image_mask(task_batch, "has_pose2d", bs).view(-1, 1)
        has_pose3d = self._task_image_mask(task_batch, "has_pose3d", bs).view(-1, 1)
        pos2d = pose_mask & pose_is_person & pose_inst_has_body2d & has_pose2d
        pos3d = fg_mask & is_person & inst_has_body3d & has_pose3d

        bbox_h = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp(min=1.0)
        target_bboxes_grid = target_bboxes / stride_tensor.view(1, -1, 1)
        pose_target_bboxes = self._select_target_instances(
            instance_bboxes,
            task_batch["batch_idx"],
            pose_target_gt_idx,
            pose_mask,
        )
        pose_target_bboxes_grid = pose_target_bboxes / stride_tensor.view(1, -1, 1)
        bbox_h_grid = (target_bboxes_grid[..., 3] - target_bboxes_grid[..., 1]).clamp(min=1.0)

        pose2d_loss = zero
        pose_vis_loss = zero
        keypoints = task_batch.get("keypoints")
        if keypoints is not None and keypoints.numel():
            keypoints = keypoints.to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            gt_kpts = self._select_target_instances(keypoints, task_batch["batch_idx"], pose_target_gt_idx, pose_mask)
            gt_kpts_grid = gt_kpts.clone()
            gt_kpts_grid[..., :2] /= stride_tensor.view(1, -1, 1, 1)
            kpt_present = gt_kpts_grid[..., 2].gt(0) & pos2d.unsqueeze(-1)
            positive = pos2d & kpt_present.any(-1)
            if positive.any():
                pose2d_loss = self._pose_xy_loss(
                    pred_pose_grid[..., :2][positive],
                    gt_kpts_grid[..., :2][positive],
                    kpt_present[positive],
                    pose_target_bboxes_grid[positive],
                    stride_tensor.view(1, -1, 1, 1).expand_as(pred_pose_grid[..., :1])[positive],
                )

                visible_target = gt_kpts_grid[..., 2].gt(0).to(pred_pose_grid.dtype)
                pose_vis_mask = positive.unsqueeze(-1).expand_as(visible_target)
                pose_vis_loss = self.bce_pose(pred_pose_grid[..., 3], visible_target)[pose_vis_mask].mean()

        pose_z_loss = zero
        bone_loss = zero
        kpts3d = task_batch.get("body_kpts_3d")
        if kpts3d is not None and kpts3d.numel():
            kpts3d = kpts3d.to(self.device).float().clone()
            kpts3d[..., 0] *= imgsz[1]
            kpts3d[..., 1] *= imgsz[0]
            gt_kpts3d = self._select_target_instances(kpts3d, task_batch["batch_idx"], target_gt_idx, fg_mask)
            gt_kpts3d_grid = gt_kpts3d.clone()
            gt_kpts3d_grid[..., :2] /= stride_tensor.view(1, -1, 1, 1)
            valid3d = gt_kpts3d[..., 3].gt(0) & pos3d.unsqueeze(-1)
            if valid3d.any():
                z_scale = bbox_h.unsqueeze(-1).clamp(min=1.0)
                pose_z_loss = F.smooth_l1_loss(pred_pose[..., 2][valid3d], gt_kpts3d[..., 2][valid3d])
                z_scale_grid = bbox_h_grid.unsqueeze(-1).clamp(min=1.0)
                pred_z_norm = pred_pose_grid[..., 2] / z_scale_grid
                gt_z_norm = gt_kpts3d_grid[..., 2] / z_scale_grid
                bone_loss = self._bone_loss(
                    pred_pose_grid[..., :2],
                    pred_z_norm,
                    gt_kpts3d_grid[..., :2],
                    gt_z_norm,
                    valid3d,
                    bbox_h_grid,
                )

        return pose2d_loss, pose_z_loss, pose_vis_loss, bone_loss

    def _pose_anchor_targets(
        self,
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        instance_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand pose supervision from sparse TAL positives to nearby bbox-centered anchors."""
        topk = int(self.pose_anchor_topk)
        if topk <= 0:
            return fg_mask, target_gt_idx

        batch_idx = batch["batch_idx"].to(self.device).long().flatten()
        classes = batch["cls"].to(self.device).long().flatten()
        flags = batch.get("instance_flags")
        has_body2d = (
            flags.to(self.device).bool()[:, 1]
            if flags is not None and flags.numel()
            else torch.ones_like(classes, dtype=torch.bool)
        )
        if not batch_idx.numel() or not instance_bboxes.numel():
            return fg_mask, target_gt_idx

        bs, num_anchors = fg_mask.shape
        out_mask = fg_mask.clone()
        out_idx = target_gt_idx.clone()
        anchors = anchor_points.to(instance_bboxes.device) * stride_tensor.to(instance_bboxes.device)
        radius = float(self.pose_anchor_radius)
        topk = min(topk, num_anchors)
        local_instance_ids = torch.zeros_like(batch_idx)
        for b in range(bs):
            image_ids = torch.where(batch_idx == b)[0]
            if image_ids.numel():
                local_instance_ids[image_ids] = torch.arange(image_ids.numel(), device=self.device, dtype=batch_idx.dtype)

        for b in range(bs):
            global_ids = torch.where(
                (batch_idx == b)
                & has_body2d
                & (classes == self.person_cls if self.person_cls >= 0 else torch.ones_like(classes, dtype=torch.bool))
            )[0]
            if not global_ids.numel():
                continue
            for global_i in global_ids:
                global_i = int(global_i)
                if global_i >= len(instance_bboxes):
                    continue
                local_i = int(local_instance_ids[global_i])
                x1, y1, x2, y2 = instance_bboxes[global_i]
                if x2 <= x1 or y2 <= y1:
                    continue
                gc = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
                gwh = torch.stack(((x2 - x1).clamp(min=1.0), (y2 - y1).clamp(min=1.0)))
                delta = (anchors - gc) / gwh
                inside = (delta.abs() <= 0.5).all(-1)
                if radius > 0:
                    inside &= delta.norm(dim=-1) <= radius
                if not inside.any():
                    continue
                dist = delta.norm(dim=-1)
                dist = torch.where(inside, dist, torch.full_like(dist, float("inf")))
                _, selected = torch.topk(-dist, k=min(topk, int(inside.sum().item())), largest=True)
                write = selected[(~out_mask[b, selected]) | (out_idx[b, selected] == local_i)]
                if not write.numel():
                    continue
                out_mask[b, write] = True
                out_idx[b, write] = local_i
        return out_mask, out_idx

    def _target_boxes_by_instance(
        self, batch: dict[str, torch.Tensor], imgsz: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return per-instance xyxy boxes in model-input pixels."""
        boxes = batch["bboxes"].to(self.device).to(dtype).clone()
        if not boxes.numel():
            return boxes.new_zeros((0, 4))
        boxes[:, [0, 2]] *= imgsz[1]
        boxes[:, [1, 3]] *= imgsz[0]
        return xywh2xyxy(boxes)

    def _pose_xy_loss(
        self,
        pred_xy: torch.Tensor,
        gt_xy: torch.Tensor,
        kpt_present: torch.Tensor,
        target_bboxes_grid: torch.Tensor,
        stride: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute 2D keypoint regression loss in bbox-normalized or grid-cell units."""
        if not kpt_present.any():
            return self._safe_zero(pred_xy)
        beta = max(float(self.pose_xy_beta), 1e-6)
        if self.pose_xy_loss == "grid":
            return F.smooth_l1_loss(pred_xy[kpt_present], gt_xy[kpt_present], beta=beta)
        if self.pose_xy_loss == "pixel":
            if stride is None:
                return F.smooth_l1_loss(pred_xy[kpt_present], gt_xy[kpt_present], beta=beta)
            error = (pred_xy - gt_xy) * stride
            return F.smooth_l1_loss(error[kpt_present], torch.zeros_like(error[kpt_present]), beta=beta)
        if self.pose_xy_loss == "mpjpe":
            error = pred_xy - gt_xy
            if stride is not None:
                error = error * stride
            distance = error.norm(dim=-1)
            visible_distance = distance[kpt_present]
            visible_weights = None
            if self.pose_mpjpe_kpt_weights is not None:
                kpt_weights = self.pose_mpjpe_kpt_weights.to(device=distance.device, dtype=distance.dtype)
                weights = kpt_weights.view(*([1] * (distance.ndim - 1)), -1).expand_as(distance)
                visible_weights = weights[kpt_present]
            if self.pose_mpjpe_hard_px > 0 and self.pose_mpjpe_hard_gain > 0:
                excess = (distance.detach() / self.pose_mpjpe_hard_px - 1.0).clamp(min=0.0)
                weights = 1.0 + self.pose_mpjpe_hard_gain * excess.pow(max(self.pose_mpjpe_hard_power, 1e-6))
                if self.pose_mpjpe_hard_max > 0:
                    weights = weights.clamp(max=self.pose_mpjpe_hard_max)
                hard_weights = weights[kpt_present]
                visible_weights = hard_weights if visible_weights is None else visible_weights * hard_weights
            if visible_weights is not None:
                visible_weights = visible_weights / visible_weights.mean().clamp(min=1e-6)
                return (visible_distance * visible_weights).mean()
            return visible_distance.mean()
        box_wh = (target_bboxes_grid[..., 2:4] - target_bboxes_grid[..., 0:2]).clamp(min=1.0).unsqueeze(1)
        error = (pred_xy - gt_xy) / box_wh
        return F.smooth_l1_loss(error[kpt_present], torch.zeros_like(error[kpt_present]), beta=beta)

    def _decode_pose25d(
        self, preds: dict[str, torch.Tensor], anchor_points: torch.Tensor, stride_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Decode raw pose maps to image-space x/y, raw z, and confidence logits."""
        pred_pose = self._decode_pose25d_grid(preds, anchor_points)
        xy = pred_pose[..., :2] * stride_tensor.view(1, -1, 1, 1)
        return torch.cat((xy, pred_pose[..., 2:4]), dim=-1)

    def _decode_pose25d_grid(
        self, preds: dict[str, torch.Tensor], anchor_points: torch.Tensor
    ) -> torch.Tensor:
        """Decode raw pose maps to stride-space x/y, raw z, and confidence logits."""
        bs = preds["pose25d"].shape[0]
        pred_pose = preds["pose25d"].permute(0, 2, 1).contiguous().view(bs, -1, *self.kpt_shape)
        xy = pred_pose[..., :2] + anchor_points.view(1, -1, 1, 2)
        return torch.cat((xy, pred_pose[..., 2:4]), dim=-1)

    def _bone_loss(
        self,
        pred_xy: torch.Tensor,
        pred_z_norm: torch.Tensor,
        gt_xy: torch.Tensor,
        gt_z_norm: torch.Tensor,
        valid3d: torch.Tensor,
        bbox_h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bone-length consistency on pseudo-3D normalized coordinates."""
        if self.kpt_shape[0] < 17:
            return self._safe_zero(pred_xy)
        root_valid = valid3d[..., 11] & valid3d[..., 12]
        if not root_valid.any():
            return self._safe_zero(pred_xy)
        pred_root = (pred_xy[..., 11:12, :] + pred_xy[..., 12:13, :]) * 0.5
        gt_root = (gt_xy[..., 11:12, :] + gt_xy[..., 12:13, :]) * 0.5
        scale = bbox_h.unsqueeze(-1).unsqueeze(-1).clamp(min=1.0)
        pred_xyz = torch.cat(((pred_xy - pred_root) / scale, pred_z_norm.unsqueeze(-1)), dim=-1)
        gt_xyz = torch.cat(((gt_xy - gt_root) / scale, gt_z_norm.unsqueeze(-1)), dim=-1)
        edges = self.bone_pairs.to(pred_xy.device)
        pred_len = (pred_xyz[..., edges[:, 0], :] - pred_xyz[..., edges[:, 1], :]).norm(dim=-1)
        gt_len = (gt_xyz[..., edges[:, 0], :] - gt_xyz[..., edges[:, 1], :]).norm(dim=-1)
        edge_valid = valid3d[..., edges[:, 0]] & valid3d[..., edges[:, 1]] & root_valid.unsqueeze(-1)
        if not edge_valid.any():
            return self._safe_zero(pred_xy)
        return F.smooth_l1_loss(pred_len[edge_valid], gt_len[edge_valid])

    def _person_mask_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        image_mask: torch.Tensor,
        assignment_cache: dict[tuple[bool, ...], tuple[dict[str, torch.Tensor], dict[str, Any], tuple]] | None = None,
    ) -> torch.Tensor:
        """Compute prototype instance-mask loss when rasterized person masks are present."""
        zero = self._safe_zero(preds["mask_coefficient"]) + self._safe_zero(preds["proto"])
        if "masks" not in batch or not torch.is_tensor(batch["masks"]):
            return zero

        task_preds, task_batch, assignment = self._get_assignment(preds, batch, image_mask, assignment_cache)
        if task_batch["batch_idx"].numel() == 0:
            return zero
        fg_mask, target_gt_idx, target_bboxes, _, _ = assignment
        if not fg_mask.any():
            return zero

        selected_cls = self._select_target_instances(task_batch["cls"].to(self.device), task_batch["batch_idx"], target_gt_idx, fg_mask)
        is_person = selected_cls.squeeze(-1).long() == self.person_cls if self.person_cls >= 0 else torch.ones_like(fg_mask)
        flags = task_batch.get("instance_flags")
        if flags is not None and flags.numel():
            selected_flags = self._select_target_instances(flags.to(self.device).bool(), task_batch["batch_idx"], target_gt_idx, fg_mask)
            inst_has_mask = selected_flags[..., 3]
        else:
            inst_has_mask = torch.ones_like(fg_mask)
        fg_mask = fg_mask & is_person & inst_has_mask
        if not fg_mask.any():
            return zero

        pred_masks = task_preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto = task_preds["proto"]
        masks = task_batch["masks"].to(self.device).float()
        if tuple(masks.shape[-2:]) != tuple(proto.shape[-2:]):
            proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)
        imgsz = torch.tensor(task_preds["feats"][0].shape[2:], device=self.device, dtype=proto.dtype) * self.det.stride[0]
        return self._calculate_person_mask_loss(
            fg_mask,
            masks,
            target_gt_idx,
            target_bboxes,
            task_batch["batch_idx"].view(-1, 1),
            proto,
            pred_masks,
            imgsz,
        )

    def _calculate_person_mask_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate person instance-mask loss from matched positive anchors."""
        _, _, mask_h, mask_w = proto.shape
        loss = self._safe_zero(proto) + self._safe_zero(pred_masks)
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if not fg_mask_i.any():
                continue
            mask_idx = target_gt_idx_i[fg_mask_i]
            if self.overlap:
                gt_mask = (masks_i == (mask_idx + 1).view(-1, 1, 1)).float()
            else:
                gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
            loss = loss + v8SegmentationLoss.single_mask_loss(
                gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i].clamp(min=1e-6)
            )
        return loss / fg_mask.sum().clamp(min=1)

    def _scene_seg_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], image_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute semantic scene loss when collate provides raster scene masks."""
        zero = self._safe_zero(preds["scene_seg"])
        target = batch.get("scene_seg")
        if not torch.is_tensor(target):
            return zero
        pred = preds["scene_seg"][image_mask].float()
        target = target.to(self.device)[image_mask].long()
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if tuple(target.shape[-2:]) != tuple(pred.shape[-2:]):
            target = F.interpolate(target[:, None].float(), pred.shape[-2:], mode="nearest")[:, 0].long()
        ignore_index = 255
        num_classes = pred.shape[1]
        invalid = (target != ignore_index) & ((target < 0) | (target >= num_classes))
        if invalid.any():
            target = target.clone()
            target[invalid] = ignore_index
        flat_target = target.reshape(-1)
        valid = flat_target != ignore_index
        if not valid.any():
            return zero
        flat_pred = pred.permute(0, 2, 3, 1).reshape(-1, num_classes)
        return F.cross_entropy(flat_pred[valid], flat_target[valid])

    def _task_image_mask(
        self, batch: dict[str, torch.Tensor], key: str, batch_size: int, default: bool = False
    ) -> torch.Tensor:
        """Return a bool image mask for a task flag, with legacy detection labels defaulting to enabled."""
        value = batch.get(key)
        if value is None:
            return torch.full((batch_size,), default, device=self.device, dtype=torch.bool)
        if torch.is_tensor(value):
            return value.to(self.device).bool().view(-1)[:batch_size]
        return torch.tensor([bool(x) for x in value], device=self.device, dtype=torch.bool)[:batch_size]

    def _get_assignment(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
        image_mask: torch.Tensor,
        cache: dict[tuple[bool, ...], tuple[dict[str, torch.Tensor], dict[str, Any], tuple]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], tuple]:
        """Return cached detection assignment for a task image subset."""
        key = self._assignment_key(image_mask)
        if cache is not None and key in cache:
            return cache[key]
        task_preds = self._select_preds_by_images(preds, image_mask)
        task_batch = self._select_batch_by_images(batch, image_mask)
        if cache is not None:
            cached_assignment = self._slice_assignment_from_superset(cache, key, task_preds, task_batch)
            if cached_assignment is not None:
                result = task_preds, task_batch, cached_assignment
                cache[key] = result
                return result
        if task_batch["batch_idx"].numel() == 0:
            empty = task_preds["boxes"].new_zeros(
                (task_preds["boxes"].shape[0], task_preds["boxes"].shape[-1]), dtype=torch.bool
            )
            assignment = (
                empty,
                task_preds["boxes"].new_zeros(empty.shape, dtype=torch.long),
                task_preds["boxes"].new_zeros((*empty.shape, 4)),
                task_preds["boxes"].new_zeros((task_preds["boxes"].shape[-1], 2)),
                task_preds["boxes"].new_ones((task_preds["boxes"].shape[-1], 1)),
            )
        else:
            assignment = self.det.get_assignment(task_preds, task_batch)
        result = task_preds, task_batch, assignment
        if cache is not None:
            cache[key] = result
        return result

    @staticmethod
    def _assignment_key(image_mask: torch.Tensor) -> tuple[bool, ...]:
        """Convert an image mask into a hashable assignment-cache key."""
        return tuple(bool(x) for x in image_mask.detach().cpu().tolist())

    @staticmethod
    def _slice_assignment_from_superset(
        cache: dict[tuple[bool, ...], tuple[dict[str, torch.Tensor], dict[str, Any], tuple]],
        key: tuple[bool, ...],
        task_preds: dict[str, torch.Tensor],
        task_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Slice a cached image-independent assignment from a cached superset batch."""
        if not any(key):
            return None
        want = torch.tensor(key, dtype=torch.bool)
        for cached_key, (_, _, assignment) in cache.items():
            if len(cached_key) != len(key):
                continue
            have = torch.tensor(cached_key, dtype=torch.bool)
            if not bool((want & ~have).any()):
                source_pos = torch.where(have)[0]
                target_pos = torch.where(want)[0]
                if not len(target_pos):
                    continue
                rel_pos = torch.searchsorted(source_pos, target_pos).to(assignment[0].device)
                fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assignment
                if fg_mask.shape[0] < int(rel_pos.max().item()) + 1:
                    continue
                return (
                    fg_mask[rel_pos],
                    target_gt_idx[rel_pos.to(target_gt_idx.device)],
                    target_bboxes[rel_pos.to(target_bboxes.device)],
                    anchor_points,
                    stride_tensor,
                )
        return None

    def _select_preds_by_images(self, preds: dict[str, torch.Tensor], image_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select prediction tensors for images enabled for a task."""
        image_mask = image_mask.to(preds["boxes"].device)
        batch_size = preds["boxes"].shape[0]
        out = {}
        for key, value in preds.items():
            if key == "feats" and isinstance(value, (list, tuple)):
                out[key] = [x[image_mask] for x in value]
            elif torch.is_tensor(value) and value.shape[:1] == (batch_size,):
                out[key] = value[image_mask]
            else:
                out[key] = value
        return out

    def _select_batch_by_images(self, batch: dict[str, Any], image_mask: torch.Tensor) -> dict[str, Any]:
        """Select and remap a batch to the images enabled for a task."""
        old_batch = int(image_mask.numel())
        device = batch["batch_idx"].device if torch.is_tensor(batch.get("batch_idx")) else self.device
        image_mask = image_mask.to(device)
        keep_idx = torch.where(image_mask)[0]
        batch_idx = batch["batch_idx"].to(device).long()
        inst_count = int(batch_idx.numel())
        inst_mask = image_mask[batch_idx] if inst_count else torch.zeros(0, device=device, dtype=torch.bool)
        remap = torch.full((old_batch,), -1, device=device, dtype=torch.long)
        remap[keep_idx] = torch.arange(len(keep_idx), device=device)

        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if key == "batch_idx":
                    out[key] = remap[batch_idx][inst_mask].to(value.dtype)
                elif key == "masks":
                    if self.overlap and value.shape[:1] == (old_batch,):
                        out[key] = value[image_mask.to(value.device)]
                    elif (not self.overlap) and value.shape[:1] == (inst_count,):
                        out[key] = value[inst_mask.to(value.device)]
                    else:
                        out[key] = value
                elif key in self.INSTANCE_KEYS and value.shape[:1] == (inst_count,):
                    out[key] = value[inst_mask.to(value.device)]
                elif value.shape[:1] == (old_batch,):
                    out[key] = value[image_mask.to(value.device)]
                else:
                    out[key] = value
            elif isinstance(value, (list, tuple)) and len(value) == old_batch:
                selected = [value[int(i)] for i in keep_idx.cpu()]
                out[key] = tuple(selected) if isinstance(value, tuple) else selected
            else:
                out[key] = value
        return out

    @staticmethod
    def _select_target_instances(
        values: torch.Tensor, batch_idx: torch.Tensor, target_gt_idx: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Gather per-instance target tensors by assigned target index for each anchor."""
        batch_idx = batch_idx.flatten().long().to(values.device)
        batch_size, num_anchors = masks.shape
        counts = torch.zeros(batch_size, dtype=torch.long, device=values.device)
        if batch_idx.numel():
            counts.scatter_add_(0, batch_idx, torch.ones_like(batch_idx))
        max_count = max(int(counts.max().item()) if counts.numel() else 0, 1)
        batched = torch.zeros((batch_size, max_count, *values.shape[1:]), device=values.device, dtype=values.dtype)
        if batch_idx.numel():
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=values.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(len(batch_idx), device=values.device) - offsets[batch_idx]
            batched[batch_idx, within_idx] = values
        gather_idx = target_gt_idx.clamp(max=max_count - 1).view(batch_size, num_anchors, *([1] * (values.ndim - 1)))
        return batched.gather(1, gather_idx.expand(batch_size, num_anchors, *values.shape[1:]))

    def _zero_aux(self, preds: dict[str, torch.Tensor]) -> torch.Tensor:
        """Keep inactive branches graph-connected for DDP and staged partial-label training."""
        zero = self._safe_zero(preds["boxes"]) + self._safe_zero(preds["scores"])
        for key in ("human_boxes", "human_scores", "pose25d", "mask_coefficient", "proto", "scene_seg"):
            if key in preds:
                zero = zero + self._safe_zero(preds[key])
        return zero

    @staticmethod
    def _safe_zero(tensor: torch.Tensor) -> torch.Tensor:
        """Return a graph-connected scalar zero without large FP16 reductions."""
        return tensor.float().sum() * 0.0


class YOLO26PS25DE2ELoss:
    """End-to-end one-to-many/one-to-one wrapper for YOLO26 PS-2.5D loss."""

    def __init__(self, model):
        """Initialize one-to-many and one-to-one branches."""
        args = getattr(model, "args", None)
        o2m_topk = int(getattr(args, "tal_topk_one2many", 10) or 10)
        o2o_topk = int(getattr(args, "tal_topk_one2one", 7) or 7)
        o2o_topk2 = int(getattr(args, "tal_topk2_one2one", 1) or 1)
        high_gt_threshold = int(getattr(args, "tal_high_gt_threshold", 0) or 0)
        metric_chunk_gt = int(getattr(args, "tal_metric_chunk_gt", 0) or 0)
        o2m_high_topk = int(getattr(args, "tal_high_gt_topk_one2many", o2m_topk) or o2m_topk)
        o2o_high_topk = int(getattr(args, "tal_high_gt_topk_one2one", o2o_topk) or o2o_topk)
        o2o_high_topk2 = int(getattr(args, "tal_high_gt_topk2_one2one", o2o_topk2) or o2o_topk2)
        self.one2many = YOLO26PS25DLoss(
            model,
            tal_topk=o2m_topk,
            tal_high_gt_threshold=high_gt_threshold,
            tal_high_gt_topk=o2m_high_topk,
            tal_metric_chunk_gt=metric_chunk_gt,
        )
        self.one2one = YOLO26PS25DLoss(
            model,
            tal_topk=o2o_topk,
            tal_topk2=o2o_topk2,
            tal_high_gt_threshold=high_gt_threshold,
            tal_high_gt_topk=o2o_high_topk,
            tal_high_gt_topk2=o2o_high_topk2,
            tal_metric_chunk_gt=metric_chunk_gt,
        )
        self.tal_topk = {
            "one2many": o2m_topk,
            "one2one": o2o_topk,
            "one2one_topk2": o2o_topk2,
            "high_gt_threshold": high_gt_threshold,
            "high_gt_one2many": o2m_high_topk,
            "high_gt_one2one": o2o_high_topk,
            "high_gt_one2one_topk2": o2o_high_topk2,
            "metric_chunk_gt": metric_chunk_gt,
        }
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2m_copy = self.o2m
        self.o2o = 0.2
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]):
        """Calculate weighted one-to-many and one-to-one losses."""
        preds = self.one2many.parse_output(preds)
        loss_one2many = self.one2many.loss(preds["one2many"], batch)
        loss_one2one = self.one2one.loss(preds["one2one"], batch)
        return (
            loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o,
            loss_one2many[1] * self.o2m + loss_one2one[1] * self.o2o,
        )

    def update(self) -> None:
        """Update one-to-many decay schedule to match E2ELoss behavior."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, updates: int) -> float:
        """Return decayed one-to-many weight."""
        return self.final_o2m + (self.o2m_copy - self.final_o2m) * math.exp(-updates / 2000)


class SemanticSegmentationLoss(nn.Module):
    """Loss function for semantic segmentation using cross-entropy and Dice terms."""

    def __init__(self, model: torch.nn.Module):
        """Initialize semantic segmentation loss."""
        super().__init__()
        m = model.model[-1]
        self.nc = m.nc
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        data_name = Path(str(getattr(model.args, "data", "") or "")).stem.lower()
        self.use_cityscapes_weight = data_name in {"cityscapes", "cityscapes8"} and self.nc == len(CITYSCAPES_WEIGHT)
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255).to(device=self.device, dtype=self.dtype)
            if self.use_cityscapes_weight:
                weight = torch.from_numpy(CITYSCAPES_WEIGHT).to(device=self.device, dtype=self.dtype)
                self.ce.register_buffer("weight", weight, persistent=False)

    def _ce_loss(self, preds, masks):
        """Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path."""
        if self.nc == 1:
            flat = masks.reshape(-1)
            valid = flat != 255
            logits = preds.reshape(-1)[valid]
            target = flat[valid].float()
        else:
            logits = preds.permute(0, 2, 3, 1).reshape(-1, self.nc)
            target = masks.reshape(-1).long()
        return self.ce(logits, target)

    def _dice_loss(self, preds, masks):
        """Compute Dice loss excluding ignore pixels."""
        if self.nc == 1:
            return self._binary_dice_loss(preds, masks)
        flat_target = masks.reshape(-1)
        valid = flat_target != 255
        if not valid.any():
            return preds.sum() * 0

        pred_soft = F.softmax(preds, dim=1)
        target = flat_target[valid].long()
        flat_pred = pred_soft.float().permute(0, 2, 3, 1).reshape(-1, self.nc)[valid]
        intersection = torch.zeros(self.nc, device=preds.device, dtype=torch.float32)
        intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
        pred_sum = flat_pred.sum(dim=0)
        target_sum = torch.bincount(target, minlength=self.nc).to(device=preds.device, dtype=torch.float32)
        cardinality = pred_sum + target_sum
        return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

    def _binary_dice_loss(self, preds, masks):
        """Compute Dice loss for single-class binary segmentation."""
        valid = (masks != 255).float()
        pred_soft = preds.squeeze(1).sigmoid()
        target = (masks == 1).float()
        intersection = (pred_soft * target * valid).sum()
        cardinality = ((pred_soft + target) * valid).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

    def forward(self, preds, batch):
        """Compute semantic segmentation loss with optional auxiliary loss."""
        aux_logits = None
        if isinstance(preds, tuple):
            preds, aux_logits = preds

        masks = batch["semantic_mask"].to(preds.device)
        if preds.shape[2:] != masks.shape[1:]:
            preds = F.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)

        ce_loss = self._ce_loss(preds, masks)
        dice_loss = self._dice_loss(preds, masks)
        total = ce_loss + dice_loss

        aux_loss = torch.tensor(0.0, device=preds.device, dtype=ce_loss.dtype)
        if aux_logits is not None:
            if aux_logits.shape[2:] != masks.shape[1:]:
                aux_logits = F.interpolate(aux_logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            aux_loss = self._ce_loss(aux_logits, masks) * 0.4
            total += aux_loss

        loss_items = torch.stack([ce_loss, dice_loss, aux_loss]).detach()
        return total * preds.shape[0], loss_items


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            packed_targets = targets[:, 1:].clone()
            packed_targets[:, 1:5].mul_(scale_tensor)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = packed_targets
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
