"""Mask BCE helpers (kept outside gres_model.model to avoid heavy imports in unit tests)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_bce_with_neg_weight(
    pred_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """BCE-with-logits; neg_weight>1 upweights negatives (including FPs) to curb over-prediction."""
    tgt = tgt_mask.float()
    if abs(float(neg_weight) - 1.0) < 1e-8:
        return F.binary_cross_entropy_with_logits(pred_mask, tgt)
    weight = torch.ones_like(tgt)
    weight = torch.where(tgt < 0.5, weight * float(neg_weight), weight)
    return F.binary_cross_entropy_with_logits(pred_mask, tgt, weight=weight)
