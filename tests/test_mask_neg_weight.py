"""Mask BCE negative weighting: higher neg_weight should raise false-positive cost."""
import torch
import torch.nn.functional as F

from gres_model.utils.mask_bce import mask_bce_with_neg_weight


def test_neg_weight_one_matches_plain_bce():
    pred = torch.tensor([[2.0, -2.0, 3.0]])
    tgt = torch.tensor([[1.0, 0.0, 0.0]])
    got = mask_bce_with_neg_weight(pred, tgt, neg_weight=1.0)
    ref = F.binary_cross_entropy_with_logits(pred, tgt)
    assert torch.allclose(got, ref)


def test_higher_neg_weight_raises_loss_on_false_positives():
    # High logit on GT=0 is a clear FP; raising neg_weight should increase loss.
    pred = torch.tensor([[2.0, -2.0, 4.0]])
    tgt = torch.tensor([[1.0, 0.0, 0.0]])
    base = mask_bce_with_neg_weight(pred, tgt, neg_weight=1.0)
    heavy = mask_bce_with_neg_weight(pred, tgt, neg_weight=3.0)
    assert float(heavy) > float(base)
