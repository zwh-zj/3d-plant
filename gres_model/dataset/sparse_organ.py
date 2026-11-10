"""Sparse-organ sampling weights and ROI masks (unit-testable, no CUDA)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def parse_sparse_organ_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize sparse_organ config; None/empty means disabled."""
    defaults = {
        "enabled": False,
        "sparse_point_threshold": 2000,
        "roi_expand_ratio": 1.5,
        "roi_min_edge_ratio": 0.05,
        "sampler_weighting": False,
        "weight_min": 1.0,
        "weight_max": 8.0,
        "point_ref": 10000.0,
        # Whether val/test/viz use the same ROI as train (avoid train-eval mismatch)
        "eval_roi": False,
        # Train: duplicate packs below the point_count threshold (train only)
        "oversample_factor": 1,
        "oversample_threshold": None,
    }
    if not cfg:
        return dict(defaults)
    # gorilla Config and dict are both supported
    if hasattr(cfg, "items") and not isinstance(cfg, dict):
        cfg = {k: cfg[k] for k in cfg}
    out = dict(defaults)
    out.update(dict(cfg))
    out["enabled"] = bool(out["enabled"])
    out["sampler_weighting"] = bool(out["sampler_weighting"])
    out["sparse_point_threshold"] = int(out["sparse_point_threshold"])
    out["roi_expand_ratio"] = float(out["roi_expand_ratio"])
    out["roi_min_edge_ratio"] = float(out["roi_min_edge_ratio"])
    out["weight_min"] = float(out["weight_min"])
    out["weight_max"] = float(out["weight_max"])
    out["point_ref"] = float(out["point_ref"])
    out["eval_roi"] = bool(out["eval_roi"])
    out["oversample_factor"] = max(1, int(out["oversample_factor"]))
    ost = out["oversample_threshold"]
    out["oversample_threshold"] = (
        int(ost) if ost is not None else int(out["sparse_point_threshold"])
    )
    return out


def clip_sparse_weight(
    point_count: float,
    weight_min: float = 1.0,
    weight_max: float = 8.0,
    point_ref: float = 10000.0,
) -> float:
    """
    w = clip(sqrt(point_ref / max(p,1)), weight_min, weight_max)。

    With default point_ref=10000: p≈10000 -> 1, p≈156 -> 8 (matches the spec numbers).
    Note: a literal 1/sqrt(p) with weight_min=1 would clamp every weight to 1; this follows the intent.
    """
    p = max(float(point_count), 1.0)
    w = math.sqrt(float(point_ref) / p)
    return float(min(weight_max, max(weight_min, w)))


def build_roi_keep_mask(
    xyz,
    must_keep,
    expand_ratio: float = 1.5,
    min_edge_ratio: float = 0.05,
):
    """
    Expand the target-point AABB and return a bool mask aligned with xyz.

    Returns all False when there are no target points (caller should fall back to full-scene sampling).
    """
    import numpy as np

    xyz = np.asarray(xyz)
    must_keep = np.asarray(must_keep, dtype=bool)
    n = xyz.shape[0]
    if n == 0 or not must_keep.any():
        return np.zeros(n, dtype=bool)

    tgt = xyz[must_keep]
    lo = tgt.min(axis=0)
    hi = tgt.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)

    scene_span = xyz.max(axis=0) - xyz.min(axis=0)
    scene_diag = float(np.linalg.norm(scene_span)) + 1e-6
    min_half = 0.5 * min_edge_ratio * scene_diag

    half = np.maximum(half * expand_ratio, min_half)
    # For a single target point half may be 0; the floor still expands symmetrically
    lo_e = center - half
    hi_e = center + half

    inside = np.all((xyz >= lo_e) & (xyz <= hi_e), axis=1)
    return inside
