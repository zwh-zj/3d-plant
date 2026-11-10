"""Point-level IoU / Precision / Recall / F1 (0.5 threshold, same as predict)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

METRIC_NAMES = ("iou", "precision", "recall", "f1")


def compute_mask_metrics(pred_prob, gt_mask, threshold: float = 0.5) -> dict[str, float]:
    """Per-sample metrics from predicted probabilities and a binary GT mask."""
    if not torch.is_tensor(pred_prob):
        pred_prob = torch.as_tensor(pred_prob, dtype=torch.float32)
    if not torch.is_tensor(gt_mask):
        gt_mask = torch.as_tensor(gt_mask, dtype=torch.float32)

    pred = (pred_prob.reshape(-1) >= threshold).float()
    gt = (gt_mask.reshape(-1) > 0.5).float()

    tp = (pred * gt).sum()
    fp = (pred * (1.0 - gt)).sum()
    fn = ((1.0 - pred) * gt).sum()

    iou = (tp / (tp + fp + fn + 1e-6)).item()
    precision = (tp / (tp + fp + 1e-6)).item()
    recall = (tp / (tp + fn + 1e-6)).item()
    f1 = (
        2.0 * precision * recall / (precision + recall + 1e-6)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def metrics_from_predict_result(result) -> list[dict[str, float]]:
    """Per-sample mask metrics from model(..., mode='predict') output."""
    pred_pmasks = result.get("pred_pmask") or []
    gt_pmasks = result.get("gt_pmask") or []
    if len(pred_pmasks) != len(gt_pmasks):
        raise ValueError(
            f"pred_pmask ({len(pred_pmasks)}) and gt_pmask ({len(gt_pmasks)}) length mismatch"
        )
    return [compute_mask_metrics(pred, gt) for pred, gt in zip(pred_pmasks, gt_pmasks)]


def stack_sample_metrics(sample_metrics: list[dict[str, float]]) -> dict[str, np.ndarray]:
    if not sample_metrics:
        empty = {k: np.array([], dtype=np.float64) for k in METRIC_NAMES}
        return empty
    return {k: np.array([m[k] for m in sample_metrics], dtype=np.float64) for k in METRIC_NAMES}


def aggregate_group_metrics(
    sample_metrics: list[dict[str, float]],
    meta_datas: list[dict],
    group_field: str,
) -> dict[str, dict[str, np.ndarray]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in METRIC_NAMES}
    )
    for idx, meta in enumerate(meta_datas):
        key = meta.get(group_field, "unknown")
        for metric in METRIC_NAMES:
            groups[key][metric].append(sample_metrics[idx][metric])
    return {
        name: {metric: np.asarray(values, dtype=np.float64) for metric, values in metrics.items()}
        for name, metrics in groups.items()
    }


def macro_mean(group_metrics: dict[str, dict[str, np.ndarray]], metric: str) -> float:
    if not group_metrics:
        return 0.0
    return float(np.mean([arr[metric].mean() for arr in group_metrics.values()]))


def overall_summary(sample_metrics: list[dict[str, float]]) -> dict[str, float]:
    stacked = stack_sample_metrics(sample_metrics)
    return {metric: float(stacked[metric].mean()) if len(stacked[metric]) else 0.0 for metric in METRIC_NAMES}


def acc_at_threshold(iou_arr: np.ndarray, threshold: float) -> float:
    if len(iou_arr) == 0:
        return 0.0
    return float((iou_arr > threshold).mean())


def format_overall_line(
    summary: dict[str, float],
    acc_50: float,
    acc_25: float,
    prefix: str = "Overall",
    miou: float | None = None,
) -> str:
    miou_val = miou if miou is not None else summary["iou"]
    return (
        f"{prefix} mIoU: {miou_val:.4f}  "
        f"P: {summary['precision']:.4f}  R: {summary['recall']:.4f}  F1: {summary['f1']:.4f}  "
        f"Acc@0.5: {acc_50:.4f}  Acc@0.25: {acc_25:.4f}"
    )


def format_group_line(name: str, group: dict[str, np.ndarray], width: int = 12) -> str:
    n = len(group["iou"])
    miou = float(group["iou"].mean())
    return (
        f"  {name:{width}s}: mIoU={miou:.4f}  "
        f"P={group['precision'].mean():.4f}  R={group['recall'].mean():.4f}  "
        f"F1={group['f1'].mean():.4f}  "
        f"Acc@0.5={acc_at_threshold(group['iou'], 0.5):.4f}  "
        f"Acc@0.25={acc_at_threshold(group['iou'], 0.25):.4f}  n={n}"
    )


def group_metrics_to_json(group_metrics: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    out = {}
    for name, metrics in group_metrics.items():
        miou = float(metrics["iou"].mean())
        out[name] = {
            "mIoU": miou,
            "iou": miou,
            "precision": float(metrics["precision"].mean()),
            "recall": float(metrics["recall"].mean()),
            "f1": float(metrics["f1"].mean()),
            "Acc_0.5": acc_at_threshold(metrics["iou"], 0.5),
            "Acc_0.25": acc_at_threshold(metrics["iou"], 0.25),
            "n": int(len(metrics["iou"])),
        }
    return out
