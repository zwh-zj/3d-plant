"""Plot training curves from TensorBoard events or logs after training."""

import glob
import json
import logging
import os
import os.path as osp
import re

import numpy as np

logger = logging.getLogger(__name__)

# Match val mIoU log lines (exclude spmIoU)
_VAL_MIOU_RE = re.compile(
    r"INFO - mIoU: ([\d.]+)\s+Acc@0\.5: ([\d.]+)\s+Acc@0\.25: ([\d.]+)"
)


def _setup_chinese_font():
    """Prefer a CJK font; matplotlib falls back if none is installed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _load_scalars_from_tensorboard(log_dir):
    """Load TensorBoard scalar curves."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir)
    ea.Reload()
    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        scalars[tag] = {
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        }
    return scalars


def _load_val_from_log(log_file):
    """Parse per-epoch val metrics from a training log."""
    if not log_file or not osp.isfile(log_file):
        return None

    epochs, miou, acc50, acc25 = [], [], [], []
    epoch_idx = 0
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _VAL_MIOU_RE.search(line)
            if not match:
                continue
            epoch_idx += 1
            epochs.append(epoch_idx)
            miou.append(float(match.group(1)))
            acc50.append(float(match.group(2)))
            acc25.append(float(match.group(3)))

    if not epochs:
        return None

    return {
        "epoch": epochs,
        "miou": miou,
        "acc50": acc50,
        "acc25": acc25,
    }


def _plot_series(ax, steps, values, label, color=None):
    """Plot one series on a subplot."""
    if not steps:
        return
    ax.plot(steps, values, label=label, linewidth=1.5, color=color)


def _is_train_loss_tag(tag):
    """Select training-loss scalars from TensorBoard."""
    return tag.startswith("train/") and "loss" in tag.lower()


def _loss_display_name(tag):
    """Turn a TensorBoard tag into a plot title / filename."""
    return tag.replace("train/", "")


def _safe_filename(name):
    """Make a filesystem-safe name fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _save_loss_curves(scalars, output_dir):
    """Save the total-loss curve and each individual loss curve."""
    import matplotlib.pyplot as plt

    loss_tags = sorted(tag for tag in scalars if _is_train_loss_tag(tag))
    if not loss_tags:
        return {}

    paths = {}
    loss_dir = osp.join(output_dir, "loss_curves")
    os.makedirs(loss_dir, exist_ok=True)

    if "train/loss" in scalars:
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_series(
            ax,
            scalars["train/loss"]["step"],
            scalars["train/loss"]["value"],
            "total loss",
            "#d62728",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Total Training Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        total_path = osp.join(output_dir, "loss_total_curve.png")
        plt.savefig(total_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["loss_total_curve"] = total_path

    for tag in loss_tags:
        name = _loss_display_name(tag)
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_series(ax, scalars[tag]["step"], scalars[tag]["value"], name, "#1f77b4")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path = osp.join(loss_dir, f"{_safe_filename(name)}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Overview: all losses on one figure for trend comparison.
    fig, ax = plt.subplots(figsize=(14, 7))
    for tag in loss_tags:
        name = _loss_display_name(tag)
        _plot_series(ax, scalars[tag]["step"], scalars[tag]["value"], name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("All Training Loss Curves")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    all_path = osp.join(output_dir, "loss_all_curves.png")
    plt.savefig(all_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    paths["loss_all_curves"] = all_path
    paths["loss_curves_dir"] = loss_dir
    paths["loss_tags"] = [_loss_display_name(tag) for tag in loss_tags]
    return paths


def save_training_curves(work_dir, log_file=None, output_dir=None):
    """
    Write training-curve PNGs and a JSON summary.

    Args:
        work_dir: experiment dir (contains events.out.tfevents.*)
        log_file: optional training .log used if TensorBoard is missing
        output_dir: optional output dir, defaults to work_dir

    Returns:
        dict: output paths and best val-metric summary
    """
    output_dir = output_dir or work_dir
    os.makedirs(output_dir, exist_ok=True)

    _setup_chinese_font()
    import matplotlib.pyplot as plt

    scalars = {}
    if glob.glob(osp.join(work_dir, "events.out.tfevents.*")):
        try:
            scalars = _load_scalars_from_tensorboard(work_dir)
        except Exception as exc:
            logger.warning("Failed to read TensorBoard events: %s", exc)

    log_val = _load_val_from_log(log_file)
    exp_name = osp.basename(osp.abspath(work_dir))

    # ---------- figure 1: val metrics + train loss + lr ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves — {exp_name}", fontsize=14)

    ax = axes[0, 0]
    if "val/mIOU" in scalars:
        _plot_series(ax, scalars["val/mIOU"]["step"], scalars["val/mIOU"]["value"], "mIoU", "#1f77b4")
    elif log_val:
        _plot_series(ax, log_val["epoch"], log_val["miou"], "mIoU", "#1f77b4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title("Val mIoU")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    ax = axes[0, 1]
    if "val/Acc_50" in scalars:
        _plot_series(ax, scalars["val/Acc_50"]["step"], scalars["val/Acc_50"]["value"], "Acc@0.5", "#ff7f0e")
        _plot_series(ax, scalars["val/Acc_25"]["step"], scalars["val/Acc_25"]["value"], "Acc@0.25", "#2ca02c")
    elif log_val:
        _plot_series(ax, log_val["epoch"], log_val["acc50"], "Acc@0.5", "#ff7f0e")
        _plot_series(ax, log_val["epoch"], log_val["acc25"], "Acc@0.25", "#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Val Accuracy")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    ax = axes[1, 0]
    for key, color, label in [
        ("train/loss", "#d62728", "loss"),
        ("train/sample_loss", "#9467bd", "sample_loss"),
        ("train/indi_loss", "#8c564b", "indi_loss"),
        ("train/mask_dice_loss", "#2ca02c", "mask_dice"),
    ]:
        if key in scalars:
            _plot_series(ax, scalars[key]["step"], scalars[key]["value"], label, color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Train Loss (sampled)")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    if "train/learning_rate" in scalars:
        _plot_series(
            ax,
            scalars["train/learning_rate"]["step"],
            scalars["train/learning_rate"]["value"],
            "LR",
            "#17becf",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("LR")
    ax.set_title("Learning Rate")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    plt.tight_layout()
    curve_main = osp.join(output_dir, "training_curves.png")
    plt.savefig(curve_main, dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- figure 2: per-crop / per-organ val mIoU ----------
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    crop_tags = sorted(k for k in scalars if k.startswith("val/crop_") and k.endswith("_mIOU"))
    organ_tags = sorted(k for k in scalars if k.startswith("val/organ_") and k.endswith("_mIOU"))

    for tag in crop_tags:
        name = tag.replace("val/crop_", "").replace("_mIOU", "")
        _plot_series(ax2, scalars[tag]["step"], scalars[tag]["value"], f"crop-{name}")

    for tag in organ_tags[:10]:
        name = tag.replace("val/organ_", "").replace("_mIOU", "")
        _plot_series(ax2, scalars[tag]["step"], scalars[tag]["value"], f"organ-{name}")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.set_title("Val mIoU by Crop / Organ")
    ax2.grid(True, alpha=0.3)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    curve_detail = osp.join(output_dir, "training_curves_detail.png")
    plt.savefig(curve_detail, dpi=150, bbox_inches="tight")
    plt.close()

    loss_curve_paths = _save_loss_curves(scalars, output_dir)

    # ---------- JSON summary ----------
    summary = {"experiment": exp_name}
    val_miou = None
    if "val/mIOU" in scalars and scalars["val/mIOU"]["value"]:
        val_miou = scalars["val/mIOU"]["value"]
        best_idx = int(np.argmax(val_miou))
        summary.update(
            {
                "best_epoch": int(scalars["val/mIOU"]["step"][best_idx]),
                "best_val_miou": float(val_miou[best_idx]),
                "latest_val_miou": float(val_miou[-1]),
            }
        )
    elif log_val and log_val["miou"]:
        best_idx = int(np.argmax(log_val["miou"]))
        summary.update(
            {
                "best_epoch": int(log_val["epoch"][best_idx]),
                "best_val_miou": float(log_val["miou"][best_idx]),
                "best_val_acc50": float(log_val["acc50"][best_idx]),
                "best_val_acc25": float(log_val["acc25"][best_idx]),
                "latest_val_miou": float(log_val["miou"][-1]),
            }
        )

    summary_path = osp.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "training_curves": curve_main,
        "training_curves_detail": curve_detail,
        "training_summary": summary_path,
        "summary": summary,
        **loss_curve_paths,
    }
