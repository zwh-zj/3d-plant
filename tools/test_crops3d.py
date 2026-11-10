"""Official Crops3D test eval (run once after training)."""

import argparse
import json
import os.path as osp
import sys
from collections import defaultdict
from pathlib import Path

# Prefer this repo's gres_model over editable MDIN-main (egg-link).
# Must force-front: easy-install.pth already lists both repos, so
# `not in sys.path` would skip insert and leave MDIN-main ahead.
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import gorilla
import numpy as np
import torch
from tqdm import tqdm

from gres_model.dataset import build_dataloader, build_dataset
from gres_model.model import MODEL
from gres_model.utils import get_root_logger
from gres_model.utils.segmentation_metrics import (
    acc_at_threshold,
    aggregate_group_metrics,
    format_group_line,
    format_overall_line,
    group_metrics_to_json,
    macro_mean,
    metrics_from_predict_result,
    overall_summary,
)


def _format_ann_id(ann_id):
    if isinstance(ann_id, (list, tuple)) and len(ann_id) == 1:
        return ann_id[0]
    return ann_id


def _print_worst_cases(sample_records, logger, topk=20):
    """Print lowest-IoU samples to help locate failures."""
    if not sample_records:
        return
    logger.info("-" * 80)
    logger.info(f"Worst {min(topk, len(sample_records))} samples by IoU:")
    for item in sorted(sample_records, key=lambda x: x["iou"])[:topk]:
        logger.info(
            f"  IoU={item['iou']:.4f}  spIoU={item['spiou']:.4f}  "
            f"crop={item['crop']}  organ={item['organ']}  "
            f"scene={item['scene_id']}  ann_id={item['ann_id']}"
        )


def get_args():
    parser = argparse.ArgumentParser("Crops3D Final Test")
    parser.add_argument("config", type=str, help="config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to best.pth")
    parser.add_argument("--out", type=str, default=None, help="result JSON output path")
    parser.add_argument(
        "--test_mode",
        type=str,
        default="canonical",
        choices=["canonical", "structured"],
        help="test set: canonical=template sentences; structured=structured text",
    )
    parser.add_argument(
        "--queries_json",
        type=str,
        default=None,
        help="override test queries_json; if set, ignore --test_mode",
    )
    parser.add_argument("--gpu_ids", type=int, default=[1], nargs="+", help="GPU ID")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="fixed 10k downsample seed (matches visualize --sample_seed); "
        "default cfg.train.seed; -1 disables the fixed seed (random downsample)",
    )
    return parser.parse_args()


def _resolve_sample_seed(cfg, args):
    """Align with visualize_crops3d / batch_visualize_test --sample_seed."""
    if args.sample_seed is not None:
        if args.sample_seed < 0:
            return None
        return int(args.sample_seed)
    return int(getattr(cfg.train, "seed", 1999))


def _resolve_queries_json(cfg, args):
    if args.queries_json:
        return args.queries_json, Path(args.queries_json).stem
    test_cfg = cfg.data.test
    if args.test_mode == "structured":
        structured_path = getattr(test_cfg, "queries_json_structured", None)
        if structured_path:
            return structured_path, "official_test_structured"
        default_path = str(Path(test_cfg.queries_json).with_name("crops3d_test_structured.json"))
        return default_path, "official_test_structured"
    return test_cfg.queries_json, "official_test"


def _print_results(
    crop_metrics,
    organ_metrics,
    sample_records,
    summary,
    acc_50,
    acc_25,
    logger,
    title="[official test set - final eval]",
    variant_metrics=None,
    miou=None,
):
    """Print test metrics (mIoU / P / R / F1)."""
    macro_crop_iou = macro_mean(crop_metrics, "iou")
    macro_organ_iou = macro_mean(organ_metrics, "iou")
    logger.info("=" * 80)
    logger.info(title)
    logger.info(format_overall_line(summary, acc_50, acc_25, prefix="Overall", miou=miou))
    logger.info(
        f"Macro crop mIoU: {macro_crop_iou:.4f}  Macro organ mIoU: {macro_organ_iou:.4f}  "
        f"Macro crop F1: {macro_mean(crop_metrics, 'f1'):.4f}  "
        f"Macro organ F1: {macro_mean(organ_metrics, 'f1'):.4f}"
    )
    if variant_metrics:
        logger.info("-" * 80)
        logger.info("Per text-variant metrics:")
        for variant, group in sorted(variant_metrics.items()):
            logger.info(format_group_line(variant, group, width=10))
    logger.info("-" * 80)
    logger.info("Per-crop metrics:")
    for crop in sorted(crop_metrics):
        logger.info(format_group_line(crop, crop_metrics[crop], width=12))
    logger.info("-" * 80)
    logger.info("Per-organ metrics:")
    for organ in sorted(organ_metrics, key=lambda x: -len(organ_metrics[x]["iou"])):
        logger.info(format_group_line(organ, organ_metrics[organ], width=16))
    _print_worst_cases(sample_records, logger)
    logger.info("=" * 80)


@torch.no_grad()
def main():
    args = get_args()
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_ids, num_gpu=args.num_gpus)

    cfg = gorilla.Config.fromfile(args.config)
    queries_json, split_name = _resolve_queries_json(cfg, args)
    sample_seed = _resolve_sample_seed(cfg, args)
    seed_tag = f"_seed{sample_seed}" if sample_seed is not None else "_seedRandom"
    logger = get_root_logger(
        log_file=args.checkpoint.replace(".pth", f"_{split_name}{seed_tag}.log")
    )
    logger.info(f"config    : {args.config}")
    logger.info(f"checkpoint: {args.checkpoint}")
    logger.info(f"test_mode : {args.test_mode}")
    logger.info(f"queries_json: {queries_json}")
    logger.info(
        f"sample_seed: {sample_seed if sample_seed is not None else 'None (random downsample)'}"
    )

    # load model
    model = MODEL(**cfg.model).cuda()
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)
    model.eval()

    test_cfg = cfg.data.test.copy()
    test_cfg.queries_json = queries_json
    dataset = build_dataset(test_cfg, logger)
    # Set before build_dataloader so DataLoader workers inherit after fork
    dataset._downsample_seed = sample_seed
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)

    pious, spious, meta_datas, sample_metrics = [], [], [], []
    progress_bar = tqdm(total=len(dataloader), desc="Test")
    for batch in dataloader:
        result = model(batch, mode="predict")
        pious.extend(result["piou"])
        spious.extend(result["spiou"])
        sample_metrics.extend(metrics_from_predict_result(result))
        meta_datas.extend(result["meta_datas"])
        progress_bar.update()
    progress_bar.close()

    pious = torch.stack(pious, dim=0).cpu().numpy()
    spious = torch.stack(spious, dim=0).cpu().numpy()

    summary = overall_summary(sample_metrics)
    miou = float(pious.mean())
    spmiou = float(spious.mean())
    acc_50 = acc_at_threshold(pious, 0.5)
    acc_25 = acc_at_threshold(pious, 0.25)

    crop_metrics = aggregate_group_metrics(sample_metrics, meta_datas, "crop")
    organ_metrics = aggregate_group_metrics(sample_metrics, meta_datas, "object_name")
    macro_crop_miou = macro_mean(crop_metrics, "iou")
    macro_organ_miou = macro_mean(organ_metrics, "iou")

    variant_groups = defaultdict(lambda: {k: [] for k in ("iou", "precision", "recall", "f1")})
    for idx, meta in enumerate(meta_datas):
        variant = meta.get("text_variant")
        if not variant:
            desc = meta.get("description", "")
            if desc.startswith("Locate the"):
                variant = "spatial"
            elif desc.startswith("Segment the"):
                variant = "relation"
            else:
                variant = "canonical"
        for metric in ("iou", "precision", "recall", "f1"):
            variant_groups[variant][metric].append(sample_metrics[idx][metric])
    variant_metrics = {
        k: {m: np.asarray(v, dtype=np.float64) for m, v in metrics.items()}
        for k, metrics in variant_groups.items()
    }

    sample_records = []
    for idx, meta in enumerate(meta_datas):
        sample_records.append(
            {
                "iou": float(sample_metrics[idx]["iou"]),
                "precision": float(sample_metrics[idx]["precision"]),
                "recall": float(sample_metrics[idx]["recall"]),
                "f1": float(sample_metrics[idx]["f1"]),
                "spiou": float(spious[idx]),
                "crop": meta.get("crop", "unknown"),
                "organ": meta.get("object_name", "unknown"),
                "scene_id": meta.get("scene_id", "?"),
                "ann_id": _format_ann_id(meta.get("ann_id", "?")),
                "description": meta.get("description", ""),
                "text_variant": meta.get("text_variant", "unknown"),
            }
        )

    title = (
        "[official test - structured text]"
        if split_name == "official_test_structured"
        else "[official test - canonical]"
    )
    _print_results(
        crop_metrics,
        organ_metrics,
        sample_records,
        summary,
        acc_50,
        acc_25,
        logger,
        title=title,
        variant_metrics=variant_metrics if len(variant_metrics) > 1 else None,
        miou=miou,
    )

    # save result JSON
    default_suffix = (
        f"_test_structured_results{seed_tag}.json"
        if split_name == "official_test_structured"
        else f"_test_results{seed_tag}.json"
    )
    out_path = args.out or args.checkpoint.replace(".pth", default_suffix)
    results = {
        "checkpoint": args.checkpoint,
        "split": split_name,
        "test_mode": args.test_mode,
        "queries_json": queries_json,
        "sample_seed": sample_seed,
        "num_samples": len(pious),
        "mIoU": miou,
        "IoU": miou,
        "Precision": summary["precision"],
        "Recall": summary["recall"],
        "F1": summary["f1"],
        "spmIoU": spmiou,
        "Acc_0.5": acc_50,
        "Acc_0.25": acc_25,
        "macro_crop_mIoU": macro_crop_miou,
        "macro_organ_mIoU": macro_organ_miou,
        "macro_crop_IoU": macro_crop_miou,
        "macro_organ_IoU": macro_organ_miou,
        "macro_crop_F1": macro_mean(crop_metrics, "f1"),
        "macro_organ_F1": macro_mean(organ_metrics, "f1"),
        "per_text_variant": group_metrics_to_json(variant_metrics),
        "per_crop": group_metrics_to_json(crop_metrics),
        "per_organ": group_metrics_to_json(organ_metrics),
        "per_text_variant_mIoU": {
            k: float(v["iou"].mean()) for k, v in variant_metrics.items()
        },
        "per_text_variant_Acc_0.5": {
            k: acc_at_threshold(v["iou"], 0.5) for k, v in variant_metrics.items()
        },
        "per_text_variant_Acc_0.25": {
            k: acc_at_threshold(v["iou"], 0.25) for k, v in variant_metrics.items()
        },
        "per_crop_mIoU": {k: float(v["iou"].mean()) for k, v in crop_metrics.items()},
        "per_crop_Acc_0.5": {
            k: acc_at_threshold(v["iou"], 0.5) for k, v in crop_metrics.items()
        },
        "per_crop_Acc_0.25": {
            k: acc_at_threshold(v["iou"], 0.25) for k, v in crop_metrics.items()
        },
        "per_organ_mIoU": {k: float(v["iou"].mean()) for k, v in organ_metrics.items()},
        "per_organ_Acc_0.5": {
            k: acc_at_threshold(v["iou"], 0.5) for k, v in organ_metrics.items()
        },
        "per_organ_Acc_0.25": {
            k: acc_at_threshold(v["iou"], 0.25) for k, v in organ_metrics.items()
        },
        "worst_cases": sorted(sample_records, key=lambda x: x["iou"])[:20],
        "samples": sample_records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
