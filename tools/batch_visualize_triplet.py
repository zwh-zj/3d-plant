"""Export triplets grouped by (scene_id, organ, ann_id): shared GT + three text preds.

- scene_gray: scheme-A luminance gray (same points as pred)
- gt / pred / compare: gray-180 + green GT; pred uses Okabe-Ito variant colors
- dirs: one folder per plant `{scene_id}_{organ}_ann{id}/`
- keep run_info.json (IoU and paths for the three queries)

Example (group-A structured test):
  python -u tools/batch_visualize_triplet.py configs/crops3d_full_baseline_fpfix_v15.yaml \\
    --checkpoint exps/crops3d_full_baseline_fpfix_v15/20260720_182330/best.pth \\
    --export_dir exps/crops3d_full_baseline_fpfix_v15/20260720_182330/vis_test_structured_triplet \\
    --queries_json data/crops3d/all_crops_full_fpfix/crops3d_test_structured.json \\
    --split test --gpu_ids 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import gorilla
import numpy as np
import torch
from tqdm import tqdm

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import visualize_crops3d as vis


def _group_key(query):
    """Triplet canonical/spatial/relation share canonical_sample_id; ann_id differs."""
    return (
        query["scene_id"],
        query.get("object_name", "organ"),
        query.get("canonical_sample_id") or query.get("sample_id"),
    )


def _canonical_ann_id(variant_map, text_variants):
    if "canonical" in variant_map:
        return variant_map["canonical"][1].get("ann_id")
    ref_variant = text_variants[0]
    return variant_map[ref_variant][1].get("ann_id")


def _collect_variant_jobs(
    dataset, text_variants, organ_filter=None, scene_filter=None,
):
    """Group by (scene, organ, canonical_sample_id); each group maps variant -> (ds_idx, query)."""
    groups = defaultdict(dict)
    for ds_idx, pack in enumerate(dataset.scene_inputs):
        for query in pack:
            if scene_filter and query.get("scene_id") != scene_filter:
                continue
            if organ_filter and query.get("object_name") != organ_filter:
                continue
            variant = query.get("text_variant") or "canonical"
            if variant not in text_variants:
                continue
            key = _group_key(query)
            groups[key][variant] = (ds_idx, query)
    return groups


def _build_shared_display(
    dataset, scene_id, ref_query, eval_sample, eval_batch, eval_result,
    args, max_n, decor_points,
):
    eval_gt_b, eval_pr_b = vis._masks_from_tensors(
        eval_batch["gt_pmasks"][0], eval_result["pred_pmask"][0]
    )
    eval_rgb = vis._rgb_from_sample(eval_sample)
    n_full = None
    n_decor_pts = 0

    if args.full_cloud:
        xyz, gt_b, _pr_b, rgb_disp, n_eval_pts, n_full, _ = vis.build_display_full_cloud(
            dataset, scene_id, ref_query, eval_batch, eval_result,
            args.sample_seed, max_display=args.max_display_points,
        )
    elif decor_points > 0:
        xyz, gt_b, _pr_b, rgb_disp, n_eval_pts, n_decor_pts = vis.build_display_with_decor(
            dataset, scene_id, ref_query, eval_batch, eval_result,
            eval_rgb, max_n, decor_points, args.sample_seed,
        )
    else:
        xyz, gt_b, _pr_b, rgb_disp, n_eval_pts, n_decor_pts = vis.build_display_eval_only(
            eval_batch, eval_gt_b, eval_pr_b, eval_rgb,
        )
    return xyz, gt_b, rgb_disp, n_full, n_eval_pts, n_decor_pts, eval_gt_b


def _variant_pred_on_shared(
    dataset, scene_id, query, eval_sample, eval_batch, eval_result,
    args, max_n, decor_points, xyz_ref, gt_ref,
):
    """Map this variant pred mask onto the shared xyz."""
    eval_gt_b, eval_pr_b = vis._masks_from_tensors(
        eval_batch["gt_pmasks"][0], eval_result["pred_pmask"][0]
    )
    eval_rgb = vis._rgb_from_sample(eval_sample)

    if args.full_cloud:
        xyz, gt_b, pr_b, rgb_disp, *_ = vis.build_display_full_cloud(
            dataset, scene_id, query, eval_batch, eval_result,
            args.sample_seed, max_display=args.max_display_points,
        )
    elif decor_points > 0:
        xyz, gt_b, pr_b, rgb_disp, *_ = vis.build_display_with_decor(
            dataset, scene_id, query, eval_batch, eval_result,
            eval_rgb, max_n, decor_points, args.sample_seed,
        )
    else:
        xyz, gt_b, pr_b, rgb_disp, *_ = vis.build_display_eval_only(
            eval_batch, eval_gt_b, eval_pr_b, eval_rgb,
        )

    if len(xyz) != len(xyz_ref):
        raise RuntimeError(
            f"display point count mismatch for {scene_id}: "
            f"shared={len(xyz_ref)} variant={len(xyz)}"
        )
    if not (gt_b == gt_ref).all():
        raise RuntimeError(f"GT mask mismatch for {scene_id} ann={query.get('ann_id')}")
    return pr_b, rgb_disp


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Batch export per-plant triplet visualization")
    parser.add_argument("config", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--export_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--queries_json", type=str, default=None)
    parser.add_argument(
        "--text_variants",
        type=str,
        nargs="+",
        default=["canonical", "spatial", "relation"],
        help="text variants to export; default group-A triplet",
    )
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--sample_seed", type=int, default=1999)
    parser.add_argument("--no_decor", action="store_true")
    parser.add_argument("--decor_points", type=int, default=None)
    parser.add_argument("--max_target_ratio", type=float, default=0.2)
    parser.add_argument("--eval_roi", action="store_true", default=None)
    parser.add_argument("--no_eval_roi", action="store_true")
    parser.add_argument("--full_cloud", action="store_true")
    parser.add_argument("--max_display_points", type=int, default=80000)
    parser.add_argument("--no_png", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--organ", type=str, default=None)
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="export one scene only, e.g. Cabbage_mvs_901_06",
    )
    args = parser.parse_args()

    text_variants = tuple(args.text_variants)
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_ids, num_gpu=args.num_gpus)
    cfg = gorilla.Config.fromfile(args.config)
    split_cfg = getattr(cfg.data, args.split)
    max_n = split_cfg.voxel_cfg.max_npoint
    eval_roi_override = False if args.no_eval_roi else args.eval_roi
    sparse_cfg = vis._resolve_sparse_organ_cfg(split_cfg, eval_roi_override=eval_roi_override)
    decor_points = 0 if args.no_decor else (
        args.decor_points if args.decor_points is not None else int(max_n * (1.0 - args.max_target_ratio))
    )
    queries_json = args.queries_json or split_cfg.queries_json

    from gres_model.dataset.crops3d_organ import Crops3DOrganDataset
    from gres_model.model import MODEL

    dataset = Crops3DOrganDataset(
        data_root=split_cfg.data_root,
        queries_json=queries_json,
        prefix=split_cfg.prefix,
        suffix=getattr(split_cfg, "suffix", "_refer.pth"),
        voxel_cfg=split_cfg.voxel_cfg,
        training=False,
        mode=split_cfg.mode,
        with_elastic=False,
        aug=False,
        use_xyz=split_cfg.use_xyz,
        max_des_len=split_cfg.max_des_len,
        lang_num_max=split_cfg.lang_num_max,
        sparse_organ=sparse_cfg,
    )

    groups = _collect_variant_jobs(
        dataset, set(text_variants),
        organ_filter=args.organ, scene_filter=args.scene_id,
    )
    keys = sorted(groups.keys())
    if args.limit is not None:
        keys = keys[: args.limit]

    export_root = Path(args.export_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    print(
        f"triplet vis: split={args.split} queries_json={queries_json} "
        f"plants={len(keys)} variants={text_variants} "
        f"eval_roi={sparse_cfg['eval_roi']} full_cloud={args.full_cloud} "
        f"decor={decor_points} -> {export_root}"
    )

    model = MODEL(**cfg.model).cuda()
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)
    model.eval()
    view = vis.VIEW_PRESETS["front"]
    manifest = []
    skipped = []

    for key in tqdm(keys, desc="vis_triplet"):
        scene_id, organ, _canonical_sample_id = key
        variant_map = groups[key]
        ann_id = _canonical_ann_id(variant_map, text_variants)
        missing = [v for v in text_variants if v not in variant_map]
        if missing:
            skipped.append({"key": key, "missing_variants": missing})
            continue

        ref_variant = text_variants[0]
        ref_ds_idx, ref_query = variant_map[ref_variant]
        ref_sample = vis._load_sample(dataset, ref_ds_idx, seed=args.sample_seed)
        ref_batch, ref_result = vis._run_predict(model, dataset, ref_sample)

        xyz, gt_b, rgb_disp, n_full, n_eval_pts, n_decor_pts, _ = _build_shared_display(
            dataset, scene_id, ref_query, ref_sample, ref_batch, ref_result,
            args, max_n, decor_points,
        )
        scene_gray = vis.rgb_to_grayscale_scheme_a(rgb_disp)
        gt_colors, _, _, _ = vis._build_compare_from_binary(
            gt_b, np.zeros(len(gt_b), dtype=bool),
        )

        out_dir = export_root / vis.triplet_folder_name(scene_id, organ, ann_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        vis.save_pcd(xyz, scene_gray, out_dir / "scene_gray.pcd")
        vis.save_pcd(xyz, gt_colors, out_dir / "gt.pcd")
        if not args.no_png:
            vis.render_png_headless(xyz, scene_gray, out_dir / "scene_gray.png", view)
            vis.render_png_headless(xyz, gt_colors, out_dir / "gt.png", view)

        variant_records = {}
        n_eval = len(ref_batch["coords_float"])
        n_target_eval = int(gt_b.sum())

        for variant in text_variants:
            ds_idx, query = variant_map[variant]
            sample = ref_sample if ds_idx == ref_ds_idx else vis._load_sample(
                dataset, ds_idx, seed=args.sample_seed
            )
            batch, result = (
                (ref_batch, ref_result)
                if ds_idx == ref_ds_idx
                else vis._run_predict(model, dataset, sample)
            )
            piou = float(result["piou"][0])
            _, eval_pr_b = vis._masks_from_tensors(
                batch["gt_pmasks"][0], result["pred_pmask"][0]
            )
            pr_b, rgb_check = _variant_pred_on_shared(
                dataset, scene_id, query, sample, batch, result,
                args, max_n, decor_points, xyz, gt_b,
            )
            if len(rgb_check) != len(rgb_disp):
                raise RuntimeError(
                    f"RGB count mismatch {scene_id} {variant}: "
                    f"{len(rgb_check)} vs {len(rgb_disp)}"
                )

            _, pred_colors, compare_colors, target_recall = vis._build_compare_from_binary(
                gt_b, pr_b, text_variant=variant,
            )
            pred_pcd = out_dir / f"pred_{variant}.pcd"
            compare_pcd = out_dir / f"compare_{variant}.pcd"
            pred_png = out_dir / f"pred_{variant}.png"
            compare_png = out_dir / f"compare_{variant}.png"

            vis.save_pcd(xyz, pred_colors, pred_pcd)
            vis.save_pcd(xyz, compare_colors, compare_pcd)
            if not args.no_png:
                vis.render_png_headless(xyz, pred_colors, pred_png, view)
                vis.render_png_headless(xyz, compare_colors, compare_png, view)

            variant_records[variant] = {
                "text_variant": variant,
                "description": query.get("description"),
                "iou": piou,
                "target_recall": target_recall,
                "pred_pcd": str(pred_pcd),
                "pred_png": str(pred_png),
                "compare_pcd": str(compare_pcd),
                "compare_png": str(compare_png),
            }

        run_info = {
            "scene_id": scene_id,
            "organ": organ,
            "ann_id": ann_id,
            "split": args.split,
            "queries_json": queries_json,
            "text_variants": list(text_variants),
            "checkpoint": args.checkpoint,
            "config": args.config,
            "sample_seed": args.sample_seed,
            "eval_roi": sparse_cfg["eval_roi"],
            "full_cloud": args.full_cloud,
            "scene_points": n_full,
            "eval_points": n_eval,
            "eval_target_points": n_target_eval,
            "display_points": len(xyz),
            "decor_points": n_decor_pts,
            "grayscale": "scheme_a_luminance_scene_gray",
            "gt_style": "classic_gray180_green",
            "pred_palette": "okabe_ito",
            "scene_gray_pcd": str(out_dir / "scene_gray.pcd"),
            "scene_gray_png": str(out_dir / "scene_gray.png"),
            "gt_pcd": str(out_dir / "gt.pcd"),
            "gt_png": str(out_dir / "gt.png"),
            "variants": variant_records,
        }
        with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2)
        manifest.append(run_info)

    manif_path = export_root / "manifest.json"
    with open(manif_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "config": args.config,
                "split": args.split,
                "queries_json": queries_json,
                "text_variants": list(text_variants),
                "layout": "triplet_per_plant",
                "eval_roi": sparse_cfg["eval_roi"],
                "num_plants": len(manifest),
                "skipped_incomplete": skipped,
                "samples": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"done: {len(manifest)} plants, skipped={len(skipped)}")
    print(f"manifest: {manif_path}")


if __name__ == "__main__":
    main()
