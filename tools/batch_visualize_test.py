"""Batch-export test visualizations (gt/pred/compare.pcd + png).

The model loads once. Default outputs match single-run visualize_crops3d.

Example (canonical templates, 337 items):
  python -u tools/batch_visualize_test.py configs/maize_only_full.yaml \\
    --checkpoint exps/maize_only_full/20260713_011340/best.pth \\
    --export_dir exps/maize_only_full/20260713_011340/vis_test_all_pcd \\
    --split test --gpu_ids 1

Example (structured text, 1029 items):
  python -u tools/batch_visualize_test.py configs/maize_only_full.yaml \\
    --checkpoint exps/maize_only_full/20260713_011340/best.pth \\
    --export_dir exps/maize_only_full/20260713_011340/vis_test_structured_pcd \\
    --queries_json data/crops3d/single_maize_full/crops3d_test_structured.json \\
    --split test --gpu_ids 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gorilla
import torch
from tqdm import tqdm

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import visualize_crops3d as vis


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Batch export test-set visualization PCDs")
    parser.add_argument("config", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--export_dir",
        type=str,
        required=True,
        help="output root; one subdirectory per query",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--queries_json",
        type=str,
        default=None,
        help="override queries_json; structured e.g. data/crops3d/single_maize_full/crops3d_test_structured.json",
    )
    parser.add_argument(
        "--text_variant",
        type=str,
        default=None,
        choices=["canonical", "spatial", "relation"],
        help="export one text kind only (structured lists); default all",
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
    parser.add_argument(
        "--with_png",
        action="store_true",
        default=True,
        help="export gt/pred/compare.png (same as visualize_crops3d; default on)",
    )
    parser.add_argument(
        "--no_png",
        action="store_true",
        help="export PCD only, no PNG",
    )
    parser.add_argument("--limit", type=int, default=None, help="export first N items (debug)")
    parser.add_argument(
        "--organ",
        type=str,
        default=None,
        help="export one organ only; default all",
    )
    args = parser.parse_args()

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

    # collect (ds_idx, query); one item per pack when lang_num_max=1
    jobs = []
    for ds_idx, pack in enumerate(dataset.scene_inputs):
        for query in pack:
            if args.organ and query.get("object_name") != args.organ:
                continue
            if args.text_variant and query.get("text_variant") != args.text_variant:
                continue
            jobs.append((ds_idx, query))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    export_root = Path(args.export_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    print(
        f"batch vis: split={args.split} queries_json={queries_json} n={len(jobs)} "
        f"text_variant={args.text_variant or 'all'} "
        f"eval_roi={sparse_cfg['eval_roi']} full_cloud={args.full_cloud} "
        f"decor={decor_points} -> {export_root}"
    )

    model = MODEL(**cfg.model).cuda()
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)
    model.eval()

    view = vis.VIEW_PRESETS["front"]
    manifest = []

    for ds_idx, query in tqdm(jobs, desc="vis_test"):
        scene_id = query["scene_id"]
        organ = query.get("object_name", "organ")
        ann_id = query.get("ann_id")
        text_variant = query.get("text_variant") or "canonical"
        # structured: many texts per scene/organ; include variant in the dir name
        out_dir = export_root / f"{scene_id}_{organ}_{text_variant}_ann{ann_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        eval_sample = vis._load_sample(dataset, ds_idx, seed=args.sample_seed)
        eval_batch, eval_result = vis._run_predict(model, dataset, eval_sample)
        piou = float(eval_result["piou"][0])
        eval_gt_b, eval_pr_b = vis._masks_from_tensors(
            eval_batch["gt_pmasks"][0], eval_result["pred_pmask"][0]
        )
        _, _, _, target_recall = vis._build_compare_from_binary(eval_gt_b, eval_pr_b)
        n_eval = len(eval_batch["coords_float"])
        n_target_eval = int(eval_gt_b.sum())
        n_full = None
        n_decor_pts = 0

        if args.full_cloud:
            xyz, gt_b, pr_b, _, n_eval_pts, n_full, _ = vis.build_display_full_cloud(
                dataset, scene_id, query, eval_batch, eval_result,
                args.sample_seed, max_display=args.max_display_points,
            )
        elif decor_points > 0:
            eval_rgb = vis._rgb_from_sample(eval_sample)
            xyz, gt_b, pr_b, _, n_eval_pts, n_decor_pts = vis.build_display_with_decor(
                dataset, scene_id, query, eval_batch, eval_result,
                eval_rgb, max_n, decor_points, args.sample_seed,
            )
        else:
            eval_rgb = vis._rgb_from_sample(eval_sample)
            xyz, gt_b, pr_b, _, n_eval_pts, n_decor_pts = vis.build_display_eval_only(
                eval_batch, eval_gt_b, eval_pr_b, eval_rgb,
            )

        gt_colors, pred_colors, compare, _ = vis._build_compare_from_binary(
            gt_b, pr_b, text_variant=text_variant,
        )
        vis.save_pcd(xyz, gt_colors, out_dir / "gt.pcd")
        vis.save_pcd(xyz, pred_colors, out_dir / "pred.pcd")
        vis.save_pcd(xyz, compare, out_dir / "compare.pcd")
        # same as tools/visualize_crops3d.py: gt / pred / compare
        if args.with_png and not args.no_png:
            vis.render_png_headless(xyz, gt_colors, out_dir / "gt.png", view)
            vis.render_png_headless(xyz, pred_colors, out_dir / "pred.png", view)
            vis.render_png_headless(xyz, compare, out_dir / "compare.png", view)

        run_info = {
            "scene_id": scene_id,
            "organ": organ,
            "ann_id": ann_id,
            "text_variant": text_variant,
            "split": args.split,
            "queries_json": queries_json,
            "iou": piou,
            "target_recall": target_recall,
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
            "description": query.get("description"),
            "gt_pcd": str(out_dir / "gt.pcd"),
            "pred_pcd": str(out_dir / "pred.pcd"),
            "compare_pcd": str(out_dir / "compare.pcd"),
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
                "text_variant_filter": args.text_variant,
                "eval_roi": sparse_cfg["eval_roi"],
                "num_samples": len(manifest),
                "samples": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    # also write a path list for copy/mount
    list_path = export_root / "pcd_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(item["compare_pcd"] + "\n")
            f.write(item["gt_pcd"] + "\n")
            f.write(item["pred_pcd"] + "\n")
    print(f"done: {len(manifest)} samples")
    print(f"manifest: {manif_path}")
    print(f"pcd_list: {list_path}")


if __name__ == "__main__":
    main()
