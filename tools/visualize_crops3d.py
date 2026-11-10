"""Visualize Crops3D segmentation: eval -> optional full-plant decor -> render_ball.so.

Examples:
  bash scripts/build_render_ball.sh   # compile render_ball.so once

  python tools/visualize_crops3d.py configs/maize_only_full.yaml \\
      --checkpoint exps/maize_only_full/20260711_111126/best.pth \\
      --scene_id Maize_1-106 --organ stem --split test \\
      --export_dir exps/.../vis --no_show

  python tools/show_3d.py --pcd_path exps/.../vis/20260711_181500_Maize_1-22_stem/gt.pcd
"""

import argparse
import json
import time
from types import SimpleNamespace
import ctypes as ct
import os
import sys
from pathlib import Path

import cv2
import gorilla
import numpy as np
import open3d as o3d
import torch

TOOLS = Path(__file__).resolve().parent

VIEW_PRESETS = {
    "oblique": SimpleNamespace(
        object_yaw_deg=-30.0, view_xangle_deg=-22.0, view_yangle_deg=0.0, zoom=1.05, pca_align=False,
    ),
    "front": SimpleNamespace(
        object_yaw_deg=-90.0, view_xangle_deg=-15.0, view_yangle_deg=0.0, zoom=1.0, pca_align=False,
    ),
}
VIEW_PRESETS["upright"] = VIEW_PRESETS["front"]
VIEW_PRESETS["default"] = VIEW_PRESETS["oblique"]

# Okabe-Ito colorblind-safe pred highlight (0-255)
OKABE_ITO_VARIANT_RGB = {
    "canonical": (0.0, 114.0, 178.0),
    "spatial": (213.0, 94.0, 0.0),
    "relation": (0.0, 158.0, 115.0),
    "attribute": (230.0, 159.0, 0.0),
    "functional": (204.0, 121.0, 167.0),
}

TEXT_VARIANT_ORDER = ("canonical", "spatial", "relation", "attribute", "functional")


def make_run_export_dir(base_dir, scene_id, organ, ann_id=None, flat=False):
    base = Path(base_dir) if base_dir else Path("exps/visualize")
    organ_part = organ or "query"
    if flat:
        out = base
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        folder = f"{ts}_{scene_id}_{organ_part}"
        if ann_id is not None:
            folder += f"_ann{ann_id}"
        out = base / folder
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_view(args):
    preset = VIEW_PRESETS[args.view_preset]
    view = SimpleNamespace(
        object_yaw_deg=args.object_yaw_deg if args.object_yaw_deg is not None else preset.object_yaw_deg,
        view_xangle_deg=args.view_xangle_deg if args.view_xangle_deg is not None else preset.view_xangle_deg,
        view_yangle_deg=args.view_yangle_deg if args.view_yangle_deg is not None else preset.view_yangle_deg,
        zoom=args.zoom if args.zoom is not None else preset.zoom,
        pca_align=getattr(preset, "pca_align", False),
    )
    print(
        f"view: preset={args.view_preset} "
        f"pca_align={view.pca_align} "
        f"object_yaw={view.object_yaw_deg:.1f}° "
        f"pitch={view.view_xangle_deg:.1f}° "
        f"yaw={view.view_yangle_deg:.1f}° zoom={view.zoom:.2f}"
    )
    return view


def compute_pca_object_rot(xyz):
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up).astype(np.float32)
    obj_rot = np.stack([right, up, forward], axis=1).astype(np.float32)

    c = xyz.astype(np.float64) - xyz.mean(axis=0)
    aligned = c.dot(obj_rot)
    high_z = c[:, 2] > np.median(c[:, 2])
    if aligned[high_z, 1].mean() < aligned[~high_z, 1].mean():
        flip = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
        obj_rot = obj_rot.dot(flip)
    return obj_rot


def _resolve_object_rot(view, xyz):
    if getattr(view, "pca_align", False):
        return compute_pca_object_rot(xyz)
    object_yaw = np.deg2rad(view.object_yaw_deg)
    return np.array([
        [np.cos(object_yaw), 0.0, -np.sin(object_yaw)],
        [0.0, 1.0, 0.0],
        [np.sin(object_yaw), 0.0, np.cos(object_yaw)],
    ], dtype=np.float32)


def _build_view_mats(view, xyz=None):
    view_xangle = np.deg2rad(view.view_xangle_deg)
    view_yangle = np.deg2rad(view.view_yangle_deg)
    obj_rot = _resolve_object_rot(view, xyz) if xyz is not None else _resolve_object_rot(view, np.zeros((1, 3)))
    view_rot = np.eye(3)
    view_rot = view_rot.dot(np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(view_xangle), -np.sin(view_xangle)],
        [0.0, np.sin(view_xangle), np.cos(view_xangle)],
    ]))
    view_rot = view_rot.dot(np.array([
        [np.cos(view_yangle), 0.0, -np.sin(view_yangle)],
        [0.0, 1.0, 0.0],
        [np.sin(view_yangle), 0.0, np.cos(view_yangle)],
    ]))
    view_rot *= view.zoom
    return obj_rot, view_rot


def _ensure_render_ball():
    so = TOOLS / "render_ball.so"
    if not so.is_file():
        raise FileNotFoundError(
            "missing tools/render_ball.so; run: bash scripts/build_render_ball.sh"
        )


def resolve_sample_index(dataset, scene_id, organ=None, ann_id=None):
    matches = []
    for i, group in enumerate(dataset.scene_inputs):
        for q in group:
            if q["scene_id"] != scene_id:
                continue
            if organ is not None and q.get("object_name") != organ:
                continue
            if ann_id is not None and q.get("ann_id") != ann_id:
                continue
            matches.append((i, q))
    if not matches:
        raise ValueError(
            f"scene_id not found: {scene_id!r}"
            + (f" organ={organ!r}" if organ else "")
            + (f" ann_id={ann_id!r}" if ann_id is not None else "")
        )
    if len(matches) > 1 and ann_id is None:
        print(
            f"note: matched {len(matches)} scene packs, using ann_id="
            f"{matches[0][1].get('ann_id')}"
        )
    return matches[0]


def _resolve_sparse_organ_cfg(split_cfg, eval_roi_override=None):
    """Parse sparse_organ from the split config; CLI may override eval_roi."""
    from gres_model.dataset.sparse_organ import parse_sparse_organ_cfg

    raw = getattr(split_cfg, "sparse_organ", None)
    cfg = parse_sparse_organ_cfg(raw)
    if eval_roi_override is not None:
        cfg["eval_roi"] = bool(eval_roi_override)
    return cfg


def _make_eval_sampler(dataset, seed):
    """Seeded downsample, aligned with dataset.compute_downsample_mask (including ROI)."""
    def _sampler(xyz, instance_label=None, object_ids=None, point_count=None):
        return dataset.compute_downsample_mask(
            xyz, instance_label, object_ids, point_count=point_count, seed=seed
        )
    return _sampler


def _rgb_norm_to_255(rgb):
    """Normalize refer.pth colors from [-1,1] or 0-255 to float32 0-255."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size == 0:
        return rgb.reshape(0, 3)
    if float(rgb.max()) <= 1.0 + 1e-3 and float(rgb.min()) >= -1.0 - 1e-3:
        return np.clip((rgb + 1.0) * 127.5, 0.0, 255.0)
    return np.clip(rgb, 0.0, 255.0)


def rgb_to_grayscale_scheme_a(rgb):
    """Scheme A: Y = 0.299R + 0.587G + 0.114B, output (N,3) float32 0-255."""
    rgb255 = _rgb_norm_to_255(rgb)
    y = 0.299 * rgb255[:, 0] + 0.587 * rgb255[:, 1] + 0.114 * rgb255[:, 2]
    y = y[:, np.newaxis]
    return np.concatenate([y, y, y], axis=1).astype(np.float32)


def _rgb_from_sample(sample):
    feat = sample[4]
    if torch.is_tensor(feat):
        return feat.detach().cpu().numpy().astype(np.float32)
    return np.asarray(feat, dtype=np.float32)


def build_pred_colors_on_gray(gray_colors, pr_b, text_variant):
    """Gray background + Okabe-Ito variant-colored pred highlight."""
    colors = gray_colors.copy()
    highlight = OKABE_ITO_VARIANT_RGB.get(
        text_variant, OKABE_ITO_VARIANT_RGB["canonical"]
    )
    colors[np.asarray(pr_b, dtype=bool)] = highlight
    return colors


def build_gt_colors_on_gray(gray_colors, gt_b, gt_rgb=(40.0, 200.0, 40.0)):
    colors = gray_colors.copy()
    colors[np.asarray(gt_b, dtype=bool)] = gt_rgb
    return colors


def build_compare_on_gray(gray_colors, gt_b, pr_b, text_variant):
    compare = gray_colors.copy()
    gt_b = np.asarray(gt_b, dtype=bool)
    pr_b = np.asarray(pr_b, dtype=bool)
    highlight = OKABE_ITO_VARIANT_RGB.get(
        text_variant, OKABE_ITO_VARIANT_RGB["canonical"]
    )
    compare[np.logical_and(gt_b, pr_b)] = (40.0, 200.0, 40.0)
    compare[np.logical_and(gt_b, ~pr_b)] = (255.0, 200.0, 0.0)
    compare[np.logical_and(~gt_b, pr_b)] = highlight
    return compare


def triplet_folder_name(scene_id, organ, ann_id):
    return f"{scene_id}_{organ}_ann{ann_id}"


def build_display_full_cloud(dataset, scan_id, query, eval_batch, eval_result, seed, max_display=80000):
    """
    Map eval-subset predictions back onto the full scene for sparse-organ viz.
    If background is too large, random-downsample to max_display; keep all GT target points.
    """
    object_ids = query["object_ids"]
    point_count = query.get("point_count")
    xyz_full, rgb_full, instance_label = _fetch_raw_scene(dataset, scan_id)

    eval_mask = dataset.compute_downsample_mask(
        xyz_full, instance_label, object_ids, point_count=point_count, seed=seed
    )
    eval_idx = np.where(eval_mask)[0]

    eval_gt_b = eval_batch["gt_pmasks"][0].detach().cpu().numpy().astype(bool).reshape(-1)
    eval_pr_b = eval_result["pred_pmask"][0].detach().cpu().numpy().reshape(-1) >= 0.5

    gt_full = _target_point_mask(instance_label, object_ids)
    pred_full = np.zeros(len(xyz_full), dtype=bool)
    if len(eval_idx) == len(eval_pr_b):
        pred_full[eval_idx] = eval_pr_b
    else:
        n = min(len(eval_idx), len(eval_pr_b))
        pred_full[eval_idx[:n]] = eval_pr_b[:n]

    # full-scene view: keep targets first, random-crop background
    keep = np.ones(len(xyz_full), dtype=bool)
    if len(xyz_full) > max_display:
        must = gt_full.copy()
        bg_idx = np.where(~must)[0]
        n_bg_keep = max(0, max_display - int(must.sum()))
        if len(bg_idx) > n_bg_keep:
            rng = np.random.default_rng(seed + 13)
            chosen = rng.choice(bg_idx, size=n_bg_keep, replace=False)
            keep = must
            keep[chosen] = True
    xyz_disp = xyz_full[keep]
    gt_disp = gt_full[keep]
    pr_disp = pred_full[keep]
    rgb_disp = rgb_full[keep]
    return (
        xyz_disp, gt_disp, pr_disp, rgb_disp,
        int(eval_mask.sum()), len(xyz_full), int(gt_full.sum()),
    )


def _fetch_raw_scene(dataset, scan_id):
    scene = dataset._fetch_scene(scan_id, dataset.sp_filenames[scan_id])
    xyz = np.asarray(scene[0], dtype=np.float32)
    rgb = np.asarray(scene[1], dtype=np.float32)
    instance_label = np.asarray(scene[4], dtype=np.int64)
    return xyz, rgb, instance_label


def _target_point_mask(instance_label, object_ids):
    mask = np.zeros(instance_label.shape[0], dtype=bool)
    for oid in object_ids:
        mask |= instance_label == oid
    return mask


def sample_decor_mask(instance_label, object_ids, eval_mask_full, n_decor, seed):
    non_target = ~_target_point_mask(instance_label, object_ids)
    candidates = np.where(non_target & ~eval_mask_full)[0]
    if len(candidates) == 0:
        candidates = np.where(~eval_mask_full)[0]
    if len(candidates) == 0 or n_decor <= 0:
        return np.zeros(instance_label.shape[0], dtype=bool)
    rng = np.random.default_rng(seed + 7)
    chosen = rng.choice(candidates, size=min(n_decor, len(candidates)), replace=False)
    decor = np.zeros(instance_label.shape[0], dtype=bool)
    decor[chosen] = True
    return decor


def build_display_with_decor(
    dataset, scan_id, query, eval_batch, eval_result, eval_rgb, max_n, decor_points, seed,
):
    """Eval points plus decorative background points that skip inference."""
    object_ids = query["object_ids"]
    point_count = query.get("point_count")
    xyz_full, rgb_full, instance_label = _fetch_raw_scene(dataset, scan_id)
    eval_mask_full = dataset.compute_downsample_mask(
        xyz_full, instance_label, object_ids, point_count=point_count, seed=seed
    )
    decor_mask_full = sample_decor_mask(instance_label, object_ids, eval_mask_full, decor_points, seed)

    eval_xyz = eval_batch["coords_float"].cpu().numpy()
    eval_gt = eval_batch["gt_pmasks"][0].detach().cpu().numpy().astype(bool).reshape(-1)
    eval_pr = eval_result["pred_pmask"][0].detach().cpu().numpy().reshape(-1) >= 0.5

    decor_xyz = xyz_full[decor_mask_full]
    decor_rgb = rgb_full[decor_mask_full]
    if len(decor_xyz):
        xyz_disp = np.vstack([eval_xyz, decor_xyz])
        rgb_disp = np.vstack([eval_rgb, decor_rgb])
        gt_disp = np.concatenate([eval_gt, np.zeros(len(decor_xyz), dtype=bool)])
        pr_disp = np.concatenate([eval_pr, np.zeros(len(decor_xyz), dtype=bool)])
    else:
        xyz_disp, rgb_disp, gt_disp, pr_disp = eval_xyz, eval_rgb, eval_gt, eval_pr

    return xyz_disp, gt_disp, pr_disp, rgb_disp, len(eval_xyz), len(decor_xyz)


def build_display_eval_only(eval_batch, eval_gt_b, eval_pr_b, eval_rgb):
    xyz = eval_batch["coords_float"].cpu().numpy()
    return xyz, eval_gt_b, eval_pr_b, eval_rgb, len(xyz), 0


def _build_compare_from_binary(gt_b, pr_b, text_variant=None):
    """Gray-180 + green GT; pred.png is red. Compare FP color still follows text type."""
    n = len(gt_b)
    gt_colors = np.full((n, 3), 180.0, dtype=np.float32)
    gt_colors[gt_b] = (40.0, 200.0, 40.0)
    pred_colors = np.full((n, 3), 180.0, dtype=np.float32)
    pred_colors[pr_b] = (220.0, 50.0, 50.0)
    if text_variant:
        fp_color = OKABE_ITO_VARIANT_RGB.get(
            text_variant, OKABE_ITO_VARIANT_RGB["canonical"]
        )
    else:
        fp_color = (80.0, 120.0, 255.0)
    compare = gt_colors.copy()
    compare[np.logical_and(gt_b, pr_b)] = (40.0, 200.0, 40.0)
    compare[np.logical_and(gt_b, ~pr_b)] = (255.0, 200.0, 0.0)
    compare[np.logical_and(~gt_b, pr_b)] = fp_color
    target_recall = float((gt_b & pr_b).sum() / max(int(gt_b.sum()), 1))
    return gt_colors, pred_colors, compare, target_recall


def _masks_from_tensors(gt_mask, pred_prob):
    gt_b = gt_mask.detach().cpu().numpy().astype(bool).reshape(-1)
    pr_b = (pred_prob > 0.5).detach().cpu().numpy().astype(bool).reshape(-1)
    return gt_b, pr_b


def _load_sample(dataset, ds_idx, seed=None):
    """Load a sample; if seed is set, fix downsample (with ROI) to match eval."""
    prev = getattr(dataset, "_downsample_seed", None)
    dataset._downsample_seed = seed
    try:
        return dataset[ds_idx]
    finally:
        dataset._downsample_seed = prev


def _run_predict(model, dataset, sample):
    batch = dataset.collate_fn([sample])
    return batch, model(batch, mode="predict")


def save_pcd(xyz, colors, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors / 255.0).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"exported: {path}")


def render_png_headless(xyz, colors, out_path, view, ballradius=3, white_bg=True, showsz=800):
    _ensure_render_ball()
    dll = np.ctypeslib.load_library("render_ball", str(TOOLS))

    xyz = xyz.astype(np.float32) - xyz.mean(axis=0)
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    xyz = xyz / (radius * 2.2) * showsz

    c0 = np.clip(colors[:, 0], 0, 255).astype(np.float32)
    c1 = np.clip(colors[:, 1], 0, 255).astype(np.float32)
    c2 = np.clip(colors[:, 2], 0, 255).astype(np.float32)
    c0 = np.require(c0, "float32", "C")
    c1 = np.require(c1, "float32", "C")
    c2 = np.require(c2, "float32", "C")

    bg = (255, 255, 255) if white_bg else (0, 0, 0)
    obj_rot, view_rot = _build_view_mats(view, xyz)
    nxyz = xyz.dot(obj_rot).dot(view_rot) + [showsz / 2, showsz / 2, 0]
    nxyz[:, 1] = showsz - nxyz[:, 1]
    ixyz = nxyz.astype("int32")

    show = np.zeros((showsz, showsz, 3), dtype="uint8")
    show[:] = bg
    dll.render_ball(
        ct.c_int(show.shape[0]),
        ct.c_int(show.shape[1]),
        show.ctypes.data_as(ct.c_void_p),
        ct.c_int(ixyz.shape[0]),
        ixyz.ctypes.data_as(ct.c_void_p),
        c0.ctypes.data_as(ct.c_void_p),
        c1.ctypes.data_as(ct.c_void_p),
        c2.ctypes.data_as(ct.c_void_p),
        ct.c_int(ballradius),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), show)
    print(f"saved png: {out_path}")


def run_show(xyz, gt_colors, pred_colors, view, ballradius, white_bg):
    _ensure_render_ball()
    os.chdir(TOOLS)
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import show_3d
    from show_3d import showpoints

    if getattr(view, "pca_align", False):
        xyz = xyz.astype(np.float32).dot(compute_pca_object_rot(xyz))
        show_3d.object_yaw = np.deg2rad(0.0)
    else:
        show_3d.object_yaw = np.deg2rad(view.object_yaw_deg)
    show_3d.view_xangle = np.deg2rad(view.view_xangle_deg)
    show_3d.view_yangle = np.deg2rad(view.view_yangle_deg)
    show_3d.zoom = view.zoom

    bg = (255, 255, 255) if white_bg else (0, 0, 0)
    print("\nkeys: L-drag rotate | wheel zoom | t=GT(green) p=Pred(red) s=shot q=quit | 1/2=views | r=reset\n")
    showpoints(
        xyz.astype(np.float32),
        c_gt=gt_colors,
        c_pred=pred_colors,
        waittime=0,
        background=bg,
        normalizecolor=False,
        ballradius=ballradius,
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Crops3D segmentation visualization")
    parser.add_argument("config", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--organ", type=str, default=None)
    parser.add_argument("--ann_id", type=int, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--export_dir", type=str, default=None)
    parser.add_argument("--flat_export", action="store_true")
    parser.add_argument("--save_png", type=str, default=None, choices=["gt", "pred", "compare"])
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--ballradius", type=int, default=3)
    parser.add_argument("--view_preset", type=str, default="front", choices=sorted(VIEW_PRESETS.keys()))
    parser.add_argument("--object_yaw_deg", type=float, default=None)
    parser.add_argument("--view_xangle_deg", type=float, default=None)
    parser.add_argument("--view_yangle_deg", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument(
        "--no_decor",
        action="store_true",
        help="do not add full-plant gray decor (eval 10k only; large organs may look isolated)",
    )
    parser.add_argument(
        "--decor_points",
        type=int,
        default=None,
        help="decor background count; default max_npoint*(1-max_target_ratio)",
    )
    parser.add_argument("--max_target_ratio", type=float, default=0.2, help="used to compute default decor count")
    parser.add_argument("--sample_seed", type=int, default=1999)
    parser.add_argument(
        "--eval_roi",
        action="store_true",
        default=None,
        help="use ROI downsample on val/test (train-aligned); default config sparse_organ.eval_roi",
    )
    parser.add_argument(
        "--no_eval_roi",
        action="store_true",
        help="force ROI off and use full-scene sample_rdn",
    )
    parser.add_argument(
        "--full_cloud",
        action="store_true",
        help="map eval preds onto the full scene (better sparse-organ viz)",
    )
    parser.add_argument(
        "--max_display_points",
        type=int,
        default=80000,
        help="background cap in full_cloud mode (targets always kept)",
    )
    args = parser.parse_args()
    view = resolve_view(args)

    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_ids, num_gpu=args.num_gpus)
    cfg = gorilla.Config.fromfile(args.config)
    split_cfg = getattr(cfg.data, args.split)
    max_n = split_cfg.voxel_cfg.max_npoint
    eval_roi_override = False if args.no_eval_roi else args.eval_roi
    sparse_cfg = _resolve_sparse_organ_cfg(split_cfg, eval_roi_override=eval_roi_override)
    decor_points = 0 if args.no_decor else (
        args.decor_points if args.decor_points is not None else int(max_n * (1.0 - args.max_target_ratio))
    )

    from gres_model.dataset.crops3d_organ import Crops3DOrganDataset
    from gres_model.model import MODEL

    dataset = Crops3DOrganDataset(
        data_root=split_cfg.data_root,
        queries_json=split_cfg.queries_json,
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
    ds_idx, query = resolve_sample_index(dataset, args.scene_id, args.organ, args.ann_id)
    print(
        f"sample: {query['scene_id']} | {query.get('crop')} | "
        f"{query.get('object_name')} | ann_id={query.get('ann_id')}"
    )
    print(f"text: {query.get('description')}")
    print(
        f"sparse_organ: enabled={sparse_cfg['enabled']} eval_roi={sparse_cfg['eval_roi']} "
        f"threshold={sparse_cfg['sparse_point_threshold']} point_count={query.get('point_count')}"
    )
    print(f"decor: {'off' if args.no_decor else decor_points} points | full_cloud={args.full_cloud}")

    eval_sample = _load_sample(dataset, ds_idx, seed=args.sample_seed)
    eval_rgb = _rgb_from_sample(eval_sample)

    model = MODEL(**cfg.model).cuda()
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)
    model.eval()
    eval_batch, eval_result = _run_predict(model, dataset, eval_sample)

    piou = float(eval_result["piou"][0])
    eval_gt_b, eval_pr_b = _masks_from_tensors(eval_batch["gt_pmasks"][0], eval_result["pred_pmask"][0])
    _, _, _, target_recall = _build_compare_from_binary(eval_gt_b, eval_pr_b)
    n_eval = len(eval_batch["coords_float"])
    n_target_eval = int(eval_gt_b.sum())
    n_full = None

    if args.full_cloud:
        xyz, gt_b, pr_b, _rgb_disp, n_eval_pts, n_full, n_target_full = build_display_full_cloud(
            dataset, query["scene_id"], query, eval_batch, eval_result,
            args.sample_seed, max_display=args.max_display_points,
        )
        n_decor_pts = 0
        print(
            f"full_cloud: scene={n_full} eval={n_eval_pts} target_full={n_target_full} "
            f"display={len(xyz)}"
        )
    elif decor_points > 0:
        xyz, gt_b, pr_b, _rgb_disp, n_eval_pts, n_decor_pts = build_display_with_decor(
            dataset, query["scene_id"], query, eval_batch, eval_result,
            eval_rgb, max_n, decor_points, args.sample_seed,
        )
        print(
            f"eval: total={n_eval} target={n_target_eval} | "
            f"display: eval={n_eval_pts} decor={n_decor_pts} total={len(xyz)}"
        )
    else:
        xyz = eval_batch["coords_float"].cpu().numpy()
        gt_b, pr_b = eval_gt_b, eval_pr_b
        n_decor_pts = 0
        print(f"eval only: total={n_eval} target={n_target_eval}")
    print(f"IoU={piou:.4f}  target_recall={target_recall:.4f}")

    gt_colors, pred_colors, compare, _ = _build_compare_from_binary(
        gt_b, pr_b, text_variant=query.get("text_variant") or "canonical"
    )

    organ_name = query.get("object_name", "organ")
    out_dir = make_run_export_dir(
        args.export_dir, args.scene_id, organ_name,
        ann_id=query.get("ann_id") if args.ann_id is not None else None,
        flat=args.flat_export,
    )
    print(f"export_dir: {out_dir}")

    need_export = args.export_dir is not None or args.save_png or not os.environ.get("DISPLAY")
    if need_export:
        save_pcd(xyz, gt_colors, out_dir / "gt.pcd")
        save_pcd(xyz, pred_colors, out_dir / "pred.pcd")
        save_pcd(xyz, compare, out_dir / "compare.pcd")
        render_png_headless(xyz, gt_colors, out_dir / "gt.png", view, args.ballradius)
        render_png_headless(xyz, pred_colors, out_dir / "pred.png", view, args.ballradius)
        render_png_headless(xyz, compare, out_dir / "compare.png", view, args.ballradius)
        run_info = {
            "scene_id": args.scene_id,
            "organ": organ_name,
            "ann_id": query.get("ann_id"),
            "split": args.split,
            "iou": piou,
            "target_recall": target_recall,
            "checkpoint": args.checkpoint,
            "config": args.config,
            "view_preset": args.view_preset,
            "object_yaw_deg": view.object_yaw_deg,
            "view_xangle_deg": view.view_xangle_deg,
            "view_yangle_deg": view.view_yangle_deg,
            "sample_seed": args.sample_seed,
            "eval_roi": sparse_cfg["eval_roi"],
            "sparse_point_threshold": sparse_cfg["sparse_point_threshold"],
            "full_cloud": args.full_cloud,
            "scene_points": n_full if args.full_cloud else None,
            "eval_points": n_eval,
            "eval_target_points": n_target_eval,
            "display_points": len(xyz),
            "decor_points": n_decor_pts,
            "description": query.get("description"),
        }
        with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2)
        print(f"saved run_info: {out_dir / 'run_info.json'}")
        if args.save_png:
            print(f"main image: {out_dir / f'{args.save_png}.png'}")

    if not args.no_show and os.environ.get("DISPLAY"):
        run_show(xyz, gt_colors, pred_colors, view, args.ballradius, white_bg=True)
    elif not args.no_show:
        print(f"no DISPLAY; skipped window. See PNG: {out_dir}")


if __name__ == "__main__":
    main()
