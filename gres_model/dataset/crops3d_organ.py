"""Crops3D text-query organ segmentation dataset (step 4)."""

import glob
import json
import logging
import math
import os.path as osp
import time
from typing import Dict, Sequence, Tuple

import numpy as np
import pointgroup_ops
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

from gorilla import is_main_process

from .sparse_organ import (
    build_roi_keep_mask,
    clip_sparse_weight,
    parse_sparse_organ_cfg,
)

# Placeholder dim for dense_maps; matches the MDIN loss interface
MAX_NUM_OBJ = 132
_DATA_LOGGER = logging.getLogger(__name__)


class Crops3DOrganDataset(Dataset):
    """
    Crops3D text-query organ segmentation dataset.

    Differences vs ScanNet Multi3DRefer:
    - lang_num_max=1: one query and one cloud load per sample
    - lang_num_max>1: pack queries of the same scene; load/transform the cloud once
    - GT masks come from organ semantic labels (scalar_sf), not instance ids
    - No scene-graph parsing; dense_maps are zeros
    """

    def __init__(
        self,
        data_root,
        queries_json,
        prefix="train",
        suffix="_refer.pth",
        voxel_cfg=None,
        training=True,
        mode=4,
        with_elastic=True,
        aug=False,
        use_xyz=True,
        logger=None,
        max_des_len=32,
        lang_num_max=1,
        debug_data=False,
        preload_scenes=False,
        sparse_organ=None,
        **kwargs,
    ):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.mode = mode
        self.with_elastic = with_elastic
        self.aug = aug
        self.use_xyz = use_xyz
        self.logger = logger
        self.max_des_len = max_des_len
        self.lang_num_max = lang_num_max
        self.debug_data = debug_data
        self.preload_scenes = preload_scenes
        self.sparse_organ = parse_sparse_organ_cfg(sparse_organ)

        # Load query JSON
        with open(queries_json, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        # Keep samples whose _refer.pth exists in this split
        self.sp_filenames = self._build_filename_map()
        self.queries = [q for q in self.queries if q["scene_id"] in self.sp_filenames]
        if len(self.queries) == 0:
            raise RuntimeError(
                f"No valid samples. Confirm {data_root}/{prefix}/*_refer.pth exists "
                f"and scene_id in queries_json matches."
            )

        self.scene_inputs = self._build_scene_inputs(self.queries, self.lang_num_max)
        # Oversample sparse packs (young stem / tassel with low point_count)
        if self.training and self.sparse_organ["enabled"]:
            self.scene_inputs = self._oversample_sparse_packs(self.scene_inputs)

        # scene_id -> numpy cloud tuple; preload once to avoid repeated disk I/O
        self.scene_table = {}
        if self.preload_scenes:
            self._preload_scenes()

        if logger and is_main_process():
            logger.info(
                f"Load Crops3D {prefix}: {len(self.queries)} queries, "
                f"{len(self.scene_inputs)} scene packs (lang_num_max={self.lang_num_max})"
                + (f", preloaded {len(self.scene_table)} scenes" if self.preload_scenes else "")
            )
            if self.training and self.sparse_organ["enabled"]:
                self._log_sparse_organ_stats()

        # RoBERTa tokenizer (same as original MDIN)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "./backbones/roberta-base/", local_files_only=True
        )

    @staticmethod
    def _build_scene_inputs(queries, lang_num_max):
        """Group queries by scene_id; at most lang_num_max per pack (ScanNet-style)."""
        lang_num_max = max(1, int(lang_num_max))
        sorted_queries = sorted(
            queries,
            key=lambda q: (q["scene_id"], q.get("ann_id", 0)),
        )
        scene_inputs = []
        scene_group = []
        current_scene_id = None
        for query in sorted_queries:
            scene_id = query["scene_id"]
            if current_scene_id != scene_id:
                if scene_group:
                    scene_inputs.append(scene_group)
                scene_group = []
                current_scene_id = scene_id
            if len(scene_group) >= lang_num_max:
                scene_inputs.append(scene_group)
                scene_group = []
            scene_group.append(query)
        if scene_group:
            scene_inputs.append(scene_group)
        return scene_inputs

    @staticmethod
    def _union_object_ids(queries):
        """Union target organ ids in a pack so downsampling can keep them."""
        merged = []
        seen = set()
        for query in queries:
            for oid in query["object_ids"]:
                oid = int(oid)
                if oid not in seen:
                    seen.add(oid)
                    merged.append(oid)
        return merged

    @staticmethod
    def _pack_point_count(queries):
        """Pack point count: min of query point_count values (missing -> 0)."""
        counts = []
        for query in queries:
            pc = query.get("point_count", 0)
            try:
                counts.append(int(pc))
            except (TypeError, ValueError):
                counts.append(0)
        return min(counts) if counts else 0

    def _oversample_sparse_packs(self, scene_inputs):
        """Duplicate scene packs below the point_count threshold (train only)."""
        cfg = self.sparse_organ
        factor = int(cfg.get("oversample_factor", 1) or 1)
        if factor <= 1:
            return scene_inputs
        thr = int(cfg.get("oversample_threshold", cfg["sparse_point_threshold"]))
        out = []
        dup_n = 0
        for pack in scene_inputs:
            p = self._pack_point_count(pack)
            if p < thr:
                out.extend([pack] * factor)
                dup_n += factor - 1
            else:
                out.append(pack)
        self._sparse_oversample_dup_count = dup_n
        return out

    def _should_use_sparse_roi(self):
        """Train may use ROI; val/test need eval_roi=true to match train."""
        cfg = self.sparse_organ
        if not cfg["enabled"]:
            return False
        if self.training:
            return True
        return bool(cfg.get("eval_roi", False))

    def _log_sparse_organ_stats(self):
        """Log sparse-pack count and sampler-weight quantiles at startup."""
        thr = self.sparse_organ["sparse_point_threshold"]
        pcs = [self._pack_point_count(pack) for pack in self.scene_inputs]
        sparse_n = sum(1 for p in pcs if p < thr)
        weights = self.get_sample_weights()
        ws = sorted(weights)
        def _pct(p):
            if not ws:
                return 0.0
            i = int(round((p / 100.0) * (len(ws) - 1)))
            return ws[i]
        dup = getattr(self, "_sparse_oversample_dup_count", 0)
        self.logger.info(
            "sparse_organ enabled: packs=%d sparse(<%d)=%d (%.1f%%); "
            "sampler_weighting=%s weight[min/p50/max]=%.3f/%.3f/%.3f; "
            "oversample_factor=%d dup_packs=%d eval_roi=%s",
            len(pcs),
            thr,
            sparse_n,
            100.0 * sparse_n / max(len(pcs), 1),
            self.sparse_organ["sampler_weighting"],
            _pct(0),
            _pct(50),
            _pct(100),
            self.sparse_organ.get("oversample_factor", 1),
            dup,
            self.sparse_organ.get("eval_roi", False),
        )

    def get_sample_weights(self):
        """One weight per scene pack for WeightedRandomSampler."""
        cfg = self.sparse_organ
        weights = []
        for pack in self.scene_inputs:
            p = self._pack_point_count(pack)
            weights.append(
                clip_sparse_weight(
                    p,
                    weight_min=cfg["weight_min"],
                    weight_max=cfg["weight_max"],
                    point_ref=cfg["point_ref"],
                )
            )
        return weights

    def _build_filename_map(self):
        """Map scene_id to _refer.pth path."""
        pattern = osp.join(self.data_root, self.prefix, f"*{self.suffix}")
        filenames = glob.glob(pattern)
        name_map = {}
        for fn in filenames:
            scene_id = osp.basename(fn).replace(self.suffix, "")
            name_map[scene_id] = fn
        return name_map

    @staticmethod
    def _to_numpy_scene(data):
        """Convert torch.load output to a numpy tuple for cache/aug."""
        arrays = []
        for item in data:
            if isinstance(item, torch.Tensor):
                arrays.append(item.cpu().numpy())
            else:
                arrays.append(np.asarray(item))
        return tuple(arrays)

    def _preload_scenes(self):
        """Load all involved _refer.pth files into scene_table at init."""
        scene_ids = sorted({q["scene_id"] for q in self.queries})
        t0 = time.time()
        total_bytes = 0
        for i, scene_id in enumerate(scene_ids, start=1):
            path = self.sp_filenames[scene_id]
            data = self._to_numpy_scene(torch.load(path, map_location="cpu"))
            self.scene_table[scene_id] = data
            total_bytes += sum(arr.nbytes for arr in data)
            if self.logger and is_main_process() and (i % 100 == 0 or i == len(scene_ids)):
                self.logger.info(
                    f"Preload Crops3D {self.prefix}: {i}/{len(scene_ids)} scenes "
                    f"({total_bytes / (1024 ** 2):.1f} MB)"
                )
        if self.logger and is_main_process():
            self.logger.info(
                f"Preload Crops3D {self.prefix} done: {len(scene_ids)} scenes, "
                f"{total_bytes / (1024 ** 2):.1f} MB, {time.time() - t0:.1f}s"
            )

    def load(self, filename, scene_id=None):
        """Load the preprocessed cloud tuple; prefer scene_table."""
        if scene_id is not None and scene_id in self.scene_table:
            return self.scene_table[scene_id]
        return self._to_numpy_scene(torch.load(filename, map_location="cpu"))

    def _fetch_scene(self, scan_id, sp_filename):
        """Fetch the cloud and copy when needed so transforms do not mutate cache."""
        scene = self.load(sp_filename, scene_id=scan_id)
        if scan_id in self.scene_table:
            scene = tuple(arr.copy() for arr in scene)
        return scene

    def transform_train(
        self, xyz, rgb, superpoint, semantic_label, instance_label, object_ids=None, point_count=None
    ):
        """Train: downsample first, then augment, to avoid elastic on millions of points."""
        # Downsample first so data_aug/elastic do not hang on million-point PLY
        valid_idxs = self.compute_downsample_mask(
            xyz, instance_label, object_ids, point_count=point_count,
            seed=getattr(self, "_downsample_seed", None),
        )
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = instance_label[valid_idxs]

        if self.aug:
            xyz_middle = self.data_aug(xyz, True, True, True)
        else:
            xyz_middle = xyz.copy()
        rgb = rgb + np.random.randn(3) * 0.1
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.0)
            xyz = self.elastic(xyz, 20, 160.0)
        xyz = xyz - xyz.min(0)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def transform_test(
        self, xyz, rgb, superpoint, semantic_label, instance_label, object_ids=None, point_count=None
    ):
        """Val/test: downsample (optional ROI) then scale; can match train protocol."""
        xyz_middle = xyz
        valid_idxs = self.compute_downsample_mask(
            xyz_middle, instance_label, object_ids, point_count=point_count,
            seed=getattr(self, "_downsample_seed", None),
        )
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = instance_label[valid_idxs]
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        """Geometric aug: jitter, flip, rotate."""
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [
                    [math.cos(theta), math.sin(theta), 0],
                    [-math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ],
            )
        return np.matmul(xyz, m)

    def sample_rdn(self, xyz, instance_label=None, object_ids=None, rng=None):
        """Random downsample; keep query organ points to avoid obj_pos OOB in loss."""
        n = xyz.shape[0]
        max_n = self.voxel_cfg.max_npoint
        if n <= max_n:
            return np.ones(n, dtype=bool)

        must_keep = np.zeros(n, dtype=bool)
        if instance_label is not None and object_ids:
            for oid in object_ids:
                must_keep |= instance_label == oid

        must_idx = np.where(must_keep)[0]
        other_idx = np.where(~must_keep)[0]

        def _choice(arr, size, replace=False):
            if rng is not None:
                return rng.choice(arr, size=size, replace=replace)
            return np.random.choice(arr, size, replace=replace)

        if len(must_idx) >= max_n:
            chosen = _choice(must_idx, max_n, replace=False)
        else:
            n_extra = max_n - len(must_idx)
            if len(other_idx) > n_extra:
                extra = _choice(other_idx, n_extra, replace=False)
            else:
                extra = other_idx
            chosen = np.concatenate([must_idx, extra]) if len(must_idx) else extra

        valid_idxs = np.zeros(n, dtype=bool)
        valid_idxs[chosen] = True
        return valid_idxs

    def compute_downsample_mask(
        self, xyz, instance_label=None, object_ids=None, point_count=None, seed=None
    ):
        """
        Shared downsample entry (train ROI / eval ROI / viz replay).
        If seed is not None, use a fixed RNG so visualize matches eval.
        """
        rng = np.random.default_rng(seed) if seed is not None else None
        return self.sample_sparse_roi(
            xyz, instance_label, object_ids, point_count=point_count, rng=rng
        )

    def sample_sparse_roi(
        self, xyz, instance_label=None, object_ids=None, point_count=None, rng=None
    ):
        """
        Sparse organ: ROI then sample_rdn; otherwise fall back to full-scene sample_rdn.
        Enabled by default in train; val/test need sparse_organ.eval_roi=true.
        """
        cfg = self.sparse_organ
        if not self._should_use_sparse_roi():
            return self.sample_rdn(xyz, instance_label, object_ids, rng=rng)

        n = xyz.shape[0]
        must_keep = np.zeros(n, dtype=bool)
        if instance_label is not None and object_ids:
            for oid in object_ids:
                must_keep |= instance_label == oid

        if point_count is None:
            point_count = int(must_keep.sum())
        else:
            point_count = int(point_count)

        if point_count >= cfg["sparse_point_threshold"]:
            return self.sample_rdn(xyz, instance_label, object_ids, rng=rng)

        if not must_keep.any():
            return self.sample_rdn(xyz, instance_label, object_ids, rng=rng)

        roi = build_roi_keep_mask(
            xyz,
            must_keep,
            expand_ratio=cfg["roi_expand_ratio"],
            min_edge_ratio=cfg["roi_min_edge_ratio"],
        )
        if not roi.any():
            return self.sample_rdn(xyz, instance_label, object_ids, rng=rng)

        # Target-first downsample on the ROI subset, then map back to the full mask
        sub_xyz = xyz[roi]
        sub_inst = instance_label[roi] if instance_label is not None else None
        sub_valid = self.sample_rdn(sub_xyz, sub_inst, object_ids, rng=rng)
        valid = np.zeros(n, dtype=bool)
        valid[np.where(roi)[0][sub_valid]] = True
        return valid

    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from PointGroup)."""
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3
        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag

    def get_ref_mask(self, instance_label, superpoint, object_ids):
        """
        Build GT masks from object_ids (organ label ids).
        After preprocess, instance_label == scalar_sf, so match all points of that organ.
        """
        ref_lbl = torch.zeros_like(instance_label, dtype=torch.float32)
        for object_id in object_ids:
            ref_lbl[instance_label == object_id] = 1.0
        gt_spmask = torch_scatter.scatter_mean(ref_lbl, superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        return ref_lbl, gt_spmask

    def get_loss_object_ids(self, object_ids, instance_label, sp_ins_label, union_sp_mask):
        """
        MDIN loss treats each id in object_ids as an independent target.
        Crops3D shoot_system is stem+leaf and should be trained as one union mask,
        so composite queries get a temporary superpoint label local to this sample.
        """
        if len(object_ids) <= 1:
            return object_ids, sp_ins_label

        composite_id = int(max(int(instance_label.max().item()), max(object_ids)) + 1)
        sp_ins_label = sp_ins_label.clone()
        sp_ins_label[union_sp_mask.bool()] = composite_id
        return [composite_id], sp_ins_label

    def get_sp_label(self, superpoint, instance_label):
        """Superpoint instance labels (majority vote of points inside each superpoint)."""
        num_sp = int(superpoint.max().item()) + 1
        sp_ins_label = torch.full((num_sp,), 199, dtype=instance_label.dtype)
        for sp_id in range(num_sp):
            mask = superpoint == sp_id
            if not mask.any():
                continue
            labels = instance_label[mask]
            vals, counts = labels.unique(return_counts=True)
            sp_ins_label[sp_id] = vals[counts.argmax()]
        return sp_ins_label

    def _empty_dense_maps(self):
        """Zero dense_maps to satisfy MDIN loss; contrastive align stays off."""
        return {
            "tokens_positive": torch.zeros(MAX_NUM_OBJ, 2),
            "positive_map": torch.zeros(MAX_NUM_OBJ, 256),
            "modify_positive_map": torch.zeros(MAX_NUM_OBJ, 256),
            "pron_positive_map": torch.zeros(MAX_NUM_OBJ, 256),
            "other_entity_map": torch.zeros(MAX_NUM_OBJ, 256),
            "auxi_entity_positive_map": torch.zeros(MAX_NUM_OBJ, 256),
            "rel_positive_map": torch.zeros(MAX_NUM_OBJ, 256),
        }

    def __len__(self):
        return len(self.scene_inputs)

    def _resolve_loss_object_ids(self, object_ids, instance_label, sp_ins_label, sp_ref_mask):
        """Keep composite sp labels in single-query mode; pack mode returns loss ids only."""
        if self.lang_num_max <= 1:
            return self.get_loss_object_ids(object_ids, instance_label, sp_ins_label, sp_ref_mask)
        return list(object_ids), sp_ins_label

    def __getitem__(self, index: int) -> Tuple:
        t0 = time.time()
        scene_input = self.scene_inputs[index]
        pack_object_ids = self._union_object_ids(scene_input)
        pack_point_count = self._pack_point_count(scene_input)

        ann_ids, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks = [], [], [], [], []
        lang_utterances, meta_datas, dense_mapss = [], [], []
        scan_id = scene_input[0]["scene_id"]
        sp_filename = self.sp_filenames[scan_id]

        t_load = time.time()
        scene = self._fetch_scene(scan_id, sp_filename)
        t_after_load = time.time()
        if self.training:
            scene = self.transform_train(
                *scene, object_ids=pack_object_ids, point_count=pack_point_count
            )
        else:
            scene = self.transform_test(
                *scene, object_ids=pack_object_ids, point_count=pack_point_count
            )
        t_after_transform = time.time()
        xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = scene

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint)
        instance_label = torch.from_numpy(instance_label).long()

        t_before_sp = time.time()
        sp_ins_label = self.get_sp_label(superpoint, instance_label)

        for query in scene_input:
            ann_id = query["ann_id"]
            object_ids = query["object_ids"]
            lang_utterance = query["description"]

            gt_pmask, gt_spmask = self.get_ref_mask(instance_label, superpoint, object_ids)
            sp_ref_mask = torch_scatter.scatter_mean(gt_pmask, superpoint, dim=-1)
            loss_object_ids, sp_ins_label = self._resolve_loss_object_ids(
                object_ids, instance_label, sp_ins_label, sp_ref_mask
            )

            meta_data = {
                "scene_id": scan_id,
                "ann_id": ann_id,
                "sample_id": query.get("sample_id", ""),
                "point_count": query.get("point_count", 0),
                "eval_type": query.get("eval_type", "organ"),
                "crop": query.get("crop", ""),
                "object_name": query.get("object_name", ""),
                "object_ids": object_ids,
                "description": lang_utterance,
                "text_variant": query.get("text_variant", "canonical"),
                "spatial": query.get("spatial", False),
                "color": query.get("color", False),
                "texture": query.get("texture", False),
                "shape": query.get("shape", False),
            }

            ann_ids.append(ann_id)
            object_idss.append(loss_object_ids)
            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
            sp_ref_masks.append(sp_ref_mask)
            lang_utterances.append(lang_utterance)
            meta_datas.append(meta_data)
            dense_mapss.append(self._empty_dense_maps())

        t_after_mask = time.time()

        if self.debug_data:
            organs = [q.get("object_name", "") for q in scene_input]
            _DATA_LOGGER.info(
                "[data] idx=%d scan=%s queries=%d organs=%s npoint=%d nsp=%d "
                "load=%.3fs transform=%.3fs mask=%.3fs total=%.3fs",
                index,
                scan_id,
                len(scene_input),
                ",".join(organs),
                coord.shape[0],
                int(superpoint.max().item()) + 1,
                t_after_load - t_load,
                t_after_transform - t_after_load,
                t_after_mask - t_before_sp,
                t_after_mask - t0,
            )

        return (
            ann_ids,
            scan_id,
            coord,
            coord_float,
            feat,
            superpoint,
            object_idss,
            gt_pmasks,
            gt_spmasks,
            sp_ref_masks,
            lang_utterances,
            sp_ins_label,
            None,
            [],
            [torch.tensor(0, dtype=torch.long)] * len(scene_input),
            [None] * len(scene_input),
            meta_datas,
            dense_mapss,
            [None] * len(scene_input),
        )

    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        """Collate a batch in the MDIN model.forward layout."""
        t0 = time.time()
        ann_ids, scan_ids, coords, coords_float, feats, superpoints = [], [], [], [], [], []
        object_idss, gt_pmasks, gt_spmasks, sp_ref_masks = [], [], [], []
        lang_utterances, sp_ins_labels, meta_datas, dense_mapss = [], [], [], []
        nsubj_inds, nsubj_names, view_dependents = [], [], []
        batch_offsets = [0]
        scenes_len = []
        superpoint_bias = 0

        for data in batch:
            (
                ann_id,
                scan_id,
                coord,
                coord_float,
                feat,
                src_superpoint,
                object_ids,
                gt_pmask,
                gt_spmask,
                sp_ref_mask,
                lang_utterance,
                sp_ins_label,
                _,
                _,
                nsubj_ind,
                nsubj_name,
                meta_data,
                dense_maps,
                view_dependent,
            ) = data

            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            scenes_len.append(len(ann_id))
            batch_offsets.append(superpoint_bias)

            ann_ids.extend(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(len(scan_ids) - 1), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            sp_ins_labels.append(sp_ins_label)
            object_idss.extend(object_ids)
            gt_pmasks.extend(gt_pmask)
            gt_spmasks.extend(gt_spmask)
            sp_ref_masks.extend(sp_ref_mask)
            meta_datas.extend(meta_data)
            view_dependents.extend(view_dependent)
            lang_utterances.extend(lang_utterance)
            dense_mapss.extend(dense_maps)
            nsubj_inds.extend(nsubj_ind)
            nsubj_names.extend(nsubj_name)

        token_dict = self.tokenizer.batch_encode_plus(
            lang_utterances, padding="longest", return_tensors="pt"
        )
        t_after_token = time.time()

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)
        scenes_len = torch.tensor(scenes_len, dtype=torch.int)
        coords = torch.cat(coords, 0)
        coords_float = torch.cat(coords_float, 0)
        feats = torch.cat(feats, 0)
        superpoints = torch.cat(superpoints, 0).long()
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)

        spatial_shape = np.clip(
            (coords.max(0)[0][1:] + 1).numpy(),
            self.voxel_cfg.spatial_shape[0],
            None,
        )
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(
            coords, len(batch), self.mode
        )
        t_after_voxel = time.time()

        if self.debug_data:
            _DATA_LOGGER.info(
                "[collate] scans=%s npoint=%d token=%.3fs voxel=%.3fs total=%.3fs",
                scan_ids,
                coords.shape[0],
                t_after_token - t0,
                t_after_voxel - t_after_token,
                t_after_voxel - t0,
            )

        return {
            "ann_ids": ann_ids,
            "scan_ids": scan_ids,
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "spatial_shape": spatial_shape,
            "feats": feats,
            "superpoints": superpoints,
            "batch_offsets": batch_offsets,
            "object_idss": object_idss,
            "gt_pmasks": gt_pmasks,
            "gt_spmasks": gt_spmasks,
            "sp_ref_masks": sp_ref_masks,
            "lang_tokenss": token_dict["input_ids"],
            "lang_masks": token_dict["attention_mask"],
            "coords_float": coords_float,
            "sp_ins_labels": sp_ins_labels,
            "sem2inss": None,
            "sem_ids": None,
            "scenes_len": scenes_len,
            "nsubj_inds": torch.stack(nsubj_inds),
            "meta_datas": meta_datas,
            "dense_maps": dense_mapss,
            "view_dependents": view_dependents,
        }
