"""Sparse-organ ROI/weight unit tests (stdlib unittest, no pytest)."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gres_model.dataset.sparse_organ import (
    build_roi_keep_mask,
    clip_sparse_weight,
    parse_sparse_organ_cfg,
)


class TestSparseOrgan(unittest.TestCase):
    def test_clip_sparse_weight_dense_near_one(self):
        w = clip_sparse_weight(10000, weight_min=1.0, weight_max=8.0)
        self.assertAlmostEqual(w, 1.0, places=6)

    def test_clip_sparse_weight_tiny_hits_max(self):
        self.assertEqual(clip_sparse_weight(1, weight_min=1.0, weight_max=8.0), 8.0)
        self.assertEqual(clip_sparse_weight(36, weight_min=1.0, weight_max=8.0), 8.0)

    def test_clip_sparse_weight_boundary_around_156(self):
        w = clip_sparse_weight(156, weight_min=1.0, weight_max=8.0)
        self.assertAlmostEqual(w, 8.0, delta=0.05)
        self.assertAlmostEqual(
            clip_sparse_weight(2500, weight_min=1.0, weight_max=8.0), 2.0, places=6
        )

    def test_parse_sparse_organ_cfg_defaults_disabled(self):
        cfg = parse_sparse_organ_cfg(None)
        self.assertFalse(cfg["enabled"])
        self.assertFalse(cfg["sampler_weighting"])
        self.assertFalse(cfg["eval_roi"])
        self.assertEqual(cfg["sparse_point_threshold"], 2000)
        self.assertEqual(cfg["oversample_factor"], 1)

    def test_parse_sparse_organ_cfg_eval_roi_and_oversample(self):
        cfg = parse_sparse_organ_cfg(
            {
                "enabled": True,
                "eval_roi": True,
                "sparse_point_threshold": 5000,
                "oversample_factor": 3,
                "oversample_threshold": 4000,
                "weight_max": 12.0,
            }
        )
        self.assertTrue(cfg["eval_roi"])
        self.assertEqual(cfg["oversample_factor"], 3)
        self.assertEqual(cfg["oversample_threshold"], 4000)
        self.assertEqual(cfg["sparse_point_threshold"], 5000)
        self.assertEqual(cfg["weight_max"], 12.0)

    def test_build_roi_keep_mask_keeps_target_drops_far_bg(self):
        tgt = np.random.RandomState(0).randn(10, 3) * 0.01
        bg = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [-80.0, -80.0, 10.0]])
        xyz = np.vstack([tgt, bg])
        must = np.zeros(len(xyz), dtype=bool)
        must[:10] = True

        mask = build_roi_keep_mask(xyz, must, expand_ratio=1.5, min_edge_ratio=0.05)
        self.assertTrue(mask[:10].all())
        self.assertLess(mask.sum(), len(xyz))
        self.assertFalse(mask[10:].all())

    def test_build_roi_keep_mask_empty_target(self):
        xyz = np.random.RandomState(1).randn(20, 3)
        must = np.zeros(20, dtype=bool)
        mask = build_roi_keep_mask(xyz, must)
        self.assertEqual(mask.shape, (20,))
        self.assertFalse(mask.any())

    def test_sample_sparse_roi_raises_target_fraction(self):
        """After sparse ROI, target fraction should beat full-scene random downsample."""
        from types import SimpleNamespace
        from gres_model.dataset.crops3d_organ import Crops3DOrganDataset

        rs = np.random.RandomState(42)
        n_bg, n_tgt = 5000, 80
        xyz = np.vstack([rs.randn(n_bg, 3) * 5.0 + 20.0, rs.randn(n_tgt, 3) * 0.02])
        inst = np.zeros(n_bg + n_tgt, dtype=np.int32)
        inst[n_bg:] = 3

        ds = Crops3DOrganDataset.__new__(Crops3DOrganDataset)
        ds.training = True
        ds.voxel_cfg = SimpleNamespace(max_npoint=500)
        ds.sparse_organ = parse_sparse_organ_cfg(
            {
                "enabled": True,
                "sparse_point_threshold": 2000,
                "roi_expand_ratio": 1.5,
                "roi_min_edge_ratio": 0.05,
            }
        )

        rs_state = np.random.get_state()
        np.random.seed(0)
        full = ds.sample_rdn(xyz, inst, [3])
        np.random.seed(0)
        roi = ds.sample_sparse_roi(xyz, inst, [3], point_count=n_tgt)
        np.random.set_state(rs_state)

        frac_full = (inst[full] == 3).mean()
        frac_roi = (inst[roi] == 3).mean()
        self.assertGreater(frac_roi, frac_full)
        self.assertTrue((inst[roi] == 3).sum() == n_tgt or roi.sum() <= 500)

    def test_eval_roi_enables_roi_when_not_training(self):
        """With eval_roi=true, val/test should also take the ROI path."""
        from types import SimpleNamespace
        from gres_model.dataset.crops3d_organ import Crops3DOrganDataset

        rs = np.random.RandomState(42)
        n_bg, n_tgt = 5000, 80
        xyz = np.vstack([rs.randn(n_bg, 3) * 5.0 + 20.0, rs.randn(n_tgt, 3) * 0.02])
        inst = np.zeros(n_bg + n_tgt, dtype=np.int32)
        inst[n_bg:] = 3

        ds = Crops3DOrganDataset.__new__(Crops3DOrganDataset)
        ds.training = False
        ds.voxel_cfg = SimpleNamespace(max_npoint=500)
        ds.sparse_organ = parse_sparse_organ_cfg(
            {"enabled": True, "eval_roi": True, "sparse_point_threshold": 2000}
        )

        rng = np.random.default_rng(0)
        roi = ds.sample_sparse_roi(xyz, inst, [3], point_count=n_tgt, rng=rng)
        frac_roi = (inst[roi] == 3).mean()

        ds2 = Crops3DOrganDataset.__new__(Crops3DOrganDataset)
        ds2.training = False
        ds2.voxel_cfg = SimpleNamespace(max_npoint=500)
        ds2.sparse_organ = parse_sparse_organ_cfg(
            {"enabled": True, "eval_roi": False, "sparse_point_threshold": 2000}
        )
        rng2 = np.random.default_rng(0)
        full = ds2.sample_sparse_roi(xyz, inst, [3], point_count=n_tgt, rng=rng2)
        frac_full = (inst[full] == 3).mean()
        self.assertGreater(frac_roi, frac_full)

    def test_oversample_sparse_packs_duplicates(self):
        """Train-time sparse packs should be duplicated by oversample_factor."""
        from types import SimpleNamespace
        from gres_model.dataset.crops3d_organ import Crops3DOrganDataset

        ds = Crops3DOrganDataset.__new__(Crops3DOrganDataset)
        ds.sparse_organ = parse_sparse_organ_cfg(
            {"enabled": True, "oversample_factor": 3, "oversample_threshold": 5000}
        )
        packs = [
            [{"point_count": 100, "scene_id": "A"}],
            [{"point_count": 8000, "scene_id": "B"}],
        ]
        out = ds._oversample_sparse_packs(packs)
        self.assertEqual(len(out), 4)  # 1*3 + 1
        self.assertEqual(getattr(ds, "_sparse_oversample_dup_count", 0), 2)


if __name__ == "__main__":
    unittest.main()
