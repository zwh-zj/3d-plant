"""Two-stage candidates: KNN pooling and token-averaged ranking then Top-K."""
import torch

from gres_model.model.sample_model import (
    SamplingModule,
    knn_maxpool,
    rank_candidates,
    superpoint_knn_mean,
    superpoint_knn_pool,
)


def test_knn_maxpool_uses_feature_neighbors_not_far_points():
    feat = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 10.0],
        ]
    )
    pooled = knn_maxpool(feat, torch.tensor([0]), k=2)
    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled[0], torch.tensor([1.0, 0.0]))


def test_rank_averages_sigmoid_over_valid_tokens_and_drops_padding():
    f_pre = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    f_txt = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    pad = torch.tensor([[False, True]])
    inds, rank_score, _ = rank_candidates(f_pre, f_txt, pad, num_proposal=1)
    logits = torch.tensor([1.0, 0.0]) / (2 ** 0.5)
    expected = torch.sigmoid(logits[0])
    assert torch.allclose(rank_score[0, 0], expected)
    assert int(inds[0, 0]) == 0


def test_sampling_module_projects_before_dot_product():
    torch.manual_seed(0)
    module = SamplingModule(num_proposal=1, pc_dim=4, lang_dim=4, d_model=4)
    with torch.no_grad():
        module.query_proj.weight.copy_(torch.eye(4))
        module.key_proj.weight.copy_(torch.eye(4))
        module.key_proj.weight.mul_(2)
    pc = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    lang = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    pad = torch.tensor([[False]])
    _, projected = module(pc, lang, None, pad)
    _, _, direct = rank_candidates(pc, lang, pad, num_proposal=1)
    assert projected.shape == direct.shape
    assert not torch.allclose(projected, direct)


def test_superpoint_knn_mean_uses_feature_neighbors_and_ignores_padding():
    feats = torch.tensor([[
        [0.0, 0.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [0.0, 0.0],
    ]])
    pad = torch.tensor([[False, False, False, True]])
    pooled = superpoint_knn_mean(feats, torch.tensor([[0]]), pad, k=2)
    assert pooled.shape == (1, 1, 2)
    assert torch.allclose(pooled[0, 0], torch.tensor([0.5, 0.0]))


def test_superpoint_knn_pool_uses_coord_neighbors_and_max():
    feats = torch.tensor([[
        [1.0, 0.0],
        [0.0, 5.0],
        [1.0, 0.0],
        [9.0, 9.0],
    ]])
    coords = torch.tensor([[
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        [0.0, 0.0, 0.0],
    ]])
    pad = torch.tensor([[False, False, False, True]])
    pooled = superpoint_knn_pool(feats, coords, torch.tensor([[0]]), pad, k=2)
    assert pooled.shape == (1, 1, 2)
    assert torch.allclose(pooled[0, 0], torch.tensor([1.0, 5.0]))
