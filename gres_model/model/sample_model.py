import torch, math
import torch.nn as nn
import numpy as np
import sys
import os
import torch.nn.functional as F
if os.path.join(os.getcwd()) not in sys.path:
    sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(os.getcwd(), "lib"))
sys.path.append(os.path.join(os.getcwd(), "lib", 'pointnet2'))

class CrossAttentionMap(nn.Module):
    def __init__(self, q_dim, k_dim, d_model):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(q_dim, d_model, bias=False)
        self.key_proj = nn.Linear(k_dim, d_model, bias=False)

    def forward(self, query, key, query_mask=None, key_padding_mask=None):
        """
        Args:
            query: (B, N, D1)
            key: (B, L, D2)
            query_mask: (B, N)
            key_padding_mask: (B, L)
        Returns:
            attn_map: (B, N, L)
        """
        # Apply projection to query and key
        Q = self.query_proj(query)  # (B, N, D_model)
        K = self.key_proj(key)  # (B, L, D_model)
        # Compute attention scores
        scores = torch.einsum('bnd,bld->bnl', Q, K) / math.sqrt(self.d_model)  # (B, N, L)
        # Apply masks
        if query_mask is not None:
            scores = scores.masked_fill(query_mask.unsqueeze(2), float('-inf'))
        # Compute attention map
        attn_map = F.softmax(scores, dim=1)
        return attn_map
    
def superpoint_knn_mean(feats, center_idx, pad_mask, k):
    """Mean-pool each FPS superpoint with its feature-space neighbors.

    feats: (B, M, D), center_idx: (B, N), pad_mask: (B, M) True on padding.
    The center index is unchanged; only the feature is replaced.
    """
    center_idx = center_idx.long()
    batch, _, dim = feats.shape
    n_center = center_idx.shape[1]
    centers = feats.gather(1, center_idx.unsqueeze(-1).expand(-1, -1, dim))
    dist = torch.cdist(centers, feats)
    dist = dist.masked_fill(pad_mask.unsqueeze(1), float("inf"))
    k_eff = min(int(k), int(feats.shape[1]))
    nn_idx = torch.topk(dist, k_eff, largest=False).indices
    flat = nn_idx.reshape(batch, -1)
    neigh = feats.gather(1, flat.unsqueeze(-1).expand(-1, -1, dim))
    neigh = neigh.view(batch, n_center, k_eff, dim)
    valid = ~pad_mask.gather(1, flat).view(batch, n_center, k_eff)
    neigh = neigh * valid.unsqueeze(-1).to(dtype=neigh.dtype)
    denom = valid.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=neigh.dtype)
    return neigh.sum(dim=-2) / denom


def superpoint_knn_pool(feats, coords, center_idx, pad_mask, k, reduce="max"):
    """Max-pool each FPS superpoint over spatial (xyz) neighbors.

    feats: (B, M, D), coords: (B, M, 3), center_idx: (B, N),
    pad_mask: (B, M) True on padding. Center indices are unchanged.
    """
    center_idx = center_idx.long()
    batch, _, dim = feats.shape
    n_center = center_idx.shape[1]
    xyz_dim = coords.shape[-1]
    centers = coords.gather(1, center_idx.unsqueeze(-1).expand(-1, -1, xyz_dim))
    dist = torch.cdist(centers, coords)
    dist = dist.masked_fill(pad_mask.unsqueeze(1), float("inf"))
    k_eff = min(int(k), int(coords.shape[1]))
    nn_idx = torch.topk(dist, k_eff, largest=False).indices
    flat = nn_idx.reshape(batch, -1)
    neigh = feats.gather(1, flat.unsqueeze(-1).expand(-1, -1, dim))
    neigh = neigh.view(batch, n_center, k_eff, dim)
    valid = ~pad_mask.gather(1, flat).view(batch, n_center, k_eff)
    if reduce == "mean":
        neigh = neigh * valid.unsqueeze(-1).to(dtype=neigh.dtype)
        denom = valid.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=neigh.dtype)
        return neigh.sum(dim=-2) / denom
    fill = torch.finfo(neigh.dtype).min
    neigh = neigh.masked_fill(~valid.unsqueeze(-1), fill)
    pooled = neigh.max(dim=-2).values
    has = valid.any(dim=-1, keepdim=True).to(dtype=pooled.dtype)
    return pooled * has


def knn_maxpool(feat, center_idx, k):
    """Feature-space KNN around FPS centers, then max-pool each neighborhood."""
    center_idx = center_idx.long()
    center_feat = feat.index_select(0, center_idx)
    k_eff = min(int(k), int(feat.shape[0]))
    dist = torch.cdist(center_feat, feat)
    nn_idx = torch.topk(dist, k_eff, largest=False).indices
    return feat.index_select(0, nn_idx.reshape(-1)).view(center_idx.shape[0], k_eff, feat.shape[-1]).max(dim=1).values


def rank_candidates(f_pre, f_txt, token_pad_mask, num_proposal):
    """Eq. (4)(5): sigmoid dot-product, mean over valid tokens, then Top-K.

    token_pad_mask: (B, L) True on padding positions.
    Returns sample indices, the ranking score r, and the mean logit used by the
    existing focal sample loss (that loss applies sigmoid itself).
    """
    logits = torch.einsum("bnd,bld->bnl", f_pre, f_txt) / math.sqrt(f_pre.shape[-1])
    valid = (~token_pad_mask.bool()).unsqueeze(1).to(dtype=logits.dtype)
    attn = torch.sigmoid(logits) * valid
    denom = valid.sum(dim=-1).clamp(min=1.0)
    rank_score = attn.sum(dim=-1) / denom
    loss_logit = (logits * valid).sum(dim=-1) / denom
    n_pre = rank_score.shape[1]
    k = min(int(num_proposal), n_pre)
    if k == n_pre:
        sample_inds = torch.arange(n_pre, device=rank_score.device).unsqueeze(0).repeat(rank_score.shape[0], 1)
    else:
        sample_inds = torch.topk(rank_score, k, dim=1).indices
    return sample_inds.int(), rank_score, loss_logit


class SamplingModule(nn.Module):
    """
    Sample object proposal.
    """
    def __init__(self, num_proposal, pc_dim, lang_dim, d_model):
        super().__init__()
        self.num_proposal = num_proposal
        self.d_model = d_model
        self.query_proj = nn.Linear(pc_dim, d_model, bias=False)
        self.key_proj = nn.Linear(lang_dim, d_model, bias=False)

    def forward(self, pc_feat, lang_feat, pc_mask, lang_mask):
        """
        Args:
            pc_feat: (B, N_pre, D) preliminary candidate features
            lang_feat: (B, L, D)
            lang_mask: (B, L) True on padding
        """
        query = self.query_proj(pc_feat)
        key = self.key_proj(lang_feat)
        scores = torch.einsum("bnd,bld->bnl", query, key) / math.sqrt(self.d_model)
        if pc_mask is not None:
            scores = scores.masked_fill(pc_mask.bool().unsqueeze(2), float("-inf"))
        attn = torch.softmax(scores, dim=1)
        valid = (~lang_mask.bool()).unsqueeze(1).to(dtype=attn.dtype)
        ref_scores = (attn * valid).sum(dim=-1)
        n_pre = ref_scores.shape[1]
        k = min(int(self.num_proposal), n_pre)
        if k == n_pre:
            sample_inds = torch.arange(n_pre, device=ref_scores.device).unsqueeze(0).repeat(ref_scores.shape[0], 1)
        else:
            sample_inds = torch.topk(ref_scores, k, dim=1).indices
        return sample_inds.int(), ref_scores

if __name__=='__main__':
    from easydict import EasyDict
    data_dict = {'point_clouds': torch.rand(16,3000,6).cuda()}
    args = EasyDict({
        'kps_fusion_dim': 256
    })
    dks_net = SamplingModule(num_proposal=512, feat_dim=32, lang_dim=256, d_model=256).cuda()
    pc_xyz = torch.rand(16, 3000, 3).cuda()
    pc_feat = torch.rand(16, 32, 3000).cuda()
    lang_feat = torch.rand(16, 256).cuda()
    data_dict['lang_hidden'] = lang_feat
    out_dict, xyz, feat = dks_net(pc_xyz, pc_feat, data_dict)
    print(xyz.shape, feat.shape)
    for key in sorted(out_dict.keys()):
        print(key, '\t', out_dict[key].shape)