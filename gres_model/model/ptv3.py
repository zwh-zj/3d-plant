"""Point Transformer V3 encoder-decoder in pure PyTorch.

Serialized z-order attention, no flash_attn. Output is one feature per input point.
"""

import torch
import torch.nn as nn

_ORDERS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0))


def serialization_code(coord, batch, order, grid_size):
    grid = torch.floor(coord / grid_size)
    grid = grid[:, list(order)]
    grid = grid - grid.min(dim=0).values
    grid = grid.clamp(min=0, max=65535).long()
    code = torch.zeros(grid.shape[0], dtype=torch.long, device=grid.device)
    for bit in range(16):
        for axis in range(3):
            code = code | (((grid[:, axis] >> bit) & 1) << (3 * bit + axis))
    return code + batch.long() * (1 << 48)


def scatter_mean_idx(src, index, dim_size):
    out = src.new_zeros(dim_size, src.shape[1])
    count = src.new_zeros(dim_size, 1)
    out.index_add_(0, index, src)
    ones = torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, ones)
    return out / count.clamp(min=1)


class SerializedAttention(nn.Module):

    def __init__(self, channels, num_heads, patch_size):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)

    def forward(self, feat):
        num_points, channels = feat.shape
        pad = (self.patch_size - num_points % self.patch_size) % self.patch_size
        if pad:
            feat = torch.cat([feat, feat.new_zeros(pad, channels)], dim=0)
        num_patches = feat.shape[0] // self.patch_size
        x = feat.view(num_patches, self.patch_size, channels)
        qkv = self.qkv(x).view(num_patches, self.patch_size, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if pad:
            valid = x.new_ones(num_patches, self.patch_size, dtype=torch.bool)
            valid[-1, -pad:] = False
            attn = attn.masked_fill(~valid[:, None, None, :], -1e4)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(num_patches * self.patch_size, channels)
        out = self.proj(out)
        if pad:
            out = out[:-pad]
        return out


class Block(nn.Module):

    def __init__(self, channels, num_heads, patch_size, order, mlp_ratio=4):
        super().__init__()
        self.order = order
        self.patch_size = patch_size
        self.cpe = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = SerializedAttention(channels, num_heads, patch_size)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, coord, feat, batch, grid_size):
        code = serialization_code(coord, batch, self.order, grid_size)
        order = torch.argsort(code)
        sorted_feat = feat[order]
        local = sorted_feat.transpose(0, 1).unsqueeze(0)
        sorted_feat = sorted_feat + self.cpe(local).squeeze(0).transpose(0, 1)
        sorted_feat = sorted_feat + self.attn(self.norm1(sorted_feat))
        sorted_feat = sorted_feat + self.mlp(self.norm2(sorted_feat))
        feat = torch.empty_like(sorted_feat)
        feat[order] = sorted_feat
        return feat


class GridPool(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, coord, feat, batch, grid_size):
        feat = self.proj(feat)
        grid = torch.floor(coord / grid_size)
        grid = grid - grid.min(dim=0).values
        grid = grid.clamp(min=0, max=65535).long()
        key = batch.long() * (1 << 48)
        key = key | (grid[:, 0] << 32) | (grid[:, 1] << 16) | grid[:, 2]
        _, inverse = torch.unique(key, return_inverse=True)
        num_pooled = int(inverse.max().item()) + 1
        pooled_feat = scatter_mean_idx(feat, inverse, num_pooled)
        pooled_coord = scatter_mean_idx(coord, inverse, num_pooled)
        pooled_batch = torch.zeros(num_pooled, dtype=torch.long, device=batch.device)
        pooled_batch.scatter_(0, inverse, batch.long())
        return pooled_coord, pooled_feat, pooled_batch, inverse


class Stage(nn.Module):

    def __init__(self, channels, depth, num_heads, patch_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(channels, num_heads, patch_size, _ORDERS[i % len(_ORDERS)])
            for i in range(depth)
        ])

    def forward(self, coord, feat, batch, grid_size):
        for block in self.blocks:
            feat = block(coord, feat, batch, grid_size)
        return feat


class DecoderStage(nn.Module):

    def __init__(self, coarse_channels, skip_channels, out_channels, depth, num_heads, patch_size):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(coarse_channels + skip_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
        self.stage = Stage(out_channels, depth, num_heads, patch_size)

    def forward(self, coarse_feat, skip_coord, skip_feat, skip_batch, inverse, grid_size):
        feat = self.fuse(torch.cat([coarse_feat[inverse], skip_feat], dim=-1))
        return self.stage(skip_coord, feat, skip_batch, grid_size)


class PointTransformerV3(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        enc_channels=(32, 64, 128, 256, 512),
        enc_depths=(2, 2, 2, 6, 2),
        enc_num_heads=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_channels=(256, 128, 64, 32),
        dec_depths=(2, 2, 2, 2),
        dec_num_heads=(16, 8, 4, 2),
        dec_patch_size=(1024, 1024, 1024, 1024),
        stride=2,
        grid_size=1.0,
    ):
        super().__init__()
        if len(dec_channels) != len(enc_channels) - 1:
            raise ValueError("decoder stages must be encoder stages minus one")
        self.grid_size = float(grid_size)
        self.stride = int(stride)
        self.embed = nn.Sequential(
            nn.Linear(in_channels, enc_channels[0]),
            nn.LayerNorm(enc_channels[0]),
            nn.GELU(),
        )
        self.enc_stages = nn.ModuleList([
            Stage(enc_channels[i], enc_depths[i], enc_num_heads[i], enc_patch_size[i])
            for i in range(len(enc_channels))
        ])
        self.downs = nn.ModuleList([
            GridPool(enc_channels[i], enc_channels[i + 1])
            for i in range(len(enc_channels) - 1)
        ])
        self.dec_stages = nn.ModuleList([
            DecoderStage(
                coarse_channels=enc_channels[-1] if i == 0 else dec_channels[i - 1],
                skip_channels=enc_channels[-(i + 2)],
                out_channels=dec_channels[i],
                depth=dec_depths[i],
                num_heads=dec_num_heads[i],
                patch_size=dec_patch_size[i],
            )
            for i in range(len(dec_channels))
        ])
        self.out_proj = nn.Linear(dec_channels[-1], out_channels)

    def forward(self, coord, feat, batch):
        feat = self.embed(feat)
        grid = self.grid_size
        skips = []
        inverses = []
        for stage_id, stage in enumerate(self.enc_stages):
            if stage_id > 0:
                coord, feat, batch, inverse = self.downs[stage_id - 1](coord, feat, batch, grid)
                inverses.append(inverse)
                grid = grid * self.stride
            feat = stage(coord, feat, batch, grid)
            skips.append((coord, feat, batch, grid))
        for dec_id, dec in enumerate(self.dec_stages):
            skip_coord, skip_feat, skip_batch, skip_grid = skips[-(dec_id + 2)]
            feat = dec(feat, skip_coord, skip_feat, skip_batch, inverses[-(dec_id + 1)], skip_grid)
        return self.out_proj(feat)
