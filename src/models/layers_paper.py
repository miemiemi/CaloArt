import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


class VolumeEmbedder(nn.Module):
    def __init__(
        self,
        input_size: list,
        patch_size: list,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.in_channels = in_channels
        self.grid_size = tuple([s // p for s, p in zip(self.input_size, self.patch_size)])
        self.left_over = tuple(
            [((g + 1) * p - s) % p for p, s, g in zip(self.patch_size, self.input_size, self.grid_size)]
        )
        self.grid_size = tuple([(s + l) // p for p, l, s in zip(self.patch_size, self.left_over, self.input_size)])
        self.num_patches = math.prod(self.grid_size)

        self.out_channels = out_channels
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        """
        input: (B, C, X, Y, Z)
        output: (B, X * Y * Z, D)
        """
        x = F.pad(x, (0, self.left_over[2], 0, self.left_over[1], 0, self.left_over[0]))
        x = self.proj(x)
        x = rearrange(x, "b c x y z -> b (x y z) c")
        return x

    def extra_repr(self):
        return (
            f"input_size={self.input_size}, \npatch_size={self.patch_size}, "
            f"\nin_channels={self.in_channels}, \nout_channels={self.out_channels},"
        )


class VolumeUnembedder(nn.Module):
    def __init__(
        self,
        output_size: list,
        patch_size: list,
        out_channels: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_size = output_size
        self.out_channels = out_channels
        self.grid_size = tuple([s // p for s, p in zip(self.output_size, self.patch_size)])
        self.left_over = tuple(
            [((g + 1) * p - s) % p for p, s, g in zip(self.patch_size, self.output_size, self.grid_size)]
        )
        self.grid_size = tuple([(s + l) // p for p, l, s in zip(self.patch_size, self.left_over, self.output_size)])
        self.num_patches = math.prod(self.grid_size)

    def forward(self, x):
        """
        input: (B, T, PX * PY * PZ * C)
        output: (B, C, X, Y, Z)
        """
        gx, gy, gz = self.grid_size
        ox, oy, oz = self.output_size
        px, py, pz = self.patch_size

        assert self.num_patches == x.shape[1], (
            f"Number of patches {x.shape[1]} does not match grid size {self.num_patches}"
        )
        x = rearrange(
            x,
            "b (x y z) (px py pz c) -> b c (x px) (y py) (z pz)",
            x=gx,
            y=gy,
            z=gz,
            px=px,
            py=py,
            pz=pz,
            c=self.out_channels,
        )
        x = x[:, :, :ox, :oy, :oz]
        return x

    def extra_repr(self):
        return f"patch_size={self.patch_size}, \noutput_size={self.output_size}"


def get_1d_sincos_pos_emb_from_grid(emb_dim, pos):
    assert emb_dim % 2 == 0

    omega = np.arange(emb_dim // 2, dtype=np.float64)
    omega /= emb_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_3d_sincos_pos_emb_from_grid(emb_dim, grid):
    assert emb_dim % 3 == 0, "Embedding dimension must be divisible by 3."

    emb_x = get_1d_sincos_pos_emb_from_grid(emb_dim // 3, grid[0])
    emb_y = get_1d_sincos_pos_emb_from_grid(emb_dim // 3, grid[1])
    emb_z = get_1d_sincos_pos_emb_from_grid(emb_dim // 3, grid[2])
    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1)
    return emb


def get_3d_sincos_pos_emb(emb_dim, grid_size):
    grid_x = np.arange(grid_size[0], dtype=np.float32)
    grid_y = np.arange(grid_size[1], dtype=np.float32)
    grid_z = np.arange(grid_size[2], dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(3, 1, grid_size[0], grid_size[1], grid_size[2])
    pos_emb = get_3d_sincos_pos_emb_from_grid(emb_dim, grid)
    return pos_emb


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = (dim // num_heads) if head_dim is None else head_dim
        self.scale = self.head_dim**-0.5
        inner_dim = self.head_dim * self.num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim),
            qkv,
        )
        attention_scores = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        outputs = einsum(attention_probs, v, "b h i j, b h j d -> b h i d")
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.to_out(outputs)
        return outputs


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
