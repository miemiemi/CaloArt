import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union # 移除了 Union[int, ...] 的需求，但在内部检查时可能用到
from einops import rearrange


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange

# class VolumeEmbedder(nn.Module):
#     def __init__(
#         self,
#         input_size: Tuple[int, int, int], 
#         patch_size: Tuple[int, int, int], 
#         in_channels: int,
#         out_channels: int,
#         bias: bool = True,
#         use_conv: bool = False,
#     ):
#         super().__init__()
        
#         # 1. 直接在 init 中进行简单的类型转换和断言
#         # 即使这里没有 helper 函数，我们最好还是转为 tuple 以确保后续计算安全
#         assert isinstance(input_size, (tuple, list)) and len(input_size) == 3, \
#             f"input_size must be a tuple/list of length 3, got {input_size}"
#         self.input_size = tuple(input_size)

#         assert isinstance(patch_size, (tuple, list)) and len(patch_size) == 3, \
#             f"patch_size must be a tuple/list of length 3, got {patch_size}"
#         self.patch_size = tuple(patch_size)
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.use_conv = use_conv

#         # 2. 计算网格尺寸 (Grid Size)
#         self.grid_size = tuple([s // p for s, p in zip(self.input_size, self.patch_size)])
        
#         # 3. 计算 Padding (Left Over)
#         self.left_over = tuple(
#             [((g + 1) * p - s) % p for p, s, g in zip(self.patch_size, self.input_size, self.grid_size)]
#         )
        
#         # 4. 修正后的网格尺寸
#         self.grid_size = tuple([(s + l) // p for p, l, s in zip(self.patch_size, self.left_over, self.input_size)])
#         self.num_patches = math.prod(self.grid_size)

#         # 5. 初始化投影层
#         if self.use_conv:
#             self.proj = nn.Conv3d(
#                 in_channels, 
#                 out_channels, 
#                 kernel_size=self.patch_size, 
#                 stride=self.patch_size, 
#                 bias=bias
#             )
#         else:
#             self.proj = nn.Linear(math.prod(self.patch_size), out_channels, bias=bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.use_conv:
#             # Padding 顺序: (Left, Right, Top, Bottom, Front, Back) -> 对应 input 的 (Z, Y, X)
#             # self.left_over 是 (X_pad, Y_pad, Z_pad)
#             padding = (0, self.left_over[2], 0, self.left_over[1], 0, self.left_over[0])
#             x = F.pad(x, padding) 
#             x = self.proj(x)
#             x = rearrange(x, "b c x y z -> b (x y z) c")
#         else:
#             x = x.squeeze(1) 
#             gx, gy, gz = self.grid_size
#             px, py, pz = self.patch_size
            
#             x = rearrange(
#                 x, 
#                 "b (x px) (y py) (z pz) -> b (x y z) (px py pz)",
#                 x=gx, y=gy, z=gz, 
#                 px=px, py=py, pz=pz
#             )
#             x = self.proj(x)
#         return x

#     def extra_repr(self) -> str:
#         return (
#             f"input_size={self.input_size}, \n"
#             f"patch_size={self.patch_size}, \n"
#             f"in_channels={self.in_channels}, \n"
#             f"out_channels={self.out_channels}, \n"
#             f"grid_size={self.grid_size}"
#         )
    

class VolumeEmbedder(nn.Module):
    def __init__(
        self,
        input_size: list,
        patch_size: list,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_conv: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.in_channels = in_channels
        self.use_conv = use_conv
        self.grid_size = tuple([s // p for s, p in zip(self.input_size, self.patch_size)])
        self.left_over = tuple(
            [((g + 1) * p - s) % p for p, s, g in zip(self.patch_size, self.input_size, self.grid_size)]
        )
        self.grid_size = tuple([(s + l) // p for p, l, s in zip(self.patch_size, self.left_over, self.input_size)])
        self.num_patches = math.prod(self.grid_size)

        self.out_channels = out_channels
        if self.use_conv:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=bias)
        else:
            # out_channels==emb_dim
            self.proj = nn.Linear(math.prod(patch_size), out_channels, bias=bias)

    def forward(self, x):
        """
        input: (B, C, X, Y, Z)
        output: (B, X * Y * Z, D)
        """
        if self.use_conv:
            x = F.pad(x, (0, self.left_over[2], 0, self.left_over[1], 0, self.left_over[0]))  # add padding if needed (padding is inversed)
            x = self.proj(x)
            # x = x.flatten(2).transpose(1, 2) # a bit faster
            x = rearrange(x, "b c x y z -> b (x y z) c")
        else:
            x = x.squeeze(1) # remove channel dimension
            x = rearrange(
                x, "b (x px) (y py) (z pz) -> b (x y z) (px py pz)",
                x=self.grid_size[0], y=self.grid_size[1], z=self.grid_size[2],
                px=self.patch_size[0], py=self.patch_size[1], pz=self.patch_size[2]
            )
            x = self.proj(x)

        return x

    def extra_repr(self):
        return f"input_size={self.input_size}, \npatch_size={self.patch_size}, \nin_channels={self.in_channels}, \nout_channels={self.out_channels},"


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
        input: (B, T, PX * PY * PZ * C) where P stands for the patch size
        output: (B, C, X, Y, Z)
        """
        gx, gy, gz = self.grid_size
        ox, oy, oz = self.output_size
        px, py, pz = self.patch_size

        # fmt: off
        assert self.num_patches == x.shape[1], f"Number of patches {x.shape[1]} does not match grid size {self.num_patches}"
        x = rearrange(x, "b (x y z) (px py pz c) -> b c (x px) (y py) (z pz)", x=gx, y=gy, z=gz, px=px, py=py, pz=pz, c=self.out_channels)
        x = x[:, :, :ox, :oy, :oz]  # limit to original input size
        # fmt: on
        return x

    def extra_repr(self):
        return f"patch_size={self.patch_size}, \noutput_size={self.output_size}"


# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_1d_sincos_pos_emb_from_grid(emb_dim, pos):
    """
    emb_dim: output dimension D for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert emb_dim % 2 == 0  # half goes to cos half goes to sin

    omega = np.arange(emb_dim // 2, dtype=np.float64)
    omega /= emb_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_emb_from_grid(emb_dim, grid):
    assert emb_dim % 3 == 0, "Embedding dimension must be divisible by 3 as each third goes to one direction X,Y,Z"

    emb_x = get_1d_sincos_pos_emb_from_grid(emb_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_emb_from_grid(emb_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_emb_from_grid(emb_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1)  # (X*Y*Z, D)
    return emb


def get_3d_sincos_pos_emb(emb_dim, grid_size):
    grid_x = np.arange(grid_size[0], dtype=np.float32)
    grid_y = np.arange(grid_size[1], dtype=np.float32)
    grid_z = np.arange(grid_size[2], dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")  # here y goes first
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
        # t is of size (batch_size, 1)
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
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim), qkv)
        attention_scores = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        outputs = einsum(attention_probs, v, "b h i j, b h j d -> b h i d")
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.to_out(outputs)

        return outputs


class MLP(nn.Module):
    """From: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py#L13"""

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


class SwishBeta(nn.Module):
    def __init__(self, beta_init=1.0):
        """Swish-β activation with learnable β parameter."""
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))  # Learnable β

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            bias=False,
            drop=None
        ):
        """
        SwiGLU-β: Swish-Gated Linear Unit with Swish-β activation.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.W1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
        self.W2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.swish_beta = SwishBeta()  # Learnable Swish-β activation

    def forward(self, x):
        x1, x2 = self.W1(x).chunk(2, dim=-1)
        return self.W2(x1 * self.swish_beta(x2))


""" WIP"""
class NAC(nn.Module):
    def __init__(self, in_dim, out_dim, init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._W_hat = nn.Parameter(torch.empty(in_dim, out_dim))
        self._M_hat = nn.Parameter(torch.empty(in_dim, out_dim))

        self.register_parameter('W_hat', self._W_hat)
        self.register_parameter('M_hat', self._M_hat)

        for param in self.parameters():
            init_fun(param)

    def forward(self, x):
        W = F.tanh(self._W_hat) * F.sigmoid(self._M_hat)
        return x.matmul(W)


class NALUUnit(nn.Module):
    def __init__(self, in_dim, out_dim, init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._G = nn.Parameter(torch.empty(in_dim, 1))
        self.register_parameter('G', self._G)
        init_fun(self._G)

        self._nac = NAC(in_dim, out_dim, init_fun=init_fun)

        self._epsilon = 1e-8

    def forward(self, x):
        g = F.sigmoid(x.matmul(self._G))

        m = torch.exp(
            self._nac(torch.log(torch.abs(x) + self._epsilon))
        )
        a = self._nac(x)

        y = g * a + (1 - g) * m

        return y


class NALU(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, n_layers=1,
                 init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._nalu_stack = nn.Sequential(*[
            NALUUnit(
                in_dim if i == 0 else hidden_dim,
                out_dim if i == n_layers - 1 else hidden_dim,
                init_fun=init_fun
            )
            for i in range(n_layers)
        ])

    def forward(self, x):
        return self._nalu_stack(x)
