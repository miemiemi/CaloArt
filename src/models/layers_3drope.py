import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..utils import manual_cast


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))
    
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
        if not isinstance(input_size, (tuple, list)) or len(input_size) != 3:
            raise ValueError(f"input_size must be a tuple/list of length 3, got {input_size}")
        if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 3:
            raise ValueError(f"patch_size must be a tuple/list of length 3, got {patch_size}")

        self.patch_size = tuple(patch_size)
        self.input_size = tuple(input_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv = use_conv

        self.grid_size = tuple([s // p for s, p in zip(self.input_size, self.patch_size)])
        self.left_over = tuple(
            [((g + 1) * p - s) % p for p, s, g in zip(self.patch_size, self.input_size, self.grid_size)]
        )
        self.grid_size = tuple([(s + l) // p for p, l, s in zip(self.patch_size, self.left_over, self.input_size)])
        self.num_patches = math.prod(self.grid_size)

        if self.use_conv:
            self.proj = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=bias,
            )
        else:
            patch_volume = math.prod(self.patch_size)
            self.proj = nn.Linear(in_channels * patch_volume, out_channels, bias=bias)

    def forward(self, x):
        """
        input: (B, C, X, Y, Z)
        output: (B, X * Y * Z, D)
        """
        x = F.pad(x, (0, self.left_over[2], 0, self.left_over[1], 0, self.left_over[0]))
        if self.use_conv:
            x = self.proj(x)
            x = rearrange(x, "b c x y z -> b (x y z) c")
        else:
            x = rearrange(
                x, "b c (x px) (y py) (z pz) -> b (x y z) (c px py pz)",
                c=self.in_channels,
                x=self.grid_size[0], y=self.grid_size[1], z=self.grid_size[2],
                px=self.patch_size[0], py=self.patch_size[1], pz=self.patch_size[2]
            )
            x = self.proj(x)

        return x

    def extra_repr(self):
        return (
            f"input_size={self.input_size}, \n"
            f"patch_size={self.patch_size}, \n"
            f"in_channels={self.in_channels}, \n"
            f"out_channels={self.out_channels}, \n"
            f"grid_size={self.grid_size}"
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


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        assert d == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(n, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [embed, torch.zeros(n, self.channels - embed.shape[1], device=embed.device)],
                dim=-1,
            )
        return embed

class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)


class RMSNorm32(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.variance_epsilon = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight is not None:
            x = x * self.weight
        return manual_cast(x, x_dtype)
