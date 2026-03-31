import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn

from src.models.layers_paper import (
    MLP,
    Attention,
    SinusoidalPositionEmbeddings,
    VolumeEmbedder,
    VolumeUnembedder,
    get_3d_sincos_pos_emb,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.t_embedding = SinusoidalPositionEmbeddings(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        return self.mlp(self.t_embedding(t))


class ConditionEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.mlp(x)


class CaloDiTPaperBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = partial(nn.GELU, approximate="tanh")
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class CaloDiTPaper(nn.Module):
    """Paper-era CaloDiT backbone copied from the earlier diffusion4sim implementation."""

    def __init__(
        self,
        input_size: tuple,
        patch_size: tuple,
        conditions_size: tuple,
        in_channels=1,
        out_channels=1,
        emb_dim=384,
        num_heads=6,
        num_layers=6,
        mlp_ratio=4,
    ):
        super().__init__()
        self.input_size = input_size
        self.conditions_size = conditions_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = emb_dim

        self.patchify = VolumeEmbedder(input_size, patch_size, in_channels, emb_dim)
        self.num_patches = self.patchify.num_patches
        self.grid_size = self.patchify.grid_size

        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim), requires_grad=False)

        self.t_embedder = TimestepEmbedder(emb_dim)
        self.c_embedders = nn.ModuleList(
            [ConditionEmbedder(c_size, emb_dim) for c_size in conditions_size]
        )

        self.blocks = nn.ModuleList(
            [CaloDiTPaperBlock(emb_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(emb_dim, patch_size, out_channels)
        self.unpatchify = VolumeUnembedder(input_size, patch_size, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        self.pos_emb.data.copy_(
            torch.from_numpy(get_3d_sincos_pos_emb(self.emb_dim, self.grid_size)).float().unsqueeze(0)
        )

        w = self.patchify.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patchify.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for c_embedder in self.c_embedders:
            nn.init.normal_(c_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(c_embedder.mlp[2].weight, std=0.02)
            nn.init.normal_(c_embedder.mlp[4].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: Tuple[torch.Tensor, ...], t: torch.Tensor):
        t = self.t_embedder(t)
        c = tuple(c_embedder(c_i) for c_embedder, c_i in zip(self.c_embedders, c))
        c = t + sum(c)

        x = self.patchify(x) + self.pos_emb
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    @property
    def example_input(self):
        x = torch.randn(1, self.in_channels, *self.input_size)
        c = tuple(torch.randn(1, dim) for dim in self.conditions_size)
        t = torch.randn(1)
        return (x, c, t)
