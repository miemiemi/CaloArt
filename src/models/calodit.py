import math
from typing import Tuple

import torch
import torch.nn as nn

from src.models.layers import (
    Attention,
    SinusoidalPositionEmbeddings,
    SwiGLU,
    VolumeEmbedder,
    VolumeUnembedder,
    get_3d_sincos_pos_emb,
)


def modulate(x, scale, shift=None):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale) + shift


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


class CaloDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, grid_size, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.seq_len = grid_size[0] * grid_size[1] * grid_size[2]
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0.0)
        self.adaLN_modulation_pos = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.adaLN_modulation_condn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, pos_emb):
        c_ada = self.adaLN_modulation_condn(c).chunk(6, dim=-1)
        pos_ada = self.adaLN_modulation_pos(pos_emb).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = list(
            map(
                lambda x, y: x.unsqueeze(1).expand(-1, self.seq_len, -1) + y.expand(c.shape[0], -1, -1),
                c_ada,
                pos_ada,
            )
        ) # (B, seq_len, hidden_size) diff with standard DiT (B, hidden_size)
        gate_msa, gate_mlp = torch.sigmoid(gate_msa), torch.sigmoid(gate_mlp)
        x = (1 - gate_msa) * x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = (1 - gate_mlp) * x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, grid_size):
        super().__init__()
        self.seq_len = grid_size[0] * grid_size[1] * grid_size[2]
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.adaLN_modulation_condn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.adaLN_modulation_pos = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )

    def forward(self, x, c, pos_emb):
        c_ada = self.adaLN_modulation_condn(c).chunk(3, dim=-1)
        pos_ada = self.adaLN_modulation_pos(pos_emb).chunk(3, dim=-1)
        gate, shift, scale = list(
            map(
                lambda x, y: x.unsqueeze(1).expand(-1, self.seq_len, -1) + y.expand(c.shape[0], -1, -1),
                c_ada,
                pos_ada,
            )
        )
        gate = torch.sigmoid(gate)
        x = (1 - gate) * x + gate * modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class CaloDiT(nn.Module):
    """Baseline CaloDiT architecture kept isolated from CaloLightningDiT."""

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
        pos_type="3DFixed",
        use_conv=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.conditions_size = conditions_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = emb_dim
        self.pos_type = pos_type

        self.patchify = VolumeEmbedder(input_size, patch_size, in_channels, emb_dim, use_conv=use_conv)
        self.num_patches = self.patchify.num_patches
        self.grid_size = self.patchify.grid_size

        if self.pos_type == "3DFixed":
            self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim), requires_grad=False)
        elif self.pos_type == "Learned":
            self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim), requires_grad=True)
        else:
            raise ValueError(f"Unsupported pos_type: {self.pos_type}")

        self.t_embedder = TimestepEmbedder(emb_dim)
        self.c_embedders = nn.ModuleList(
            [ConditionEmbedder(c_size, emb_dim) for c_size in conditions_size]
        )

        self.blocks = nn.ModuleList(
            [CaloDiTBlock(emb_dim, num_heads, self.grid_size, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(emb_dim, patch_size, out_channels, self.grid_size)
        self.unpatchify = VolumeUnembedder(input_size, patch_size, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.pos_type == "3DFixed":
            self.pos_emb.data.copy_(
                torch.from_numpy(get_3d_sincos_pos_emb(self.emb_dim, self.grid_size)).float().unsqueeze(0)
            )
        elif self.pos_type == "Learned":
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

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
            nn.init.constant_(block.adaLN_modulation_condn[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_condn[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation_pos[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_pos[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation_condn[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation_condn[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation_pos[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation_pos[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: Tuple[torch.Tensor, ...], t: torch.Tensor):
        t = self.t_embedder(t)
        c = tuple(c_embedder(c_i) for c_embedder, c_i in zip(self.c_embedders, c))
        c = t + sum(c)

        x = self.patchify(x) + self.pos_emb
        for block in self.blocks:
            x = block(x, c, self.pos_emb)
        x = self.final_layer(x, c, self.pos_emb)
        x = self.unpatchify(x)
        return x

    @property
    def example_input(self):
        x = torch.randn(1, self.in_channels, *self.input_size)
        c = tuple(torch.randn(1, dim) for dim in self.conditions_size)
        t = torch.randn(1)
        return (x, c, t)
