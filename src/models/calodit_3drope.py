import math
from functools import partial
from typing import *

import numpy as np

import torch
import torch.nn as nn

from src.utils import str_to_dtype, convert_module_to
from src.models.modules import MultiHeadAttention
from src.models.rope import RotaryPositionEmbedder
from src.models.layers_3drope import (
    AbsolutePositionEmbedder,
    LayerNorm32,
    RMSNorm32,
    SwiGLUFFN,
    VolumeEmbedder,
    VolumeUnembedder,
)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    @torch.compile
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class ContinuousConditionEmbedder(TimestepEmbedder):
    """
    PixArt-style embedder for continuous conditions.

    Each scalar entry is embedded independently with the same sinusoidal
    frequency basis used for timesteps, then the per-dimension embeddings are
    concatenated back together. For an input of shape `(B, D_in)`, the output
    shape is `(B, D_in * hidden_size)`.
    """
    def __init__(self, input_size, hidden_size, frequency_embedding_size=256):
        if hidden_size % input_size != 0:
            raise ValueError(
                f"Continuous condition output dim {hidden_size} must be divisible by "
                f"its input size {input_size} ."
            )
        super().__init__(
            hidden_size=hidden_size // input_size,
            frequency_embedding_size=frequency_embedding_size,
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.outdim is call outdim because PixArt called it, its not the final output size

        self.outdim = hidden_size // input_size 
        
    def forward(self, s, bs: Optional[int] = None):
        if s.ndim == 1:
            s = s[:, None]
        if s.ndim != 2:
            # here we can take (B, D_in)
            raise ValueError(f"Expected a 2D condition tensor, got shape {tuple(s.shape)}.")

        if bs is None:
            bs = s.shape[0]
        if s.shape[0] != bs:
            if bs % s.shape[0] != 0:
                raise ValueError(f"Cannot broadcast condition batch {s.shape[0]} to target batch {bs}.")
            s = s.repeat(bs // s.shape[0], 1)

        if s.shape[1] != self.input_size:
            raise ValueError(
                f"Expected condition width {self.input_size}, got {s.shape[1]}."
            )

        batch_size, dims = s.shape
        s = s.reshape(batch_size * dims)
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        return s_emb.reshape(batch_size, dims * self.outdim)


class DiscreteConditionEmbedder(nn.Module):
    def __init__(self, num_embeddings, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, hidden_size)

    def forward(self, x):
        return self.embedding(x)


class CaloLightningDiTBlock(nn.Module):
    """
    3D transformer block used by CaloLightningDiT.

    This block extends a LightningDiT-style design to 3D calorimeter tokens
    and supports shared AdaLN modulation across blocks. 

    Notes:
        - Some options are experimental and may not be enabled in the final
          training configuration.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0), 
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        share_mod: bool = False,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.use_rmsnorm = use_rmsnorm
        if not use_rmsnorm:
            self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
            self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm32(channels, eps=1e-6)
            self.norm2 = RMSNorm32(channels, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = SwiGLUFFN(channels, mlp_hidden_dim, drop=proj_drop)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.scale_shift_table = nn.Parameter(torch.randn(6, channels) / channels ** 0.5)

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.share_mod:
            mod = mod.view(mod.shape[0], 6, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + mod
            ).to(dtype=mod.dtype).unbind(dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, phases=phases)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    @torch.compile
    def forward(self, x: torch.Tensor, mod: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, phases, use_reentrant=False)
        else:
            return self._forward(x, mod, phases)

class FinalLayer(nn.Module):
    """
    Final token-to-patch prediction head for CaloLightningDiT.

    The layer first normalizes each token, then uses AdaLN-style conditioning
    to produce per-sample shift, scale, and gate parameters from the global
    modulation vector. The normalized tokens are affine-modulated, blended
    with the original tokens through a sigmoid gate, and finally projected to
    `patch_volume * out_channels` so they can be unpatchified back into the
    3D calorimeter volume.

    Compared with the vanilla DiT output head, this version keeps an explicit
    gated interpolation between the residual stream and the conditioned branch
    before the final linear projection.
    """
    def __init__(
        self,
        channels: int,
        patch_size: Tuple[int, int, int], 
        out_channels: int,
        use_checkpoint: bool = False,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_rmsnorm = use_rmsnorm
        if not use_rmsnorm:
            self.norm_final = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm32(channels, eps=1e-6)
            
        if isinstance(patch_size, int):
            patch_vol = patch_size
        else:
            patch_vol = math.prod(patch_size)

        self.linear = nn.Linear(channels, patch_vol * out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        # mod: (B, C)
        # linear projection -> (B, 3C)
        shift, scale, gate = self.adaLN_modulation(mod).chunk(3, dim=1)
        
        gate = torch.sigmoid(gate)
        h = self.norm_final(x)
        
        # (B, 1, C)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Gating
        x = (1 - gate.unsqueeze(1)) * x + gate.unsqueeze(1) * h
        x = self.linear(x)
        return x

    @torch.compile
    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        return self._forward(x, mod)

class CaloLightningDiT(nn.Module):
    """
    3D CaloDiT backbone with configurable positional encoding and AdaLN
    conditioning.

    The model consumes calorimeter volumes with shape
    `(N, in_channels, R, PHI, Z)`, patchifies them into 3D tokens, applies a
    stack of CaloLightningDiT blocks, and unpatchifies the output back to the
    original spatial layout.

    Positional encoding:
        - `ape`: absolute positional embeddings on the patch grid.
        - `rope`: rotary positional encoding computed from the same 3D patch
          grid used by patch embedding.

    The patch grid size is:
        `(R / pR, PHI / pPHI, Z / pZ)`
    where `patch_size = (pR, pPHI, pZ)`.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int], 
        patch_size: Tuple[int, int, int], 
        conditions_size: tuple,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        pe_mode: Literal["ape", "rope", "ape+rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'float32',
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rmsnorm: bool = False,
        use_conv: bool = False,
        condition_embed_dims: Optional[Sequence[int]] = None,
        **kwargs
    ):
        super().__init__()
        # 1. 基础参数保存
        self.input_size = tuple(input_size)
        self.patch_size = tuple(patch_size)
        self.conditions_size = conditions_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.use_rmsnorm = use_rmsnorm
        self.dtype = str_to_dtype(dtype) if isinstance(dtype, str) else dtype

        # 2. 核心修改：先初始化 Patch Embedder
        # 我们直接使用 VolumeEmbedder 来处理 grid_size 的计算逻辑
        # 这样确保 RoPE 的网格和实际 Patch 的网格 100% 对齐
        self.patch_embedder = VolumeEmbedder(
            input_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            out_channels=self.model_channels,
            use_conv=use_conv
        )
        
        # 3. 获取正确的 grid_size (D, H, W)
        self.grid_size = self.patch_embedder.grid_size
        self.num_patches = self.patch_embedder.num_patches

        # ----------------------------------------------------------------
        # 4. 位置编码逻辑 (Position Embedding)
        # 支持三种模式："ape" / "rope" / "ape+rope"
        # 参考 JiT (Lightning-DiT)：APE 加到 token, RoPE 旋转 Q/K，两者可共存
        # ----------------------------------------------------------------
        self._use_ape = pe_mode in ("ape", "ape+rope")
        self._use_rope = pe_mode in ("rope", "ape+rope")

        # 生成 3D 网格坐标 (APE / RoPE 都需要)
        coords = torch.meshgrid(
            *[torch.arange(s) for s in self.grid_size],
            indexing='ij'
        )
        # stack -> (D, H, W, 3) -> flatten -> (N_patches, 3)
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)

        if self._use_ape:
            # Absolute Position Embedding
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            pos_emb = pos_embedder(coords)  # (N_patches, model_channels)
            self.register_buffer("pos_emb", pos_emb.unsqueeze(0))  # (1, N, D)
        else:
            self.pos_emb = None

        if self._use_rope:
            # Rotary Position Embedding — 预计算相位/频率表
            rope_embedder = RotaryPositionEmbedder(self.model_channels // self.num_heads, 3)
            rope_phases = rope_embedder(coords)
            self.register_buffer("rope_phases", rope_phases)
        else:
            self.rope_phases = None

        # ----------------------------------------------------------------
        # 5. 其他组件初始化
        # ----------------------------------------------------------------
        self.t_embedder = TimestepEmbedder(model_channels)
        self.num_condition_components = len(self.conditions_size)
        self.has_label_condition = self.num_condition_components == 4
        self.condition_embed_dims = tuple(int(dim) for dim in condition_embed_dims)
        assert len(self.condition_embed_dims) == self.num_condition_components
        assert sum(self.condition_embed_dims) == self.model_channels

        self.energy_embedder = ContinuousConditionEmbedder(
            self.conditions_size[0], self.condition_embed_dims[0]
        )
        self.phi_embedder = ContinuousConditionEmbedder(
            self.conditions_size[1], self.condition_embed_dims[1]
        )
        self.theta_embedder = ContinuousConditionEmbedder(
            self.conditions_size[2], self.condition_embed_dims[2]
        )
        if self.has_label_condition:
            self.label_embedder = DiscreteConditionEmbedder(
                self.conditions_size[3], self.condition_embed_dims[3]
            )
        else:
            self.label_embedder = None

        # 共享调制层 (如果启用)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            CaloLightningDiTBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=self._use_rope,
                rope_freq=rope_freq, # Block 内部通常不需要 freq，只需要 rope_phases，视具体实现而定
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                attn_drop=self.attn_drop,
                proj_drop=self.proj_drop,
                use_rmsnorm=self.use_rmsnorm,
            )
            for _ in range(num_blocks)
        ])

        # 6. 输出层与 Unpatchify
        # 修正：FinalLayer 的输出必须匹配 Unpatchify 的输入要求 (Patch体积 * C_out)
        self.final_layer = FinalLayer(
            channels=model_channels,
            patch_size=self.patch_size,
            out_channels=out_channels,
            use_checkpoint=self.use_checkpoint,
            use_rmsnorm=self.use_rmsnorm,
        )
        
        self.unpatchify = VolumeUnembedder(
            output_size=self.input_size, # 还原回原始尺寸
            patch_size=self.patch_size,
            out_channels=out_channels
        )

        self.initialize_weights()
        self.convert_to(self.dtype)

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to(self, dtype: torch.dtype) -> None:
        """
        Convert the torso of the model to the specified dtype.
        """
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, (nn.Linear, nn.Conv3d)):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Initialize patch embedder like nn.Linear (instead of nn.Conv2d):
            w = self.patch_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.patch_embedder.proj.bias, 0)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Initialize condition embedders:
            for c_embedder in (self.energy_embedder, self.phi_embedder, self.theta_embedder):
                nn.init.normal_(c_embedder.mlp[0].weight, std=0.02)
                nn.init.normal_(c_embedder.mlp[2].weight, std=0.02)
            if self.label_embedder is not None:
                nn.init.normal_(self.label_embedder.embedding.weight, std=0.02)

            # share_mod modulation initialize ? if grad boom maybe i should initialize them to 0

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.normal_(self.adaLN_modulation[-1].weight, std=0.02)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # C. Final Layer (关键：Zero-Init)
            # 1. Zero-out 线性投影层，使得初始输出为 0 (类似高斯噪声的均值)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

            # if hasattr(self.final_layer, 'adaLN_modulation'):
            #     nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            #     nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        elif self.initialization == 'scaled':
            # Initialize transformer layers:
            pass

    def _embed_label_condition(self, label: torch.Tensor) -> torch.Tensor:
        if self.label_embedder is None:
            raise RuntimeError("Label embedder is not initialized for this model.")
        if label.shape[-1] == 1:
            label_ids = label.reshape(-1).long()
        else:
            label_ids = label.argmax(dim=-1).long()
        return self.label_embedder(label_ids)

    def _embed_conditions(self, c: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(c) != self.num_condition_components:
            raise ValueError(
                f"Expected {self.num_condition_components} condition tensors, got {len(c)}."
            )

        condition_embs = [
            self.energy_embedder(c[0], bs=c[0].shape[0]),
            self.phi_embedder(c[1], bs=c[0].shape[0]),
            self.theta_embedder(c[2], bs=c[0].shape[0]),
        ]
        if self.has_label_condition:
            condition_embs.append(self._embed_label_condition(c[3]))
        return torch.cat(condition_embs, dim=-1)

    def forward(self, x: torch.Tensor, c: Tuple[torch.Tensor, ...], t: torch.Tensor):
            """
            x: (N, in_channels, R, PHI, Z) tensor of showers
            c: ((N, 1), (N, 2), (N, 1)) or ((N, 1), (N, 2), (N, 1), (N, K))
               tuple of conditioning tensors
            t: (N,) tensor of diffusion timesteps
            """
            # 1. 计算基础 Embeddings
            t_emb = self.t_embedder(t) # Shape: (N, D)

            # 条件元组使用独立的 embedder 编码，并沿特征维做等宽拼接
            c_cond = self._embed_conditions(c)

            # 2. 生成全局原始条件向量 (Global Conditioning Vector)
            # 条件之间不再相加，而是先 concat，再与 timestep embedding 融合
            c_global = t_emb + c_cond

            # 3. 准备骨干网络专用条件 (Backbone Conditioning)
            if self.share_mod:
                # 如果启用共享调制，在此处统一投影到 6*D (或者 Block 需要的维度)
                c_backbone = self.adaLN_modulation(c_global)
            else:
                # 如果独立调制，保持原始维度 D，交给 Block 内部的 MLP 处理
                c_backbone = c_global

            # 4. Patchify (Image -> Tokens)
            x = self.patch_embedder(x)

            # 5. 位置编码处理
            # APE：直接加到 token embedding 上
            if self._use_ape:
                x = x + self.pos_emb

            # RoPE：准备相位信息传给 Attention（Q/K 旋转）
            rope = self.rope_phases if self._use_rope else None

            # 6. Transformer Blocks 循环
            # 关键点：这里传入 c_backbone
            for block in self.blocks:
                x = block(x, c_backbone, phases=rope)

            # 7. Final Layer
            # 关键点：这里传入原始的 c_global (维度 D)
            # 因为 FinalLayer 定义了自己的映射层 (adaLN_modulation_condn)，它期望输入维度为 model_channels
            x = self.final_layer(x, c_global)    

            # 8. Unpatchify (Tokens -> Image)
            x = self.unpatchify(x)
            
            return x
    
    @property
    def example_input(self):
        x = torch.randn(1, self.in_channels, *self.input_size)
        c = []
        for idx, dim in enumerate(self.conditions_size):
            if self.has_label_condition and idx == len(self.conditions_size) - 1:
                label = torch.zeros(1, dim)
                label[0, 0] = 1
                c.append(label)
            else:
                c.append(torch.randn(1, dim))
        t = torch.randn(1)
        return (x, tuple(c), t)
