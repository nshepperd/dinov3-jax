from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from dinov3_jax.config import Dinov3VitConfig
from dinov3_jax.layers.attention import Dinov3VitAttention
from dinov3_jax.layers.layer_scale import Dinov3VitLayerScale
from dinov3_jax.layers.mlp import Dinov3VitGatedMLP, Dinov3VitMLP
from dinov3_jax.layers.rms_norm import LayerNorm


class Dinov3VitLayer(eqx.Module):
    """Single transformer block matching HF DINOv3ViTLayer."""

    norm1: LayerNorm
    attention: Dinov3VitAttention
    layer_scale1: Dinov3VitLayerScale
    norm2: LayerNorm
    mlp: Dinov3VitMLP | Dinov3VitGatedMLP
    layer_scale2: Dinov3VitLayerScale

    def __init__(
        self, config: Dinov3VitConfig, use_flash_attn: bool = True, dtype=jnp.float32
    ):
        self.norm1 = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )
        self.attention = Dinov3VitAttention(
            config, use_flash_attn=use_flash_attn, dtype=dtype
        )
        self.layer_scale1 = Dinov3VitLayerScale(config.hidden_size)
        self.norm2 = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, dtype=dtype
        )
        if config.use_gated_mlp:
            self.mlp = Dinov3VitGatedMLP(config, dtype=dtype)
        else:
            self.mlp = Dinov3VitMLP(config, dtype=dtype)
        self.layer_scale2 = Dinov3VitLayerScale(config.hidden_size)

    def __call__(
        self,
        hidden_states: Array,
        position_embeddings: tuple[Array, Array],
    ) -> Array:
        # Attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(
            hidden_states, position_embeddings=position_embeddings
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = hidden_states + residual

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
