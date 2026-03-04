from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

import eepynox.utils as eu
from dinov3_jax.config import Dinov3VitConfig
from dinov3_jax.layers.rms_norm import LayerNorm
from dinov3_jax.layers.attention import Dinov3VitAttention
from dinov3_jax.layers.layer_scale import Dinov3VitLayerScale
from dinov3_jax.layers.mlp import Dinov3VitMLP, Dinov3VitGatedMLP


class Dinov3VitLayer(eqx.Module):
    """Single transformer block matching HF DINOv3ViTLayer."""

    norm1: LayerNorm
    attention: Dinov3VitAttention
    layer_scale1: Dinov3VitLayerScale
    norm2: LayerNorm
    mlp: Dinov3VitMLP | Dinov3VitGatedMLP
    layer_scale2: Dinov3VitLayerScale

    def __init__(self, config: Dinov3VitConfig, use_flash_attn: bool = True):
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov3VitAttention(config, use_flash_attn=use_flash_attn)
        self.layer_scale1 = Dinov3VitLayerScale(config.hidden_size)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_gated_mlp:
            self.mlp = Dinov3VitGatedMLP(config)
        else:
            self.mlp = Dinov3VitMLP(config)
        self.layer_scale2 = Dinov3VitLayerScale(config.hidden_size)

    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = "") -> Dinov3VitLayer:
        norm1 = self.norm1.load_state_dict(state_dict, prefix=prefix + "norm1.")
        attention = self.attention.load_state_dict(state_dict, prefix=prefix + "attention.")
        layer_scale1 = self.layer_scale1.load_state_dict(state_dict, prefix=prefix + "layer_scale1.")
        norm2 = self.norm2.load_state_dict(state_dict, prefix=prefix + "norm2.")
        mlp = self.mlp.load_state_dict(state_dict, prefix=prefix + "mlp.")
        layer_scale2 = self.layer_scale2.load_state_dict(state_dict, prefix=prefix + "layer_scale2.")
        return eu.replace(
            self,
            norm1=norm1, attention=attention, layer_scale1=layer_scale1,
            norm2=norm2, mlp=mlp, layer_scale2=layer_scale2,
        )

    def __call__(
        self,
        hidden_states: Array,
        position_embeddings: tuple[Array, Array],
    ) -> Array:
        # Attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, position_embeddings=position_embeddings)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = hidden_states + residual

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
