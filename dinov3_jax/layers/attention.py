from __future__ import annotations

import math

import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import eepynox.utils as eu
from eepynox.nn.linear import Linear
from dinov3_jax.config import Dinov3VitConfig
from dinov3_jax.layers.rope import apply_rotary_pos_emb


class Dinov3VitAttention(eqx.Module):
    """Multi-head self-attention with separate Q, K, V, O projections."""

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    use_flash_attn: bool = eqx.field(static=True)

    def __init__(self, config: Dinov3VitConfig, use_flash_attn: bool = True):
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.use_flash_attn = use_flash_attn

        self.q_proj = Linear(self.embed_dim, self.embed_dim, use_bias=config.query_bias)
        self.k_proj = Linear(self.embed_dim, self.embed_dim, use_bias=config.key_bias)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, use_bias=config.value_bias)
        self.o_proj = Linear(self.embed_dim, self.embed_dim, use_bias=config.proj_bias)

    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = "") -> Dinov3VitAttention:
        q_proj = self.q_proj.load_state_dict(state_dict, prefix=prefix + "q_proj.")
        k_proj = self.k_proj.load_state_dict(state_dict, prefix=prefix + "k_proj.")
        v_proj = self.v_proj.load_state_dict(state_dict, prefix=prefix + "v_proj.")
        o_proj = self.o_proj.load_state_dict(state_dict, prefix=prefix + "o_proj.")
        return eu.replace(self, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj)

    def __call__(
        self,
        hidden_states: Array,
        position_embeddings: tuple[Array, Array],
    ) -> Array:
        B, N, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (B, num_heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_flash_attn:
            attn_out = self._flash_attention(q, k, v)
        else:
            attn_out = self._eager_attention(q, k, v)

        # Reshape back to (B, N, hidden_size)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, -1)
        attn_out = self.o_proj(attn_out)
        return attn_out

    def _eager_attention(self, q: Array, k: Array, v: Array) -> Array:
        """Standard scaled dot-product attention."""
        scale = math.sqrt(self.head_dim)
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        return jnp.matmul(attn_weights, v)

    def _flash_attention(self, q: Array, k: Array, v: Array) -> Array:
        """Flash attention via flash_attn_jax."""
        from flash_attn_jax import flash_mha

        dtype = q.dtype
        # flash_mha expects (B, N, H, D) layout
        q = q.transpose(0, 2, 1, 3).astype(jnp.float16)
        k = k.transpose(0, 2, 1, 3).astype(jnp.float16)
        v = v.transpose(0, 2, 1, 3).astype(jnp.float16)
        out = flash_mha(q, k, v).astype(dtype)
        # Back to (B, H, N, D)
        return out.transpose(0, 2, 1, 3)
