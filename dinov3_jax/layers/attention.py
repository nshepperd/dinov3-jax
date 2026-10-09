from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from dinov3_jax.config import Dinov3VitConfig
from dinov3_jax.eepynox.nn.linear import Linear
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

    def __init__(
        self, config: Dinov3VitConfig, use_flash_attn: bool = True, dtype=jnp.float32
    ):
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.use_flash_attn = use_flash_attn

        self.q_proj = Linear(
            self.embed_dim, self.embed_dim, use_bias=config.query_bias, dtype=dtype
        )
        self.k_proj = Linear(
            self.embed_dim, self.embed_dim, use_bias=config.key_bias, dtype=dtype
        )
        self.v_proj = Linear(
            self.embed_dim, self.embed_dim, use_bias=config.value_bias, dtype=dtype
        )
        self.o_proj = Linear(
            self.embed_dim, self.embed_dim, use_bias=config.proj_bias, dtype=dtype
        )

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

        # Both attention paths take (B, N, num_heads, head_dim)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        if self.use_flash_attn:
            attn_out = self._flash_attention(q, k, v)
        else:
            attn_out = self._eager_attention(q, k, v)

        # Reshape back to (B, N, hidden_size)
        attn_out = attn_out.reshape(B, N, -1)
        attn_out = self.o_proj(attn_out)
        return attn_out

    def _eager_attention(self, q: Array, k: Array, v: Array) -> Array:
        """Scaled dot-product attention via XLA.

        Logits and softmax are computed in float32 (layer 0 logits reach ~1e6,
        which overflows float16).
        """
        return jax.nn.dot_product_attention(q, k, v)

    def _flash_attention(self, q: Array, k: Array, v: Array) -> Array:
        """Flash attention via fa4_jax."""
        from fa4_jax import flash_attn

        dtype = q.dtype
        out = flash_attn(
            q.astype(jnp.float16), k.astype(jnp.float16), v.astype(jnp.float16)
        )
        return out.astype(dtype)
