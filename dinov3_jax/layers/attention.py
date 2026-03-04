from typing import Optional, Tuple

import einops
import equinox as eqx
import jax.numpy as jnp
from flash_attn_jax import flash_mha
from jaxtyping import Array, Float

import eepynox.utils as eu
from eepynox.nn.linear import Linear


def rope_rotate_half(x: Array) -> Array:
    """Rotate half of the dimensions for RoPE."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def rope_apply(x: Array, sin: Array, cos: Array) -> Array:
    """Apply RoPE rotation to input tensor."""
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(eqx.Module):
    """Linear layer with masked bias for K (keys) in QKV projection."""
    weight: Float[Array, "out_features in_features"] | None
    bias: Float[Array, "out_features"] | None
    bias_mask: Float[Array, "out_features"] | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super().__init__()
        assert out_features % 3 == 0
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = None
        self.bias = None
        self.bias_mask = None

    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        assert state_dict[prefix + "weight"].shape == (self.out_features, self.in_features)
        weight = state_dict.pop(prefix + "weight").astype(jnp.float32)
        
        if self.use_bias:
            assert state_dict[prefix + "bias"].shape == (self.out_features,)
            bias = state_dict.pop(prefix + "bias").astype(jnp.float32)
        else:
            bias = None
        
        # Create bias mask (NaN for K, 1 for Q and V)
        k_size = self.out_features // 3
        bias_mask = jnp.ones(self.out_features)
        bias_mask = bias_mask.at[k_size:2*k_size].set(jnp.nan)
        
        return eu.replace(self, weight=weight, bias=bias, bias_mask=bias_mask)

    def __call__(self, x: Array) -> Array:
        y = x @ self.weight.T
        
        if self.use_bias:
            assert self.bias is not None and self.bias_mask is not None
            masked_bias = self.bias * self.bias_mask
            # Replace NaN with 0 for computation
            masked_bias = jnp.where(jnp.isnan(masked_bias), 0, masked_bias)
            y = y + masked_bias
        
        return y


class SelfAttention(eqx.Module):
    """Multi-head self-attention with optional RoPE."""
    qkv: Linear | LinearKMaskedBias
    proj: Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        mask_k_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        
        # QKV projection
        if mask_k_bias:
            self.qkv = LinearKMaskedBias(dim, dim * 3, use_bias=qkv_bias)
        else:
            self.qkv = Linear(dim, dim * 3, use_bias=qkv_bias)
        
        # Output projection
        self.proj = Linear(dim, dim, use_bias=proj_bias)
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        qkv = self.qkv.load_state_dict(state_dict, prefix=prefix + "qkv.")
        proj = self.proj.load_state_dict(state_dict, prefix=prefix + "proj.")
        return eu.replace(self, qkv=qkv, proj=proj)

    def apply_rope(
        self, 
        q: Array, 
        k: Array, 
        rope: Optional[Tuple[Array, Array]]
    ) -> Tuple[Array, Array]:
        """Apply RoPE to queries and keys."""
        if rope is None:
            return q, k
        
        sin, cos = rope
        rope_dtype = sin.dtype
        
        # Cast to rope dtype for computation
        q_dtype = q.dtype
        k_dtype = k.dtype
        q = q.astype(rope_dtype)
        k = k.astype(rope_dtype)
        
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        
        # Apply RoPE to non-prefix tokens
        if prefix > 0:
            q_prefix = q[:, :, :prefix, :]
            q_rope = rope_apply(q[:, :, prefix:, :], sin, cos)
            q = jnp.concatenate([q_prefix, q_rope], axis=-2)
            
            k_prefix = k[:, :, :prefix, :]
            k_rope = rope_apply(k[:, :, prefix:, :], sin, cos)
            k = jnp.concatenate([k_prefix, k_rope], axis=-2)
        else:
            q = rope_apply(q, sin, cos)
            k = rope_apply(k, sin, cos)
        
        # Cast back to original dtype
        q = q.astype(q_dtype)
        k = k.astype(k_dtype)
        
        return q, k
    
    def compute_attention(
        self,
        qkv: Array,
        rope: Optional[Tuple[Array, Array]] = None
    ) -> Array:
        """Compute multi-head attention using flash attention."""
        B, N, _ = qkv.shape

        q, k, v = einops.rearrange(qkv, 'b n (p h d) -> p b h n d', p=3, h=self.num_heads, d=self.head_dim)

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        dtype = q.dtype
        q = einops.rearrange(q, 'b h n d -> b n h d').astype(jnp.float16)
        k = einops.rearrange(k, 'b h n d -> b n h d').astype(jnp.float16)
        v = einops.rearrange(v, 'b h n d -> b n h d').astype(jnp.float16)
        x = flash_mha(q, k, v).astype(dtype)
        x = einops.rearrange(x, 'b n h d -> b n (h d)', b=B, n=N)

        return x
    
    def __call__(
        self,
        x: Array,
        rope: Optional[Tuple[Array, Array]] = None
    ) -> Array:
        """Forward pass of self-attention."""
        qkv = self.qkv(x)
        attn_out = self.compute_attention(qkv, rope=rope)
        x = self.proj(attn_out)
        return x
