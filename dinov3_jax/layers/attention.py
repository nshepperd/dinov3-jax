import math
from typing import Optional, Tuple, List

import jax
import jax.numpy as jnp
from jaxtorch import Module, Context, init
from jaxtorch.nn import Linear, Dropout
from flash_attn_jax import flash_mha
import einops


def rope_rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotate half of the dimensions for RoPE."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def rope_apply(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray) -> jnp.ndarray:
    """Apply RoPE rotation to input tensor."""
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(Module):
    """Linear layer with masked bias for K (keys) in QKV projection."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        assert out_features % 3 == 0
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight
        self.weight = init.glorot_normal(out_features, in_features)
        
        # Initialize bias with masking for K
        if bias:
            self.bias = init.zeros(out_features)
            # Create bias mask (NaN for K, 1 for Q and V)
            k_size = out_features // 3
            bias_mask = jnp.ones(out_features)
            bias_mask = bias_mask.at[k_size:2*k_size].set(jnp.nan)
            self.bias_mask = bias_mask
        else:
            self.bias = None
            self.bias_mask = None
    
    def forward(self, cx: Context, x: jnp.ndarray) -> jnp.ndarray:
        y = x @ cx[self.weight].T
        
        if self.bias is not None:
            masked_bias = cx[self.bias] * self.bias_mask
            # Replace NaN with 0 for computation
            masked_bias = jnp.where(jnp.isnan(masked_bias), 0, masked_bias)
            y = y + masked_bias
        
        return y


class SelfAttention(Module):
    """Multi-head self-attention with optional RoPE."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,  # Ignored in JAX
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim
        
        # QKV projection
        if mask_k_bias:
            self.qkv = LinearKMaskedBias(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        
        # Attention dropout
        self.attn_drop = Dropout(attn_drop) if attn_drop > 0 else None
        
        # Output projection
        self.proj = Linear(dim, dim, bias=proj_bias)
        self.proj_drop = Dropout(proj_drop) if proj_drop > 0 else None
    
    def apply_rope(
        self, 
        q: jnp.ndarray, 
        k: jnp.ndarray, 
        rope: Optional[Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        cx: Context,
        qkv: jnp.ndarray,
        attn_bias: Optional[jnp.ndarray] = None,
        rope: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    ) -> jnp.ndarray:
        """Compute multi-head attention."""
        B, N, _ = qkv.shape
        
        # Reshape QKV
        q,k,v = einops.rearrange(qkv, 'b n (p h d) -> p b h n d', p=3, h=self.num_heads, d=self.head_dim)
        
        # Apply RoPE if provided
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        
        # Compute attention scores
        assert attn_bias is None
        dtype = q.dtype
        if True: # use flash
            q = einops.rearrange(q, 'b h n d -> b n h d').astype(jnp.float16)  # flash_mha format
            k = einops.rearrange(k, 'b h n d -> b n h d').astype(jnp.float16)
            v = einops.rearrange(v, 'b h n d -> b n h d').astype(jnp.float16)
            x = flash_mha(q, k, v).astype(dtype)
            x = einops.rearrange(x, 'b n h d -> b n (h d)',b=B,n=N)
        else:
            attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # B, heads, N, N
            
            # Apply attention bias if provided
            if attn_bias is not None:
                attn = attn + attn_bias
            
            # Softmax
            attn = jax.nn.softmax(attn, axis=-1)
            
            # Apply attention dropout
            if self.attn_drop is not None and cx.mode == "train":
                attn = self.attn_drop(cx, attn)
            
            # Apply attention to values
            x = attn @ v  # B, heads, N, head_dim
            
            # Reshape output
            x = x.transpose(0, 2, 1, 3).reshape(B, N, self.dim)
        
        return x
    
    def forward(
        self,
        cx: Context,
        x: jnp.ndarray,
        attn_bias: Optional[jnp.ndarray] = None,
        rope: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    ) -> jnp.ndarray:
        """Forward pass of self-attention."""
        qkv = self.qkv(cx, x)
        attn_out = self.compute_attention(cx, qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(cx, attn_out)
        
        if self.proj_drop is not None:
            x = self.proj_drop(cx, x)
        
        return x
    
    def forward_list(
        self,
        cx: Context,
        x_list: List[jnp.ndarray],
        attn_bias: Optional[jnp.ndarray] = None,
        rope_list: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]] = None
    ) -> List[jnp.ndarray]:
        """Forward pass for a list of inputs (used for multi-crop training)."""
        if rope_list is None:
            rope_list = [None] * len(x_list)
        
        assert len(x_list) == len(rope_list)
        
        # Process each input separately
        outputs = []
        for x, rope in zip(x_list, rope_list):
            qkv = self.qkv(cx, x)
            attn_out = self.compute_attention(cx, qkv, attn_bias=attn_bias, rope=rope)
            outputs.append(attn_out)
        
        # Apply projection to all outputs
        projected = []
        for out in outputs:
            x = self.proj(cx, out)
            if self.proj_drop is not None:
                x = self.proj_drop(cx, x)
            projected.append(x)
        
        return projected