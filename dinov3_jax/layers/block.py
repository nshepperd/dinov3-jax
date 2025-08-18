from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jaxtorch import Module, Context
from jaxtorch.nn import LayerNorm, GELU, Identity

from .attention import SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale


class SelfAttentionBlock(Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., Module] = GELU,
        norm_layer: Callable[..., Module] = LayerNorm,
        attn_class: Callable[..., Module] = SelfAttention,
        ffn_layer: Callable[..., Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,  # Ignored in JAX
    ):
        super().__init__()
        
        # First block: attention
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        
        # Second block: feed-forward
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        
        # Stochastic depth (drop path)
        self.drop_path_rate = drop_path
    
    def _apply_drop_path(
        self,
        cx: Context,
        x: jnp.ndarray,
        residual: jnp.ndarray,
        scale: float = 1.0
    ) -> jnp.ndarray:
        """Apply stochastic depth (drop path) during training."""
        if cx.mode == "train" and self.drop_path_rate > 0:
            # Random binary mask for dropping paths
            keep_prob = 1 - self.drop_path_rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = cx.random.bernoulli(keep_prob, shape=shape)
            random_tensor = random_tensor / keep_prob
            return x + residual * random_tensor * scale
        else:
            return x + residual * scale
    
    def forward(
        self,
        cx: Context,
        x_or_x_list: Union[jnp.ndarray, List[jnp.ndarray]],
        rope_or_rope_list: Optional[Union[Tuple, List[Tuple]]] = None
    ) -> Union[jnp.ndarray, List[jnp.ndarray]]:
        """Forward pass for single tensor or list of tensors."""
        
        # Handle single tensor case
        if isinstance(x_or_x_list, jnp.ndarray):
            return self._forward_single(cx, x_or_x_list, rope_or_rope_list)
        
        # Handle list case
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None] * len(x_or_x_list)
            return self._forward_list(cx, x_or_x_list, rope_or_rope_list)
        
        else:
            raise TypeError("Input must be jnp.ndarray or list of jnp.ndarray")
    
    def _forward_single(
        self,
        cx: Context,
        x: jnp.ndarray,
        rope: Optional[Tuple] = None
    ) -> jnp.ndarray:
        """Forward pass for a single tensor."""
        
        # Self-attention block
        x_norm = self.norm1(cx, x)
        attn_out = self.attn(cx, x_norm, rope=rope)
        attn_out = self.ls1(cx, attn_out)
        x = self._apply_drop_path(cx, x, attn_out)
        
        # Feed-forward block
        x_norm = self.norm2(cx, x)
        ffn_out = self.mlp(cx, x_norm)
        ffn_out = self.ls2(cx, ffn_out)
        x = self._apply_drop_path(cx, x, ffn_out)
        
        return x
    
    def _forward_list(
        self,
        cx: Context,
        x_list: List[jnp.ndarray],
        rope_list: List[Optional[Tuple]]
    ) -> List[jnp.ndarray]:
        """Forward pass for a list of tensors (multi-crop training)."""
        
        outputs = []
        
        # Process attention for all inputs
        x_norm_list = [self.norm1(cx, x) for x in x_list]
        attn_out_list = self.attn.forward_list(cx, x_norm_list, rope_list=rope_list)
        
        # Apply layer scale and residual
        for x, attn_out in zip(x_list, attn_out_list):
            attn_out = self.ls1(cx, attn_out)
            x_attn = self._apply_drop_path(cx, x, attn_out)
            outputs.append(x_attn)
        
        # Process feed-forward for all outputs
        final_outputs = []
        for x_attn in outputs:
            x_norm = self.norm2(cx, x_attn)
            ffn_out = self.mlp(cx, x_norm)
            ffn_out = self.ls2(cx, ffn_out)
            x_ffn = self._apply_drop_path(cx, x_attn, ffn_out)
            final_outputs.append(x_ffn)
        
        return final_outputs