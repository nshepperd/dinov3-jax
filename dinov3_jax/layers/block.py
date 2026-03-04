from functools import partial
from typing import Literal, Optional, Tuple

import equinox as eqx
from jaxtyping import Array

from .attention import SelfAttention
from .ffn_layers import Mlp, SwiGLUFFN
from .layer_scale import LayerScale
from .rms_norm import LayerNorm, RMSNorm

import eepynox.utils as eu


class SelfAttentionBlock(eqx.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    norm1: LayerNorm | RMSNorm
    attn: SelfAttention
    ls1: LayerScale | None
    norm2: LayerNorm | RMSNorm
    mlp: Mlp | SwiGLUFFN
    ls2: LayerScale | None

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: Optional[float] = None,
        norm_layer: Literal['layernorm', 'layernormbf16', 'rmsnorm'] = 'layernorm',
        ffn_layer: Literal['mlp', 'swiglu', 'swiglu32', 'swiglu64', 'swiglu128'] = 'mlp',
        mask_k_bias: bool = False,
    ):
        super().__init__()

        f_norm_layer = {
            'layernorm': partial(LayerNorm, eps=1e-6),
            'layernormbf16': partial(LayerNorm, eps=1e-5),
            'rmsnorm': RMSNorm
        }[norm_layer]
        f_ffn_layer = {
            'mlp': Mlp,
            'swiglu': SwiGLUFFN,
            'swiglu32': partial(SwiGLUFFN, align_to=32),
            'swiglu64': partial(SwiGLUFFN, align_to=64),
            'swiglu128': partial(SwiGLUFFN, align_to=128),
        }[ffn_layer]

        self.norm1 = f_norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else None
        
        # Second block: feed-forward
        self.norm2 = f_norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = f_ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else None
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        norm1 = self.norm1.load_state_dict(state_dict, prefix=prefix + "norm1.")
        attn = self.attn.load_state_dict(state_dict, prefix=prefix + "attn.")
        
        if self.ls1:
            ls1 = self.ls1.load_state_dict(state_dict, prefix=prefix + "ls1.")
        else:
            ls1 = None
        
        norm2 = self.norm2.load_state_dict(state_dict, prefix=prefix + "norm2.")
        mlp = self.mlp.load_state_dict(state_dict, prefix=prefix + "mlp.")
        
        if self.ls2:
            ls2 = self.ls2.load_state_dict(state_dict, prefix=prefix + "ls2.")
        else:
            ls2 = None
        
        return eu.replace(
            self,
            norm1=norm1,
            attn=attn,
            ls1=ls1,
            norm2=norm2,
            mlp=mlp,
            ls2=ls2,
        )

    def __call__(
        self,
        x: Array,
        rope: Tuple | None = None,
    ) -> Array:
        # Self-attention block
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, rope=rope)
        if self.ls1:
            attn_out = self.ls1(attn_out)
        x = x + attn_out
        
        # Feed-forward block
        x_norm = self.norm2(x)
        ffn_out = self.mlp(x_norm)
        if self.ls2:
            ffn_out = self.ls2(ffn_out)
        x = x + ffn_out
        
        return x