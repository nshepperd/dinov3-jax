import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jaxtorch import Module, Context, init
from jaxtorch.nn import LayerNorm, ModuleList, Identity, GELU

from .layers import (
    PatchEmbed,
    RopePositionEmbedding,
    SelfAttention,
    SelfAttentionBlock,
    Mlp,
    SwiGLUFFN,
    RMSNorm,
    LayerScale,
)

logger = logging.getLogger("dinov3_jax")

# Layer dictionaries for configuration
ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(LayerNorm, eps=1e-6),
    "layernormbf16": partial(LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


class DinoVisionTransformer(Module):
    """DINOv3 Vision Transformer implementation in JAX."""
    
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: Optional[float] = None,
        pos_embed_rope_jitter_coords: Optional[float] = None,
        pos_embed_rope_rescale_coords: Optional[float] = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any = None,  # Ignored in JAX
        **ignored_kwargs,
    ):
        super().__init__()
        
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        
        norm_layer_cls = norm_layer_dict[norm_layer]
        
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        
        # Class token
        self.cls_token = init.normal(1, 1, embed_dim, stddev=0.02)
        
        # Storage tokens (registers)
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = init.normal(1, n_storage_tokens, embed_dim, stddev=0.02)
        
        # RoPE position embedding
        logger.info(f"using base={pos_embed_rope_base} for rope")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope")
        
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
        )
        
        # Transformer blocks
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        
        self.blocks = ModuleList()
        for i in range(depth):
            block = SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
            )
            self.blocks.append(block)
        
        # Final normalization
        self.norm = norm_layer_cls(embed_dim)
        
        # Optional untied norms
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            self.cls_norm = norm_layer_cls(embed_dim)
        
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls(embed_dim)
        
        # Head (identity for feature extraction)
        self.head = Identity()
        
        # Mask token for masked modeling
        self.mask_token = init.zeros(1, embed_dim)
    
    def prepare_tokens_with_masks(
        self,
        cx: Context,
        x: jnp.ndarray,
        masks: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Tuple[int, int]]:
        """Prepare tokens from image patches with optional masking."""
        
        x = self.patch_embed(cx, x)
        B, H, W, _ = x.shape
        x = x.reshape(B, H * W, -1)
        
        # Apply mask token if masks provided
        if masks is not None:
            mask_token = cx[self.mask_token].astype(x.dtype)
            x = jnp.where(masks[:, :, None], mask_token[None, :, :], x)
            cls_token = cx[self.cls_token]
        else:
            cls_token = cx[self.cls_token] + 0 * cx[self.mask_token]
        
        # Prepare storage tokens
        if self.n_storage_tokens > 0:
            storage_tokens = cx[self.storage_tokens]
        else:
            storage_tokens = jnp.empty((1, 0, cls_token.shape[-1]), dtype=cls_token.dtype)
        
        # Concatenate cls token, storage tokens, and patch tokens
        cls_token = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        storage_tokens = jnp.broadcast_to(storage_tokens, (B, self.n_storage_tokens, self.embed_dim))
        
        x = jnp.concatenate([cls_token, storage_tokens, x], axis=1)
        
        return x, (H, W)
    
    def forward_features_list(
        self,
        cx: Context,
        x_list: List[jnp.ndarray],
        masks_list: List[Optional[jnp.ndarray]]
    ) -> List[Dict[str, jnp.ndarray]]:
        """Forward pass for a list of inputs (multi-crop training)."""
        
        # Prepare tokens for each input
        x_tokens = []
        rope_hw = []
        for x, masks in zip(x_list, masks_list):
            tokens, hw_tuple = self.prepare_tokens_with_masks(cx, x, masks)
            x_tokens.append(tokens)
            rope_hw.append(hw_tuple)
        
        # Process through transformer blocks
        for block in self.blocks:
            # Compute RoPE for each input
            rope_list = []
            for H, W in rope_hw:
                rope_sincos = self.rope_embed(cx, H=H, W=W)
                rope_list.append(rope_sincos)
            
            # Forward through block
            x_tokens = block(cx, x_tokens, rope_list)
        
        # Apply final normalization and prepare outputs
        outputs = []
        for idx, (x, masks) in enumerate(zip(x_tokens, masks_list)):
            # Apply appropriate normalization
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and cx.mode == "train" and idx == 1:
                    # Local crops get local norm
                    x_norm_cls_reg = self.local_cls_norm(cx, x[:, :self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(cx, x[:, :self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(cx, x[:, :self.n_storage_tokens + 1])
                x_norm_patch = self.norm(cx, x[:, self.n_storage_tokens + 1:])
            else:
                x_norm = self.norm(cx, x)
                x_norm_cls_reg = x_norm[:, :self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1:]
            
            # Prepare output dictionary
            output = {
                "x_norm_clstoken": x_norm_cls_reg[:, 0],
                "x_storage_tokens": x_norm_cls_reg[:, 1:],
                "x_norm_patchtokens": x_norm_patch,
                "x_prenorm": x,
                "masks": masks,
            }
            outputs.append(output)
        
        return outputs
    
    def forward_features(
        self,
        cx: Context,
        x: Union[jnp.ndarray, List[jnp.ndarray]],
        masks: Optional[Union[jnp.ndarray, List[jnp.ndarray]]] = None
    ) -> Union[Dict[str, jnp.ndarray], List[Dict[str, jnp.ndarray]]]:
        """Forward features extraction."""
        
        if isinstance(x, jnp.ndarray):
            # Single input
            if masks is None or isinstance(masks, jnp.ndarray):
                return self.forward_features_list(cx, [x], [masks])[0]
            else:
                raise TypeError("Masks must be jnp.ndarray for single input")
        else:
            # List of inputs
            if masks is None:
                masks = [None] * len(x)
            return self.forward_features_list(cx, x, masks)
    
    def get_intermediate_layers(
        self,
        cx: Context,
        x: jnp.ndarray,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]]:
        """Get intermediate layer outputs."""
        
        x, (H, W) = self.prepare_tokens_with_masks(cx, x)
        
        # Determine which blocks to take
        total_blocks = len(self.blocks)
        if isinstance(n, int):
            blocks_to_take = list(range(total_blocks - n, total_blocks))
        else:
            blocks_to_take = list(n)
        
        # Forward through blocks
        outputs = []
        for i, block in enumerate(self.blocks):
            rope_sincos = self.rope_embed(cx, H=H, W=W)
            x = block(cx, x, rope_sincos)
            
            if i in blocks_to_take:
                outputs.append(x)
        
        assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
        
        # Apply normalization if requested
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(cx, out[:, :self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(cx, out[:, self.n_storage_tokens + 1:])
                    outputs_normed.append(jnp.concatenate([x_norm_cls_reg, x_norm_patch], axis=1))
                else:
                    outputs_normed.append(self.norm(cx, out))
            outputs = outputs_normed
        
        # Extract different token types
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1:self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1:] for out in outputs]
        
        # Reshape if requested
        if reshape:
            B = x.shape[0]
            h, w = H, W
            outputs = [
                out.reshape(B, h, w, -1).transpose(0, 3, 1, 2)
                for out in outputs
            ]
        
        # Return based on flags
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        else:
            return tuple(zip(outputs, class_tokens, extra_tokens))
    
    def forward(
        self,
        cx: Context,
        *args,
        is_training: bool = False,
        **kwargs
    ) -> Union[List[Dict[str, jnp.ndarray]], jnp.ndarray]:
        """Forward pass."""
        
        ret = self.forward_features(cx, *args, **kwargs)
        if is_training:
            return ret
        else:
            # Return class token for inference
            if isinstance(ret, dict):
                return self.head(cx, ret["x_norm_clstoken"])
            else:
                # If list, return first element's class token
                return self.head(cx, ret[0]["x_norm_clstoken"])


# Model constructors
def vit_small(patch_size=16, **kwargs):
    """Vision Transformer Small variant."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    """Vision Transformer Base variant."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    """Vision Transformer Large variant."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    """Vision Transformer SO400M variant."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    """Vision Transformer Huge2 variant."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """Vision Transformer Giant2 variant."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    """Vision Transformer 7B variant."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model

# Pretrained model constructors
def dinov3_vits16():
    vits_16_kwargs = dict(
        pos_embed_rope_dtype="fp32",         # default is "bf16"
        pos_embed_rope_rescale_coords=2,     # default is None
        embed_dim=384,                        # default is 768
        num_heads=6,                          # default is 12
        layerscale_init=1.0e-05,            # default is None
        norm_layer="layernormbf16",         # default is "layernorm"
        n_storage_tokens=4,                  # default is 0
        mask_k_bias=True                     # default is False
    )
    return DinoVisionTransformer(**vits_16_kwargs)

def dinov3_vitl16():
    kwargs = dict(
        pos_embed_rope_dtype="fp32",         # default is "bf16"
        pos_embed_rope_rescale_coords=2,     # default is None
        embed_dim=1024,                       # default is 768
        num_heads=16,                         # default is 12
        depth=24,                             # default is 12
        layerscale_init=1.0e-05,            # default is None
        norm_layer="layernormbf16",         # default is "layernorm"
        n_storage_tokens=4,                  # default is 0
        mask_k_bias=True                     # default is False
    )
    return DinoVisionTransformer(**kwargs)