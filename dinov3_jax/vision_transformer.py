from __future__ import annotations

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import eepynox.utils as eu
from dinov3_jax.layers.rms_norm import LayerNorm
from eepynox.nn.activation import Identity

from .config import DinoV3Config
from .layers import (
    Mlp,
    PatchEmbed,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
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

dtype_dict: dict[str, jnp.dtype] = {
    "fp32": jnp.dtype(jnp.float32),
    "fp16": jnp.dtype(jnp.float16),
    "bf16": jnp.dtype(jnp.bfloat16),
}


class DinoVisionTransformer(eqx.Module):
    """DINOv3 Vision Transformer implementation in JAX."""
    
    patch_embed: PatchEmbed
    cls_token: Array | None
    storage_tokens: Array | None
    rope_embed: RopePositionEmbedding
    blocks: list[SelfAttentionBlock]
    norm: RMSNorm | LayerNorm
    cls_norm: RMSNorm | LayerNorm | None
    local_cls_norm: RMSNorm | LayerNorm | None
    head: Identity
    mask_token: Array | None

    config: DinoV3Config = eqx.field(static=True)
    # n_storage_tokens: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        config: DinoV3Config,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__()
        self.config = config
        
        # self.num_features = self.embed_dim = config.embed_dim
        # self.n_blocks = config.depth
        # self.num_heads = config.num_heads
        # self.patch_size = config.patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            flatten_embedding=False,
        )
        
        # Class token
        # self.cls_token = init.normal(1, 1, embed_dim, stddev=0.02)
        self.cls_token = None
        
        # Storage tokens (registers)
        # self.n_storage_tokens = config.n_storage_tokens
        self.storage_tokens = None
        # if self.n_storage_tokens > 0:
        #     self.storage_tokens = init.normal(1, n_storage_tokens, embed_dim, stddev=0.02)
        
        # RoPE position embedding
        logger.info(f"using base={config.pos_embed_rope_base} for rope")
        logger.info(f"using min_period={config.pos_embed_rope_min_period} for rope")
        logger.info(f"using max_period={config.pos_embed_rope_max_period} for rope")
        logger.info(f"using normalize_coords={config.pos_embed_rope_normalize_coords} for rope")
        logger.info(f"using shift_coords={config.pos_embed_rope_shift_coords} for rope")
        logger.info(f"using rescale_coords={config.pos_embed_rope_rescale_coords} for rope")
        logger.info(f"using jitter_coords={config.pos_embed_rope_jitter_coords} for rope")
        logger.info(f"using dtype={config.pos_embed_rope_dtype} for rope")
        
        self.rope_embed = RopePositionEmbedding(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            base=config.pos_embed_rope_base,
            min_period=config.pos_embed_rope_min_period,
            max_period=config.pos_embed_rope_max_period,
            normalize_coords=config.pos_embed_rope_normalize_coords,
            shift_coords=config.pos_embed_rope_shift_coords,
            jitter_coords=config.pos_embed_rope_jitter_coords,
            rescale_coords=config.pos_embed_rope_rescale_coords,
            dtype=dtype_dict[config.pos_embed_rope_dtype],
        )
        
        # Transformer blocks
        logger.info(f"using {config.ffn_layer} layer as FFN")
        
        self.blocks = []
        for i in range(config.depth):
            block = SelfAttentionBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                ffn_ratio=config.ffn_ratio,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                ffn_bias=config.ffn_bias,
                norm_layer=config.norm_layer,
                act_layer='gelu',
                ffn_layer=config.ffn_layer,
                init_values=config.layerscale_init,
                mask_k_bias=config.mask_k_bias,
            )
            self.blocks.append(block)
        
        norm_layer_cls = norm_layer_dict[config.norm_layer]

        # Final normalization
        self.norm = norm_layer_cls(config.embed_dim)
        
        # Optional untied norms
        # self.untie_cls_and_patch_norms = config.untie_cls_and_patch_norms
        if config.untie_cls_and_patch_norms:
            self.cls_norm = norm_layer_cls(config.embed_dim)
        else:
            self.cls_norm = None
        
        # self.untie_global_and_local_cls_norm = config.untie_global_and_local_cls_norm
        if config.untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls(config.embed_dim)
        else:
            self.local_cls_norm = None
        
        # Head (identity for feature extraction)
        self.head = Identity()
        
        # Mask token for masked modeling
        # self.mask_token = init.zeros(1, embed_dim)
        self.mask_token = None
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix='') -> DinoVisionTransformer:
        patch_embed = self.patch_embed.load_state_dict(state_dict, prefix=prefix + 'patch_embed.')
        cls_token = state_dict[prefix + 'cls_token']
        storage_tokens = state_dict.get(prefix + 'storage_tokens', None)
        rope_embed = self.rope_embed #.load_state_dict(state_dict, prefix=prefix + 'rope_embed.')
        blocks = []
        for i, block in enumerate(self.blocks):
            block_loaded = block.load_state_dict(state_dict, prefix=prefix + f'blocks.{i}.')
            blocks.append(block_loaded)
        norm = self.norm.load_state_dict(state_dict, prefix=prefix + 'norm.')
        cls_norm = None
        if self.cls_norm is not None:
            cls_norm = self.cls_norm.load_state_dict(state_dict, prefix=prefix + 'cls_norm.')
        local_cls_norm = None
        if self.local_cls_norm is not None:
            local_cls_norm = self.local_cls_norm.load_state_dict(state_dict, prefix=prefix + 'local_cls_norm.')
        mask_token = state_dict.get(prefix + 'mask_token', None)
        return eu.replace(
            self,
            patch_embed=patch_embed,
            cls_token=cls_token,
            storage_tokens=storage_tokens,
            rope_embed=rope_embed,
            blocks=blocks,
            norm=norm,
            cls_norm=cls_norm,
            local_cls_norm=local_cls_norm,
            mask_token=mask_token,
        )

    def prepare_tokens_with_masks(
        self,
        x: Array,
        masks: Optional[Array] = None
    ) -> Tuple[Array, Tuple[int, int]]:
        """Prepare tokens from image patches with optional masking."""
        
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.reshape(B, H * W, -1)
        
        # Apply mask token if masks provided
        if masks is not None:
            mask_token = self.mask_token.astype(x.dtype)
            x = jnp.where(masks[:, :, None], mask_token[None, :, :], x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        
        # Prepare storage tokens
        if self.config.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = jnp.empty((1, 0, cls_token.shape[-1]), dtype=cls_token.dtype)
        
        # Concatenate cls token, storage tokens, and patch tokens
        cls_token = jnp.broadcast_to(cls_token, (B, 1, self.config.embed_dim))
        storage_tokens = jnp.broadcast_to(storage_tokens, (B, self.config.n_storage_tokens, self.config.embed_dim))
        
        x = jnp.concatenate([cls_token, storage_tokens, x], axis=1)
        
        return x, (H, W)
    
    def forward_features(
        self,
        x: Array,
        masks: Optional[Array] = None
    ) -> Dict[str, Array]:
        """Forward pass for a list of inputs (multi-crop training)."""
        
        # Prepare tokens for each input
        tokens, rope_hw = self.prepare_tokens_with_masks(x, masks)
        
        # Process through transformer blocks
        for block in self.blocks:
            # Compute RoPE for each input
            H,W = rope_hw
            rope = self.rope_embed(H=H, W=W)
            # Forward through block
            tokens = block(tokens, rope)
        
        x, masks = tokens, masks
        # Apply final normalization and prepare outputs
        # Apply appropriate normalization
        if self.config.untie_cls_and_patch_norms or self.config.untie_global_and_local_cls_norm:
            # if self.config.untie_global_and_local_cls_norm and cx.mode == "train" and idx == 1:
            #     # Local crops get local norm
            #     x_norm_cls_reg = self.local_cls_norm(cx, x[:, :self.config.n_storage_tokens + 1])
            if self.config.untie_cls_and_patch_norms:
                assert self.cls_norm is not None
                x_norm_cls_reg = self.cls_norm(x[:, :self.config.n_storage_tokens + 1])
            else:
                x_norm_cls_reg = self.norm(x[:, :self.config.n_storage_tokens + 1])
            x_norm_patch = self.norm(x[:, self.config.n_storage_tokens + 1:])
        else:
            x_norm = self.norm(x)
            x_norm_cls_reg = x_norm[:, :self.config.n_storage_tokens + 1]
            x_norm_patch = x_norm[:, self.config.n_storage_tokens + 1:]
        
        # Prepare output dictionary
        output = {
            "x_norm_clstoken": x_norm_cls_reg[:, 0],
            "x_storage_tokens": x_norm_cls_reg[:, 1:],
            "x_norm_patchtokens": x_norm_patch,
            "x_prenorm": x,
            "masks": masks,
        }
        return output
    
    def get_intermediate_layers(
        self,
        x: Array,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[Array, Tuple[Array, ...]]]:
        """Get intermediate layer outputs."""
        
        x, (H, W) = self.prepare_tokens_with_masks(x)
        
        # Determine which blocks to take
        total_blocks = len(self.blocks)
        if isinstance(n, int):
            blocks_to_take = list(range(total_blocks - n, total_blocks))
        else:
            blocks_to_take = list(n)
        
        # Forward through blocks
        outputs = []
        for i, block in enumerate(self.blocks):
            rope_sincos = self.rope_embed(H=H, W=W)
            x = block(x, rope_sincos)
            
            if i in blocks_to_take:
                outputs.append(x)
        
        assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
        
        # Apply normalization if requested
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.config.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, :self.config.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.config.n_storage_tokens + 1:])
                    outputs_normed.append(jnp.concatenate([x_norm_cls_reg, x_norm_patch], axis=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        
        # Extract different token types
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1:self.config.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.config.n_storage_tokens + 1:] for out in outputs]
        
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
    
    def __call__(
        self,
        *args,
        is_training: bool = False,
        **kwargs
    ) -> Dict[str, Array] | Array:
        """Forward pass."""
        
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            # Return class token for inference
            if isinstance(ret, dict):
                return self.head(ret["x_norm_clstoken"])
            else:
                # If list, return first element's class token
                return self.head(ret[0]["x_norm_clstoken"])

# Model constructors
def vit_small(patch_size=16, **kwargs):
    """Vision Transformer Small variant."""
    config = DinoV3Config(
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4.0,
        patch_size=patch_size,
    )
    model = DinoVisionTransformer(
        config=config,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    """Vision Transformer Base variant."""
    config = DinoV3Config(
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        patch_size=patch_size,
    )
    model = DinoVisionTransformer(
        config=config,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    """Vision Transformer Large variant."""
    config = DinoV3Config(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4.0,
        patch_size=patch_size,
    )
    model = DinoVisionTransformer(
        config=config,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    """Vision Transformer SO400M variant."""
    config = DinoV3Config(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
    )
    model = DinoVisionTransformer(
        config=config,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    """Vision Transformer Huge2 variant."""
    config = DinoV3Config(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4.0,
    )
    model = DinoVisionTransformer(
        config=config,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """Vision Transformer Giant2 variant."""
    config = DinoV3Config(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4.0,
    )
    model = DinoVisionTransformer(
        config=config,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    """Vision Transformer 7B variant."""
    config = DinoV3Config(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
    )
    model = DinoVisionTransformer(
        config=config,
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
    return DinoVisionTransformer(config=DinoV3Config.model_validate(vits_16_kwargs), dtype=jnp.bfloat16)

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
    return DinoVisionTransformer(config=DinoV3Config.model_validate(kwargs), dtype=jnp.bfloat16)