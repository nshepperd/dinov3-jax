from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from dinov3_jax.eepynox.nn.conv2d import Conv2d
from dinov3_jax.eepynox.nn.param import Param
from dinov3_jax.config import Dinov3VitConfig


class Dinov3VitEmbeddings(eqx.Module):
    """Construct the CLS token, mask token, register tokens and patch embeddings."""

    cls_token: Param  # (1, 1, hidden_size)
    mask_token: Param  # (1, 1, hidden_size)
    register_tokens: Param | None  # (1, num_register_tokens, hidden_size)
    patch_embeddings: Conv2d
    num_register_tokens: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, config: Dinov3VitConfig, dtype=jnp.float32):
        self.num_register_tokens = config.num_register_tokens
        self.hidden_size = config.hidden_size
        self.cls_token = Param((1, 1, config.hidden_size), dtype)
        self.mask_token = Param((1, 1, config.hidden_size), dtype)
        self.register_tokens = (
            Param((1, config.num_register_tokens, config.hidden_size), dtype)
            if config.num_register_tokens > 0
            else None
        )
        self.patch_embeddings = Conv2d(
            in_features=config.num_channels,
            out_features=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            dtype=dtype,
        )

    def __call__(
        self, pixel_values: Array, bool_masked_pos: Array | None = None
    ) -> Array:
        batch_size = pixel_values.shape[0]

        # (B, C, H, W) -> (B, hidden_size, H', W') -> (B, num_patches, hidden_size)
        patch_embeddings = self.patch_embeddings(pixel_values)
        B, C, H, W = patch_embeddings.shape
        patch_embeddings = patch_embeddings.reshape(B, C, H * W).transpose(0, 2, 1)

        if bool_masked_pos is not None:
            mask_token = jnp.broadcast_to(self.mask_token(), patch_embeddings.shape)
            patch_embeddings = jnp.where(
                bool_masked_pos[:, :, None], mask_token, patch_embeddings
            )

        # Prepend CLS + register tokens
        assert self.register_tokens is not None
        cls_token = jnp.broadcast_to(self.cls_token(), (batch_size, 1, self.hidden_size))
        register_tokens = jnp.broadcast_to(
            self.register_tokens(), (batch_size, self.num_register_tokens, self.hidden_size)
        )
        embeddings = jnp.concatenate([cls_token, register_tokens, patch_embeddings], axis=1)
        return embeddings
