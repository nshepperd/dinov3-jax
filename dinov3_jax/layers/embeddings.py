from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import eepynox.utils as eu
from eepynox.nn.conv2d import Conv2d
from dinov3_jax.config import Dinov3VitConfig


class Dinov3VitEmbeddings(eqx.Module):
    """Construct the CLS token, mask token, register tokens and patch embeddings."""

    cls_token: Array | None  # (1, 1, hidden_size)
    mask_token: Array | None  # (1, 1, hidden_size)
    register_tokens: Array | None  # (1, num_register_tokens, hidden_size)
    patch_embeddings: Conv2d
    num_register_tokens: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, config: Dinov3VitConfig):
        self.num_register_tokens = config.num_register_tokens
        self.hidden_size = config.hidden_size
        self.cls_token = None
        self.mask_token = None
        self.register_tokens = None
        self.patch_embeddings = Conv2d(
            in_features=config.num_channels,
            out_features=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = "") -> Dinov3VitEmbeddings:
        cls_token = state_dict.pop(prefix + "cls_token")
        mask_token = state_dict.pop(prefix + "mask_token")
        register_tokens = state_dict.pop(prefix + "register_tokens") if self.num_register_tokens > 0 else None
        patch_embeddings = self.patch_embeddings.load_state_dict(state_dict, prefix=prefix + "patch_embeddings.")
        return eu.replace(
            self,
            cls_token=cls_token,
            mask_token=mask_token,
            register_tokens=register_tokens,
            patch_embeddings=patch_embeddings,
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
            mask_token = jnp.broadcast_to(self.mask_token, patch_embeddings.shape)
            patch_embeddings = jnp.where(
                bool_masked_pos[:, :, None], mask_token, patch_embeddings
            )

        # Prepend CLS + register tokens
        assert self.cls_token is not None and self.register_tokens is not None
        cls_token = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.hidden_size))
        register_tokens = jnp.broadcast_to(
            self.register_tokens, (batch_size, self.num_register_tokens, self.hidden_size)
        )
        embeddings = jnp.concatenate([cls_token, register_tokens, patch_embeddings], axis=1)
        return embeddings
