from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from dinov3_jax.config import Dinov3VitConfig


class Dinov3VitRopePositionEmbedding(eqx.Module):
    """Rotary Position Embedding for DINOv3 ViT, matching HF implementation."""

    inv_freq: Array  # (head_dim // 4,) — not trainable
    head_dim: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    rope_theta: float = eqx.field(static=True)

    def __init__(self, config: Dinov3VitConfig):
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.patch_size = config.patch_size
        self.rope_theta = config.rope_theta
        self.inv_freq = 1.0 / (
            config.rope_theta ** jnp.arange(0, 1, 4 / self.head_dim, dtype=jnp.float32)
        )

    def __call__(self, pixel_values: Array) -> tuple[Array, Array]:
        """Compute cos/sin position embeddings from pixel_values shape.

        Args:
            pixel_values: (B, C, H, W) input images.

        Returns:
            (cos, sin) each of shape (num_patches, head_dim).
        """
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # Patch center coordinates normalized to [-1, +1]
        coords_h = jnp.arange(0.5, num_patches_h, dtype=jnp.float32) / num_patches_h
        coords_w = jnp.arange(0.5, num_patches_w, dtype=jnp.float32) / num_patches_w
        # (H, W, 2) -> (H*W, 2)
        grid_h, grid_w = jnp.meshgrid(coords_h, coords_w, indexing="ij")
        coords = jnp.stack([grid_h, grid_w], axis=-1).reshape(-1, 2)
        # Shift [0, 1] -> [-1, +1]
        coords = 2.0 * coords - 1.0

        # (num_patches, 2, head_dim//4) -> (num_patches, head_dim//2) -> (num_patches, head_dim)
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.reshape(angles.shape[0], -1)  # flatten last two dims
        angles = jnp.tile(angles, (1, 2))  # double to head_dim

        cos = jnp.cos(angles)
        sin = jnp.sin(angles)
        return cos, sin


def rotate_half(x: Array) -> Array:
    """Rotates half the hidden dims of the input."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: Array, k: Array, cos: Array, sin: Array
) -> tuple[Array, Array]:
    """Apply rotary position embeddings to Q and K, only on patch tokens.

    Prefix tokens (CLS + register) are left unmodified.

    Args:
        q, k: (B, num_heads, seq_len, head_dim)
        cos, sin: (num_patches, head_dim)
    """
    num_tokens = q.shape[-2]
    num_patches = cos.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_prefix, q_patches = jnp.split(q, [num_prefix_tokens], axis=-2)
    k_prefix, k_patches = jnp.split(k, [num_prefix_tokens], axis=-2)

    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = jnp.concatenate([q_prefix, q_patches], axis=-2)
    k = jnp.concatenate([k_prefix, k_patches], axis=-2)
    return q, k
