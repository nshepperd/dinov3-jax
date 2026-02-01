"""Test RoPE position encoding compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp

import torch
import torch.nn as nn

from dinov3_jax.layers.rope_position_encoding import RopePositionEmbedding as RoPEJAX
from dinov3.layers.rope_position_encoding import RopePositionEmbedding as RoPEPyTorch
jax.config.update("jax_default_matmul_precision", "highest")


def test_rope_basic_generation():
    """Test that RoPE generates sin/cos embeddings with correct shapes and values."""

    # Configuration
    embed_dim = 384
    num_heads = 6
    base = 100.0
    H, W = 14, 14  # For 224x224 image with patch_size=16

    # Create PyTorch module
    rope_pt = RoPEPyTorch(
        embed_dim=embed_dim,
        num_heads=num_heads,
        base=base,
        dtype=torch.float32,
    )

    # Create JAX module
    rope_jax = RoPEJAX(
        embed_dim=embed_dim,
        num_heads=num_heads,
        base=base,
        dtype=jnp.float32,
    )

    # Generate embeddings PyTorch
    sin_pt, cos_pt = rope_pt(H=H, W=W)
    sin_pt_np = sin_pt.detach().numpy()
    cos_pt_np = cos_pt.detach().numpy()

    # Generate embeddings JAX
    sin_jax, cos_jax = rope_jax(H=H, W=W)
    sin_jax_np = np.array(sin_jax)
    cos_jax_np = np.array(cos_jax)

    # Check shapes (RoPE outputs D_head = embed_dim // num_heads)
    D_head = embed_dim // num_heads
    expected_shape = (H * W, D_head)
    assert sin_jax.shape == expected_shape, f"Expected {expected_shape}, got {sin_jax.shape}"
    assert cos_jax.shape == expected_shape, f"Expected {expected_shape}, got {cos_jax.shape}"
    assert sin_pt.shape == expected_shape, f"Expected {expected_shape}, got {sin_pt.shape}"
    assert cos_pt.shape == expected_shape, f"Expected {expected_shape}, got {cos_pt.shape}"

    # Compare outputs
    np.testing.assert_allclose(sin_jax_np, sin_pt_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cos_jax_np, cos_pt_np, rtol=1e-5, atol=1e-6)
    print("✓ RoPE basic generation matches between JAX and PyTorch")


def test_rope_different_sizes():
    """Test RoPE with different image sizes."""

    embed_dim = 768
    num_heads = 12
    base = 100.0

    test_sizes = [
        (14, 14),  # Standard 224x224 with patch_size=16
        (7, 7),    # 112x112 with patch_size=16
        (28, 28),  # 448x448 with patch_size=16
        (16, 12),  # Non-square
    ]

    for H, W in test_sizes:
        # Create modules
        rope_pt = RoPEPyTorch(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=base,
            dtype=torch.float32,
        )

        rope_jax = RoPEJAX(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=base,
            dtype=jnp.float32,
        )

        # Generate embeddings
        sin_pt, cos_pt = rope_pt(H=H, W=W)
        sin_jax, cos_jax = rope_jax(H=H, W=W)

        # Compare
        np.testing.assert_allclose(np.array(sin_jax), sin_pt.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.array(cos_jax), cos_pt.numpy(), rtol=1e-5, atol=1e-6)
        print(f"✓ RoPE with H={H}, W={W} matches")


def test_rope_with_periods():
    """Test RoPE using min_period and max_period instead of base."""

    embed_dim = 256
    num_heads = 8
    min_period = 2.0
    max_period = 10000.0
    H, W = 10, 10

    # Create PyTorch module with periods
    rope_pt = RoPEPyTorch(
        embed_dim=embed_dim,
        num_heads=num_heads,
        base=None,
        min_period=min_period,
        max_period=max_period,
        dtype=torch.float32,
    )

    # Create JAX module with periods
    rope_jax = RoPEJAX(
        embed_dim=embed_dim,
        num_heads=num_heads,
        base=None,
        min_period=min_period,
        max_period=max_period,
        dtype=jnp.float32,
    )

    # Generate embeddings
    sin_pt, cos_pt = rope_pt(H=H, W=W)
    sin_jax, cos_jax = rope_jax(H=H, W=W)

    # Compare
    np.testing.assert_allclose(np.array(sin_jax), sin_pt.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(cos_jax), cos_pt.numpy(), rtol=1e-5, atol=1e-6)
    print("✓ RoPE with min/max periods matches")


def test_rope_normalize_coords():
    """Test different coordinate normalization strategies."""

    embed_dim = 384
    num_heads = 6
    base = 100.0

    normalize_modes = ["separate", "max", "min"]

    for normalize_mode in normalize_modes:
        H, W = 16, 12  # Non-square to test normalization

        # Create modules
        rope_pt = RoPEPyTorch(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=base,
            normalize_coords=normalize_mode,
            dtype=torch.float32,
        )

        rope_jax = RoPEJAX(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=base,
            normalize_coords=normalize_mode,
            dtype=jnp.float32,
        )

        # Generate embeddings
        sin_pt, cos_pt = rope_pt(H=H, W=W)
        sin_jax, cos_jax = rope_jax(H=H, W=W)

        # Compare
        np.testing.assert_allclose(np.array(sin_jax), sin_pt.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.array(cos_jax), cos_pt.numpy(), rtol=1e-5, atol=1e-6)
        print(f"✓ RoPE with normalize_coords='{normalize_mode}' matches")


def test_rope_dtype_handling():
    """Test RoPE with different data types."""

    embed_dim = 128
    num_heads = 4
    base = 100.0
    H, W = 7, 7

    dtypes = [
        (torch.float32, jnp.float32),
        (torch.bfloat16, jnp.bfloat16),
    ]

    for dtype_pt, dtype_jax in dtypes:
        # Create modules
        rope_pt = RoPEPyTorch(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=base,
            dtype=dtype_pt,
        )

        rope_jax = RoPEJAX(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=base,
            dtype=dtype_jax,
        )

        # Generate embeddings
        sin_pt, cos_pt = rope_pt(H=H, W=W)
        sin_jax, cos_jax = rope_jax(H=H, W=W)

        # Check dtypes
        assert sin_jax.dtype == dtype_jax
        assert cos_jax.dtype == dtype_jax
        assert sin_pt.dtype == dtype_pt
        assert cos_pt.dtype == dtype_pt

        # Compare (use float32 for comparison to avoid bfloat16 precision issues)
        sin_pt_f32 = sin_pt.float().numpy()
        cos_pt_f32 = cos_pt.float().numpy()
        sin_jax_f32 = np.array(sin_jax.astype(jnp.float32))
        cos_jax_f32 = np.array(cos_jax.astype(jnp.float32))

        # Relaxed tolerance for bfloat16
        tol = 1e-3 if dtype_pt == torch.bfloat16 else 1e-5
        np.testing.assert_allclose(sin_jax_f32, sin_pt_f32, rtol=tol, atol=tol)
        np.testing.assert_allclose(cos_jax_f32, cos_pt_f32, rtol=tol, atol=tol)
        print(f"✓ RoPE with dtype={dtype_pt} matches")


if __name__ == "__main__":
    test_rope_basic_generation()
    test_rope_different_sizes()
    test_rope_with_periods()
    test_rope_normalize_coords()
    test_rope_dtype_handling()
    print("\n✅ All RoPE tests passed!")
