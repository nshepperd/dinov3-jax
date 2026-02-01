"""Test PatchEmbed module compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp

import torch
import torch.nn as nn

# Import implementations
from dinov3_jax.layers.patch_embed import PatchEmbed as PatchEmbedJAX
from dinov3.layers.patch_embed import PatchEmbed as PatchEmbedPyTorch

def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    """Convert a PyTorch state dict to JAX format."""
    return {k: jnp.array(v.detach().cpu().numpy()) for k, v in state_dict.items()}

def test_patch_embed_forward():
    """Test that PatchEmbed produces identical outputs in JAX and PyTorch."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    img_size = 224
    patch_size = 16
    in_chans = 3
    embed_dim = 768
    batch_size = 2
    
    # Create random input image
    x_np = np.random.randn(batch_size, in_chans, img_size, img_size).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_jax = jnp.array(x_np)
    
    # Create PyTorch module and input
    layer_pt = PatchEmbedPyTorch(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten_embedding=True,  # Test flattened version first
    )
    
    # Create JAX module
    layer_jax = PatchEmbedJAX(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten_embedding=True,
    ).load_state_dict(convert_state_dict(layer_pt.state_dict()))
    
    # Forward pass PyTorch
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()
    
    # Forward pass JAX
    output_jax = layer_jax(x_jax)
    
    # Check output shape
    expected_num_patches = (img_size // patch_size) ** 2
    expected_shape = (batch_size, expected_num_patches, embed_dim)
    assert output_jax.shape == expected_shape, f"Expected shape {expected_shape}, got {output_jax.shape}"
    assert output_pt.shape == expected_shape, f"Expected shape {expected_shape}, got {output_pt.shape}"
    
    # Compare outputs (allow slightly higher tolerance for numerical precision)
    np.testing.assert_allclose(output_jax, output_pt_np, rtol=1e-5, atol=2e-6)