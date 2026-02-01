"""Test RMSNorm compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp

import torch
import torch.nn as nn

from dinov3_jax.layers.rms_norm import RMSNorm as RMSNormJAX
from dinov3.layers.rms_norm import RMSNorm as RMSNormPyTorch
jax.config.update("jax_default_matmul_precision", "highest")

def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    """Convert a PyTorch state dict to JAX format."""
    return {k: jnp.array(v.detach().cpu().numpy()) for k, v in state_dict.items()}

def test_rms_norm_forward():
    """Test that RMSNorm produces identical outputs in JAX and PyTorch."""
    
    # Configuration
    dim = 768
    eps = 1e-5
    batch_size = 2
    seq_len = 197
    
    # Create random input
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_jax = jnp.array(x_np)
    
    layer_pt = RMSNormPyTorch(dim, eps=eps)
    layer_jax = RMSNormJAX(dim, eps=eps)
    
    # Copy PyTorch weights to JAX
    layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))
    
    # Forward pass PyTorch
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()
    
    # Forward pass JAX
    output_jax = layer_jax(x_jax)
    
    # Compare outputs
    np.testing.assert_allclose(output_jax, output_pt_np, rtol=1e-5, atol=1e-6)


def test_rms_norm_different_shapes():
    """Test RMSNorm with various input shapes."""
    
    test_configs = [
        (1, 1, 768),      # Single token
        (4, 196, 384),    # Standard ViT
        (2, 1024, 1024),  # Large dimension
        (8, 49, 256),     # Small patches
    ]
    
    for batch_size, seq_len, dim in test_configs:
        x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_jax = jnp.array(x_np)
        
        layer_pt = RMSNormPyTorch(dim)
        layer_jax = RMSNormJAX(dim)
        layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))        

        with torch.no_grad():
            output_pt = layer_pt(x_pt)
        output_jax = layer_jax(x_jax)
        
        np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-5, atol=1e-6)
