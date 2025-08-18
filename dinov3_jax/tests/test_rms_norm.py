"""Test RMSNorm compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp
from jaxtorch import Context

import torch
import torch.nn as nn

from dinov3_jax.layers.rms_norm import RMSNorm as RMSNormJAX
from dinov3.layers.rms_norm import RMSNorm as RMSNormPyTorch
jax.config.update("jax_default_matmul_precision", "highest")


def test_rms_norm_forward():
    """Test that RMSNorm produces identical outputs in JAX and PyTorch."""
    
    # Configuration
    dim = 768
    eps = 1e-5
    batch_size = 2
    seq_len = 197
    
    # Create random input
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    
    # Create PyTorch module
    layer_pt = RMSNormPyTorch(dim, eps=eps)
    x_pt = torch.from_numpy(x_np)
    
    # Create JAX module
    layer_jax = RMSNormJAX(dim, eps=eps)
    x_jax = jnp.array(x_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(42)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy PyTorch weights to JAX
    weight_pt = layer_pt.weight.detach().numpy()
    cx[layer_jax.weight] = jnp.array(weight_pt)
    
    # Forward pass PyTorch
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()
    
    # Forward pass JAX
    output_jax = layer_jax(cx, x_jax)
    output_jax_np = np.array(output_jax)
    
    # Compare outputs
    np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-5, atol=1e-6)
    print("✓ RMSNorm forward pass matches")


def test_rms_norm_gradient():
    """Test that RMSNorm gradients match between JAX and PyTorch."""
    
    # Configuration
    dim = 256
    eps = 1e-5
    batch_size = 1
    seq_len = 49
    
    # Create random input and target
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    target_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    
    # PyTorch setup
    layer_pt = RMSNormPyTorch(dim, eps=eps)
    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    target_pt = torch.from_numpy(target_np)
    
    # JAX setup
    layer_jax = RMSNormJAX(dim, eps=eps)
    x_jax = jnp.array(x_np)
    target_jax = jnp.array(target_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(42)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy weights
    cx[layer_jax.weight] = jnp.array(layer_pt.weight.detach().numpy())
    
    # PyTorch forward and backward
    output_pt = layer_pt(x_pt)
    loss_pt = ((output_pt - target_pt) ** 2).mean()
    loss_pt.backward()
    
    # Get PyTorch gradients
    weight_grad_pt = layer_pt.weight.grad.numpy()
    x_grad_pt = x_pt.grad.numpy()
    
    # JAX forward and gradient
    def loss_fn(params, x, target):
        cx_inner = Context(params, key)
        output = layer_jax(cx_inner, x)
        return jnp.mean((output - target) ** 2)
    
    loss_jax, grads_jax = jax.value_and_grad(loss_fn)(cx.params, x_jax, target_jax)
    weight_grad_jax = np.array(grads_jax[layer_jax.weight.name])
    
    # Compute input gradient
    def loss_fn_x(x, params, target):
        cx_inner = Context(params, key)
        output = layer_jax(cx_inner, x)
        return jnp.mean((output - target) ** 2)
    
    x_grad_jax = jax.grad(loss_fn_x)(x_jax, cx.params, target_jax)
    x_grad_jax_np = np.array(x_grad_jax)
    
    # Compare gradients
    np.testing.assert_allclose(weight_grad_jax, weight_grad_pt, rtol=1e-4, atol=1e-6)
    print("✓ RMSNorm weight gradients match")
    
    np.testing.assert_allclose(x_grad_jax_np, x_grad_pt, rtol=1e-4, atol=1e-6)
    print("✓ RMSNorm input gradients match")


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
        
        layer_pt = RMSNormPyTorch(dim)
        x_pt = torch.from_numpy(x_np)
        
        layer_jax = RMSNormJAX(dim)
        x_jax = jnp.array(x_np)
        
        key = jax.random.PRNGKey(123)
        params = {}
        cx = Context(params, key)
        layer_jax.setup(cx)
        
        cx[layer_jax.weight] = jnp.array(layer_pt.weight.detach().numpy())
        
        with torch.no_grad():
            output_pt = layer_pt(x_pt)
        output_jax = layer_jax(cx, x_jax)
        
        np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-5, atol=1e-6)
        print(f"✓ RMSNorm with shape ({batch_size}, {seq_len}, {dim}) matches")


if __name__ == "__main__":
    test_rms_norm_forward()
    test_rms_norm_gradient()
    test_rms_norm_different_shapes()
    print("\n✅ All RMSNorm tests passed!")