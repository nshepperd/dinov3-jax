"""Test PatchEmbed module compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp
from jaxtorch import Context

import torch
import torch.nn as nn

# Import implementations
from dinov3_jax.layers.patch_embed import PatchEmbed as PatchEmbedJAX
from dinov3.layers.patch_embed import PatchEmbed as PatchEmbedPyTorch
jax.config.update("jax_default_matmul_precision", "highest")


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
    
    # Create PyTorch module and input
    layer_pt = PatchEmbedPyTorch(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten_embedding=True,  # Test flattened version first
    )
    x_pt = torch.from_numpy(x_np)
    
    # Create JAX module
    layer_jax = PatchEmbedJAX(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten_embedding=True,
    )
    x_jax = jnp.array(x_np)
    
    # Initialize JAX parameters
    key = jax.random.PRNGKey(42)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy PyTorch weights to JAX
    # Conv2d weight: PyTorch shape is (out_channels, in_channels, kernel_h, kernel_w)
    # JAX Conv2d expects the same format
    conv_weight_pt = layer_pt.proj.weight.detach().numpy()
    conv_bias_pt = layer_pt.proj.bias.detach().numpy()
    
    cx[layer_jax.proj.weight] = jnp.array(conv_weight_pt)
    cx[layer_jax.proj.bias] = jnp.array(conv_bias_pt)
    
    # Forward pass PyTorch
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()
    
    # Forward pass JAX
    output_jax = layer_jax(cx, x_jax)
    output_jax_np = np.array(output_jax)
    
    # Check output shape
    expected_num_patches = (img_size // patch_size) ** 2
    expected_shape = (batch_size, expected_num_patches, embed_dim)
    assert output_jax.shape == expected_shape, f"Expected shape {expected_shape}, got {output_jax.shape}"
    assert output_pt.shape == expected_shape, f"Expected shape {expected_shape}, got {output_pt.shape}"
    
    # Compare outputs (allow slightly higher tolerance for numerical precision)
    np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-5, atol=2e-6)
    print("✓ PatchEmbed forward pass (flattened) matches between JAX and PyTorch")


def test_patch_embed_unflattened():
    """Test PatchEmbed with flatten_embedding=False."""
    
    # Configuration
    img_size = 224
    patch_size = 16
    in_chans = 3
    embed_dim = 384
    batch_size = 1
    
    # Create random input
    x_np = np.random.randn(batch_size, in_chans, img_size, img_size).astype(np.float32)
    
    # Create modules with flatten_embedding=False
    layer_pt = PatchEmbedPyTorch(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten_embedding=False,
    )
    x_pt = torch.from_numpy(x_np)
    
    layer_jax = PatchEmbedJAX(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten_embedding=False,
    )
    x_jax = jnp.array(x_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(123)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy weights
    cx[layer_jax.proj.weight] = jnp.array(layer_pt.proj.weight.detach().numpy())
    cx[layer_jax.proj.bias] = jnp.array(layer_pt.proj.bias.detach().numpy())
    
    # Forward passes
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()
    
    output_jax = layer_jax(cx, x_jax)
    output_jax_np = np.array(output_jax)
    
    # Check shape (should be B, H, W, C)
    H = W = img_size // patch_size
    expected_shape = (batch_size, H, W, embed_dim)
    assert output_jax.shape == expected_shape, f"Expected shape {expected_shape}, got {output_jax.shape}"
    assert output_pt.shape == expected_shape, f"Expected shape {expected_shape}, got {output_pt.shape}"
    
    # Compare outputs (allow slightly higher tolerance for numerical precision)
    np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-5, atol=2e-6)
    print("✓ PatchEmbed forward pass (unflattened) matches between JAX and PyTorch")


def test_patch_embed_different_sizes():
    """Test PatchEmbed with various image and patch sizes."""
    
    test_configs = [
        (224, 16, 768),  # Standard ViT
        (224, 14, 384),  # Different patch size
        (384, 16, 768),  # Larger image
        (96, 8, 512),    # Smaller image and patch
    ]
    
    for img_size, patch_size, embed_dim in test_configs:
        # Create input
        batch_size = 2
        in_chans = 3
        x_np = np.random.randn(batch_size, in_chans, img_size, img_size).astype(np.float32)
        
        # Create modules
        layer_pt = PatchEmbedPyTorch(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=True,
        )
        x_pt = torch.from_numpy(x_np)
        
        layer_jax = PatchEmbedJAX(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=True,
        )
        x_jax = jnp.array(x_np)
        
        # Initialize JAX
        key = jax.random.PRNGKey(456)
        params = {}
        cx = Context(params, key)
        layer_jax.setup(cx)
        
        # Copy weights
        cx[layer_jax.proj.weight] = jnp.array(layer_pt.proj.weight.detach().numpy())
        cx[layer_jax.proj.bias] = jnp.array(layer_pt.proj.bias.detach().numpy())
        
        # Forward passes
        with torch.no_grad():
            output_pt = layer_pt(x_pt)
        output_pt_np = output_pt.numpy()
        
        output_jax = layer_jax(cx, x_jax)
        output_jax_np = np.array(output_jax)
        
        # Compare (allow slightly higher tolerance for numerical precision)
        np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-5, atol=2e-6)
        print(f"✓ PatchEmbed with img_size={img_size}, patch_size={patch_size}, embed_dim={embed_dim} matches")


def test_patch_embed_gradient():
    """Test that PatchEmbed gradients match between JAX and PyTorch."""
    
    # Configuration
    img_size = 112  # Smaller for faster gradient computation
    patch_size = 16
    in_chans = 3
    embed_dim = 256
    batch_size = 1
    
    # Create random input and target
    x_np = np.random.randn(batch_size, in_chans, img_size, img_size).astype(np.float32)
    num_patches = (img_size // patch_size) ** 2
    target_np = np.random.randn(batch_size, num_patches, embed_dim).astype(np.float32)
    
    # PyTorch setup
    layer_pt = PatchEmbedPyTorch(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten_embedding=True,
    )
    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    target_pt = torch.from_numpy(target_np)
    
    # JAX setup
    layer_jax = PatchEmbedJAX(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten_embedding=True,
    )
    x_jax = jnp.array(x_np)
    target_jax = jnp.array(target_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(789)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy PyTorch weights to JAX
    conv_weight_pt_init = layer_pt.proj.weight.detach().numpy()
    conv_bias_pt_init = layer_pt.proj.bias.detach().numpy()
    cx[layer_jax.proj.weight] = jnp.array(conv_weight_pt_init)
    cx[layer_jax.proj.bias] = jnp.array(conv_bias_pt_init)
    
    # PyTorch forward and backward
    output_pt = layer_pt(x_pt)
    loss_pt = ((output_pt - target_pt) ** 2).mean()
    loss_pt.backward()
    
    # Get PyTorch gradients
    weight_grad_pt = layer_pt.proj.weight.grad.numpy()
    bias_grad_pt = layer_pt.proj.bias.grad.numpy()
    x_grad_pt = x_pt.grad.numpy()
    
    # JAX forward and gradient
    def loss_fn(params, x, target):
        cx_inner = Context(params, key)
        output = layer_jax(cx_inner, x)
        return jnp.mean((output - target) ** 2)
    
    # Compute gradients with respect to params
    loss_jax, grads_jax = jax.value_and_grad(loss_fn)(cx.params, x_jax, target_jax)
    weight_grad_jax = np.array(grads_jax[layer_jax.proj.weight.name])
    bias_grad_jax = np.array(grads_jax[layer_jax.proj.bias.name])
    
    # Compute input gradient
    def loss_fn_x(x, params, target):
        cx_inner = Context(params, key)
        output = layer_jax(cx_inner, x)
        return jnp.mean((output - target) ** 2)
    
    x_grad_jax = jax.grad(loss_fn_x)(x_jax, cx.params, target_jax)
    x_grad_jax_np = np.array(x_grad_jax)
    
    # Compare gradients
    np.testing.assert_allclose(weight_grad_jax, weight_grad_pt, rtol=1e-4, atol=1e-6)
    print("✓ PatchEmbed weight gradients match")
    
    np.testing.assert_allclose(bias_grad_jax, bias_grad_pt, rtol=1e-4, atol=1e-6)
    print("✓ PatchEmbed bias gradients match")
    
    np.testing.assert_allclose(x_grad_jax_np, x_grad_pt, rtol=1e-4, atol=1e-6)
    print("✓ PatchEmbed input gradients match")
    
    # Compare loss values
    loss_pt_val = loss_pt.item()
    loss_jax_val = float(loss_jax)
    np.testing.assert_allclose(loss_jax_val, loss_pt_val, rtol=1e-5, atol=1e-6)
    print("✓ Loss values match")


if __name__ == "__main__":
    test_patch_embed_forward()
    test_patch_embed_unflattened()
    test_patch_embed_different_sizes()
    test_patch_embed_gradient()
    print("\n✅ All PatchEmbed tests passed!")