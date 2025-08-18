"""Test Attention layers compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp
from jaxtorch import Context

import torch
import torch.nn as nn

from dinov3_jax.layers.attention import SelfAttention as SelfAttentionJAX
from dinov3_jax.layers.attention import LinearKMaskedBias as LinearKMaskedBiasJAX
from dinov3.layers.attention import SelfAttention as SelfAttentionPyTorch
from dinov3.layers.attention import LinearKMaskedBias as LinearKMaskedBiasPyTorch
jax.config.update("jax_default_matmul_precision", "highest")


def test_self_attention_forward():
    """Test that SelfAttention produces identical outputs in JAX and PyTorch."""
    
    # Configuration
    dim = 768
    num_heads = 12
    batch_size = 2
    seq_len = 197
    
    # Create random input
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    
    # Create PyTorch module
    layer_pt = SelfAttentionPyTorch(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        proj_bias=True,
    )
    x_pt = torch.from_numpy(x_np)
    
    # Create JAX module
    layer_jax = SelfAttentionJAX(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        proj_bias=True,
    )
    x_jax = jnp.array(x_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(42)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy PyTorch weights to JAX
    cx[layer_jax.qkv.weight] = jnp.array(layer_pt.qkv.weight.detach().numpy())
    cx[layer_jax.qkv.bias] = jnp.array(layer_pt.qkv.bias.detach().numpy())
    cx[layer_jax.proj.weight] = jnp.array(layer_pt.proj.weight.detach().numpy())
    cx[layer_jax.proj.bias] = jnp.array(layer_pt.proj.bias.detach().numpy())
    
    # Forward pass PyTorch
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()
    
    # Forward pass JAX
    output_jax = layer_jax(cx, x_jax)
    output_jax_np = np.array(output_jax)
    
    # Compare outputs (relaxed tolerance for attention)
    np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-4, atol=1e-5)
    print("✓ SelfAttention forward pass matches")


def test_self_attention_with_rope():
    """Test SelfAttention with RoPE position embeddings."""
    
    # Configuration
    dim = 384
    num_heads = 6
    batch_size = 1
    seq_len = 196  # 14x14 patches
    
    # Create input
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    
    # Create RoPE embeddings (simplified for testing)
    D_head = dim // num_heads
    rope_sin = np.random.randn(seq_len, D_head).astype(np.float32)
    rope_cos = np.random.randn(seq_len, D_head).astype(np.float32)
    rope_pt = (torch.from_numpy(rope_sin), torch.from_numpy(rope_cos))
    rope_jax = (jnp.array(rope_sin), jnp.array(rope_cos))
    
    # Create modules
    layer_pt = SelfAttentionPyTorch(dim=dim, num_heads=num_heads, qkv_bias=True)
    x_pt = torch.from_numpy(x_np)
    
    layer_jax = SelfAttentionJAX(dim=dim, num_heads=num_heads, qkv_bias=True)
    x_jax = jnp.array(x_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(123)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy weights
    cx[layer_jax.qkv.weight] = jnp.array(layer_pt.qkv.weight.detach().numpy())
    cx[layer_jax.qkv.bias] = jnp.array(layer_pt.qkv.bias.detach().numpy())
    cx[layer_jax.proj.weight] = jnp.array(layer_pt.proj.weight.detach().numpy())
    cx[layer_jax.proj.bias] = jnp.array(layer_pt.proj.bias.detach().numpy())
    
    # Forward passes with RoPE
    with torch.no_grad():
        output_pt = layer_pt(x_pt, rope=rope_pt)
    output_jax = layer_jax(cx, x_jax, rope=rope_jax)
    
    # Compare
    np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-4, atol=1e-5)
    print("✓ SelfAttention with RoPE matches")


def test_linear_k_masked_bias():
    """Test LinearKMaskedBias layer."""
    
    # Configuration
    in_features = 768
    out_features = 768 * 3  # QKV projection
    batch_size = 2
    seq_len = 49
    
    # Create input
    x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
    
    # Create PyTorch module
    layer_pt = LinearKMaskedBiasPyTorch(in_features, out_features, bias=True)
    x_pt = torch.from_numpy(x_np)
    
    # Create JAX module
    layer_jax = LinearKMaskedBiasJAX(in_features, out_features, bias=True)
    x_jax = jnp.array(x_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(456)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy weights and bias mask
    cx[layer_jax.weight] = jnp.array(layer_pt.weight.detach().numpy())
    cx[layer_jax.bias] = jnp.array(layer_pt.bias.detach().numpy())
    # Note: bias_mask should be the same (NaN for K portion)
    
    # Forward passes
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_jax = layer_jax(cx, x_jax)
    
    # Compare outputs
    np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-5, atol=1e-6)
    print("✓ LinearKMaskedBias forward pass matches")


def test_self_attention_different_configs():
    """Test SelfAttention with various configurations."""
    
    test_configs = [
        (256, 4, True, True),    # Small with biases
        (768, 12, False, False), # No biases
        (1024, 16, True, False), # Mixed biases
    ]
    
    for dim, num_heads, qkv_bias, proj_bias in test_configs:
        batch_size = 1
        seq_len = 49
        x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        
        # Create modules
        layer_pt = SelfAttentionPyTorch(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias
        )
        x_pt = torch.from_numpy(x_np)
        
        layer_jax = SelfAttentionJAX(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias
        )
        x_jax = jnp.array(x_np)
        
        # Initialize JAX
        key = jax.random.PRNGKey(789)
        params = {}
        cx = Context(params, key)
        layer_jax.setup(cx)
        
        # Copy weights
        cx[layer_jax.qkv.weight] = jnp.array(layer_pt.qkv.weight.detach().numpy())
        if qkv_bias:
            cx[layer_jax.qkv.bias] = jnp.array(layer_pt.qkv.bias.detach().numpy())
        cx[layer_jax.proj.weight] = jnp.array(layer_pt.proj.weight.detach().numpy())
        if proj_bias:
            cx[layer_jax.proj.bias] = jnp.array(layer_pt.proj.bias.detach().numpy())
        
        # Forward passes
        with torch.no_grad():
            output_pt = layer_pt(x_pt)
        output_jax = layer_jax(cx, x_jax)
        
        # Compare
        np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-4, atol=1e-5)
        print(f"✓ SelfAttention (dim={dim}, heads={num_heads}, qkv_bias={qkv_bias}, proj_bias={proj_bias}) matches")


def test_self_attention_gradient():
    """Test that SelfAttention gradients match."""
    
    # Configuration (smaller for faster gradient computation)
    dim = 256
    num_heads = 4
    batch_size = 1
    seq_len = 16
    
    # Create input and target
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    target_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    
    # PyTorch setup
    layer_pt = SelfAttentionPyTorch(dim=dim, num_heads=num_heads, qkv_bias=True)
    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    target_pt = torch.from_numpy(target_np)
    
    # JAX setup
    layer_jax = SelfAttentionJAX(dim=dim, num_heads=num_heads, qkv_bias=True)
    x_jax = jnp.array(x_np)
    target_jax = jnp.array(target_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(111)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy weights
    cx[layer_jax.qkv.weight] = jnp.array(layer_pt.qkv.weight.detach().numpy())
    cx[layer_jax.qkv.bias] = jnp.array(layer_pt.qkv.bias.detach().numpy())
    cx[layer_jax.proj.weight] = jnp.array(layer_pt.proj.weight.detach().numpy())
    cx[layer_jax.proj.bias] = jnp.array(layer_pt.proj.bias.detach().numpy())
    
    # PyTorch forward and backward
    output_pt = layer_pt(x_pt)
    loss_pt = ((output_pt - target_pt) ** 2).mean()
    loss_pt.backward()
    
    # JAX forward and gradient
    def loss_fn(params, x, target):
        cx_inner = Context(params, key)
        output = layer_jax(cx_inner, x)
        return jnp.mean((output - target) ** 2)
    
    loss_jax, grads_jax = jax.value_and_grad(loss_fn)(cx.params, x_jax, target_jax)
    
    # Compare loss values
    np.testing.assert_allclose(float(loss_jax), loss_pt.item(), rtol=1e-4, atol=1e-5)
    print("✓ SelfAttention loss values match")
    
    # Compare parameter gradients (relaxed tolerance for attention gradients)
    qkv_grad_jax = np.array(grads_jax[layer_jax.qkv.weight.name])
    qkv_grad_pt = layer_pt.qkv.weight.grad.numpy()
    np.testing.assert_allclose(qkv_grad_jax, qkv_grad_pt, rtol=1e-3, atol=1e-5)
    print("✓ SelfAttention gradients match")


if __name__ == "__main__":
    test_self_attention_forward()
    test_self_attention_with_rope()
    test_linear_k_masked_bias()
    test_self_attention_different_configs()
    test_self_attention_gradient()
    print("\n✅ All Attention tests passed!")