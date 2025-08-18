"""Test FFN layers compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp
from jaxtorch import Context

import torch
import torch.nn as nn
from functools import partial

from dinov3_jax.layers.ffn_layers import Mlp as MlpJAX
from dinov3_jax.layers.ffn_layers import SwiGLUFFN as SwiGLUFFNJAX
from dinov3.layers.ffn_layers import Mlp as MlpPyTorch
from dinov3.layers.ffn_layers import SwiGLUFFN as SwiGLUFFNPyTorch
jax.config.update("jax_default_matmul_precision", "highest")


def test_mlp_forward():
    """Test that Mlp produces identical outputs in JAX and PyTorch."""

    # Configuration
    in_features = 768
    hidden_features = 3072
    out_features = 768
    batch_size = 2
    seq_len = 197
    
    # Create random input
    x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
    
    # Create PyTorch module
    layer_pt = MlpPyTorch(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        bias=True,
    )
    x_pt = torch.from_numpy(x_np)
    
    # Create JAX module
    layer_jax = MlpJAX(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        bias=True,
    )
    x_jax = jnp.array(x_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(42)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy PyTorch weights to JAX
    cx[layer_jax.fc1.weight] = jnp.array(layer_pt.fc1.weight.detach().numpy())
    cx[layer_jax.fc1.bias] = jnp.array(layer_pt.fc1.bias.detach().numpy())
    cx[layer_jax.fc2.weight] = jnp.array(layer_pt.fc2.weight.detach().numpy())
    cx[layer_jax.fc2.bias] = jnp.array(layer_pt.fc2.bias.detach().numpy())
    
    # Forward pass PyTorch
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()
    
    # Forward pass JAX
    output_jax = layer_jax(cx, x_jax)
    output_jax_np = np.array(output_jax)
    
    # Compare outputs
    np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-5, atol=1e-6)
    print("✓ Mlp forward pass matches")


def test_swiglu_forward():
    """Test that SwiGLUFFN produces identical outputs in JAX and PyTorch."""
    
    # Configuration
    in_features = 768
    hidden_features = 2048
    out_features = 768
    align_to = 8
    batch_size = 2
    seq_len = 49
    
    # Create random input
    x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
    
    # Create PyTorch module
    layer_pt = SwiGLUFFNPyTorch(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        align_to=align_to,
        bias=True,
    )
    x_pt = torch.from_numpy(x_np)
    
    # Create JAX module
    layer_jax = SwiGLUFFNJAX(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        align_to=align_to,
        bias=True,
    )
    x_jax = jnp.array(x_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(123)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy PyTorch weights to JAX
    cx[layer_jax.w1.weight] = jnp.array(layer_pt.w1.weight.detach().numpy())
    cx[layer_jax.w1.bias] = jnp.array(layer_pt.w1.bias.detach().numpy())
    cx[layer_jax.w2.weight] = jnp.array(layer_pt.w2.weight.detach().numpy())
    cx[layer_jax.w2.bias] = jnp.array(layer_pt.w2.bias.detach().numpy())
    cx[layer_jax.w3.weight] = jnp.array(layer_pt.w3.weight.detach().numpy())
    cx[layer_jax.w3.bias] = jnp.array(layer_pt.w3.bias.detach().numpy())
    
    # Forward pass PyTorch
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()
    
    # Forward pass JAX
    output_jax = layer_jax(cx, x_jax)
    output_jax_np = np.array(output_jax)
    
    # Compare outputs
    np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-5, atol=1e-6)
    print("✓ SwiGLUFFN forward pass matches")


def test_swiglu_different_alignments():
    """Test SwiGLUFFN with different alignment values."""
    
    alignments = [8, 32, 64, 128]
    in_features = 384
    out_features = 384
    batch_size = 1
    seq_len = 196
    
    for align_to in alignments:
        x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
        
        # Create modules with specific alignment
        layer_pt = SwiGLUFFNPyTorch(
            in_features=in_features,
            out_features=out_features,
            align_to=align_to,
            bias=True,
        )
        x_pt = torch.from_numpy(x_np)
        
        layer_jax = SwiGLUFFNJAX(
            in_features=in_features,
            out_features=out_features,
            align_to=align_to,
            bias=True,
        )
        x_jax = jnp.array(x_np)
        
        # Initialize JAX
        key = jax.random.PRNGKey(456)
        params = {}
        cx = Context(params, key)
        layer_jax.setup(cx)
        
        # Copy weights
        cx[layer_jax.w1.weight] = jnp.array(layer_pt.w1.weight.detach().numpy())
        cx[layer_jax.w1.bias] = jnp.array(layer_pt.w1.bias.detach().numpy())
        cx[layer_jax.w2.weight] = jnp.array(layer_pt.w2.weight.detach().numpy())
        cx[layer_jax.w2.bias] = jnp.array(layer_pt.w2.bias.detach().numpy())
        cx[layer_jax.w3.weight] = jnp.array(layer_pt.w3.weight.detach().numpy())
        cx[layer_jax.w3.bias] = jnp.array(layer_pt.w3.bias.detach().numpy())
        
        # Forward passes
        with torch.no_grad():
            output_pt = layer_pt(x_pt)
        output_jax = layer_jax(cx, x_jax)
        
        # Compare
        np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-5, atol=1e-6)
        print(f"✓ SwiGLUFFN with align_to={align_to} matches")


def test_mlp_gradient():
    """Test that Mlp gradients match."""
    
    # Configuration
    in_features = 256
    hidden_features = 512
    out_features = 256
    batch_size = 1
    seq_len = 49
    
    # Create input and target
    x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
    target_np = np.random.randn(batch_size, seq_len, out_features).astype(np.float32)
    
    # PyTorch setup
    layer_pt = MlpPyTorch(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
    )
    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    target_pt = torch.from_numpy(target_np)
    
    # JAX setup
    layer_jax = MlpJAX(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
    )
    x_jax = jnp.array(x_np)
    target_jax = jnp.array(target_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(789)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Copy weights
    cx[layer_jax.fc1.weight] = jnp.array(layer_pt.fc1.weight.detach().numpy())
    cx[layer_jax.fc1.bias] = jnp.array(layer_pt.fc1.bias.detach().numpy())
    cx[layer_jax.fc2.weight] = jnp.array(layer_pt.fc2.weight.detach().numpy())
    cx[layer_jax.fc2.bias] = jnp.array(layer_pt.fc2.bias.detach().numpy())
    
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
    
    # Compare loss
    np.testing.assert_allclose(float(loss_jax), loss_pt.item(), rtol=1e-5, atol=1e-6)
    print("✓ Mlp loss values match")
    
    # Compare gradients
    fc1_weight_grad_jax = np.array(grads_jax[layer_jax.fc1.weight.name])
    fc1_weight_grad_pt = layer_pt.fc1.weight.grad.numpy()
    np.testing.assert_allclose(fc1_weight_grad_jax, fc1_weight_grad_pt, rtol=1e-4, atol=1e-6)
    print("✓ Mlp gradients match")


def test_forward_list():
    """Test forward_list method for multi-crop training."""
    
    # Configuration
    in_features = 384
    hidden_features = 1536
    out_features = 384
    
    # Create list of inputs with different sequence lengths
    x_list_np = [
        np.random.randn(2, 196, in_features).astype(np.float32),  # Global crop
        np.random.randn(2, 49, in_features).astype(np.float32),   # Local crop 1
        np.random.randn(2, 49, in_features).astype(np.float32),   # Local crop 2
    ]
    
    # Test Mlp forward_list
    layer_jax = MlpJAX(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
    )
    
    key = jax.random.PRNGKey(111)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    x_list_jax = [jnp.array(x) for x in x_list_np]
    outputs = layer_jax.forward_list(cx, x_list_jax)
    
    # Check outputs
    assert len(outputs) == len(x_list_jax)
    for i, (out, inp) in enumerate(zip(outputs, x_list_jax)):
        assert out.shape == inp.shape[:2] + (out_features,)
    print("✓ Mlp forward_list works correctly")
    
    # Test SwiGLUFFN forward_list
    layer_swiglu = SwiGLUFFNJAX(
        in_features=in_features,
        out_features=out_features,
    )
    
    params = {}
    cx = Context(params, key)
    layer_swiglu.setup(cx)
    
    outputs_swiglu = layer_swiglu.forward_list(cx, x_list_jax)
    
    assert len(outputs_swiglu) == len(x_list_jax)
    for out, inp in zip(outputs_swiglu, x_list_jax):
        assert out.shape == inp.shape[:2] + (out_features,)
    print("✓ SwiGLUFFN forward_list works correctly")


if __name__ == "__main__":
    test_mlp_forward()
    test_swiglu_forward()
    test_swiglu_different_alignments()
    test_mlp_gradient()
    test_forward_list()
    print("\n✅ All FFN layer tests passed!")