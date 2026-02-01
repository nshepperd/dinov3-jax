"""Test FFN layers compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import torch
import torch.nn as nn
from functools import partial

from dinov3_jax.layers.ffn_layers import Mlp as MlpJAX
from dinov3_jax.layers.ffn_layers import SwiGLUFFN as SwiGLUFFNJAX
from dinov3.layers.ffn_layers import Mlp as MlpPyTorch
from dinov3.layers.ffn_layers import SwiGLUFFN as SwiGLUFFNPyTorch
jax.config.update("jax_default_matmul_precision", "highest")


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    """Convert a PyTorch state dict to JAX format."""
    return {k: jnp.array(v.detach().cpu().numpy()) for k, v in state_dict.items()}


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

    # Copy PyTorch weights to JAX
    layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))

    # Forward pass PyTorch
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()

    # Forward pass JAX
    output_jax = layer_jax(x_jax)
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

    # Copy PyTorch weights to JAX
    layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))

    # Forward pass PyTorch
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()

    # Forward pass JAX
    output_jax = layer_jax(x_jax)
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

        # Copy weights
        layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))

        # Forward passes
        with torch.no_grad():
            output_pt = layer_pt(x_pt)
        output_jax = layer_jax(x_jax)

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

    # Copy weights
    layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))

    # PyTorch forward and backward
    output_pt = layer_pt(x_pt)
    loss_pt = ((output_pt - target_pt) ** 2).mean()
    loss_pt.backward()

    # JAX forward and gradient using equinox filter
    @eqx.filter_value_and_grad
    def loss_fn(model, x, target):
        output = model(x)
        return jnp.mean((output - target) ** 2)

    loss_jax, grads_jax = loss_fn(layer_jax, x_jax, target_jax)

    # Compare loss
    np.testing.assert_allclose(float(loss_jax), loss_pt.item(), rtol=1e-5, atol=1e-6)
    print("✓ Mlp loss values match")

    # Compare gradients (fc1 weight)
    fc1_weight_grad_jax = np.array(grads_jax.fc1.weight)
    fc1_weight_grad_pt = layer_pt.fc1.weight.grad.numpy()
    np.testing.assert_allclose(fc1_weight_grad_jax, fc1_weight_grad_pt, rtol=1e-4, atol=1e-6)
    print("✓ Mlp gradients match")


if __name__ == "__main__":
    test_mlp_forward()
    test_swiglu_forward()
    test_swiglu_different_alignments()
    test_mlp_gradient()
    print("\n✅ All FFN layer tests passed!")
