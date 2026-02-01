"""Test LayerScale module compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import torch
import torch.nn as nn

from dinov3_jax.layers.layer_scale import LayerScale as LayerScaleJAX
from dinov3.layers.layer_scale import LayerScale as LayerScalePyTorch
jax.config.update("jax_default_matmul_precision", "highest")


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    """Convert a PyTorch state dict to JAX format."""
    return {k: jnp.array(v.detach().cpu().numpy()) for k, v in state_dict.items()}


def test_layer_scale_forward():
    """Test that LayerScale produces identical outputs in JAX and PyTorch."""

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    dim = 768
    init_value = 1e-4
    batch_size = 2
    seq_len = 197

    # Create random input
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

    # Create PyTorch module and input
    layer_pt = LayerScalePyTorch(dim, init_values=init_value)
    x_pt = torch.from_numpy(x_np)

    # Create JAX module
    layer_jax = LayerScaleJAX(dim, init_values=init_value)
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
    print("✓ LayerScale forward pass matches between JAX and PyTorch")

    # Test with different initialization values
    init_value_2 = 0.1
    layer_pt_2 = LayerScalePyTorch(dim, init_values=init_value_2)
    layer_jax_2 = LayerScaleJAX(dim, init_values=init_value_2)

    # Copy weights
    layer_jax_2 = layer_jax_2.load_state_dict(convert_state_dict(layer_pt_2.state_dict()))

    # Forward passes
    with torch.no_grad():
        output_pt_2 = layer_pt_2(x_pt)
    output_pt_2_np = output_pt_2.numpy()

    output_jax_2 = layer_jax_2(x_jax)
    output_jax_2_np = np.array(output_jax_2)

    # Compare
    np.testing.assert_allclose(output_jax_2_np, output_pt_2_np, rtol=1e-5, atol=1e-6)
    print("✓ LayerScale with different init value matches")


def test_layer_scale_gradient():
    """Test that LayerScale gradients match between JAX and PyTorch."""

    # Configuration
    dim = 256
    init_value = 1e-3
    batch_size = 1
    seq_len = 49

    # Create random input and target
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    target_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

    # PyTorch setup
    layer_pt = LayerScalePyTorch(dim, init_values=init_value)
    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    target_pt = torch.from_numpy(target_np)

    # JAX setup
    layer_jax = LayerScaleJAX(dim, init_values=init_value)
    x_jax = jnp.array(x_np)
    target_jax = jnp.array(target_np)

    # Copy PyTorch weights to JAX
    layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))

    # PyTorch forward and backward
    output_pt = layer_pt(x_pt)
    loss_pt = ((output_pt - target_pt) ** 2).mean()
    loss_pt.backward()

    # Get PyTorch gradients
    gamma_grad_pt = layer_pt.gamma.grad.numpy()
    x_grad_pt = x_pt.grad.numpy()

    # JAX forward and gradient using equinox filter
    @eqx.filter_value_and_grad
    def loss_fn(model, x, target):
        output = model(x)
        return jnp.mean((output - target) ** 2)

    loss_jax, grads_jax = loss_fn(layer_jax, x_jax, target_jax)
    gamma_grad_jax = np.array(grads_jax.gamma)

    # Also compute input gradient
    def loss_fn_x(x, model, target):
        output = model(x)
        return jnp.mean((output - target) ** 2)

    x_grad_jax = jax.grad(loss_fn_x)(x_jax, layer_jax, target_jax)
    x_grad_jax_np = np.array(x_grad_jax)

    # Compare gradients
    np.testing.assert_allclose(gamma_grad_jax, gamma_grad_pt, rtol=1e-4, atol=1e-6)
    print("✓ LayerScale parameter gradients match")

    np.testing.assert_allclose(x_grad_jax_np, x_grad_pt, rtol=1e-4, atol=1e-6)
    print("✓ LayerScale input gradients match")

    # Compare loss values
    loss_pt_val = loss_pt.item()
    loss_jax_val = float(loss_jax)
    np.testing.assert_allclose(loss_jax_val, loss_pt_val, rtol=1e-5, atol=1e-6)
    print("✓ Loss values match")


if __name__ == "__main__":
    test_layer_scale_forward()
    test_layer_scale_gradient()
    print("\n✅ All LayerScale tests passed!")
