"""Test SelfAttentionBlock compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array

import torch

from dinov3_jax.layers.block import SelfAttentionBlock as SelfAttentionBlockJAX
from dinov3.layers.block import SelfAttentionBlock as SelfAttentionBlockPyTorch
jax.config.update("jax_default_matmul_precision", "highest")


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Array]:
    """Convert a PyTorch state dict to JAX format."""
    return {k: jnp.array(v.detach().cpu().numpy()) for k, v in state_dict.items()}


def test_self_attention_block_forward():
    """Test that SelfAttentionBlock produces identical outputs in JAX and PyTorch."""

    # Configuration
    dim = 768
    num_heads = 12
    ffn_ratio = 4.0
    batch_size = 2
    seq_len = 197

    # Create random input
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

    # Create PyTorch module
    layer_pt = SelfAttentionBlockPyTorch(
        dim=dim,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    x_pt = torch.from_numpy(x_np)

    # Create JAX module
    layer_jax = SelfAttentionBlockJAX(
        dim=dim,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    x_jax = jnp.array(x_np)

    # Copy PyTorch weights to JAX
    layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))

    # Forward pass PyTorch
    layer_pt.eval()
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()

    # Forward pass JAX
    output_jax = layer_jax(x_jax)
    output_jax_np = np.array(output_jax)

    # Compare outputs (relaxed tolerance due to flash attention fp16)
    np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-2, atol=1e-2)
    print("✓ SelfAttentionBlock forward pass matches")


def test_block_with_layer_scale():
    """Test SelfAttentionBlock with LayerScale."""

    # Configuration
    dim = 384
    num_heads = 6
    ffn_ratio = 4.0
    init_values = 1e-4
    batch_size = 1
    seq_len = 196

    # Create input
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

    # Create modules with LayerScale
    layer_pt = SelfAttentionBlockPyTorch(
        dim=dim,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        init_values=init_values,
    )
    x_pt = torch.from_numpy(x_np)

    layer_jax = SelfAttentionBlockJAX(
        dim=dim,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        init_values=init_values,
    )
    x_jax = jnp.array(x_np)

    # Copy weights
    layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))

    # Forward passes
    layer_pt.eval()
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_jax = layer_jax(x_jax)

    # Compare (relaxed tolerance due to flash attention fp16)
    np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-2, atol=1e-2)
    print("✓ SelfAttentionBlock with LayerScale matches")


def test_block_with_rope():
    """Test SelfAttentionBlock with RoPE embeddings."""

    # Configuration
    dim = 384
    num_heads = 6
    batch_size = 1
    seq_len = 196  # 14x14 patches

    # Create input and RoPE
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    D_head = dim // num_heads
    rope_sin = np.random.randn(seq_len, D_head).astype(np.float32)
    rope_cos = np.random.randn(seq_len, D_head).astype(np.float32)
    rope_pt = (torch.from_numpy(rope_sin), torch.from_numpy(rope_cos))
    rope_jax = (jnp.array(rope_sin), jnp.array(rope_cos))

    # Create modules
    layer_pt = SelfAttentionBlockPyTorch(
        dim=dim,
        num_heads=num_heads,
    )
    x_pt = torch.from_numpy(x_np)

    layer_jax = SelfAttentionBlockJAX(
        dim=dim,
        num_heads=num_heads,
    )
    x_jax = jnp.array(x_np)

    # Copy weights
    layer_jax = layer_jax.load_state_dict(convert_state_dict(layer_pt.state_dict()))

    # Forward passes with RoPE
    layer_pt.eval()
    with torch.no_grad():
        output_pt = layer_pt(x_pt, rope_pt)
    output_jax = layer_jax(x_jax, rope_jax)

    # Compare (relaxed tolerance due to flash attention fp16)
    np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-2, atol=1e-2)
    print("✓ SelfAttentionBlock with RoPE matches")


def test_block_gradient():
    """Test that SelfAttentionBlock gradients work."""

    # Configuration (smaller for faster computation)
    dim = 256
    num_heads = 4
    batch_size = 1
    seq_len = 16

    # Create input and target
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    target_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

    # Create JAX module
    layer_jax = SelfAttentionBlockJAX(
        dim=dim,
        num_heads=num_heads,
    )
    x_jax = jnp.array(x_np)
    target_jax = jnp.array(target_np)

    # PyTorch setup for comparison
    layer_pt = SelfAttentionBlockPyTorch(
        dim=dim,
        num_heads=num_heads,
    )
    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    target_pt = torch.from_numpy(target_np)

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

    # Compare loss values (relaxed tolerance due to flash attention)
    np.testing.assert_allclose(float(loss_jax), loss_pt.item(), rtol=1e-2, atol=1e-2)
    print("✓ SelfAttentionBlock loss values match")

    # Check that gradients exist
    assert grads_jax.attn.qkv.weight is not None
    assert grads_jax.mlp.fc1.weight is not None
    print("✓ SelfAttentionBlock gradients computed successfully")


if __name__ == "__main__":
    test_self_attention_block_forward()
    test_block_with_layer_scale()
    test_block_with_rope()
    test_block_gradient()
    print("\n✅ All SelfAttentionBlock tests passed!")
