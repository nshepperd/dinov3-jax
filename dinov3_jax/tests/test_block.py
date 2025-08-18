"""Test SelfAttentionBlock compatibility between JAX and PyTorch implementations."""

import numpy as np
import jax
import jax.numpy as jnp
from jaxtorch import Context
from jaxtorch.nn import LayerNorm, GELU

import torch
import torch.nn as nn
from functools import partial

from dinov3_jax.layers.block import SelfAttentionBlock as SelfAttentionBlockJAX
from dinov3_jax.layers.ffn_layers import Mlp as MlpJAX
from dinov3.layers.block import SelfAttentionBlock as SelfAttentionBlockPyTorch
from dinov3.layers.ffn_layers import Mlp as MlpPyTorch
jax.config.update("jax_default_matmul_precision", "highest")


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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        ffn_layer=MlpPyTorch,
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
        norm_layer=partial(LayerNorm, eps=1e-6),
        act_layer=GELU,
        ffn_layer=MlpJAX,
    )
    x_jax = jnp.array(x_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(42)
    params = {}
    cx = Context(params, key, mode="eval")
    layer_jax.setup(cx)
    
    # Copy PyTorch weights to JAX
    # Attention weights
    cx[layer_jax.attn.qkv.weight] = jnp.array(layer_pt.attn.qkv.weight.detach().numpy())
    cx[layer_jax.attn.qkv.bias] = jnp.array(layer_pt.attn.qkv.bias.detach().numpy())
    cx[layer_jax.attn.proj.weight] = jnp.array(layer_pt.attn.proj.weight.detach().numpy())
    cx[layer_jax.attn.proj.bias] = jnp.array(layer_pt.attn.proj.bias.detach().numpy())
    
    # MLP weights
    cx[layer_jax.mlp.fc1.weight] = jnp.array(layer_pt.mlp.fc1.weight.detach().numpy())
    cx[layer_jax.mlp.fc1.bias] = jnp.array(layer_pt.mlp.fc1.bias.detach().numpy())
    cx[layer_jax.mlp.fc2.weight] = jnp.array(layer_pt.mlp.fc2.weight.detach().numpy())
    cx[layer_jax.mlp.fc2.bias] = jnp.array(layer_pt.mlp.fc2.bias.detach().numpy())
    
    # Norm weights
    cx[layer_jax.norm1.weight] = jnp.array(layer_pt.norm1.weight.detach().numpy())
    cx[layer_jax.norm1.bias] = jnp.array(layer_pt.norm1.bias.detach().numpy())
    cx[layer_jax.norm2.weight] = jnp.array(layer_pt.norm2.weight.detach().numpy())
    cx[layer_jax.norm2.bias] = jnp.array(layer_pt.norm2.bias.detach().numpy())
    
    # Forward pass PyTorch
    layer_pt.eval()
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_pt_np = output_pt.numpy()
    
    # Forward pass JAX
    output_jax = layer_jax(cx, x_jax)
    output_jax_np = np.array(output_jax)
    
    # Compare outputs (relaxed tolerance for full block)
    np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-4, atol=1e-5)
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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        ffn_layer=MlpPyTorch,
    )
    x_pt = torch.from_numpy(x_np)
    
    layer_jax = SelfAttentionBlockJAX(
        dim=dim,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        init_values=init_values,
        norm_layer=partial(LayerNorm, eps=1e-6),
        act_layer=GELU,
        ffn_layer=MlpJAX,
    )
    x_jax = jnp.array(x_np)
    
    # Initialize JAX
    key = jax.random.PRNGKey(123)
    params = {}
    cx = Context(params, key, mode="eval")
    layer_jax.setup(cx)
    
    # Copy all weights including LayerScale
    cx[layer_jax.attn.qkv.weight] = jnp.array(layer_pt.attn.qkv.weight.detach().numpy())
    cx[layer_jax.attn.qkv.bias] = jnp.array(layer_pt.attn.qkv.bias.detach().numpy())
    cx[layer_jax.attn.proj.weight] = jnp.array(layer_pt.attn.proj.weight.detach().numpy())
    cx[layer_jax.attn.proj.bias] = jnp.array(layer_pt.attn.proj.bias.detach().numpy())
    
    cx[layer_jax.mlp.fc1.weight] = jnp.array(layer_pt.mlp.fc1.weight.detach().numpy())
    cx[layer_jax.mlp.fc1.bias] = jnp.array(layer_pt.mlp.fc1.bias.detach().numpy())
    cx[layer_jax.mlp.fc2.weight] = jnp.array(layer_pt.mlp.fc2.weight.detach().numpy())
    cx[layer_jax.mlp.fc2.bias] = jnp.array(layer_pt.mlp.fc2.bias.detach().numpy())
    
    cx[layer_jax.norm1.weight] = jnp.array(layer_pt.norm1.weight.detach().numpy())
    cx[layer_jax.norm1.bias] = jnp.array(layer_pt.norm1.bias.detach().numpy())
    cx[layer_jax.norm2.weight] = jnp.array(layer_pt.norm2.weight.detach().numpy())
    cx[layer_jax.norm2.bias] = jnp.array(layer_pt.norm2.bias.detach().numpy())
    
    # LayerScale parameters
    cx[layer_jax.ls1.gamma] = jnp.array(layer_pt.ls1.gamma.detach().numpy())
    cx[layer_jax.ls2.gamma] = jnp.array(layer_pt.ls2.gamma.detach().numpy())
    
    # Forward passes
    layer_pt.eval()
    with torch.no_grad():
        output_pt = layer_pt(x_pt)
    output_jax = layer_jax(cx, x_jax)
    
    # Compare
    np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-4, atol=1e-5)
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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        ffn_layer=MlpPyTorch,
    )
    x_pt = torch.from_numpy(x_np)
    
    layer_jax = SelfAttentionBlockJAX(
        dim=dim,
        num_heads=num_heads,
        norm_layer=partial(LayerNorm, eps=1e-6),
        act_layer=GELU,
        ffn_layer=MlpJAX,
    )
    x_jax = jnp.array(x_np)
    
    # Initialize and copy weights
    key = jax.random.PRNGKey(456)
    params = {}
    cx = Context(params, key, mode="eval")
    layer_jax.setup(cx)
    
    # Copy all weights (simplified - assuming default initialization matches)
    for name, param in layer_jax.gen_named_parameters():
        pt_name = name.replace("_", ".")
        if hasattr(layer_pt, pt_name.split(".")[0]):
            pt_module = layer_pt
            for part in pt_name.split(".")[:-1]:
                pt_module = getattr(pt_module, part)
            pt_param = getattr(pt_module, pt_name.split(".")[-1])
            if hasattr(pt_param, "detach"):
                cx[param] = jnp.array(pt_param.detach().numpy())
    
    # Forward passes with RoPE
    layer_pt.eval()
    with torch.no_grad():
        output_pt = layer_pt(x_pt, rope_pt)
    output_jax = layer_jax(cx, x_jax, rope_jax)
    
    # Compare
    np.testing.assert_allclose(np.array(output_jax), output_pt.numpy(), rtol=1e-4, atol=1e-5)
    print("✓ SelfAttentionBlock with RoPE matches")


def test_block_forward_list():
    """Test SelfAttentionBlock with list inputs (multi-crop)."""
    
    # Configuration
    dim = 384
    num_heads = 6
    
    # Create list of inputs
    x_list_np = [
        np.random.randn(2, 196, dim).astype(np.float32),  # Global crop
        np.random.randn(2, 49, dim).astype(np.float32),   # Local crop 1
        np.random.randn(2, 49, dim).astype(np.float32),   # Local crop 2
    ]
    
    # Create JAX module
    layer_jax = SelfAttentionBlockJAX(
        dim=dim,
        num_heads=num_heads,
        norm_layer=partial(LayerNorm, eps=1e-6),
        act_layer=GELU,
        ffn_layer=MlpJAX,
    )
    
    # Initialize
    key = jax.random.PRNGKey(789)
    params = {}
    cx = Context(params, key, mode="eval")
    layer_jax.setup(cx)
    
    # Forward with list
    x_list_jax = [jnp.array(x) for x in x_list_np]
    outputs = layer_jax(cx, x_list_jax)
    
    # Check outputs
    assert isinstance(outputs, list)
    assert len(outputs) == len(x_list_jax)
    for out, inp in zip(outputs, x_list_jax):
        assert out.shape == inp.shape
    print("✓ SelfAttentionBlock forward_list works correctly")


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
        norm_layer=partial(LayerNorm, eps=1e-6),
        act_layer=GELU,
        ffn_layer=MlpJAX,
    )
    x_jax = jnp.array(x_np)
    target_jax = jnp.array(target_np)
    
    # Initialize
    key = jax.random.PRNGKey(111)
    params = {}
    cx = Context(params, key)
    layer_jax.setup(cx)
    
    # Define loss function
    def loss_fn(params, x, target):
        cx_inner = Context(params, key, mode="eval")
        output = layer_jax(cx_inner, x)
        return jnp.mean((output - target) ** 2)
    
    # Compute gradients
    loss_value, grads = jax.value_and_grad(loss_fn)(cx.params, x_jax, target_jax)
    
    # Check that gradients exist and have correct shapes
    assert len(grads) > 0
    for param_name, grad in grads.items():
        assert grad.shape == cx.params[param_name].shape
    
    print("✓ SelfAttentionBlock gradients computed successfully")
    print(f"  Loss value: {float(loss_value):.6f}")


if __name__ == "__main__":
    test_self_attention_block_forward()
    test_block_with_layer_scale()
    test_block_with_rope()
    test_block_forward_list()
    test_block_gradient()
    print("\n✅ All SelfAttentionBlock tests passed!")