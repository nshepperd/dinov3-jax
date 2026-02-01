"""Utilities for converting PyTorch weights to JAX format."""

import jax
import jax.numpy as jnp
import torch


def to_jax(tensor: torch.Tensor) -> jax.Array:
    """Convert a PyTorch tensor to a JAX array."""
    if tensor.dtype == torch.bfloat16:
        return jnp.array(tensor.detach().cpu().to(torch.float32).numpy()).astype(jnp.bfloat16)
    return jnp.array(tensor.detach().cpu().numpy())

def convert_pytorch_to_jax(pytorch_state_dict: dict[str, torch.Tensor]) -> dict[str, jax.Array]:
    """
    Convert PyTorch state dict to JAX parameter format.
    
    Args:
        pytorch_state_dict: PyTorch model state dictionary
    
    Returns:
        JAX parameter dictionary
    """
    return {key: to_jax(value) for key, value in pytorch_state_dict.items()}