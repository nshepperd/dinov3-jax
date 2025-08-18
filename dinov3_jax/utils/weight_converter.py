"""Utilities for converting PyTorch weights to JAX format."""

import re
from typing import Dict, Any

import torch
import jax.numpy as jnp
import numpy as np


def convert_pytorch_to_jax(pytorch_state_dict: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    """
    Convert PyTorch state dict to JAX parameter format.
    
    Args:
        pytorch_state_dict: PyTorch model state dictionary
    
    Returns:
        JAX parameter dictionary compatible with jaxtorch
    """
    jax_params = {}
    
    for key, value in pytorch_state_dict.items():
        # Convert torch tensor to numpy then JAX
        if hasattr(value, "numpy"): 
            # Handle BFloat16 by converting to float32 first
            if value.dtype == torch.bfloat16:
                np_value = value.detach().cpu().float().numpy()
            else:
                np_value = value.detach().cpu().numpy()
        else:
            np_value = np.array(value)
        
        # Handle specific layer conversions
        jax_key = convert_key_name(key)
        
        # Convert weights based on layer type
        if "proj.weight" in key and "patch_embed" in key:
            # Conv2d weight: PyTorch is (out, in, h, w), keep same for JAX Conv2d
            jax_value = jnp.array(np_value)
        elif ".weight" in key and ("fc" in key or "proj" in key or "qkv" in key or "w1" in key or "w2" in key or "w3" in key):
            # Linear weight: PyTorch is (out, in), keep same for JAX Linear
            jax_value = jnp.array(np_value)
        else:
            # Default: direct conversion
            jax_value = jnp.array(np_value)
        
        jax_params[jax_key] = jax_value
    
    return jax_params


def convert_key_name(pytorch_key: str) -> str:
    """
    Convert PyTorch parameter name to JAX/jaxtorch format.
    
    Args:
        pytorch_key: PyTorch parameter key name
    
    Returns:
        Converted JAX parameter key name
    """
    # Direct mappings
    key = pytorch_key
    
    # Handle block indices (blocks.0 -> blocks.0)
    # Keep the same format
    
    # Handle attention layers
    key = key.replace("attn.qkv.", "attn.qkv.")
    key = key.replace("attn.proj.", "attn.proj.")
    
    # Handle MLP layers
    key = key.replace("mlp.fc1.", "mlp.fc1.")
    key = key.replace("mlp.fc2.", "mlp.fc2.")
    
    # Handle SwiGLU layers
    key = key.replace("mlp.w1.", "mlp.w1.")
    key = key.replace("mlp.w2.", "mlp.w2.")
    key = key.replace("mlp.w3.", "mlp.w3.")
    
    # Handle layer scale
    key = key.replace("ls1.gamma", "ls1.gamma")
    key = key.replace("ls2.gamma", "ls2.gamma")
    
    # Handle normalization layers
    # LayerNorm uses 'weight' and 'bias' in both PyTorch and JAX
    # RMSNorm uses 'weight' in both
    
    return key


def load_pytorch_weights(jax_model, pytorch_state_dict: Dict[str, Any], cx):
    """
    Load PyTorch weights into a JAX model.
    
    Args:
        jax_model: JAX model instance
        pytorch_state_dict: PyTorch state dictionary
        cx: JAX Context object containing parameters
    
    Returns:
        Updated context with loaded weights
    """
    # Convert PyTorch weights to JAX format
    jax_params = convert_pytorch_to_jax(pytorch_state_dict)
    
    # Map converted parameters to model parameter names
    for param_name, param in jax_model.gen_named_parameters():
        if param_name in jax_params:
            cx.params[param.name] = jax_params[param_name]
        else:
            # Try alternative naming patterns
            alt_name = param_name.replace("_", ".")
            if alt_name in jax_params:
                cx.params[param.name] = jax_params[alt_name]
            else:
                print(f"Warning: Parameter {param_name} not found in PyTorch state dict")
    
    return cx