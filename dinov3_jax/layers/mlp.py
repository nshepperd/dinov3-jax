from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from dinov3_jax.eepynox.nn.linear import Linear
from dinov3_jax.config import Dinov3VitConfig


class Dinov3VitMLP(eqx.Module):
    """Standard MLP (up_proj -> act -> down_proj)."""

    up_proj: Linear
    down_proj: Linear
    hidden_act: str = eqx.field(static=True)

    def __init__(self, config: Dinov3VitConfig, dtype=jnp.float32):
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias, dtype=dtype)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, use_bias=config.mlp_bias, dtype=dtype)
        self.hidden_act = config.hidden_act

    def __call__(self, x: Array) -> Array:
        x = self.up_proj(x)
        x = _activate(x, self.hidden_act)
        x = self.down_proj(x)
        return x


class Dinov3VitGatedMLP(eqx.Module):
    """Gated MLP with SiLU (SwiGLU): gate_proj * up_proj -> down_proj."""

    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    hidden_act: str = eqx.field(static=True)

    def __init__(self, config: Dinov3VitConfig, dtype=jnp.float32):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias, dtype=dtype)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias, dtype=dtype)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, use_bias=config.mlp_bias, dtype=dtype)
        self.hidden_act = config.hidden_act

    def __call__(self, x: Array) -> Array:
        gate = _activate(self.gate_proj(x), self.hidden_act)
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def _activate(x: Array, act: str) -> Array:
    if act == "gelu":
        return jax.nn.gelu(x, approximate=False)
    elif act == "silu":
        return jax.nn.silu(x)
    elif act == "relu":
        return jax.nn.relu(x)
    else:
        raise ValueError(f"Unknown activation: {act}")
