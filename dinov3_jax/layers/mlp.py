from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import dinov3_jax.eepynox.utils as eu
from dinov3_jax.eepynox.nn.linear import Linear
from dinov3_jax.config import Dinov3VitConfig


class Dinov3VitMLP(eqx.Module):
    """Standard MLP (up_proj -> act -> down_proj)."""

    up_proj: Linear
    down_proj: Linear
    hidden_act: str = eqx.field(static=True)

    def __init__(self, config: Dinov3VitConfig):
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, use_bias=config.mlp_bias)
        self.hidden_act = config.hidden_act

    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = "") -> Dinov3VitMLP:
        up_proj = self.up_proj.load_state_dict(state_dict, prefix=prefix + "up_proj.")
        down_proj = self.down_proj.load_state_dict(state_dict, prefix=prefix + "down_proj.")
        return eu.replace(self, up_proj=up_proj, down_proj=down_proj)

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

    def __init__(self, config: Dinov3VitConfig):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, use_bias=config.mlp_bias)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, use_bias=config.mlp_bias)
        self.hidden_act = config.hidden_act

    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = "") -> Dinov3VitGatedMLP:
        gate_proj = self.gate_proj.load_state_dict(state_dict, prefix=prefix + "gate_proj.")
        up_proj = self.up_proj.load_state_dict(state_dict, prefix=prefix + "up_proj.")
        down_proj = self.down_proj.load_state_dict(state_dict, prefix=prefix + "down_proj.")
        return eu.replace(self, gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)

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
