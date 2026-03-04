from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import dinov3_jax.eepynox.utils as eu


class Dinov3VitLayerScale(eqx.Module):
    """Layer-wise learnable scaling parameter."""

    lambda1: Array | None  # (hidden_size,)
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        self.lambda1 = None
        self.dim = dim

    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = "") -> Dinov3VitLayerScale:
        assert state_dict[prefix + "lambda1"].shape == (self.dim,)
        lambda1 = state_dict.pop(prefix + "lambda1")
        return eu.replace(self, lambda1=lambda1)

    def __call__(self, x: Array) -> Array:
        return x * self.lambda1
