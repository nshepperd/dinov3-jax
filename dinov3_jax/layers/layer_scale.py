from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from dinov3_jax.eepynox.nn.param import Param


class Dinov3VitLayerScale(eqx.Module):
    """Layer-wise learnable scaling parameter."""

    lambda1: Param  # (hidden_size,)
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        self.lambda1 = Param((dim,))
        self.dim = dim

    def __call__(self, x: Array) -> Array:
        return x * self.lambda1()
