import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
import jax

import dinov3_jax.eepynox.utils as eu

class GELU(eqx.Module):
    """Gaussian Error Linear Unit (GELU) activation function."""
    def __call__(self, x: Array) -> Array:
        return jax.nn.gelu(x, approximate=False)

class Identity(eqx.Module):
    """Identity activation function."""
    def __call__(self, x: Array) -> Array:
        return x