import equinox as eqx
import jax
from jaxtyping import Array


class GELU(eqx.Module):
    """Gaussian Error Linear Unit (GELU) activation function."""
    def __call__(self, x: Array) -> Array:
        return jax.nn.gelu(x, approximate=False)

class Identity(eqx.Module):
    """Identity activation function."""
    def __call__(self, x: Array) -> Array:
        return x