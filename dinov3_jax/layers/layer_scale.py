import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import eepynox.utils as eu


class LayerScale(eqx.Module):
    """Layer-wise learnable scaling parameter."""
    gamma: Array
    dim: int = eqx.field(static=True)
    init_values: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dim = dim
        self.init_values = init_values
        self.gamma= jnp.full((dim,), init_values, dtype=dtype)
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        """Load state dict into the LayerScale module."""
        assert state_dict[prefix + 'gamma'].shape == (self.dim,)
        gamma = state_dict.pop(prefix + 'gamma')
        return eu.replace(self, gamma=gamma)

    def __call__(self, x: Array) -> Array:
        return x * self.gamma