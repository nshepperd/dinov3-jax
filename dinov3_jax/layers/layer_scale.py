from typing import Union

import jax.numpy as jnp
from jaxtorch import Module, Context, init


class LayerScale(Module):
    """Layer-wise learnable scaling parameter."""
    
    def __init__(
        self,
        dim: int,
        init_values: Union[float, jnp.ndarray] = 1e-5,
        inplace: bool = False,  # Ignored in JAX (no in-place operations)
        device=None,  # Ignored in JAX
    ):
        super().__init__()
        self.dim = dim
        self.init_values = init_values
        
        # Initialize gamma parameter
        if isinstance(init_values, float):
            self.gamma = init.const(jnp.full((dim,), init_values))
        else:
            self.gamma = init.const(init_values)
    
    def forward(self, cx: Context, x: jnp.ndarray) -> jnp.ndarray:
        return x * cx[self.gamma]