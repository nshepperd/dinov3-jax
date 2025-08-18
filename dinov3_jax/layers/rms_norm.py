import jax
import jax.numpy as jnp
from jaxtorch import Module, Context, init


class RMSNorm(Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # Initialize weight parameter
        self.weight = init.ones(dim)
    
    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute RMS normalization."""
        return x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
    
    def forward(self, cx: Context, x: jnp.ndarray) -> jnp.ndarray:
        # Normalize in float32 for stability
        x_float32 = x.astype(jnp.float32)
        output = self._norm(x_float32)
        output = output.astype(x.dtype)
        return output * cx[self.weight]