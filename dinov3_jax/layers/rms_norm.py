import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
from dinov3_jax.eepynox.nn.param import Param


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization."""
    weight: Param
    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5, dtype=jnp.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Param((dim,), dtype, value=jnp.ones(dim))
        self.dtype = jnp.dtype(dtype)

    def _norm(self, x: Array) -> Array:
        """Compute RMS normalization."""
        return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
    
    def __call__(self, x: Array) -> Array:
        # Normalize in float32 for stability
        x_float32 = x.astype(jnp.float32)
        output = self._norm(x_float32) * self.weight()
        return output.astype(x.dtype)

class LayerNorm(eqx.Module):
    weight: Param
    bias: Param
    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5, dtype=jnp.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Param((dim,), dtype, value=jnp.ones(dim))
        self.bias = Param((dim,), dtype, value=jnp.zeros(dim))
        self.dtype = jnp.dtype(dtype)

    def __call__(self, x: Array) -> Array:
        # Normalize in float32 for stability
        x_float32 = x.astype(jnp.float32)
        mu = jnp.mean(x_float32, axis=-1, keepdims=True)
        sigma = jnp.sqrt(jnp.mean((x_float32 - mu) ** 2, axis=-1, keepdims=True) + self.eps)
        normalized = (x_float32 - mu) / sigma
        output = normalized * self.weight() + self.bias()
        return output.astype(x.dtype)