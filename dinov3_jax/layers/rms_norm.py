import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import eepynox.utils as eu


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization."""
    weight: Float[Array, "c"]
    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5, dtype=jnp.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = jnp.ones(dim, dtype=dtype)
        self.dtype = jnp.dtype(dtype)
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        assert state_dict[prefix + "weight"].shape == (self.dim,)
        return eu.replace(self, weight=state_dict.pop(prefix + "weight").astype(self.dtype))

    def _norm(self, x: Array) -> Array:
        """Compute RMS normalization."""
        return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
    
    def __call__(self, x: Array) -> Array:
        # Normalize in float32 for stability
        x_float32 = x.astype(jnp.float32)
        output = self._norm(x_float32) * self.weight
        return output.astype(x.dtype)

class LayerNorm(eqx.Module):
    weight: Float[Array, "c"]
    bias: Float[Array, "c"]
    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5, dtype=jnp.float32):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = jnp.ones(dim, dtype=dtype)
        self.bias = jnp.zeros(dim, dtype=dtype)
        self.dtype = jnp.dtype(dtype)
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        assert state_dict[prefix + "weight"].shape == (self.dim,)
        assert state_dict[prefix + "bias"].shape == (self.dim,)
        weight = state_dict.pop(prefix + "weight").astype(self.dtype)
        bias = state_dict.pop(prefix + "bias").astype(self.dtype)
        return eu.replace(self, weight=weight, bias=bias)
    
    def state_dict(self, prefix: str = "") -> dict[str, Array]:
        return {
            prefix + "weight": self.weight,
            prefix + "bias": self.bias,
        }
    
    def __call__(self, x: Array) -> Array:
        # Normalize in float32 for stability
        x_float32 = x.astype(jnp.float32)
        mu = jnp.mean(x_float32, axis=-1, keepdims=True)
        sigma = jnp.sqrt(jnp.mean((x_float32 - mu) ** 2, axis=-1, keepdims=True) + self.eps)
        normalized = (x_float32 - mu) / sigma
        output = normalized * self.weight + self.bias
        return output.astype(x.dtype)