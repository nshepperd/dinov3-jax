import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray
import equinox as eqx
import dinov3_jax.eepynox.utils as eu
import math

class Linear(eqx.Module):
    weight: Float[Array, "out_features in_features"] | None
    bias: Float[Array, "out_features"] | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, dtype=jnp.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None
        self.dtype = jnp.dtype(dtype)
        self.use_bias = use_bias

    def init_weights(self, key: PRNGKeyArray):
        A = 1.0/math.sqrt(self.in_features)
        weight = jax.random.normal(key, (self.out_features, self.in_features), dtype=self.dtype) * A
        if self.use_bias:
            bias = jnp.zeros((self.out_features,), dtype=self.dtype)
        else:
            bias = None
        return eu.replace(self, weight=weight, bias=bias)
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        assert state_dict[prefix + "weight"].shape == (self.out_features, self.in_features)
        weight = state_dict.pop(prefix + "weight").astype(self.dtype)
        if self.use_bias:
            assert state_dict[prefix + "bias"].shape == (self.out_features,)
            bias = state_dict.pop(prefix + "bias").astype(self.dtype)
        else:
            bias = None
        return eu.replace(self, weight=weight, bias=bias)
    
    def state_dict(self, prefix: str = "") -> dict[str, Array]:
        assert self.weight is not None
        sd = {prefix + "weight": self.weight}
        if self.use_bias:
            assert self.bias is not None
            sd[prefix + "bias"] = self.bias
        return sd

    def __call__(self, x: Float[Array, "... in_features"]) -> Float[Array, "... out_features"]:
        assert self.weight is not None
        y = jnp.dot(x, jnp.transpose(self.weight))
        if self.use_bias:
            assert self.bias is not None
            y = y + self.bias
        return y