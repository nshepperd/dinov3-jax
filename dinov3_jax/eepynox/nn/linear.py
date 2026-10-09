import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

import dinov3_jax.eepynox.utils as eu
from dinov3_jax.eepynox.nn.param import Param


class Linear(eqx.Module):
    weight: Param
    bias: Param | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dtype=jnp.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Param((out_features, in_features), dtype)
        self.bias = Param((out_features,), dtype) if use_bias else None
        self.dtype = jnp.dtype(dtype)
        self.use_bias = use_bias

    def init_weights(self, key: PRNGKeyArray):
        A = 1.0 / math.sqrt(self.in_features)
        weight = self.weight.with_value(
            jax.random.normal(key, self.weight.shape, dtype=self.dtype) * A
        )
        if self.bias is not None:
            bias = self.bias.with_value(jnp.zeros(self.bias.shape, dtype=self.dtype))
        else:
            bias = None
        return eu.replace(self, weight=weight, bias=bias)

    def __call__(
        self, x: Float[Array, "... in_features"]
    ) -> Float[Array, "... out_features"]:
        y = jnp.dot(x, jnp.transpose(self.weight()))
        if self.bias is not None:
            y = y + self.bias()
        return y
