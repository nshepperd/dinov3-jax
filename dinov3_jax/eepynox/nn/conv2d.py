import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

import dinov3_jax.eepynox.utils as eu
from dinov3_jax.eepynox.nn.param import Param


class Conv2d(eqx.Module):
    weight: Param
    bias: Param | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    padding: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        dtype=jnp.float32,
    ):
        super().__init__()
        assert in_features % groups == 0
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Param(
            (out_features, in_features // groups, kernel_size, kernel_size), dtype
        )
        self.bias = Param((out_features,), dtype) if use_bias else None
        self.dtype = jnp.dtype(dtype)
        self.use_bias = use_bias

    def init_weights(self, key: PRNGKeyArray):
        A = math.sqrt(
            self.groups / (self.in_features * self.kernel_size * self.kernel_size)
        )
        weight = self.weight.with_value(
            jax.random.uniform(
                key, self.weight.shape, minval=-A, maxval=A, dtype=self.dtype
            )
        )
        if self.bias is not None:
            bias = self.bias.with_value(
                jax.random.uniform(
                    key, self.bias.shape, minval=-A, maxval=A, dtype=self.dtype
                )
            )
        else:
            bias = None
        return eu.replace(self, weight=weight, bias=bias)

    def __call__(
        self, x: Float[Array, "... in_features h w"]
    ) -> Float[Array, "... out_features h_out w_out"]:
        y = jax.lax.conv_general_dilated(
            x.astype(self.dtype),
            self.weight(),
            window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
            rhs_dilation=(self.dilation, self.dilation),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias()[:, None, None]
        return y.astype(x.dtype)
