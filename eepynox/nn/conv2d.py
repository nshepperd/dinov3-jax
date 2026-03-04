import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray
import equinox as eqx
import eepynox.utils as eu
import math


class Conv2d(eqx.Module):
    weight: Float[Array, "out_features in_features kh kw"] | None
    bias: Float[Array, "out_features"] | None
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
        self.weight = None
        self.bias = None
        self.dtype = jnp.dtype(dtype)
        self.use_bias = use_bias

    def init_weights(self, key: PRNGKeyArray):
        A = math.sqrt(
            self.groups / (self.in_features * self.kernel_size * self.kernel_size)
        )
        weight = jax.random.uniform(
            key,
            (
                self.out_features,
                self.in_features // self.groups,
                self.kernel_size,
                self.kernel_size,
            ),
            minval=-A,
            maxval=A,
            dtype=self.dtype,
        )
        if self.use_bias:
            bias = jax.random.uniform(
                key, (self.out_features,), minval=-A, maxval=A, dtype=self.dtype
            )
        else:
            bias = None
        return eu.replace(self, weight=weight, bias=bias)

    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        assert state_dict[prefix + "weight"].shape == (
            self.out_features,
            self.in_features // self.groups,
            self.kernel_size,
            self.kernel_size,
        )
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

    def __call__(
        self, x: Float[Array, "... in_features h w"]
    ) -> Float[Array, "... out_features h_out w_out"]:
        assert self.weight is not None
        y = jax.lax.conv_general_dilated(
            x.astype(self.dtype),
            self.weight,
            window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
            rhs_dilation=(self.dilation, self.dilation),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=self.groups,
        )
        if self.use_bias:
            assert self.bias is not None
            y = y + self.bias[:,None, None]
        return y.astype(x.dtype)
