from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import eepynox.utils as eu
from eepynox.nn.activation import GELU
from eepynox.nn.linear import Linear


class Mlp(eqx.Module):
    """Multi-layer perceptron with activation and dropout."""
    fc1: Linear
    act: GELU
    fc2: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = Linear(in_features, hidden_features, use_bias=bias)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features, use_bias=bias)
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        fc1 = self.fc1.load_state_dict(state_dict, prefix=prefix + "fc1.")
        fc2 = self.fc2.load_state_dict(state_dict, prefix=prefix + "fc2.")
        return eu.replace(self, fc1=fc1, fc2=fc2)

    def __call__(self, x: Array) -> Array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SwiGLUFFN(eqx.Module):
    """SwiGLU feed-forward network (gated linear unit with SiLU activation)."""
    w1: Linear
    w2: Linear
    w3: Linear


    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        align_to: int = 8,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # Compute aligned hidden dimension
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        
        self.w1 = Linear(in_features, swiglu_hidden_features, use_bias=bias)
        self.w2 = Linear(in_features, swiglu_hidden_features, use_bias=bias)
        self.w3 = Linear(swiglu_hidden_features, out_features, use_bias=bias)
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        w1 = self.w1.load_state_dict(state_dict, prefix=prefix + "w1.")
        w2 = self.w2.load_state_dict(state_dict, prefix=prefix + "w2.")
        w3 = self.w3.load_state_dict(state_dict, prefix=prefix + "w3.")
        return eu.replace(self, w1=w1, w2=w2, w3=w3)

    def __call__(self, x: Array) -> Array:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = jax.nn.silu(x1) * x2
        return self.w3(hidden)
