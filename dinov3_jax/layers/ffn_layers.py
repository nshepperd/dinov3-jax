from typing import Optional, Callable, List
from functools import partial

import jax
import jax.numpy as jnp
from jaxtorch import Module, Context
from jaxtorch.nn import Linear, Dropout, GELU, SiLU


class Mlp(Module):
    """Multi-layer perceptron with activation and dropout."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., Module] = GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,  # Ignored in JAX
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, bias=bias)
        self.drop = Dropout(drop) if drop > 0 else None
    
    def forward(self, cx: Context, x: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(cx, x)
        x = self.act(cx, x)
        
        if self.drop is not None:
            x = self.drop(cx, x)
        
        x = self.fc2(cx, x)
        
        if self.drop is not None:
            x = self.drop(cx, x)
        
        return x
    
    def forward_list(self, cx: Context, x_list: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Process a list of inputs (for multi-crop training)."""
        outputs = []
        for x in x_list:
            outputs.append(self.forward(cx, x))
        return outputs


class SwiGLUFFN(Module):
    """SwiGLU feed-forward network (gated linear unit with SiLU activation)."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., Module]] = None,  # Not used, kept for compatibility
        drop: float = 0.0,  # Not used in original implementation
        bias: bool = True,
        align_to: int = 8,
        device=None,  # Ignored in JAX
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # Compute aligned hidden dimension
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        
        self.w1 = Linear(in_features, swiglu_hidden_features, bias=bias)
        self.w2 = Linear(in_features, swiglu_hidden_features, bias=bias)
        self.w3 = Linear(swiglu_hidden_features, out_features, bias=bias)
    
    def forward(self, cx: Context, x: jnp.ndarray) -> jnp.ndarray:
        x1 = self.w1(cx, x)
        x2 = self.w2(cx, x)
        hidden = jax.nn.silu(x1) * x2
        return self.w3(cx, hidden)
    
    def forward_list(self, cx: Context, x_list: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Process a list of inputs (for multi-crop training)."""
        outputs = []
        for x in x_list:
            outputs.append(self.forward(cx, x))
        return outputs