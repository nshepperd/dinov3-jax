import math
from typing import Literal, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class RopePositionEmbedding(eqx.Module):
    """
    RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights.
    Supports two parametrizations: either using `base` or `min_period` and `max_period`.
    """
    periods: Float[Array, "D_head//4"]

    base: Optional[float] = eqx.field(static=True)
    min_period: Optional[float] = eqx.field(static=True)
    max_period: Optional[float] = eqx.field(static=True)
    D_head: int = eqx.field(static=True)
    normalize_coords: Literal["min", "max", "separate"] = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        dtype: Optional[jnp.dtype] = None,
        device: Optional[str] = None,  # Ignored in JAX
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")
        
        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.dtype = dtype or jnp.float32
        
        if self.base is not None:
            periods = self.base ** (
                2 * jnp.arange(self.D_head // 4, dtype=self.dtype) / (self.D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = jnp.linspace(0, 1, self.D_head // 4, dtype=self.dtype)
            periods = base ** exponents
            periods = periods / base
            periods = periods * self.max_period
        
        self.periods = periods
    
    def __call__(self, H: int, W: int) -> tuple[Array, Array]:
        dtype = self.dtype
        periods = self.periods
        
        # Prepare coords in range [0, 1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = jnp.arange(0.5, H, dtype=dtype) / max_HW
            coords_w = jnp.arange(0.5, W, dtype=dtype) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = jnp.arange(0.5, H, dtype=dtype) / min_HW
            coords_w = jnp.arange(0.5, W, dtype=dtype) / min_HW
        elif self.normalize_coords == "separate":
            coords_h = jnp.arange(0.5, H, dtype=dtype) / H
            coords_w = jnp.arange(0.5, W, dtype=dtype) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        
        # Create grid and flatten
        coords_h_grid, coords_w_grid = jnp.meshgrid(coords_h, coords_w, indexing="ij")
        coords = jnp.stack([coords_h_grid, coords_w_grid], axis=-1)  # [H, W, 2]
        coords = coords.reshape(-1, 2)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.reshape(-1, self.D_head // 2)  # [HW, D//2]
        angles = jnp.tile(angles, 2)  # [HW, D]
        cos = jnp.cos(angles)  # [HW, D]
        sin = jnp.sin(angles)  # [HW, D]
        
        return (sin, cos)  # 2 * [HW, D]