import math
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtorch import Module, Context, Param


class RopePositionEmbedding(Module):
    """
    RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights.
    Supports two parametrizations: either using `base` or `min_period` and `max_period`.
    """
    
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
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
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype or jnp.float32
        
        # Create periods parameter (will be initialized in setup)
        self.periods = Param((D_head // 4,))
    
    def setup(self, cx: Context):
        super().setup(cx)
        self._init_weights(cx)
    
    def _init_weights(self, cx: Context):
        D_head = self.D_head
        if self.base is not None:
            periods = self.base ** (
                2 * jnp.arange(D_head // 4, dtype=self.dtype) / (D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = jnp.linspace(0, 1, D_head // 4, dtype=self.dtype)
            periods = base ** exponents
            periods = periods / base
            periods = periods * self.max_period
        
        cx[self.periods] = periods
    
    def forward(self, cx: Context, *, H: int, W: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        dtype = self.dtype
        periods = cx[self.periods]
        
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
        
        # Apply coordinate transformations during training
        if cx.mode == "train":
            # Shift coords
            if self.shift_coords is not None:
                shift_hw = cx.random.uniform(
                    shape=(2,),
                    minval=-self.shift_coords,
                    maxval=self.shift_coords
                ).astype(dtype)
                coords = coords + shift_hw[None, :]
            
            # Jitter coords
            if self.jitter_coords is not None:
                jitter_max = np.log(self.jitter_coords)
                jitter_min = -jitter_max
                jitter_hw = jnp.exp(
                    cx.random.uniform(shape=(2,), minval=jitter_min, maxval=jitter_max)
                ).astype(dtype)
                coords = coords * jitter_hw[None, :]
            
            # Rescale coords
            if self.rescale_coords is not None:
                rescale_max = np.log(self.rescale_coords)
                rescale_min = -rescale_max
                rescale_hw = jnp.exp(
                    cx.random.uniform(shape=(1,), minval=rescale_min, maxval=rescale_max)
                ).astype(dtype)
                coords = coords * rescale_hw
        
        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.reshape(-1, self.D_head // 2)  # [HW, D//2]
        angles = jnp.tile(angles, 2)  # [HW, D]
        cos = jnp.cos(angles)  # [HW, D]
        sin = jnp.sin(angles)  # [HW, D]
        
        return (sin, cos)  # 2 * [HW, D]