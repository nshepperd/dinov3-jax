import math
from typing import Union, Tuple

import jax
import jax.numpy as jnp
from jaxtorch import Module, Context, init
from jaxtorch.nn import Conv2d, LayerNorm


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D) or (B,H,W,D)
    
    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer class.
        flatten_embedding: Whether to flatten spatial dimensions.
    """
    
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type = None,
        flatten_embedding: bool = True,
    ):
        super().__init__()
        
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )
        
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        
        # Conv2d layer for patch projection
        # Convert tuple to integers if needed for stride
        stride_val = patch_HW[0] if isinstance(patch_HW, tuple) else patch_HW
        self.proj = Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_HW, 
            stride=stride_val,
            padding=0,
            bias=True
        )
        
        # Optional normalization
        self.use_norm = norm_layer is not None
        if self.use_norm:
            self.norm = norm_layer(embed_dim)
    
    def setup(self, cx: Context):
        super().setup(cx)
        # Custom initialization for patch embedding
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        cx[self.proj.weight] = jax.random.uniform(
            cx.rng.split(), 
            cx[self.proj.weight].shape,
            minval=-math.sqrt(k),
            maxval=math.sqrt(k)
        )
        if self.proj.bias is not None:
            cx[self.proj.bias] = jax.random.uniform(
                cx.rng.split(),
                cx[self.proj.bias].shape,
                minval=-math.sqrt(k),
                maxval=math.sqrt(k)
            )
    
    def forward(self, cx: Context, x: jnp.ndarray) -> jnp.ndarray:
        B, C, H, W = x.shape
        
        # Apply convolution to extract patches
        x = self.proj(cx, x)  # B, embed_dim, H', W'
        _, _, H_out, W_out = x.shape
        
        # Reshape: (B, embed_dim, H', W') -> (B, H'*W', embed_dim)
        x = x.reshape(B, self.embed_dim, -1)  # B, embed_dim, H'*W'
        x = x.transpose(0, 2, 1)  # B, H'*W', embed_dim
        
        # Apply normalization if specified
        if self.use_norm:
            x = self.norm(cx, x)
        
        # Optionally reshape to spatial format
        if not self.flatten_embedding:
            x = x.reshape(B, H_out, W_out, self.embed_dim)  # B, H, W, embed_dim
        
        return x