import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import eepynox.utils as eu
from eepynox.nn.conv2d import Conv2d


class PatchEmbed(eqx.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D) or (B,H,W,D)
    
    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        flatten_embedding: Whether to flatten spatial dimensions.
    """
    proj: Conv2d
    img_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    patches_resolution: tuple[int, int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    in_chans: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    flatten_embedding: bool = eqx.field(static=True)

    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten_embedding: bool = True,
    ):
        super().__init__()
        
        patch_grid_size = (
            img_size // patch_size,
            img_size // patch_size,
        )
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        
        # Conv2d layer for patch projection
        # Convert tuple to integers if needed for stride
        stride_val = patch_size
        self.proj = Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size,
            stride=stride_val,
            padding=0,
            use_bias=True
        )
    
    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = ""):
        proj = self.proj.load_state_dict(state_dict, prefix + "proj.")
        return eu.replace(self, proj=proj)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, C, H, W = x.shape
        
        # Apply convolution to extract patches
        x = self.proj(x)  # B, embed_dim, H', W'
        _, _, H_out, W_out = x.shape
        
        # Reshape: (B, embed_dim, H', W') -> (B, H'*W', embed_dim)
        x = x.reshape(B, self.embed_dim, -1)  # B, embed_dim, H'*W'
        x = x.transpose(0, 2, 1)  # B, H'*W', embed_dim
        
        # Optionally reshape to spatial format
        if not self.flatten_embedding:
            x = x.reshape(B, H_out, W_out, self.embed_dim)  # B, H, W, embed_dim
        
        return x