from pydantic import BaseModel, ConfigDict
from typing import Optional, Literal

class DinoV3Config(BaseModel):
    model_config = ConfigDict(frozen=True)
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    pos_embed_rope_base: float = 100.0
    pos_embed_rope_min_period: Optional[float] = None
    pos_embed_rope_max_period: Optional[float] = None
    pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate"
    pos_embed_rope_shift_coords: Optional[float] = None
    pos_embed_rope_jitter_coords: Optional[float] = None
    pos_embed_rope_rescale_coords: Optional[float] = None
    pos_embed_rope_dtype: str = "bf16"
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    drop_path_rate: float = 0.0
    layerscale_init: Optional[float] = None
    norm_layer: Literal["layernorm", "layernormbf16", "rmsnorm"] = "layernorm"
    ffn_layer: Literal["mlp", "swiglu", "swiglu32", "swiglu64", "swiglu128"] = "mlp"
    ffn_bias: bool = True
    proj_bias: bool = True
    n_storage_tokens: int = 0
    mask_k_bias: bool = False
    untie_cls_and_patch_norms: bool = False
    untie_global_and_local_cls_norm: bool = False