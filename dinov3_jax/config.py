from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Dinov3VitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    rope_theta: float = 100.0
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    layerscale_value: float = 1.0
    use_gated_mlp: bool = False
    num_register_tokens: int = 0

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> Dinov3VitConfig:
        """Load config from a HuggingFace model directory containing config.json."""
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            data = json.load(f)
        # Filter to only fields this config knows about
        valid_fields = cls.model_fields.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
