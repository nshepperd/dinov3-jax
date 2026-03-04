from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from dinov3_jax.config import Dinov3VitConfig
from dinov3_jax.model import Dinov3VitModel


def load_dinov3(
    model_path: str | Path,
    dtype: jnp.dtype = jnp.float32,
    use_flash_attn: bool = True,
) -> Dinov3VitModel:
    """Load a DINOv3 model from a HuggingFace model directory.

    Args:
        model_path: Path to HF model directory (containing config.json + model.safetensors).
        dtype: Target dtype for model parameters.
        use_flash_attn: Whether to use flash attention (True) or eager attention (False).

    Returns:
        Dinov3VitModel with loaded weights.
    """
    model_path = Path(model_path)

    # 1. Load config
    config = Dinov3VitConfig.from_pretrained(model_path)

    # 2. Construct model (weights initialized to None)
    model = Dinov3VitModel(config, use_flash_attn=use_flash_attn)

    # 3. Load safetensors weights
    weights_path = model_path / "model.safetensors"
    np_params = load_file(str(weights_path))

    # 4. Convert numpy arrays to JAX arrays with target dtype
    jax_params: dict[str, jnp.ndarray] = {}
    for key, value in np_params.items():
        jax_params[key] = jnp.array(value, dtype=dtype)

    # 5. Load into model via load_state_dict
    model = model.load_state_dict(jax_params)

    return model
