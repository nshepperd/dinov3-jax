# dinov3-jax Handoff Document

## What This Is

JAX/Equinox port of DINOv3 (self-supervised vision transformer), structured to follow the HuggingFace `transformers` implementation. Inference-only — `load_dinov3(model_path)` reads an HF model directory (`config.json` + `model.safetensors`), builds `Dinov3VitModel`, and loads weights via `load_state_dict(model, state_dict)`, which fills every `Param` from the state dict key matching its module path (e.g. `layer.0.attention.q_proj.weight`). Interactive visualizer (`vis_dpg.py`) lets you click patches to see cosine similarity heatmaps; images can be pasted with Ctrl+V.

## What Still Needs Work

### API Design

**`get_intermediate_layers` return type varies by a boolean flag.** Returns `tuple[Array, ...]` or `tuple[tuple[Array, Array], ...]` depending on `return_class_token`. A named return type would be clearer.

### Usability

**`vis_dpg.py` hardcodes the model path.** `MODEL_PATH` must be edited in source. Needs `argparse`.

**`fa4-jax` is a required dependency with no automatic fallback.** Flash attention comes from `fa4-jax` (git dep; needs Python ≥3.12 and the `jax-tvm-ffi==0.1.2` pin for jax 0.8.x, see `[tool.uv]` in `pyproject.toml`). `use_flash_attn=False` gives an eager path and the import is lazy, but the default `use_flash_attn=True` doesn't fall back if the import or kernel launch fails. (On CPU, `fa4_jax` uses its pure-jnp reference backend on its own.)

**No weight download story.** Users must already have an HF model directory locally. Nothing downloads it (no `hf_hub`) and there's no documentation on where to get weights.

**README is empty.** No install instructions, no quickstart, no API docs.

## File Map

```
dinov3_jax/
  config.py              - Dinov3VitConfig (pydantic, from HF config.json)
  model.py               - Dinov3VitModel, Dinov3VitOutput, get_intermediate_layers
  loading.py             - load_dinov3: HF directory -> loaded model
  layers/
    embeddings.py        - Conv2d patch embedding + CLS/register tokens
    attention.py         - Dinov3VitAttention (RoPE; flash or eager)
    layer.py             - Dinov3VitLayer (norm->attn->scale->norm->mlp->scale)
    mlp.py               - Dinov3VitMLP and Dinov3VitGatedMLP
    rms_norm.py          - RMSNorm and LayerNorm
    rope.py              - RoPE position embedding (no learnable params)
    layer_scale.py       - Learnable per-layer scaling
  utils/
    pjit.py              - pjit decorator
  eepynox/               - Custom Equinox utilities
    utils.py             - replace(), new(), mapmod()
    nn/param.py          - Param (one weight array), load_state_dict(), state_dict()
    nn/linear.py         - Linear layer
    nn/conv2d.py         - Conv2d layer
    nn/activation.py     - Activations
    test_util.py         - Layer-output collection for torch/eqx comparison
    label_collect.py     - label() primitive for tagging intermediates
    debug.py             - debugpy post-mortem helpers

vis_dpg.py               - Interactive DearPyGui feature similarity viewer
tests/
  test_hf_equivalence.py - Equivalence tests against HF transformers
```
