"""Tests comparing JAX DINOv3 model against HuggingFace PyTorch model with real weights."""

import numpy as np
import jax.numpy as jnp
import torch
import pytest

from dinov3_jax import load_dinov3

VITB16_PATH = "/data/models/dinov3-vitb16-pretrain-lvd1689m"


def _load_hf_model(model_path: str):
    """Load HF PyTorch model."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def models():
    """Load both JAX and HF models once for the module."""
    jax_model = load_dinov3(VITB16_PATH, dtype=jnp.float32, use_flash_attn=False)
    hf_model = _load_hf_model(VITB16_PATH)
    return jax_model, hf_model


@pytest.fixture
def sample_input():
    """Generate deterministic sample input."""
    rng = np.random.RandomState(42)
    return rng.randn(1, 3, 224, 224).astype(np.float32)


class TestEmbeddings:
    def test_patch_embeddings(self, models, sample_input):
        jax_model, hf_model = models

        x_jax = jnp.array(sample_input)
        x_pt = torch.from_numpy(sample_input)

        # JAX embeddings
        jax_emb = jax_model.embeddings(x_jax)

        # HF embeddings
        with torch.no_grad():
            hf_emb = hf_model.embeddings(x_pt)

        np.testing.assert_allclose(
            np.array(jax_emb), hf_emb.numpy(), atol=1e-5,
            err_msg="Embeddings output mismatch",
        )


class TestRoPE:
    def test_rope_embeddings(self, models, sample_input):
        jax_model, hf_model = models

        x_jax = jnp.array(sample_input)
        x_pt = torch.from_numpy(sample_input)

        cos_jax, sin_jax = jax_model.rope_embeddings(x_jax)

        with torch.no_grad():
            cos_pt, sin_pt = hf_model.rope_embeddings(x_pt)

        np.testing.assert_allclose(
            np.array(cos_jax), cos_pt.numpy(), atol=1e-6,
            err_msg="RoPE cos mismatch",
        )
        np.testing.assert_allclose(
            np.array(sin_jax), sin_pt.numpy(), atol=1e-6,
            err_msg="RoPE sin mismatch",
        )


class TestSingleLayer:
    def test_first_layer(self, models, sample_input):
        jax_model, hf_model = models

        x_jax = jnp.array(sample_input)
        x_pt = torch.from_numpy(sample_input)

        # Get embeddings
        jax_emb = jax_model.embeddings(x_jax)
        cos_jax, sin_jax = jax_model.rope_embeddings(x_jax)

        with torch.no_grad():
            hf_emb = hf_model.embeddings(x_pt)
            cos_pt, sin_pt = hf_model.rope_embeddings(x_pt)

        # Run first layer
        jax_out = jax_model.layer[0](jax_emb, position_embeddings=(cos_jax, sin_jax))

        with torch.no_grad():
            hf_out = hf_model.layer[0](hf_emb, position_embeddings=(cos_pt, sin_pt))

        np.testing.assert_allclose(
            np.array(jax_out), hf_out.numpy(), atol=1e-3,
            err_msg="First layer output mismatch",
        )


class TestFullModel:
    def test_forward(self, models, sample_input):
        """Full forward pass equivalence."""
        jax_model, hf_model = models

        x_jax = jnp.array(sample_input)
        x_pt = torch.from_numpy(sample_input)

        jax_out = jax_model(x_jax)

        with torch.no_grad():
            hf_out = hf_model(x_pt)

        np.testing.assert_allclose(
            np.array(jax_out.last_hidden_state),
            hf_out.last_hidden_state.numpy(),
            atol=1e-4,
            err_msg="last_hidden_state mismatch",
        )
        np.testing.assert_allclose(
            np.array(jax_out.pooler_output),
            hf_out.pooler_output.numpy(),
            atol=1e-4,
            err_msg="pooler_output mismatch",
        )

    def test_output_shapes(self, models, sample_input):
        jax_model, hf_model = models

        jax_out = jax_model(jnp.array(sample_input))

        # B=1, seq_len = 1 (CLS) + 4 (register) + 196 (patches) = 201
        assert jax_out.last_hidden_state.shape == (1, 201, 768)
        assert jax_out.pooler_output.shape == (1, 768)

    def test_different_resolution(self, models):
        """Test with non-default resolution to verify dynamic RoPE."""
        jax_model, hf_model = models

        rng = np.random.RandomState(123)
        x_np = rng.randn(1, 3, 448, 448).astype(np.float32)

        jax_out = jax_model(jnp.array(x_np))

        with torch.no_grad():
            hf_out = hf_model(torch.from_numpy(x_np))

        np.testing.assert_allclose(
            np.array(jax_out.last_hidden_state),
            hf_out.last_hidden_state.numpy(),
            atol=1e-4,
            err_msg="Different resolution last_hidden_state mismatch",
        )
