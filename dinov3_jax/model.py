from __future__ import annotations
from dinov3_jax.utils.pjit import pjit
import jax

from dataclasses import dataclass
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import dinov3_jax.eepynox.utils as eu
from dinov3_jax.config import Dinov3VitConfig
from dinov3_jax.layers.embeddings import Dinov3VitEmbeddings
from dinov3_jax.layers.rope import Dinov3VitRopePositionEmbedding
from dinov3_jax.layers.layer import Dinov3VitLayer
from dinov3_jax.layers.rms_norm import LayerNorm


@jax.tree_util.register_dataclass
@dataclass
class Dinov3VitOutput:
    """Output of Dinov3VitModel, matching HF BaseModelOutputWithPooling."""

    last_hidden_state: Array  # (B, 1 + num_register + num_patches, hidden_size)
    pooler_output: Array  # (B, hidden_size)


class Dinov3VitModel(eqx.Module):
    """DINOv3 Vision Transformer matching the HuggingFace structure."""

    embeddings: Dinov3VitEmbeddings
    rope_embeddings: Dinov3VitRopePositionEmbedding
    layer: list[Dinov3VitLayer]
    norm: LayerNorm
    config: Dinov3VitConfig = eqx.field(static=True)

    def __init__(self, config: Dinov3VitConfig, use_flash_attn: bool = True):
        self.config = config
        self.embeddings = Dinov3VitEmbeddings(config)
        self.rope_embeddings = Dinov3VitRopePositionEmbedding(config)
        self.layer = [Dinov3VitLayer(config, use_flash_attn=use_flash_attn) for _ in range(config.num_hidden_layers)]
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def load_state_dict(self, state_dict: dict[str, Array], prefix: str = "") -> Dinov3VitModel:
        embeddings = self.embeddings.load_state_dict(state_dict, prefix=prefix + "embeddings.")
        layers = []
        for i, layer_module in enumerate(self.layer):
            layers.append(layer_module.load_state_dict(state_dict, prefix=prefix + f"layer.{i}."))
        norm = self.norm.load_state_dict(state_dict, prefix=prefix + "norm.")
        return eu.replace(self, embeddings=embeddings, layer=layers, norm=norm)

    @pjit
    def __call__(self, pixel_values: Array) -> Dinov3VitOutput:
        hidden_states = self.embeddings(pixel_values)
        position_embeddings = self.rope_embeddings(pixel_values)

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)

        sequence_output = self.norm(hidden_states)
        pooler_output = sequence_output[:, 0, :]

        return Dinov3VitOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooler_output,
        )

    @pjit(static_argnames=["n", "reshape", "return_class_token", "norm"])
    def get_intermediate_layers(
        self,
        pixel_values: Array,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple[Array, ...] | tuple[tuple[Array, Array], ...]:
        """Get intermediate layer outputs.

        Args:
            pixel_values: (B, C, H, W) input.
            n: Number of last layers to return, or list of layer indices.
            reshape: If True, reshape patch tokens to (B, C, H', W').
            return_class_token: If True, also return CLS token per layer.
            norm: If True, apply final LayerNorm to each output.
        """
        hidden_states = self.embeddings(pixel_values)
        position_embeddings = self.rope_embeddings(pixel_values)

        total_layers = len(self.layer)
        if isinstance(n, int):
            layers_to_take = set(range(total_layers - n, total_layers))
        else:
            layers_to_take = set(n)

        outputs = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
            if i in layers_to_take:
                outputs.append(hidden_states)

        if norm:
            outputs = [self.norm(out) for out in outputs]

        num_prefix = 1 + self.config.num_register_tokens
        class_tokens = [out[:, 0] for out in outputs]
        patch_tokens = [out[:, num_prefix:] for out in outputs]

        if reshape:
            _, _, H, W = pixel_values.shape
            h = H // self.config.patch_size
            w = W // self.config.patch_size
            patch_tokens = [
                pt.reshape(pt.shape[0], h, w, -1).transpose(0, 3, 1, 2)
                for pt in patch_tokens
            ]

        if return_class_token:
            return tuple(zip(patch_tokens, class_tokens))
        return tuple(patch_tokens)
