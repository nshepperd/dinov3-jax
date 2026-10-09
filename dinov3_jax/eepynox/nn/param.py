from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import dinov3_jax.eepynox.utils as eu


class Param(eqx.Module):
    """A single weight array, loaded from the state dict key given by its module path.

    A Param at `model.layer[0].attention.q_proj.weight` loads from
    `"layer.0.attention.q_proj.weight"`. Call it to get the array.
    """

    value: Array | None
    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: jnp.dtype | None = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], dtype=None, value: Array | None = None):
        self.shape = tuple(shape)
        self.dtype = None if dtype is None else jnp.dtype(dtype)
        self.value = None if value is None else self._check(value)

    def _check(self, value: Array) -> Array:
        if value.shape != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {value.shape}")
        if self.dtype is not None:
            value = value.astype(self.dtype)
        return value

    def with_value(self, value: Array) -> Param:
        return eu.replace(self, value=self._check(value))

    def load_state_dict(self, state_dict: Mapping[str, Array], path: str) -> Param:
        try:
            return self.with_value(state_dict[path])
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None

    def __call__(self) -> Array:
        assert self.value is not None, "Param not loaded"
        return self.value


def _is_param(x) -> bool:
    return isinstance(x, Param)


def _path_str(path: jax.tree_util.KeyPath) -> str:
    return jax.tree_util.keystr(path, simple=True, separator=".")


def load_state_dict[T](
    module: T, state_dict: Mapping[str, Array], strict: bool = True
) -> T:
    """Returns `module` with every Param loaded from `state_dict` by its module path.

    Missing keys raise KeyError. With `strict`, keys in `state_dict` that no
    Param consumed raise ValueError.
    """
    used = set()

    def load(path, x):
        if not isinstance(x, Param):
            return x
        key = _path_str(path)
        used.add(key)
        return x.load_state_dict(state_dict, key)

    module = jax.tree_util.tree_map_with_path(load, module, is_leaf=_is_param)
    if strict and (unexpected := state_dict.keys() - used):
        raise ValueError(f"Unexpected keys in state dict: {sorted(unexpected)}")
    return module


def state_dict(module) -> dict[str, Array]:
    """Inverse of `load_state_dict`: maps each Param's module path to its array."""
    leaves = jax.tree_util.tree_leaves_with_path(module, is_leaf=_is_param)
    return {_path_str(path): x() for path, x in leaves if isinstance(x, Param)}
