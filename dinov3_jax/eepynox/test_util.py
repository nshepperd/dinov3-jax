from jax import Array
import dataclasses
from functools import partial
from typing import Callable, TypeVar, Any

import torch
import equinox as eqx
import dinov3_jax.eepynox.utils as eu
import jax

def collect_layers(model: torch.nn.Module, *args, **kwargs) -> tuple[Any, dict[str, tuple[tuple[Any,...], Any]]]:
    """Calls a pytorch module with given args and kwargs,
    and collects the inputs and outputs of all submodules using hooks.
    """
    collected: dict[str, tuple[tuple[Any,...], Any]] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            collected[name] = (inputs, output)
        return hook

    for name, module in model.named_modules():
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)

    try:
        output = model(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    return output, collected

class WrapModule(eqx.Module):
    name: str = eqx.field(static=True)
    wrapped: eqx.Module
    out: dict[str, tuple[tuple[Any,...], Any]] = eqx.field(static=True)

    def __init__(self, name: str, wrapped: eqx.Module, out: dict[str, tuple[tuple[Any,...], Any]]):
        self.name = name
        self.wrapped = wrapped
        self.out = out
    
    def __call__(self, *args, **kwargs):
        out = self.wrapped(*args, **kwargs)
        self.out[self.name] = (args, out)
        # if isinstance(out, Array):
        #     out = label(self.name, out)
        return out

def collect_layers_eqx(
    model: eqx.Module,
    *args,
    **kwargs
) -> tuple[Array, dict[str, tuple[tuple[Any,...], Any]]]:
    """Calls an equinox module with given args and kwargs,
    and collects the inputs and outputs of all submodules.
    """
    collected: dict[str, tuple[tuple[Any,...], Any]] = {}

    def wrap_module(path: jax.tree_util.KeyPath, module: eqx.Module) -> eqx.Module:
        name = jax.tree_util.keystr(path, separator='.', simple=True)
        return WrapModule(name, module, collected)

    wrapped_model = eu.mapmod_with_path(wrap_module, model)
    output = wrapped_model(*args, **kwargs)
    return output, collected