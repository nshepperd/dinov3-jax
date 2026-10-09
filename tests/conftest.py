import os

import jax
import lovely_jax as lj
import lovely_tensors as lt
import pytest
import torch

from dinov3_jax.eepynox.debug import maybe_debugpy_postmortem

jax.config.update("jax_default_matmul_precision", "highest")
# Persist compiled executables (and with them XLA's GPU autotuning results)
# across runs; first compiles of the eager model otherwise take ~15-25s each.
if not jax.config.jax_compilation_cache_dir:
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

# Reduce memory allocation
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
torch.cuda.memory.set_per_process_memory_fraction(0.4)
lj.monkey_patch()
lt.monkey_patch()


@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact(call: pytest.CallInfo):
    print(f"pytest_exception_interact called with call: {call}")
    if call.when == "call" and call.excinfo and call.excinfo._excinfo:
        maybe_debugpy_postmortem(call.excinfo._excinfo)
        print("Invoked debugpy postmortem debugger.")
