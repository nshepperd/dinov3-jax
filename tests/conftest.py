import os
import sys

import jax
import pytest
import torch

import lovely_jax as lj
import lovely_tensors as lt


from dinov3_jax.eepynox.debug import maybe_debugpy_postmortem

jax.config.update("jax_default_matmul_precision", "highest")

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
