import os
import sys

import jax
import pytest
import torch

import lovely_jax as lj
import lovely_tensors as lt
lj.monkey_patch()
lt.monkey_patch()

from eepynox.debug import maybe_debugpy_postmortem

jax.config.update("jax_default_matmul_precision", "highest")

# Reduce memory allocation
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
torch.cuda.memory.set_per_process_memory_fraction(0.4)

# try:
#     import debugpy
#     import pydevd
#     import threading
#     if debugpy.is_client_connected():
#         if sys.gettrace() is None:
#             py_db = pydevd.get_global_debugger()
#             thread = threading.current_thread()
#             py_db.set_trace_for_frame_and_parents(thread.ident, sys._getframe())
# except ImportError:
#     pass

@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact(call: pytest.CallInfo):
    print(f"pytest_exception_interact called with call: {call}")
    if call.when == 'call' and call.excinfo and call.excinfo._excinfo:
        maybe_debugpy_postmortem(call.excinfo._excinfo)
        print("Invoked debugpy postmortem debugger.")