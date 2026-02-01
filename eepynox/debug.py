"""Generally useful debugging utilities."""
from functools import wraps
import sys
from typing import Callable, TypeVar
import threading


def maybe_debugpy_postmortem(excinfo):
    """Make the debugpy debugger enter and stop at a raised exception.

    excinfo: A (type(e), e, e.__traceback__) tuple. See sys.exc_info()
    """
    try:
        import debugpy
        pm = debugpy.post_mortem
    except (ImportError, AttributeError) as e:
        print(f"[DEBUGPY PM] Could not import debugpy.postmortem: {e}")
        return
    
    pm(excinfo, as_uncaught=True)
    

    # exc_type, exc_value, exc_tb = excinfo

    # try:
    #     import debugpy
    #     import pydevd  # type: ignore
    # except ImportError:
    #     # If pydevd isn't available, no debugger attached; do nothing.
    #     return
    # if not debugpy.is_client_connected():
    #     return

    # py_db = pydevd.get_global_debugger()
    # thread = threading.current_thread()

    # additional_info = py_db.set_additional_thread_info(thread)
    # additional_info.is_tracing += 1
    # thread_info = None
    # saved_trace = False
    # if hasattr(sys, 'monitoring'):
    #     from _pydevd_sys_monitoring._pydevd_sys_monitoring import _get_thread_info
    #     thread_info = _get_thread_info(True, 1)
    #     saved_trace = thread_info.trace
    #     thread_info.trace = False

    # try:
    #     py_db.stop_on_unhandled_exception(py_db, thread, additional_info, excinfo)
    # finally:
    #     additional_info.is_tracing -= 1
    #     if thread_info is not None:
    #         thread_info.trace = saved_trace
            


def debugpy_pm_tb():
    maybe_debugpy_postmortem(sys.exc_info())


T = TypeVar("T", bound=Callable)

class debugpy_pm:
    """A decorator and context manager that causes exceptions that escape this context to trigger postmortem debugging immediately."""

    def __call__(self, func: T) -> T:
        import inspect

        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def wrapper(*args, **kwargs):  # type: ignore
                try:
                    async for item in func(*args, **kwargs):
                        yield item
                except Exception:
                    maybe_debugpy_postmortem(sys.exc_info())
                    raise

            return wrapper  # type: ignore
        elif inspect.iscoroutinefunction(func):

            @wraps(func)
            async def wrapper(*args, **kwargs):  # type: ignore
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    maybe_debugpy_postmortem(sys.exc_info())
                    raise

            return wrapper  # type: ignore
        else:

            @wraps(func)
            def wrapper(*args, **kwargs):  # type: ignore
                try:
                    return func(*args, **kwargs)
                except Exception:
                    maybe_debugpy_postmortem(sys.exc_info())
                    raise

            return wrapper  # type: ignore

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            maybe_debugpy_postmortem((exc_type, exc_value, traceback))
