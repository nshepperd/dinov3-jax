"""Generally useful debugging utilities."""

import sys
from collections.abc import Callable
from functools import wraps
from typing import TypeVar


def maybe_debugpy_postmortem(excinfo):
    """Make the debugpy debugger enter and stop at a raised exception.

    excinfo: A (type(e), e, e.__traceback__) tuple. See sys.exc_info()
    """
    try:
        import debugpy  # noqa: T100

        pm = debugpy.trigger_exception_handler
    except (ImportError, AttributeError) as e:
        print(f"[DEBUGPY PM] Could not import debugpy.postmortem: {e}")
        return

    pm(excinfo, as_uncaught=True)


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
