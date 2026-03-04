from typing import overload, ParamSpec, TypeVar, Callable
import functools
import jax

P = ParamSpec("P")
R = TypeVar("R")

@overload
def pjit(func: Callable[P, R], **kwargs) -> Callable[P, R]: ...
@overload
def pjit(func: None = None, **kwargs) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

def pjit(func=None, **kwargs):
    if func is None:
        return lambda f: pjit(f, **kwargs)
    jitted = jax.jit(func, **kwargs)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return jitted(*args, **kwargs)
    return wrapper
