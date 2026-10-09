import functools
from collections.abc import Callable
from typing import overload

import jax


@overload
def pjit[**P, R](func: Callable[P, R], **kwargs) -> Callable[P, R]: ...
@overload
def pjit[**P, R](
    func: None = None, **kwargs
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def pjit(func=None, **kwargs):
    if func is None:
        return lambda f: pjit(f, **kwargs)
    jitted = jax.jit(func, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return jitted(*args, **kwargs)

    return wrapper
