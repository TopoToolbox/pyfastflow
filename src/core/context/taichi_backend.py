"""
Taichi backend implementation of Parameter, DeviceFunction and Kernel.

Author: B.G (07/2026)
"""

from types import FunctionType
from typing import Any

import numpy as np
import taichi as ti

from ..pool.base import Pool
from .base import DeviceFunction, Kernel, Parameter, resolve_binding


def _specialize(template, bindings: dict[str, Any]):
    """
    Clone `template`'s code object with a globals dict where sentinels are
    replaced by resolved bindings. Shared by DeviceFunction and Kernel -
    they only differ in the ti.func/ti.kernel decorator applied on top.

    Author: B.G (07/2026)
    """
    resolved = {name: resolve_binding(value) for name, value in bindings.items()}
    source = getattr(template, "__wrapped__", template)
    func_globals = dict(source.__globals__)
    func_globals.update(resolved)

    specialised = FunctionType(
        source.__code__,
        func_globals,
        source.__name__,
        source.__defaults__,
        source.__closure__,
    )
    specialised.__kwdefaults__ = source.__kwdefaults__
    specialised.__annotations__ = dict(source.__annotations__)
    specialised.__doc__ = source.__doc__
    specialised.__qualname__ = source.__qualname__
    return specialised


def _numpy_dtype(dtype):
    """
    Map a Taichi dtype to the numpy dtype used for host-side (de)serialization.

    Author: B.G (07/2026)
    """
    if dtype == ti.u8:
        return np.uint8
    if dtype == ti.i32:
        return np.int32
    if dtype == ti.i64:
        return np.int64
    return np.float32


class TaichiParameter(Parameter):
    """
    Parameter backed by a Taichi scalar/const value or a pooled TaichiDataHandle.

    Author: B.G (07/2026)
    """

    SUPPORTED_MODES = frozenset({"const", "scalar", "field"})

    def __init__(self, name: str, *, dtype, mode: str, value, pool: Pool, n_flat: int | None = None):
        """
        Declare and initialize one parameter. "scalar"/"field" modes allocate
        pooled storage immediately via `pool`; "const" stays a plain python value.

        Author: B.G (07/2026)
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"{name}: mode must be one of {sorted(self.SUPPORTED_MODES)}, got {mode!r}")

        self.name = name
        self.dtype = dtype
        self.mode = mode
        self._pool = pool
        self._const_value: Any = None
        self._handle = None

        if mode == "scalar":
            self._handle = pool.get_data(dtype, ())
        elif mode == "field":
            if n_flat is None:
                raise ValueError(f"{name}: field mode requires n_flat")
            self._handle = pool.get_data(dtype, (n_flat,))

        self.set(value)

    def get(self):
        """
        Author: B.G (07/2026)
        """
        return self._const_value if self.mode == "const" else self._handle

    def set(self, value) -> None:
        """
        Author: B.G (07/2026)
        """
        if self.mode == "const":
            self._const_value = _numpy_dtype(self.dtype)(value).item()
        elif self.mode == "scalar":
            self._handle.data[None] = value
        else:  # field
            arr = np.asarray(value, dtype=_numpy_dtype(self.dtype)).reshape(-1)
            self._handle.data.from_numpy(arr)

    def destroy(self) -> None:
        """
        Author: B.G (07/2026)
        """
        if self._handle is not None:
            self._pool.release_data(self._handle)
            self._handle = None


class TaichiDeviceFunction(DeviceFunction):
    """
    DeviceFunction backed by a compiled ti.func.

    compile() clones the template's code object with a globals dict where
    sentinels are replaced by resolved bindings, then decorates with ti.func -
    same specialization mechanism as the legacy CallableFactory.compile.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled):
        self.name = name
        self._compiled = compiled

    @classmethod
    def compile(cls, template, *, bindings: dict[str, Any]) -> "TaichiDeviceFunction":
        """
        Author: B.G (07/2026)
        """
        specialised = _specialize(template, bindings)
        return cls(specialised.__name__, ti.func(specialised))

    @property
    def compiled(self):
        """
        Author: B.G (07/2026)
        """
        return self._compiled

    def __call__(self, *args, **kwargs):
        """
        A ti.func cannot be invoked from host Python - it only runs inside
        kernel/func scope, where callers use `.compiled` directly.

        Author: B.G (07/2026)
        """
        raise RuntimeError(f"DeviceFunction '{self.name}' is only callable from kernel/func scope, not host Python")


class TaichiKernel(Kernel):
    """
    Kernel backed by a compiled ti.kernel.

    The template's own signature should only declare data-field arguments
    (e.g. `def template(out: ti.template()): ...`) - any params/helpers it
    needs are resolved into the kernel body via `bindings`, so callers only
    ever pass data fields at call time, never params/helpers.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled):
        self.name = name
        self._compiled = compiled

    @classmethod
    def compile(cls, template, *, bindings: dict[str, Any]) -> "TaichiKernel":
        """
        Author: B.G (07/2026)
        """
        specialised = _specialize(template, bindings)
        return cls(specialised.__name__, ti.kernel(specialised))

    @property
    def compiled(self):
        """
        Author: B.G (07/2026)
        """
        return self._compiled

    def __call__(self, *args, **kwargs):
        """
        Launches the compiled kernel. Args are data fields only.

        Author: B.G (07/2026)
        """
        return self._compiled(*args, **kwargs)
