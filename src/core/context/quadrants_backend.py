"""
Quadrants backend implementation of Parameter, DeviceFunction and Kernel.

Mirrors taichi_backend.py; the specialization mechanism (clone template code
object, patch globals, decorate) is unchanged between the two - verified
empirically before writing this file. One addition specific to this backend:
Kernel templates may type their own data-field arguments as `qd.Tensor`,
which accepts either a field- or ndarray-backed value at call time with no
change to the compiled template - Taichi has no equivalent for this.

Author: B.G (07/2026)
"""

from types import FunctionType
from typing import Any

import numpy as np
import quadrants as qd

from ..pool.base import Pool
from .base import DeviceFunction, Kernel, Parameter, resolve_binding


def _numpy_dtype(dtype):
    """
    Map a Quadrants dtype to the numpy dtype used for host-side (de)serialization.

    Author: B.G (07/2026)
    """
    if dtype == qd.u8:
        return np.uint8
    if dtype == qd.i32:
        return np.int32
    if dtype == qd.i64:
        return np.int64
    return np.float32


class QuadrantsParameter(Parameter):
    """
    Parameter backed by a Quadrants scalar/const value or a pooled QuadrantsDataHandle.

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


def _specialize(template, bindings: dict[str, Any]):
    """
    Clone `template`'s code object with a globals dict where sentinels are
    replaced by resolved bindings. Shared by DeviceFunction and Kernel -
    they only differ in the qd.func/qd.kernel decorator applied on top.

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


class QuadrantsDeviceFunction(DeviceFunction):
    """
    DeviceFunction backed by a compiled qd.func.

    Only field-backed Parameters/DeviceFunctions can be resolved into a
    template this way - Quadrants rejects ndarrays referenced as globals
    (verified: raises "Ndarray used in kernel scope but not registered as
    a kernel parameter" even for the unwrapped object).

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled):
        self.name = name
        self._compiled = compiled

    @classmethod
    def compile(cls, template, *, bindings: dict[str, Any]) -> "QuadrantsDeviceFunction":
        """
        Author: B.G (07/2026)
        """
        specialised = _specialize(template, bindings)
        return cls(specialised.__name__, qd.func(specialised))

    @property
    def compiled(self):
        """
        Author: B.G (07/2026)
        """
        return self._compiled

    def __call__(self, *args, **kwargs):
        """
        A qd.func cannot be invoked from host Python - it only runs inside
        kernel/func scope, where callers use `.compiled` directly.

        Author: B.G (07/2026)
        """
        raise RuntimeError(f"DeviceFunction '{self.name}' is only callable from kernel/func scope, not host Python")


class QuadrantsKernel(Kernel):
    """
    Kernel backed by a compiled qd.kernel.

    The template's own data-field arguments should be typed `qd.Tensor` -
    unlike Taichi's ti.template()/ti.types.ndarray() split, a single
    qd.Tensor-typed template accepts either a field- or ndarray-backed
    value at call time (verified). Params/helpers still arrive via
    bindings, resolved into the kernel body - callers only ever pass data
    fields at call time.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled):
        self.name = name
        self._compiled = compiled

    @classmethod
    def compile(cls, template, *, bindings: dict[str, Any]) -> "QuadrantsKernel":
        """
        Author: B.G (07/2026)
        """
        specialised = _specialize(template, bindings)
        return cls(specialised.__name__, qd.kernel(specialised))

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
