"""
Shared machinery for backends that compile python function templates by
patching globals on a cloned code object (Taichi, Quadrants). Not used by
backends without that mechanism (e.g. the cupy/RawKernel backend).

Author: B.G (07/2026)
"""

from types import FunctionType
from typing import Any

import numpy as np

from .base import Parameter, resolve_binding


def specialize_closure(template, bindings: dict[str, Any]) -> FunctionType:
    """
    Clone `template`'s code object with a globals dict where sentinels are
    replaced by resolved bindings.

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


class ClosureParamDeviceView:
    """
    Device-facing view of a Parameter for closure backends: `.get` and (unless
    const) `.set_node` are the raw compiled device funcs (ti.func / qd.func),
    so a template body traces `p.get(i)` / `p.set_node(i, v)` as plain
    attribute lookups + func calls, uniform across modes. Const params expose
    no `.set_node` (read-only), so touching it in a kernel raises at trace time.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, get_fn, set_fn=None):
        self._name = name
        self.get = get_fn
        if set_fn is not None:
            self.set_node = set_fn


class ClosureBackendParameter(Parameter):
    """
    Parameter backed by a const value or a pooled DataHandle, for backends
    sharing the closure-specialization mechanism. Subclasses pin `_numpy_dtype`
    and implement `device_view` with their own func decorator.

    Author: B.G (07/2026)
    """

    SUPPORTED_MODES = frozenset({"const", "scalar", "field"})

    def __init__(self, name: str, *, dtype, mode: str, value, pool, n_flat: int | None = None, solo: bool = False):
        """
        Declare and initialize one parameter. "scalar"/"field" modes allocate
        pooled storage immediately via `pool`; "const" stays a plain python
        value. solo=True (const only) lets the parameter be read bare in a
        template body (no .get()) - it resolves to a compile-time literal.

        Author: B.G (07/2026)
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"{name}: mode must be one of {sorted(self.SUPPORTED_MODES)}, got {mode!r}")
        if solo and mode != "const":
            raise ValueError(f"{name}: solo access is const-only, got mode {mode!r}")

        self.name = name
        self.dtype = dtype
        self.mode = mode
        self.solo = solo
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

    @staticmethod
    def _numpy_dtype(dtype):
        """
        Map a backend dtype to the numpy dtype used for host-side (de)serialization.
        Overridden per backend.

        Author: B.G (07/2026)
        """
        raise NotImplementedError

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
            self._const_value = self._numpy_dtype(self.dtype)(value).item()
        elif self.mode == "scalar":
            self._handle.data[None] = value
        else:  # field
            arr = np.asarray(value, dtype=self._numpy_dtype(self.dtype)).reshape(-1)
            self._handle.data.from_numpy(arr)

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            raise ValueError(f"{self.name}: const parameter is read-only")
        if self.mode == "scalar":
            self._handle.data[None] = value
        else:  # field
            self._handle.data[node] = value

    def destroy(self) -> None:
        """
        Author: B.G (07/2026)
        """
        if self._handle is not None:
            self._pool.release_data(self._handle)
            self._handle = None
