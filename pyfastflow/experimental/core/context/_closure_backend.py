"""
Shared machinery for backends that compile python function templates by
patching globals on a cloned code object (Taichi, Quadrants). Not used by
backends without that mechanism (e.g. the cupy/RawKernel backend).

Author: B.G (07/2026)
"""

from types import FunctionType
from typing import Any, ClassVar

import numpy as np

from .base import (
    Bag,
    DeviceFunction,
    DeviceFunctionBuilder,
    Kernel,
    KernelBuilder,
    Parameter,
    attach_meta,
    filter_bindings,
    resolve_binding,
)


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
    sharing the closure-specialization mechanism (Taichi, Quadrants).
    Subclasses pin `_backend` (the `ti`/`qd` module) - `_numpy_dtype` and
    `_build_device_view` are shared here since `ti.u8/i32/i64`,
    `ti.func`/`ti.static` and their `qd.` equivalents are identical in name
    and behaviour across both modules.

    Author: B.G (07/2026)
    """

    SUPPORTED_MODES = frozenset({"const", "scalar", "field"})
    _backend: ClassVar[Any]

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
        self._device_view: "ClosureParamDeviceView | None" = None

        if mode == "scalar":
            self._handle = pool.get_data(dtype, ())
        elif mode == "field":
            if n_flat is None:
                raise ValueError(f"{name}: field mode requires n_flat")
            self._handle = pool.get_data(dtype, (n_flat,))

        self.set(value)

    @classmethod
    def _numpy_dtype(cls, dtype):
        """
        Map a backend dtype (`ti.*`/`qd.*`) to the numpy dtype used for
        host-side (de)serialization.

        Author: B.G (07/2026)
        """
        backend = cls._backend
        if dtype == backend.u8:
            return np.uint8
        if dtype == backend.i32:
            return np.int32
        if dtype == backend.i64:
            return np.int64
        return np.float32

    def get(self):
        """
        The python value for const mode, the backing DataHandle otherwise.

        Author: B.G (07/2026)
        """
        return self._const_value if self.mode == "const" else self._handle

    def set(self, value) -> None:
        """
        Overwrite the whole value: a cast python scalar for const, a device
        write for scalar, a full host->device copy for field.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            self._const_value = self._numpy_dtype(self.dtype)(value).item()
            self._device_view = None  # a cached view would bake the stale literal
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
        Return any pooled storage to the pool. const mode owns none, so this
        is a no-op there.

        Author: B.G (07/2026)
        """
        if self._handle is not None:
            self._pool.release_data(self._handle)
            self._handle = None
            self._device_view = None  # a cached view closes over the released handle

    def device_view(self) -> ClosureParamDeviceView:
        """
        Cached uniform device accessor for this parameter. Built once via
        `_build_device_view` (backend-specific) and memoized on the instance:
        rebuilding it per compile is pointless python work at O(params x
        kernels) since the ti.func/qd.func objects are identical every time.
        Invalidated (not refreshed) by set() on const mode and by destroy(),
        the two places the baked literal/handle a cached view depends on can
        change; scalar/field set() writes through the handle a cached view
        already points at, so no invalidation is needed there.

        Author: B.G (07/2026)
        """
        if self._device_view is None:
            self._device_view = self._build_device_view()
        return self._device_view

    def _build_device_view(self) -> ClosureParamDeviceView:
        """
        Compile this parameter's uniform device accessors as backend funcs
        (ti.func / qd.func).

        get(node) dispatches on mode at trace time via `_backend.static`, so
        only the taken arm compiles: const returns a baked literal, scalar
        reads HANDLE[None], field reads HANDLE[node]. set_node(node, val) is
        built only for scalar/field (const is read-only, exposes no setter).
        MODE/VALUE/HANDLE are plain python values, not backend values.

        Const getters bake VALUE as a compile-time literal: a later .set()
        needs the view (and anything binding it) rebuilt to take effect.

        Called at most once per instance, memoized by device_view().

        Author: B.G (07/2026)
        """
        backend = self._backend
        mode = self.mode
        value = self._const_value
        handle = self._handle.data if self._handle is not None else None

        def get_template(node):
            if STATIC(MODE == "const"):
                return VALUE
            elif STATIC(MODE == "scalar"):
                return HANDLE[None]
            else:
                return HANDLE[node]

        get_fn = backend.func(
            specialize_closure(get_template, {"MODE": mode, "VALUE": value, "HANDLE": handle, "STATIC": backend.static})
        )

        set_fn = None
        if mode != "const":

            def set_node_template(node, val):
                if STATIC(MODE == "scalar"):
                    HANDLE[None] = val
                else:
                    HANDLE[node] = val

            set_fn = backend.func(
                specialize_closure(set_node_template, {"MODE": mode, "HANDLE": handle, "STATIC": backend.static})
            )

        return ClosureParamDeviceView(self.name, get_fn, set_fn)


class ClosureDeviceFunction(DeviceFunction):
    """
    DeviceFunction backed by a compiled closure-backend func (ti.func /
    qd.func). Built by ClosureDeviceFunctionBuilder subclasses.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled):
        self.name = name
        self._compiled = compiled

    @property
    def compiled(self):
        """
        The raw ti.func/qd.func, for binding into another template's body.

        Author: B.G (07/2026)
        """
        return self._compiled

    def __call__(self, *args, **kwargs):
        """
        A ti.func/qd.func only runs inside kernel/func scope; callers use
        `.compiled` there.

        Author: B.G (07/2026)
        """
        raise RuntimeError(f"DeviceFunction '{self.name}' is only callable from kernel/func scope, not host Python")


class ClosureKernel(Kernel):
    """
    Kernel backed by a compiled closure-backend kernel (ti.kernel /
    qd.kernel). Built by ClosureKernelBuilder subclasses.

    The template's own signature declares data-field arguments only, e.g.
    `def template(out: ti.template()): ...` - see base.py's module docstring.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled):
        self.name = name
        self._compiled = compiled

    @property
    def compiled(self):
        """
        The raw ti.kernel/qd.kernel behind this Kernel's __call__.

        Author: B.G (07/2026)
        """
        return self._compiled

    def __call__(self, *args, **kwargs):
        """
        Launches the compiled kernel. Args are data fields only.

        Author: B.G (07/2026)
        """
        return self._compiled(*args, **kwargs)


def _check_const_only(bindings: dict[str, Any]) -> None:
    """
    Enforce the framework rule (base.py module docstring): a device helper
    only ever binds const-mode Parameters - any data it needs is passed to it
    as an explicit argument by the calling kernel instead. Recurses into
    bound Bags so a scalar/field param nested inside one is also caught.

    Author: B.G (07/2026)
    """
    for name, value in bindings.items():
        if isinstance(value, Parameter):
            if value.mode != "const":
                raise ValueError(
                    f"device helper: bound parameter '{name}' has mode {value.mode!r}, but a device "
                    "helper may only bind const parameters - pass the data as an explicit argument to "
                    "the helper instead"
                )
        elif isinstance(value, Bag):
            _check_const_only(dict(value.items()))


class ClosureDeviceFunctionBuilder(DeviceFunctionBuilder):
    """
    Builds a ClosureDeviceFunction: specialize the ingested def with bound
    globals, decorate with `_backend.func`. Subclasses pin `_backend`.

    Author: B.G (07/2026)
    """

    _backend: ClassVar[Any]

    def compile(self) -> ClosureDeviceFunction:
        """
        Check the const-only rule, inject the referenced bindings into the
        template's globals, decorate the result as a device func.

        Author: B.G (07/2026)
        """
        _check_const_only(self._bindings)
        specialised = specialize_closure(self._template, filter_bindings(self._template, self._bindings))
        fn = ClosureDeviceFunction(specialised.__name__, self._backend.func(specialised))
        attach_meta(fn, self._template, self._bindings)
        return fn


class ClosureKernelBuilder(KernelBuilder):
    """
    Builds a ClosureKernel: specialize the ingested def with bound globals,
    decorate with `_backend.kernel`. Subclasses pin `_backend`.

    Author: B.G (07/2026)
    """

    _backend: ClassVar[Any]

    def compile(self) -> ClosureKernel:
        """
        Inject the referenced bindings into the template's globals and
        decorate the result as a launchable kernel.

        Author: B.G (07/2026)
        """
        specialised = specialize_closure(self._template, filter_bindings(self._template, self._bindings))
        krn = ClosureKernel(specialised.__name__, self._backend.kernel(specialised))
        attach_meta(krn, self._template, self._bindings)
        return krn
