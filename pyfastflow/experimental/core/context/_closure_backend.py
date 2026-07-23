"""
Machinery shared by the two backends whose templates are python functions:
Taichi and Quadrants.

Specialization works by rebuilding the template function around a globals dict
that carries the bound objects, so a name like `phys` in the template body
resolves to the bound Bag when the backend traces it. The rebuilt function
is then decorated with ti.func/qd.func or ti.kernel/qd.kernel.

The two backends can share all of this because the pieces used here - func,
kernel, static, u8, i32, i64 - carry the same names and the same behaviour in
both modules. A backend subclass therefore only pins `_backend` to the ti or qd
module; nothing else varies.

cupy does not appear here: CUDA source text has no globals to patch, and that
backend substitutes into the source directly instead.

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
    Rebuild `template` as a new function whose globals carry the resolved
    bindings, leaving the original untouched.

    The code object is reused as-is; only the globals differ, which is what
    makes a name in the template body resolve to a bound object. Defaults,
    annotations and the rest are copied over so the result still introspects
    like the template it came from.

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
    What a Parameter looks like from inside device code.

    `.get` and `.set_node` are compiled device funcs, so a template body reads
    `p.get(i)` and writes `p.set_node(i, v)` the same way whatever the
    parameter's mode. A const parameter is read-only and carries no `.set_node`
    at all, which turns a write to one into a trace-time error.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, get_fn, set_fn=None):
        self._name = name
        self.get = get_fn
        if set_fn is not None:
            self.set_node = set_fn


class ClosureBackendParameter(Parameter):
    """
    Parameter backed by a const python value or by pooled device storage.

    Concrete backends subclass this and pin `_backend` to their module; the
    dtype mapping and the device view are written once here against the names
    both modules share.

    Author: B.G (07/2026)
    """

    SUPPORTED_MODES = frozenset({"const", "scalar", "field"})
    _backend: ClassVar[Any]

    def __init__(self, name: str, *, dtype, mode: str, value, pool, n_flat: int | None = None, solo: bool = False):
        """
        Declare one parameter and give it its initial value.

        scalar and field take pooled storage straight away; const stays a plain
        python value. solo=True, available on const only, lets the parameter be
        read bare in a template body - written `p` rather than `p.get(i)` -
        because it resolves to a compile-time literal.

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
        This parameter's device accessor, built on first use and kept.

        The compiled funcs come out identical every time, so one view serves
        every kernel that binds this parameter. Two things can invalidate it -
        set() on a const mode, which changes the literal baked into the getter,
        and destroy(), which releases the storage it reads. Both drop the view
        so the next caller rebuilds; neither reaches kernels compiled earlier
        (see base.py, "Lifetime of a compiled object"). A scalar or field set()
        needs no invalidation, writing through the very storage the view reads.

        Author: B.G (07/2026)
        """
        if self._device_view is None:
            self._device_view = self._build_device_view()
        return self._device_view

    def _build_device_view(self) -> ClosureParamDeviceView:
        """
        Compile this parameter's device accessors as backend funcs.

        get(node) branches on the mode through `_backend.static`, which
        resolves at trace time, so only one arm survives into the generated
        code: a baked literal for const, HANDLE[None] for scalar, HANDLE[node]
        for field. set_node is built for scalar and field only. MODE, VALUE and
        HANDLE are ordinary python values spliced in as globals.

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
    A device helper compiled to a ti.func or qd.func.

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
    A launchable kernel compiled to a ti.kernel or qd.kernel.

    Its call signature is the template's own, which declares data arguments
    only - `def template(out: ti.template()): ...` - since bound objects reach
    the body through globals instead. See base.py.

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
    Raise unless every Parameter reachable in `bindings` is const mode.

    This enforces the rule in base.py that a device helper binds const
    parameters only, passing any data through explicit arguments instead.
    Bound Bags are searched too, so a scalar or field parameter tucked inside
    one does not slip past.

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
    Compiles an ingested def into a device helper. Subclasses pin `_backend`.

    Author: B.G (07/2026)
    """

    _backend: ClassVar[Any]

    def compile(self) -> ClosureDeviceFunction:
        """
        Check the const-only rule, splice the referenced bindings into the
        template's globals, and compile the result as a device func.

        Author: B.G (07/2026)
        """
        _check_const_only(self._bindings)
        specialised = specialize_closure(self._template, filter_bindings(self._template, self._bindings))
        fn = ClosureDeviceFunction(specialised.__name__, self._backend.func(specialised))
        attach_meta(fn, self._template, self._bindings)
        return fn


class ClosureKernelBuilder(KernelBuilder):
    """
    Compiles an ingested def into a launchable kernel. Subclasses pin
    `_backend`.

    Author: B.G (07/2026)
    """

    _backend: ClassVar[Any]

    def compile(self) -> ClosureKernel:
        """
        Splice the referenced bindings into the template's globals and compile
        the result as a launchable kernel.

        Author: B.G (07/2026)
        """
        specialised = specialize_closure(self._template, filter_bindings(self._template, self._bindings))
        krn = ClosureKernel(specialised.__name__, self._backend.kernel(specialised))
        attach_meta(krn, self._template, self._bindings)
        return krn
