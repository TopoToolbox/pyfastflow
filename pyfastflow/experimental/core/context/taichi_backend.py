"""
Taichi backend implementation of Parameter, DeviceFunction, Kernel and their
builders.

Author: B.G (07/2026)
"""

import numpy as np
import taichi as ti

from ._closure_backend import ClosureBackendParameter, ClosureParamDeviceView, specialize_closure
from .base import DeviceFunction, DeviceFunctionBuilder, Kernel, KernelBuilder, attach_meta


class TaichiParameter(ClosureBackendParameter):
    """
    Parameter backed by a Taichi const value or a pooled TaichiDataHandle.

    Author: B.G (07/2026)
    """

    @staticmethod
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

    def device_view(self) -> ClosureParamDeviceView:
        """
        Compile this parameter's uniform device accessors as ti.funcs.

        get(node) dispatches on mode at ti.func trace time via ti.static, so
        only the taken arm compiles: const returns a baked literal, scalar
        reads HANDLE[None], field reads HANDLE[node]. set_node(node, val) is
        built only for scalar/field (const is read-only, exposes no setter).
        MODE/VALUE/HANDLE are plain python values, not Taichi values.

        Const getters bake VALUE as a compile-time literal: a later .set()
        needs the view (and anything binding it) rebuilt to take effect.

        Author: B.G (07/2026)
        """
        mode = self.mode
        value = self._const_value
        handle = self._handle.data if self._handle is not None else None

        def get_template(node):
            if ti.static(MODE == "const"):
                return VALUE
            elif ti.static(MODE == "scalar"):
                return HANDLE[None]
            else:
                return HANDLE[node]

        get_fn = ti.func(specialize_closure(get_template, {"MODE": mode, "VALUE": value, "HANDLE": handle}))

        set_fn = None
        if mode != "const":

            def set_node_template(node, val):
                if ti.static(MODE == "scalar"):
                    HANDLE[None] = val
                else:
                    HANDLE[node] = val

            set_fn = ti.func(specialize_closure(set_node_template, {"MODE": mode, "HANDLE": handle}))

        return ClosureParamDeviceView(self.name, get_fn, set_fn)


class TaichiDeviceFunction(DeviceFunction):
    """
    DeviceFunction backed by a compiled ti.func. Built by TaichiDeviceFunctionBuilder.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled):
        self.name = name
        self._compiled = compiled

    @property
    def compiled(self):
        """
        Author: B.G (07/2026)
        """
        return self._compiled

    def __call__(self, *args, **kwargs):
        """
        A ti.func only runs inside kernel/func scope; callers use `.compiled` there.

        Author: B.G (07/2026)
        """
        raise RuntimeError(f"DeviceFunction '{self.name}' is only callable from kernel/func scope, not host Python")


class TaichiKernel(Kernel):
    """
    Kernel backed by a compiled ti.kernel. Built by TaichiKernelBuilder.

    The template's own signature declares data-field arguments only (e.g.
    `def template(out: ti.template()): ...`); params/helpers arrive via bind()
    and are resolved into the kernel body, so callers pass data fields only.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled):
        self.name = name
        self._compiled = compiled

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


class TaichiDeviceFunctionBuilder(DeviceFunctionBuilder):
    """
    Builds a TaichiDeviceFunction: specialize the ingested def with bound
    globals, decorate with ti.func.

    Author: B.G (07/2026)
    """

    def compile(self) -> TaichiDeviceFunction:
        """
        Author: B.G (07/2026)
        """
        specialised = specialize_closure(self._template, self._bindings)
        fn = TaichiDeviceFunction(specialised.__name__, ti.func(specialised))
        attach_meta(fn, self._template, self._bindings)
        return fn


class TaichiKernelBuilder(KernelBuilder):
    """
    Builds a TaichiKernel: specialize the ingested def with bound globals,
    decorate with ti.kernel.

    Author: B.G (07/2026)
    """

    def compile(self) -> TaichiKernel:
        """
        Author: B.G (07/2026)
        """
        specialised = specialize_closure(self._template, self._bindings)
        krn = TaichiKernel(specialised.__name__, ti.kernel(specialised))
        attach_meta(krn, self._template, self._bindings)
        return krn
