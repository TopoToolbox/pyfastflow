"""
Quadrants backend implementation of Parameter, DeviceFunction, Kernel and
their builders.

Mirrors taichi_backend.py; both share the closure-specialization mechanism
via _closure_backend.py. One difference specific to this backend: Kernel
templates may type their own data-field arguments as `qd.Tensor`, which
accepts either a field- or ndarray-backed value at call time with no change
to the compiled template - Taichi has no equivalent for this. Note that
field-mode Parameters must be field-backed (they close over the field as a
global): Quadrants rejects ndarrays referenced as globals inside a func.

Author: B.G (07/2026)
"""

import numpy as np
import quadrants as qd

from ._closure_backend import ClosureBackendParameter, ClosureParamDeviceView, specialize_closure
from .base import DeviceFunction, DeviceFunctionBuilder, Kernel, KernelBuilder, attach_meta


class QuadrantsParameter(ClosureBackendParameter):
    """
    Parameter backed by a Quadrants const value or a pooled QuadrantsDataHandle.

    Author: B.G (07/2026)
    """

    @staticmethod
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

    def device_view(self) -> ClosureParamDeviceView:
        """
        Compile this parameter's uniform device accessors as qd.funcs.

        get(node) dispatches on mode at qd.func trace time via qd.static, so
        only the taken arm compiles: const returns a baked literal, scalar
        reads HANDLE[None], field reads HANDLE[node]. set_node(node, val) is
        built only for scalar/field (const is read-only, exposes no setter).
        MODE/VALUE/HANDLE are plain python values, not Quadrants values.

        Author: B.G (07/2026)
        """
        mode = self.mode
        value = self._const_value
        handle = self._handle.data if self._handle is not None else None

        def get_template(node):
            if qd.static(MODE == "const"):
                return VALUE
            elif qd.static(MODE == "scalar"):
                return HANDLE[None]
            else:
                return HANDLE[node]

        get_fn = qd.func(specialize_closure(get_template, {"MODE": mode, "VALUE": value, "HANDLE": handle}))

        set_fn = None
        if mode != "const":

            def set_node_template(node, val):
                if qd.static(MODE == "scalar"):
                    HANDLE[None] = val
                else:
                    HANDLE[node] = val

            set_fn = qd.func(specialize_closure(set_node_template, {"MODE": mode, "HANDLE": handle}))

        return ClosureParamDeviceView(self.name, get_fn, set_fn)


class QuadrantsDeviceFunction(DeviceFunction):
    """
    DeviceFunction backed by a compiled qd.func. Built by QuadrantsDeviceFunctionBuilder.

    Only field-backed Parameters/DeviceFunctions can be resolved into a
    template this way - Quadrants rejects ndarrays referenced as globals.

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
        A qd.func only runs inside kernel/func scope; callers use `.compiled` there.

        Author: B.G (07/2026)
        """
        raise RuntimeError(f"DeviceFunction '{self.name}' is only callable from kernel/func scope, not host Python")


class QuadrantsKernel(Kernel):
    """
    Kernel backed by a compiled qd.kernel. Built by QuadrantsKernelBuilder.

    The template's own data-field arguments should be typed `qd.Tensor` - a
    single qd.Tensor-typed template accepts either a field- or ndarray-backed
    value at call time. Params/helpers arrive via bind() and are resolved into
    the kernel body; callers pass data fields only.

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


class QuadrantsDeviceFunctionBuilder(DeviceFunctionBuilder):
    """
    Builds a QuadrantsDeviceFunction: specialize the ingested def with bound
    globals, decorate with qd.func.

    Author: B.G (07/2026)
    """

    def compile(self) -> QuadrantsDeviceFunction:
        """
        Author: B.G (07/2026)
        """
        specialised = specialize_closure(self._template, self._bindings)
        fn = QuadrantsDeviceFunction(specialised.__name__, qd.func(specialised))
        attach_meta(fn, self._template, self._bindings)
        return fn


class QuadrantsKernelBuilder(KernelBuilder):
    """
    Builds a QuadrantsKernel: specialize the ingested def with bound globals,
    decorate with qd.kernel.

    Author: B.G (07/2026)
    """

    def compile(self) -> QuadrantsKernel:
        """
        Author: B.G (07/2026)
        """
        specialised = specialize_closure(self._template, self._bindings)
        krn = QuadrantsKernel(specialised.__name__, qd.kernel(specialised))
        attach_meta(krn, self._template, self._bindings)
        return krn
