"""
Quadrants backend implementation of Parameter, DeviceFunction, Kernel and
their builders.

Shares the closure-specialization mechanism with taichi_backend.py via
_closure_backend.py. One difference specific to this backend: Kernel
templates may type their own data-field arguments as `qd.Tensor`, which
accepts either a field- or ndarray-backed value at call time with no change
to the compiled template - Taichi has no equivalent for this. Note that
field-mode Parameters must be field-backed (they close over the field as a
global): Quadrants rejects ndarrays referenced as globals inside a func.

Author: B.G (07/2026)
"""

import quadrants as qd

from ._closure_backend import (
    ClosureBackendParameter,
    ClosureDeviceFunctionBuilder,
    ClosureKernelBuilder,
)


class QuadrantsParameter(ClosureBackendParameter):
    """
    Parameter backed by a Quadrants const value or a pooled QuadrantsDataHandle.

    Author: B.G (07/2026)
    """

    _backend = qd


class QuadrantsDeviceFunctionBuilder(ClosureDeviceFunctionBuilder):
    """
    Builds a ClosureDeviceFunction: specialize the ingested def with bound
    globals, decorate with qd.func. Only field-backed Parameters/DeviceFunctions
    can be resolved into a template this way - Quadrants rejects ndarrays
    referenced as globals.

    Author: B.G (07/2026)
    """

    _backend = qd


class QuadrantsKernelBuilder(ClosureKernelBuilder):
    """
    Builds a ClosureKernel: specialize the ingested def with bound globals,
    decorate with qd.kernel. The template's own data-field arguments should be
    typed `qd.Tensor` - a single qd.Tensor-typed template accepts either a
    field- or ndarray-backed value at call time.

    Author: B.G (07/2026)
    """

    _backend = qd
