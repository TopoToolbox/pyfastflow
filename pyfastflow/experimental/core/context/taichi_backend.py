"""
Taichi backend implementation of Parameter, DeviceFunction, Kernel and their
builders.

Author: B.G (07/2026)
"""

import taichi as ti

from ._closure_backend import (
    ClosureBackendParameter,
    ClosureDeviceFunctionBuilder,
    ClosureKernelBuilder,
)


class TaichiParameter(ClosureBackendParameter):
    """
    Parameter backed by a Taichi const value or a pooled TaichiDataHandle.

    Author: B.G (07/2026)
    """

    _backend = ti


class TaichiDeviceFunctionBuilder(ClosureDeviceFunctionBuilder):
    """
    Builds a ClosureDeviceFunction: specialize the ingested def with bound
    globals, decorate with ti.func.

    Author: B.G (07/2026)
    """

    _backend = ti


class TaichiKernelBuilder(ClosureKernelBuilder):
    """
    Builds a ClosureKernel: specialize the ingested def with bound globals,
    decorate with ti.kernel.

    Author: B.G (07/2026)
    """

    _backend = ti
