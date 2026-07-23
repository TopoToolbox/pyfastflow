"""
Taichi backend implementation of Parameter, DeviceFunction, Kernel and their
builders.

Author: B.G (07/2026)
"""

import taichi as ti

from ._closure_backend import (
    ClosureBackendParameter,
    ClosureDeviceFunction,
    ClosureDeviceFunctionBuilder,
    ClosureKernel,
    ClosureKernelBuilder,
)


class TaichiParameter(ClosureBackendParameter):
    """
    Parameter backed by a Taichi const value or a pooled TaichiDataHandle.

    Author: B.G (07/2026)
    """

    _backend = ti


class TaichiDeviceFunction(ClosureDeviceFunction):
    """
    DeviceFunction backed by a compiled ti.func. Built by TaichiDeviceFunctionBuilder.

    Author: B.G (07/2026)
    """


class TaichiKernel(ClosureKernel):
    """
    Kernel backed by a compiled ti.kernel. Built by TaichiKernelBuilder.

    Author: B.G (07/2026)
    """


class TaichiDeviceFunctionBuilder(ClosureDeviceFunctionBuilder):
    """
    Builds a TaichiDeviceFunction: specialize the ingested def with bound
    globals, decorate with ti.func.

    Author: B.G (07/2026)
    """

    _backend = ti


class TaichiKernelBuilder(ClosureKernelBuilder):
    """
    Builds a TaichiKernel: specialize the ingested def with bound globals,
    decorate with ti.kernel.

    Author: B.G (07/2026)
    """

    _backend = ti
