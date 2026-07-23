"""
Taichi implementations of Parameter and the two builders.

Templates are python defs. A builder specializes one by rebuilding it with the
bound objects spliced into its globals, then hands the result to ti.func or
ti.kernel; _closure_backend.py holds that machinery, shared with Quadrants.
Everything below just names Taichi as the backend to use.

Kernel templates declare their data arguments the usual Taichi way, typically
ti.template(). Bound Parameters and helpers are not arguments - see base.py.

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
    Compiles a device helper with ti.func.

    Author: B.G (07/2026)
    """

    _backend = ti


class TaichiKernelBuilder(ClosureKernelBuilder):
    """
    Compiles a launchable kernel with ti.kernel.

    Author: B.G (07/2026)
    """

    _backend = ti
