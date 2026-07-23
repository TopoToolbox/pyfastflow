"""
Quadrants implementations of Parameter and the two builders.

Templates are python defs, specialized by splicing bound objects into their
globals before handing them to qd.func or qd.kernel - the mechanism in
_closure_backend.py, shared with Taichi.

Two things differ from Taichi. Kernel templates may type their data arguments
qd.Tensor, and one such template then accepts either a field- or ndarray-backed
value at call time; Taichi has no equivalent. Field-mode Parameters, on the
other hand, must be field-backed, because they reach device code as globals and
Quadrants rejects an ndarray referenced as a global inside a func.

Caching
-------
Leave Quadrants' src_ll fast cache off - do not mark templates compiled here
with @qd.pure or qd.kernel(fastcache=True), even to skip the python-side AST
trace. That cache keys a kernel on its source text, re-read from disk by file
path and line range (_fast_caching/function_hasher.py), together with argument
and config hashes. It never inspects __globals__, and globals are exactly what
distinguishes one specialization from another here. Two compiles of one
template with different bound consts, or a different helper under the same
name, hash identically; a hit then skips AST transformation, so nothing
downstream can catch the mismatch.

The IR-keyed caches are safe and stay on: Quadrants' own offline_cache, and
Taichi's, hash generated IR, which contains the baked literals and therefore
tells specializations apart.

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
    Compiles a device helper with qd.func. Parameters bound into it must be
    field-backed, since Quadrants rejects an ndarray referenced as a global.

    Author: B.G (07/2026)
    """

    _backend = qd


class QuadrantsKernelBuilder(ClosureKernelBuilder):
    """
    Compiles a launchable kernel with qd.kernel. Type the template's data
    arguments qd.Tensor to accept field- or ndarray-backed values alike.

    Author: B.G (07/2026)
    """

    _backend = qd
