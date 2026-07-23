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

DO NOT enable Quadrants' src_ll fast cache (`@qd.pure`, or
`qd.kernel(fastcache=True)`) on templates compiled through this framework -
not even as a way to skip the python-side AST trace. That cache keys on the
kernel's *source text*, re-read from disk by file path and line range
(_fast_caching/function_hasher.py), plus arg and config hashes. It never looks
at __globals__ - and injecting globals is precisely how specialize_closure
distinguishes one specialization from another. Two compiles of the same
template with different bound consts (or a different helper under the same
name) therefore produce the same key, and a fast-cache hit skips AST
transformation entirely, so nothing downstream catches the mismatch.

Currently inert: that path is gated on is_pure, and the builders here call
plain qd.kernel / qd.func. The two IR-keyed caches (Quadrants' offline_cache,
and Taichi's) are safe by contrast - baked literals are part of the IR they
hash, so they discriminate specializations correctly.

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
