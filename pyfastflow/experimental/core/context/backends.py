"""
Per-backend wiring shared by every factory (make_grid, make_noise, ...).

A factory needs two things to build its Parameters against a chosen backend
name: the backend module itself (for a cupy-only helper that needs e.g.
`np.int32`, or a closure block that needs `ti`/`qd` directly), and the
backend's own dtype objects keyed by the short names factories write their
Parameters with ("i32", "i64", "f32", "u8", "u32"). backend_classes() is the
one place that knows the mapping from a backend name to those things, so a
factory does not carry its own copy of the same if-ladder.

Picking which private block module implements a factory's device code
("_closure_blocks" vs "_cupy_blocks") stays the caller's job - a factory owns
its own blocks, this module does not know they exist.

Author: B.G (07/2026)
"""

import numpy as np


def backend_classes(backend: str):
    """
    Look up the module, Parameter subclass and dtype table for one backend.

    Parameters
    ----------
    backend : str
        "taichi", "quadrants" or "cupy".

    Returns
    -------
    module : module or None
        `ti`/`qd` for the closure backends, `None` for cupy - cupy blocks
        call plain C, never a bound backend module.
    ParameterCls : type
        The backend's Parameter subclass.
    unused : None
        Reserved, always None - kept so `_, ParamCls, _, dtypes =
        backend_classes(backend)` call sites stay stable.
    dtypes : dict
        Maps "i32"/"i64"/"f32"/"u8"/"u32" to that backend's own dtype
        objects (ti.*/qd.* for the closure backends, numpy dtypes for cupy).

    No blocks module is returned - each factory (grid, noise, ...) has its
    own private block module and picks it itself.

    Author: B.G (07/2026)
    """
    if backend == "taichi":
        import taichi as ti

        from .taichi_backend import TaichiParameter

        return ti, TaichiParameter, None, {
            "i32": ti.i32, "i64": ti.i64, "f32": ti.f32, "u8": ti.u8, "u32": ti.u32,
        }
    if backend == "quadrants":
        import quadrants as qd

        from .quadrants_backend import QuadrantsParameter

        return qd, QuadrantsParameter, None, {
            "i32": qd.i32, "i64": qd.i64, "f32": qd.f32, "u8": qd.u8, "u32": qd.u32,
        }
    if backend == "cupy":
        from .cupy_backend import CupyParameter

        return None, CupyParameter, None, {
            "i32": np.int32, "i64": np.int64, "f32": np.float32, "u8": np.uint8, "u32": np.uint32,
        }
    raise ValueError(f"unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")
