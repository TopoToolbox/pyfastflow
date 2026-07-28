"""
Per-backend wiring shared by every Bag factory (make_grid, make_noise, ...).

A factory needs three things to build its Parameters and Helpers against a
chosen backend name: the backend module itself (for a block that needs e.g.
`backend_mod.abs`), the Parameter/HelperBuilder classes to construct with,
and the backend's own dtype objects keyed by the short names factories write
their Parameters with ("i32", "f32", "u8", "u32"). backend_classes() is the
one place that knows the mapping from a backend name to those three things,
so a factory does not carry its own copy of the same if-ladder.

Picking which private block module implements a factory's device code
("_closure_blocks" vs "_cupy_blocks") stays the caller's job - a factory owns
its own blocks, this module does not know they exist.

make_helper() is the HelperBuilder assembly every block module's
build_helpers() performs once per block.

Author: B.G (07/2026)
"""

from typing import Any

import numpy as np

from .base import HelperBuilder


def backend_classes(backend: str):
    """
    (backend_module_or_None, ParameterCls, HelperBuilderCls, dtypes) for one
    backend name.

    `backend_module_or_None` is `ti`/`qd` for the closure backends, `None` for
    cupy - cupy blocks call plain C, never a bound backend module. `dtypes`
    maps "i32"/"f32"/"u8"/"u32" to that backend's own dtype objects (ti.*/qd.*
    for the closure backends, numpy dtypes for cupy) - every name either
    make_grid or make_noise currently needs, plus the obvious siblings.

    No blocks module is returned - each factory (grid, noise, ...) has its own
    private block module and picks it itself.

    Author: B.G (07/2026)
    """
    if backend == "taichi":
        import taichi as ti

        from .taichi_backend import TaichiHelperBuilder, TaichiParameter

        return ti, TaichiParameter, TaichiHelperBuilder, {
            "i32": ti.i32, "f32": ti.f32, "u8": ti.u8, "u32": ti.u32,
        }
    if backend == "quadrants":
        import quadrants as qd

        from .quadrants_backend import QuadrantsHelperBuilder, QuadrantsParameter

        return qd, QuadrantsParameter, QuadrantsHelperBuilder, {
            "i32": qd.i32, "f32": qd.f32, "u8": qd.u8, "u32": qd.u32,
        }
    if backend == "cupy":
        from .cupy_backend import CupyHelperBuilder, CupyParameter

        return None, CupyParameter, CupyHelperBuilder, {
            "i32": np.int32, "f32": np.float32, "u8": np.uint8, "u32": np.uint32,
        }
    raise ValueError(f"unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")


def make_helper(HelperCls, template, **binds: Any) -> HelperBuilder:
    """
    A HelperBuilder ingesting `template` with every entry of `binds` applied.

    The assembly every block module's build_helpers() needs for each of its
    blocks, so a new block module does not carry its own copy.

    Author: B.G (07/2026)
    """
    builder = HelperCls().ingest(template)
    for name, obj in binds.items():
        builder.bind(name, obj)
    return builder
