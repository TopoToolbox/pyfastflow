"""
Per-backend wiring shared by every Bag factory (make_grid, make_noise, ...).

A factory needs three things to build its Parameters and Helpers against a
chosen backend name: the backend module itself (for a block that needs e.g.
`backend_mod.abs`), the Parameter/HelperBuilder classes to construct with,
and the backend's own dtype objects keyed by the short names factories write
their Parameters with ("i32", "i64", "f32", "u8", "u32"). backend_classes() is
the one place that knows the mapping from a backend name to those three
things, so a factory does not carry its own copy of the same if-ladder.

Picking which private block module implements a factory's device code
("_closure_blocks" vs "_cupy_blocks") stays the caller's job - a factory owns
its own blocks, this module does not know they exist.

make_helper() is the HelperBuilder assembly every block module's
build_helpers() performs once per block; make_kernel() is its twin for
KernelBuilder, first exercised by ops/_closure_blocks.py and
ops/_cupy_blocks.py, which - unlike grid/noise - build kernels directly
rather than only helpers. param_need()/helper_need()/bag_need() are the
small, reusable Need (need.py) constructors a factory converting its own
internal wiring to strict Needs (CompileBuilder(strict_needs=True) - see
compile.py) reaches for at each bind site - see grid/_closure_blocks.py and
grid/_cupy_blocks.py for the first factory doing so (Parameter/HelperBuilder
binds only), and noise/_closure_blocks.py and noise/_cupy_blocks.py for the
first use of bag_need() (binding a whole grid Bag under one name).

Author: B.G (07/2026)
"""

from typing import Any

import numpy as np

from .compile import HelperBuilder, KernelBuilder
from .need import Kind, Need


def backend_classes(backend: str):
    """
    (backend_module_or_None, ParameterCls, HelperBuilderCls, dtypes) for one
    backend name.

    `backend_module_or_None` is `ti`/`qd` for the closure backends, `None` for
    cupy - cupy blocks call plain C, never a bound backend module. `dtypes`
    maps "i32"/"i64"/"f32"/"u8"/"u32" to that backend's own dtype objects
    (ti.*/qd.* for the closure backends, numpy dtypes for cupy) - every name
    make_grid/make_noise/make_depressions currently needs ("i64" for
    depressions' bitpacked basin_saddle/outlet buffers), plus the obvious
    siblings.

    No blocks module is returned - each factory (grid, noise, ...) has its own
    private block module and picks it itself.

    Author: B.G (07/2026)
    """
    if backend == "taichi":
        import taichi as ti

        from .taichi_backend import TaichiHelperBuilder, TaichiParameter

        return ti, TaichiParameter, TaichiHelperBuilder, {
            "i32": ti.i32, "i64": ti.i64, "f32": ti.f32, "u8": ti.u8, "u32": ti.u32,
        }
    if backend == "quadrants":
        import quadrants as qd

        from .quadrants_backend import QuadrantsHelperBuilder, QuadrantsParameter

        return qd, QuadrantsParameter, QuadrantsHelperBuilder, {
            "i32": qd.i32, "i64": qd.i64, "f32": qd.f32, "u8": qd.u8, "u32": qd.u32,
        }
    if backend == "cupy":
        from .cupy_backend import CupyHelperBuilder, CupyParameter

        return None, CupyParameter, CupyHelperBuilder, {
            "i32": np.int32, "i64": np.int64, "f32": np.float32, "u8": np.uint8, "u32": np.uint32,
        }
    raise ValueError(f"unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")


def make_helper(HelperCls, template, *, strict_needs: bool = False, **binds: Any) -> HelperBuilder:
    """
    A HelperBuilder ingesting `template` with every entry of `binds` applied.

    The assembly every block module's build_helpers() needs for each of its
    blocks, so a new block module does not carry its own copy.

    `strict_needs`, forwarded to `HelperCls(strict_needs=...)` (see
    compile.py, CompileBuilder), is what a factory converting its own wiring
    to Need turns on; default False keeps every existing caller's permissive
    `bind(name, obj)` behaviour unchanged. A `binds` entry whose value is
    already a Need (need.py) - typically built via param_need()/helper_need()
    just below - is declared on the builder via `.need()` and then bound
    through that Need's own `.value`, the declare-then-bind sequence a
    strict_needs=True builder requires; any other value is bound directly,
    exactly as before Need existed. This dispatch runs the same whether or
    not `strict_needs` is set, so a caller may mix Need and raw bindings on
    one builder freely - `strict_needs` only changes what a *raw*, non-Need
    binding requires (see CompileBuilder._bind_raw).

    Author: B.G (07/2026)
    """
    builder = HelperCls(strict_needs=strict_needs).ingest(template)
    for name, obj in binds.items():
        if isinstance(obj, Need):
            builder.need(obj)
            builder.bind(name, obj.value)
        else:
            builder.bind(name, obj)
    return builder


def make_kernel(KernelCls, template, *, strict_needs: bool = False, **binds: Any) -> KernelBuilder:
    """
    make_helper's twin for kernels: a KernelBuilder ingesting `template` with
    every entry of `binds` applied, same dispatch (a Need value is declared
    then bound through; anything else binds directly), same `strict_needs`
    meaning. Not compiled here - call .compile() on the result, same as a
    KernelBuilder built by hand.

    Author: B.G (08/2026)
    """
    builder = KernelCls(strict_needs=strict_needs).ingest(template)
    for name, obj in binds.items():
        if isinstance(obj, Need):
            builder.need(obj)
            builder.bind(name, obj.value)
        else:
            builder.bind(name, obj)
    return builder


def param_need(name: str, param) -> Need:
    """
    A fresh `Need(name, kind=Kind.PARAM)`, bound to `param`, with dtype/modes
    taken from `param`'s own current dtype/mode.

    For a factory's internal wiring converted to strict Needs (see
    make_helper): each call builds and binds a brand-new Need, never reuses
    one across bind sites - Need.bind() freezes a kind=PARAM Need after one
    bind (need.py), so the same Need object cannot back two different
    `param_need()` call sites the way flow's own caller-facing Needs are
    meant to be shared by identity; this constructs a distinct, single-use
    Need per site instead, mirroring the existing `seed_n = Need(...);
    seed_n.bind(...)` pattern in flow/_closure_receivers.py's build_rand_unit.

    Author: B.G (08/2026)
    """
    return Need(name, kind=Kind.PARAM, dtype=param.dtype, modes={param.mode}).bind(param)


def helper_need(name: str, helper: HelperBuilder) -> Need:
    """
    A fresh `Need(name, kind=Kind.HELPER)`, bound to `helper`. See
    param_need's docstring - same single-use-per-site reasoning applies.

    Author: B.G (08/2026)
    """
    return Need(name, kind=Kind.HELPER).bind(helper)


def bag_need(name: str, bag, contains) -> Need:
    """
    A fresh `Need(name, kind=Kind.BAG, contains=contains)`, bound to `bag`.

    For binding a whole Bag under one name (e.g. `GRID=grid` in
    noise/_closure_blocks.py, where a template reaches it by dotted path -
    `GRID.nx.get(0)`) rather than one Parameter/HelperBuilder - see need.py's
    module docstring for what `contains` (a list of sub-Needs the bag must
    satisfy by member name) checks. Unlike param_need/helper_need, the
    sub-Needs in `contains` are supplied by the caller rather than derived
    automatically - which of a Bag's members a given template actually reads
    varies per bind site (row/col here only ever read `grid.nx`; a Perlin
    `at` also reads `grid.ny`), so there is no single "whatever this bag
    currently holds" contract to infer the way a lone Parameter's dtype/mode
    can be read straight off it.

    Author: B.G (08/2026)
    """
    return Need(name, kind=Kind.BAG, contains=list(contains)).bind(bag)
