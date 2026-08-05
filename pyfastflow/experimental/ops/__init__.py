"""
Small device-op Bag factories built on the backend-agnostic core (see
../core/context: parameter.py for Parameter, compile.py for HelperBuilder,
bag.py for Bag) - bit packing, missing math, generic flat-buffer kernels, a
grid-aware slope helper, a parallel inclusive scan / stream compaction, and
plain reductions.

Like make_grid/make_noise there is no stateful context class: every factory
below builds its Bag (or, for make_scan/make_reduce, a small plain object
holding one) once and hands it back. Two kinds of factory:

  Helper-only (make_bitpack, make_math, make_slope, make_block_reduce)
    returns a Bag of HelperBuilders - device-only recipes, bound into
    whatever kernel needs them, never compiled standalone (see
    ../core/context/compile.py, HelperBuilder).
  Kernel-builder (make_elementwise)
    returns a Bag of unbuilt KernelBuilders - call .compile() on the member
    you want before launching it.
  Host-orchestrated (make_scan, make_reduce)
    returns a plain object wrapping one or more compiled Kernels/Routines
    plus device-side 0-d scalar Parameters for their results - not
    representable as a single Bag of interchangeable members, since calling
    them runs a short host-side sequence (see Scan/Reduce below).

make_scan and make_reduce both read their result off a device Parameter with
no host sync (for another kernel to bind and read via .get(0)) and also
expose a syncing host getter - the caller picks which cost to pay, per call.

cupy backend notes
-------------------
`.inclusive` on cupy is `cp.cumsum`, not a hand-rolled Blelloch scan: CUB's
own DeviceScan is already the accelerator cupy dispatches to by default on
this build (`cupy._core._accelerator.get_reduction_accelerators()` and
`get_routine_accelerators()` both return `[1]` == CUB, cupy 14.1.1), so
reimplementing the tree here would only reproduce what cupy already calls.
make_reduce's sum/min/max/argmin are `cp.sum`/`cp.min`/`cp.max`/`cp.argmin`
for the same reason. Both write their device Parameter via a plain cupy
device-to-device slice assignment (`param.get().data[...] = result`), which
issues no host sync.

Taichi/Quadrants build their own Blelloch up-sweep/down-sweep via an
internal RoutineBuilder compiled unfused, so every step is its own kernel
launch and the global barrier each one needs is real, and their own
atomic-accumulate kernels for reduce - see _closure_blocks.py.

_closure_blocks.py/_cupy_blocks.py's own internal wiring goes through a Need
(need.py) now, every HelperBuilder/KernelBuilder built `strict_needs=True`
(compile.py) - the third factory converted, after grid/ and noise/, and the
first with real KernelBuilder construction of its own (make_kernel,
make_helper's twin, is exercised here for the first time - see
backends.py). build_scan_routine's RoutineBuilder is never given a
bind_bag() bag - none of its KernelBuilders bind anything at all (see
_closure_blocks.py's build_scan_routine).
build_bitpack now returns a Bag rather than a plain dict - make_reduce reads
its own internal use of it (pack/unpack_index) the same way make_bitpack's
external caller would. Internal only - every factory's own signature and
every Bag/Scan/Reduce's member names/types are unchanged.

make_scan's/make_reduce's returned Parameters (Scan.count_param,
Reduce.{sum,min,max,argmin}_param) stay bare, not wrapped in a Need - a
deliberate decision, not an oversight. Need's contract models a caller
building an object and handing it to a factory that binds it later at a
distance (see need.py's module docstring); this is the opposite direction,
the factory building the object and handing it to a caller who may or may
not ever bind it anywhere. Wrapping it here would check nothing (there is no
bind-site yet for a mismatch to be caught at) and would just be a returned
Need standing in for a returned Parameter with the same information. A
caller wanting to bind one of these into its own strict_needs=True builder
already can, today, with no change here: param_need(name, scan.count_param)
constructs exactly the Need such a builder needs, the same way a caller
already wraps any other bare Parameter it receives (e.g. grid.nx) - nothing
about make_scan/make_reduce is different from any other Bag member in this
respect, Scan/Reduce just are not Bags.

Author: B.G (07/2026)
"""

import functools

import numpy as np

from ..core.context.backends import backend_classes
from ..core.context.bag import Bag


def _blocks_for(backend: str):
    """
    The private block module implementing one of this package's factories
    for a given backend name: the closure blocks (shared by Taichi and
    Quadrants) or the cupy blocks.

    Author: B.G (07/2026)
    """
    if backend in ("taichi", "quadrants"):
        from . import _closure_blocks as blocks
    elif backend == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"ops: unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")
    return blocks


# ---------------------------------------------------------------------------
# bitpack / math / elementwise / slope
# ---------------------------------------------------------------------------


def make_bitpack(backend: str) -> Bag:
    """
    pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32: pack a
    float and an int32 index into one i64 so that an atomic_min over the
    packed value behaves as a lexicographic argmin over (value, index) - see
    _closure_blocks.build_bitpack / _cupy_blocks.build_bitpack.

    build_bitpack already returns a Bag (both backends) - see its own
    docstring for why (its result is also threaded internally by
    make_reduce, not just returned here) - so this returns it directly.

    Author: B.G (07/2026)
    """
    _, _, HelperCls, _ = backend_classes(backend)
    blocks = _blocks_for(backend)
    return blocks.build_bitpack(HelperCls)


def make_math(backend: str) -> Bag:
    """
    atan(x) and nextafter(x, y) (f32), filling in for the two functions
    Taichi/Quadrants/CUDA device code has no direct equivalent for - see
    _closure_blocks.build_math / _cupy_blocks.build_math.

    Author: B.G (07/2026)
    """
    _, _, HelperCls, _ = backend_classes(backend)
    blocks = _blocks_for(backend)
    helpers = blocks.build_math(HelperCls)
    return Bag(helpers)


def make_elementwise(backend: str) -> Bag:
    """
    swap, add_B_to_A, add_B_to_weighted_A, weighted_mean_B_in_A, arange,
    multiply_by_scalar over a flat buffer, as unbuilt KernelBuilders - call
    .compile() on the member you want. Buffers (and, on cupy, the buffer
    length) are kernel call arguments, not bound Parameters - see
    parameter.py, "Data at call time, configuration at compile time".

    Author: B.G (07/2026)
    """
    backend_mod, _, _, _ = backend_classes(backend)
    KernelCls = _kernel_cls(backend)
    blocks = _blocks_for(backend)
    if backend == "cupy":
        kernels = blocks.build_elementwise(KernelCls)
    else:
        kernels = blocks.build_elementwise(KernelCls, backend, backend_mod)
    return Bag(kernels)


def make_slope(backend: str, grid: Bag) -> Bag:
    """
    sumslope_downstream(z, i): sum of (z[i]-z[j])/dx over every downstream
    neighbour of i. slope_dir(z, i, k): the signed slope towards neighbour
    k, 0 where there is none. Both walk `grid`'s own
    neighbour/dx/n_neighbours surface (see _closure_blocks.build_slope /
    _cupy_blocks.build_slope) rather than a hardcoded 4-neighbour loop, so
    they follow whatever topology/boundary/nodata `grid` was built with.

    Author: B.G (07/2026)
    """
    _, _, HelperCls, _ = backend_classes(backend)
    blocks = _blocks_for(backend)
    helpers = blocks.build_slope(HelperCls, grid)
    return Bag(helpers)


def make_block_reduce(backend: str) -> Bag:
    """
    cupy only: `sum(val)`, one cub::BlockReduce<float, 128>::Sum() per
    calling CUDA block - see _cupy_blocks.build_block_reduce. Raises on
    Taichi/Quadrants, which have no block-level primitive this wraps.

    The first compile that reaches this triggers a one-time jitify header
    cache warm-up for <cub/block/block_reduce.cuh>, roughly two minutes; that
    is expected, not a hang.

    Author: B.G (07/2026)
    """
    if backend != "cupy":
        raise ValueError(f"make_block_reduce: only supported on cupy, got {backend!r}")
    _, _, HelperCls, _ = backend_classes(backend)
    from . import _cupy_blocks as blocks

    return Bag(blocks.build_block_reduce(HelperCls))


def _kernel_cls(backend: str):
    """
    The KernelBuilder class for `backend` - not exposed by backend_classes(),
    which only returns HelperBuilder, so factories that build kernels
    (make_elementwise, make_scan, make_reduce) look it up here instead.

    Author: B.G (07/2026)
    """
    if backend == "taichi":
        from ..core.context.taichi_backend import TaichiKernelBuilder

        return TaichiKernelBuilder
    if backend == "quadrants":
        from ..core.context.quadrants_backend import QuadrantsKernelBuilder

        return QuadrantsKernelBuilder
    if backend == "cupy":
        from ..core.context.cupy_backend import CupyKernelBuilder

        return CupyKernelBuilder
    raise ValueError(f"unknown backend {backend!r}")


def _routine_cls(backend: str):
    """
    The RoutineBuilder class for `backend` - only Taichi/Quadrants need one
    (make_scan's Blelloch tree); cupy's scan is cp.cumsum, no routine
    involved.

    Author: B.G (07/2026)
    """
    if backend == "taichi":
        from ..core.context.taichi_backend import TaichiRoutineBuilder

        return TaichiRoutineBuilder
    if backend == "quadrants":
        from ..core.context.quadrants_backend import QuadrantsRoutineBuilder

        return QuadrantsRoutineBuilder
    raise ValueError(f"unknown closure backend {backend!r}")


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


class Scan:
    """
    Inclusive prefix-sum and flag-based stream compaction over a fixed-size
    i32 buffer, as a plain python object rather than a Routine (see
    ../ops/__init__.py's module docstring) - built once by make_scan for one
    backend/size, reused for every call.

    `.inclusive(input_handle, output_handle)` fills `output_handle` with the
    inclusive prefix sum of `input_handle`. `.compact(flags_handle,
    ids_handle)` scatters the indices where `flags_handle` is nonzero into
    `ids_handle[0:count]` and returns `count` as a python int - the legacy
    parallel_scan_compact semantics.

    `count_param` is a device-side 0-d scalar Parameter holding the count
    from the most recent `.compact()` call, readable by another kernel via
    `.get(0)` with no host sync; `.count()` is the syncing host equivalent.
    Both exist because the caller, not this object, knows which cost is
    worth paying at a given call site.

    Author: B.G (07/2026)
    """

    def __init__(self, inclusive_fn, compact_fn, count_param):
        self._inclusive_fn = inclusive_fn
        self._compact_fn = compact_fn
        self.count_param = count_param

    def inclusive(self, input_handle, output_handle) -> None:
        self._inclusive_fn(input_handle, output_handle)

    def compact(self, flags_handle, ids_handle) -> int:
        return self._compact_fn(flags_handle, ids_handle)

    def count(self) -> int:
        return int(self.count_param.read())


def make_scan(backend: str, pool, n: int) -> Scan:
    """
    Build one Scan over i32 buffers of length `n` (compaction's `ids`/`flags`
    are naturally i32-shaped, and the internal work buffer needs a fixed
    dtype picked once at build time - see the class docstring).

    Taichi/Quadrants: `.inclusive` drives an internal Blelloch
    RoutineBuilder over a `next_pow2(n)` work buffer, allocated once here -
    see _closure_blocks.build_scan_routine, and its docstring for why every
    step is its own kernel. cupy: `.inclusive` is `cp.cumsum` - see this
    module's docstring.

    Author: B.G (07/2026)
    """
    _, ParamCls, _, dtypes = backend_classes(backend)
    KernelCls = _kernel_cls(backend)

    if backend == "cupy":
        import cupy as cp

        from . import _cupy_blocks as blocks

        scan_out_h = pool.get_data(np.int32, (n,))
        count_p = ParamCls("SCAN_COUNT", dtype=dtypes["i32"], mode="scalar", value=0, pool=pool)
        scatter_kernel = blocks.build_scatter_kernel(KernelCls).compile()
        block = 256
        grid_dim = (n + block - 1) // block

        def inclusive_fn(input_handle, output_handle):
            cp.cumsum(input_handle.data, out=output_handle.data)

        def compact_fn(flags_handle, ids_handle):
            cp.cumsum(flags_handle.data, out=scan_out_h.data)
            count_p.get().data[...] = scan_out_h.data[n - 1 : n]
            count = int(count_p.read())
            if count <= 0:
                return 0
            scatter_kernel(flags_handle.data, scan_out_h.data, ids_handle.data, n, grid=grid_dim, block=block)
            return count

        return Scan(inclusive_fn, compact_fn, count_p)

    # Taichi / Quadrants
    backend_mod, _, _, _ = backend_classes(backend)
    RoutineBuilderCls = _routine_cls(backend)
    blocks = _blocks_for(backend)
    work_size = blocks.next_pow2(n)

    routine, scan_in_default, work_h, scan_out_default = blocks.build_scan_routine(
        RoutineBuilderCls, KernelCls, backend, backend_mod, pool, n, work_size
    )
    count_p = ParamCls("SCAN_COUNT", dtype=dtypes["i32"], mode="scalar", value=0, pool=pool)
    read_count_kernel, scatter_kernel = blocks.build_count_and_scatter_kernels(
        KernelCls, backend, backend_mod, count_p, n
    )
    scan_out_scratch = pool.get_data(dtypes["i32"], (n,))

    def inclusive_fn(input_handle, output_handle):
        routine(input_handle.data, work_h.data, output_handle.data)

    def compact_fn(flags_handle, ids_handle):
        routine(flags_handle.data, work_h.data, scan_out_scratch.data)
        read_count_kernel(scan_out_scratch.data)
        count = int(count_p.read())
        if count <= 0:
            return 0
        scatter_kernel(flags_handle.data, scan_out_scratch.data, ids_handle.data)
        return count

    return Scan(inclusive_fn, compact_fn, count_p)


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


class Reduce:
    """
    sum/min/max/argmin over a fixed-size f32 buffer, each backed by its own
    device-side 0-d scalar Parameter (readable by another kernel via
    `.get(0)` with no host sync) plus a syncing host getter - built once by
    make_reduce for one backend/size, reused for every call.

    Every parameter holds the same thing on every backend - `argmin_param` a
    bare i64 index, not the packed (value, index) pair the Taichi/Quadrants
    reduction accumulates into on its way there - so a kernel reading one
    needs to know nothing about which backend produced it.

    Author: B.G (07/2026)
    """

    def __init__(self, sum_p, min_p, max_p, argmin_p, run, host):
        self.sum_param = sum_p
        self.min_param = min_p
        self.max_param = max_p
        self.argmin_param = argmin_p
        self._run = run
        self._host = host

    def sum(self, handle) -> None:
        self._run["sum"](handle)

    def min(self, handle) -> None:
        self._run["min"](handle)

    def max(self, handle) -> None:
        self._run["max"](handle)

    def argmin(self, handle) -> None:
        self._run["argmin"](handle)

    def sum_value(self) -> float:
        return self._host["sum"]()

    def min_value(self) -> float:
        return self._host["min"]()

    def max_value(self) -> float:
        return self._host["max"]()

    def argmin_value(self) -> int:
        return self._host["argmin"]()


def make_reduce(backend: str, pool, n: int) -> Reduce:
    """
    Build one Reduce over f32 buffers of length `n`.

    cupy: each op is `cp.sum`/`cp.min`/`cp.max`/`cp.argmin`, written into its
    device Parameter via an async device-to-device copy - see this module's
    docstring. Taichi/Quadrants: one atomic-accumulate kernel per op (sum via
    Taichi/Quadrants' automatic atomic `+=`, min/max via atomic_min/max,
    argmin via atomic_min over ops.bitpack's packed (value, index) i64 - see
    _closure_blocks.build_reduce_kernels), each preceded by writing the
    identity element into that op's Parameter from the host.

    Author: B.G (07/2026)
    """
    _, ParamCls, HelperCls, dtypes = backend_classes(backend)

    if backend == "cupy":
        import cupy as cp

        sum_p = ParamCls("REDUCE_SUM", dtype=dtypes["f32"], mode="scalar", value=0.0, pool=pool)
        min_p = ParamCls("REDUCE_MIN", dtype=dtypes["f32"], mode="scalar", value=float("inf"), pool=pool)
        max_p = ParamCls("REDUCE_MAX", dtype=dtypes["f32"], mode="scalar", value=float("-inf"), pool=pool)
        argmin_p = ParamCls("REDUCE_ARGMIN", dtype=np.int64, mode="scalar", value=0, pool=pool)

        def run_sum(handle):
            sum_p.get().data[...] = cp.sum(handle.data)

        def run_min(handle):
            min_p.get().data[...] = cp.min(handle.data)

        def run_max(handle):
            max_p.get().data[...] = cp.max(handle.data)

        def run_argmin(handle):
            argmin_p.get().data[...] = cp.argmin(handle.data).astype(cp.int64)

        run = {"sum": run_sum, "min": run_min, "max": run_max, "argmin": run_argmin}
        host = {
            "sum": lambda: float(sum_p.get().data.get()),
            "min": lambda: float(min_p.get().data.get()),
            "max": lambda: float(max_p.get().data.get()),
            "argmin": lambda: int(argmin_p.get().data.get()),
        }
        return Reduce(sum_p, min_p, max_p, argmin_p, run, host)

    # Taichi / Quadrants
    backend_mod, _, _, _ = backend_classes(backend)
    KernelCls = _kernel_cls(backend)
    blocks = _blocks_for(backend)

    sum_p = ParamCls("REDUCE_SUM", dtype=dtypes["f32"], mode="scalar", value=0.0, pool=pool)
    min_p = ParamCls("REDUCE_MIN", dtype=dtypes["f32"], mode="scalar", value=float("inf"), pool=pool)
    max_p = ParamCls("REDUCE_MAX", dtype=dtypes["f32"], mode="scalar", value=float("-inf"), pool=pool)
    argmin_p = ParamCls("REDUCE_ARGMIN", dtype=backend_mod.i64, mode="scalar", value=0, pool=pool)
    # internal: the atomic_min accumulator, holding a packed (value, index)
    # pair that argmin_unpack_kernel resolves into argmin_p's bare index.
    argmin_packed_p = ParamCls("REDUCE_ARGMIN_PACKED", dtype=backend_mod.i64, mode="scalar", value=0, pool=pool)

    bitpack = blocks.build_bitpack(HelperCls)
    sum_kernel, min_kernel, max_kernel, argmin_kernel, argmin_unpack_kernel = blocks.build_reduce_kernels(
        KernelCls,
        backend,
        backend_mod,
        bitpack.pack,
        bitpack.unpack_index,
        sum_p.get().data,
        min_p.get().data,
        max_p.get().data,
        argmin_packed_p.get().data,
        argmin_p.get().data,
        n,
    )

    # the argmin identity is pack(+inf, 0): always beaten by any real value.
    _argmin_identity = _closure_pack_identity()

    def run_sum(handle):
        sum_p.set(0.0)
        sum_kernel(handle.data)

    def run_min(handle):
        min_p.set(float("inf"))
        min_kernel(handle.data)

    def run_max(handle):
        max_p.set(float("-inf"))
        max_kernel(handle.data)

    def run_argmin(handle):
        argmin_packed_p.set(_argmin_identity)
        argmin_kernel(handle.data)
        argmin_unpack_kernel()

    run = {"sum": run_sum, "min": run_min, "max": run_max, "argmin": run_argmin}
    host = {
        "sum": lambda: float(sum_p.get().to_numpy()),
        "min": lambda: float(min_p.get().to_numpy()),
        "max": lambda: float(max_p.get().to_numpy()),
        "argmin": lambda: int(argmin_p.get().to_numpy()),
    }
    return Reduce(sum_p, min_p, max_p, argmin_p, run, host)


def _closure_pack_identity() -> int:
    """
    pack(+inf, 0), computed host-side with the same bit arithmetic as
    _closure_blocks._pack_tmpl - the identity element atomic_min starts
    argmin's accumulator field at, so any real (value, index) pair wins.

    Author: B.G (07/2026)
    """
    f = np.float32(float("inf"))
    u = int(f.view(np.uint32))
    # +inf has a clear sign bit, so flip_float_bits' positive branch applies:
    # invert every bit.
    f_enc = np.uint32(~np.uint32(u))
    i_enc = np.uint32(0)
    packed = (np.int64(f_enc) << np.int64(32)) | np.int64(i_enc)
    flipped_upper = (~packed) & (np.int64(0xFFFFFFFF) << np.int64(32))
    unchanged_lower = packed & np.int64(0xFFFFFFFF)
    return int(flipped_upper | unchanged_lower)
