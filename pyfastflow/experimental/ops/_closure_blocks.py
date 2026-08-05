"""
Taichi/Quadrants (closure) block templates behind ops's make_* factories.

Mirrors the split ../grid/_closure_blocks.py and ../noise/_closure_blocks.py
already use: every private block is a plain python def, picked - never
branched on inside one function body - by the build_* function that wires it,
and every public helper binds the private blocks it needs BY NAME so a block
reached from two composites is specialized once per compile and shared at
both call sites (see ../core/context/compile.py, _SpecializeCtx).

`_BK` is the bound backend module (ti or qd) - both expose the same bare
names (bit_cast, select, cast, atan2, atomic_min, ...), so one template body
serves both backends without branching on which one it is.

Kernel templates (make_elementwise, make_scan, make_reduce) differ from
helpers in one respect: a kernel's data arguments need a real type
annotation, and Taichi/Quadrants spell that differently (`ti.template()` vs
`qd.Tensor`) - see _tensor_annotation. A scan step's stride is never bound at
all: the template factory closes over it as a plain python int, so it is baked
in without going through bind()/rebind() - see build_scan_routine.

Every bind in this module goes through a Need (param_need/helper_need/
bag_need, see backends.py) and every HelperBuilder/KernelBuilder is
constructed strict_needs=True - the first module converted with real
KernelBuilder use (make_helper's twin, make_kernel, is exercised here for
the first time). build_scan_routine's RoutineBuilder is never given a
bind_bag() bag - none of its KernelBuilders bind anything at all (see that
function's own note); only the KernelBuilders it is built from are
converted. build_bitpack returns a Bag rather than a plain dict, since its result
is threaded internally (build_reduce_kernels' pack/unpack_index arguments)
as well as returned publicly by make_bitpack.

Author: B.G (07/2026)
"""

import functools

import numpy as np

from ..core.context.backends import bag_need, helper_need, make_helper, make_kernel, param_need
from ..core.context.bag import Bag
from ..core.context.need import Kind, Need
from ..core.context.routine import RoutineBuilder

# ---------------------------------------------------------------------------
# bitpack
# ---------------------------------------------------------------------------


def _flip_float_bits_tmpl(f):
    u = _BK.bit_cast(f, _BK.u32)
    return _BK.select(u & _BK.u32(0x80000000) != 0, u ^ _BK.u32(0x80000000), ~u)


def _unflip_float_bits_tmpl(u):
    restored = _BK.select(u & _BK.u32(0x80000000) != 0, ~u, u ^ _BK.u32(0x80000000))
    return _BK.bit_cast(restored, _BK.f32)


def _pack_tmpl(f, i):
    f_enc = _FLIP(f)
    i_enc = _BK.bit_cast(i, _BK.u32)
    packed = (_BK.cast(f_enc, _BK.i64) << 32) | _BK.cast(i_enc, _BK.i64)
    flipped_upper = (~packed) & (_BK.i64(0xFFFFFFFF) << 32)
    unchanged_lower = packed & _BK.i64(0xFFFFFFFF)
    return flipped_upper | unchanged_lower


def _unpack_raw_tmpl(packed):
    flipped_upper = (~packed) & (_BK.i64(0xFFFFFFFF) << 32)
    unchanged_lower = packed & _BK.i64(0xFFFFFFFF)
    return flipped_upper | unchanged_lower


def _unpack_value_tmpl(packed):
    u = _UNPACKRAW(packed)
    f_enc = _BK.cast(u >> 32, _BK.u32)
    return _UNFLIP(f_enc)


def _unpack_index_tmpl(packed):
    u = _UNPACKRAW(packed)
    i_enc = _BK.cast(u & _BK.i64(0xFFFFFFFF), _BK.u32)
    return _BK.bit_cast(i_enc, _BK.i32)


def build_bitpack(HelperCls) -> Bag:
    """
    pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32: the
    IEEE-754 bit-flip trick that makes an i64 atomic_min double as a
    lexicographic argmin over (float, int).

    Every bind goes through a Need (helper_need, see backends.py) and every
    HelperBuilder is constructed strict_needs=True - see
    grid/_closure_blocks.py's build_helpers for the reference conversion.
    `_BK` needs no bind at all - auto-injected, see
    core/context/_closure_backend.py's module docstring.

    Returns a Bag, not a plain dict: make_bitpack returns this directly, and
    make_reduce's own internal reuse of it (build_reduce_kernels' pack_helper/
    unpack_index_helper arguments, below) reads it the same way, by attribute
    - the honest type for a named group of HelperBuilders that gets threaded
    internally as well as handed back to an outside caller.

    Author: B.G (07/2026)
    """
    mk = functools.partial(make_helper, HelperCls, strict_needs=True)
    flip = mk(_flip_float_bits_tmpl)
    unflip = mk(_unflip_float_bits_tmpl)
    pack = mk(_pack_tmpl, _FLIP=helper_need("_FLIP", flip))
    unpack_raw = mk(_unpack_raw_tmpl)
    unpack_value = mk(
        _unpack_value_tmpl,
        _UNPACKRAW=helper_need("_UNPACKRAW", unpack_raw),
        _UNFLIP=helper_need("_UNFLIP", unflip),
    )
    unpack_index = mk(_unpack_index_tmpl, _UNPACKRAW=helper_need("_UNPACKRAW", unpack_raw))
    return Bag({"pack": pack, "unpack_value": unpack_value, "unpack_index": unpack_index})


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def _atan_tmpl(x):
    return _BK.atan2(x, 1.0)


def _nextafter_tmpl(x, y):
    result = y
    if x != y:
        sign_mask = _BK.bit_cast(_BK.cast(-0.0, _BK.f32), _BK.u32)
        ix = _BK.bit_cast(x, _BK.u32)
        if x == 0.0:
            ix = (_BK.bit_cast(y, _BK.u32) & sign_mask) | _BK.cast(1, _BK.u32)
        elif (x > 0.0) == (y > x):
            ix += _BK.cast(1, _BK.u32)
        else:
            ix -= _BK.cast(1, _BK.u32)
        result = _BK.bit_cast(ix, _BK.f32)
    return result


def build_math(HelperCls):
    """
    atan(x) via atan2(x, 1); nextafter(x, y), one ULP of f32 towards y via
    IEEE-754 bit-twiddling (no libm nextafter on GPU).

    `strict_needs=True` for consistency (see build_bitpack); neither template
    references anything but `_BK`, which needs no bind - auto-injected.

    Author: B.G (07/2026)
    """
    mk = functools.partial(make_helper, HelperCls, strict_needs=True)
    atan = mk(_atan_tmpl)
    nextafter = mk(_nextafter_tmpl)
    return {"atan": atan, "nextafter": nextafter}


# ---------------------------------------------------------------------------
# elementwise (kernel builders, returned unbuilt)
# ---------------------------------------------------------------------------


def _tensor_annotation(backend_mod, backend: str):
    """
    The data-argument annotation a kernel template needs on this closure
    backend: `ti.template()` for Taichi, `qd.Tensor` for Quadrants (accepts a
    field or an ndarray - see quadrants_backend.py's module docstring).

    Author: B.G (07/2026)
    """
    return backend_mod.template() if backend == "taichi" else backend_mod.Tensor


def build_elementwise(KernelCls, backend: str, backend_mod):
    """
    swap/add_B_to_A/add_B_to_weighted_A/weighted_mean_B_in_A/arange/
    multiply_by_scalar over a flat buffer, as unbuilt KernelBuilders - the
    caller's own .compile() specializes each to whatever field/ndarray it is
    first launched against.

    Every KernelBuilder is constructed strict_needs=True (see grid/
    _closure_blocks.py's build_helpers for the reference conversion) though
    none of these six actually bind anything - `swap`'s `_BK.grouped(...)`
    needs no explicit bind either, auto-injected like every other closure
    template's `_BK`. `backend_mod` stays a plain argument here for
    `_tensor_annotation` - the one use Stage 1's auto-injection does not
    cover.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    F = backend_mod.f32

    def swap_template(array1: T, array2: T):
        for idx in _BK.grouped(array1):
            temp = array1[idx]
            array1[idx] = array2[idx]
            array2[idx] = temp

    def add_B_to_A_template(array1: T, array2: T):
        for i in array1:
            array1[i] += array2[i]

    def add_B_to_weighted_A_template(array1: T, array2: T, weight: F):
        for i in array1:
            array1[i] += array2[i] * weight

    def weighted_mean_B_in_A_template(array1: T, array2: T, weight: F):
        for i in array1:
            array1[i] = array2[i] * weight + array1[i] * (1 - weight)

    def arange_template(array: T):
        for i in array:
            array[i] = i

    def multiply_by_scalar_template(A: T, scalar: T):
        for i in A:
            A[i] *= scalar

    mkk = functools.partial(make_kernel, KernelCls, strict_needs=True)
    return {
        "swap": mkk(swap_template),
        "add_B_to_A": mkk(add_B_to_A_template),
        "add_B_to_weighted_A": mkk(add_B_to_weighted_A_template),
        "weighted_mean_B_in_A": mkk(weighted_mean_B_in_A_template),
        "arange": mkk(arange_template),
        "multiply_by_scalar": mkk(multiply_by_scalar_template),
    }


# ---------------------------------------------------------------------------
# slope
# ---------------------------------------------------------------------------


def _sumslope_downstream_tmpl(z, i):
    sumslope = 0.0
    for k in range(_GRID.n_neighbours.get(0)):
        j = _GRID.neighbour(i, k)
        if j > -1:
            if z[j] < z[i]:
                sumslope += (z[i] - z[j]) / _GRID.dx.get(0)
    return sumslope


def _slope_dir_tmpl(z, i, k):
    j = _GRID.neighbour(i, k)
    slope = 0.0
    if j > -1:
        slope = (z[i] - z[j]) / _GRID.dx.get(0)
    return slope


def build_slope(HelperCls, grid: Bag):
    """
    sumslope_downstream(z, i): sum of (z[i]-z[j])/dx over every downstream
    neighbour. slope_dir(z, i, k): the signed slope towards neighbour k, 0
    where there is none. Both walk the grid's own neighbour/dx/n_neighbours
    surface, so they follow whatever topology/boundary/nodata `grid` was
    built with.

    `_GRID=grid` - a whole Bag bound under one name - goes through a
    Kind.BAG Need (bag_need, see backends.py), same as noise/_closure_blocks.
    py's `GRID=grid`; both templates read the same three members, so one
    `contains` list is built once and reused across the two `bag_need()`
    calls below (safe - a sub-Need is only ever read via `._check()`, never
    itself bound, so sharing the list does not risk the "frozen after one
    bind" issue a top-level Need would have).

    Author: B.G (07/2026)
    """
    mk = functools.partial(make_helper, HelperCls, strict_needs=True)
    grid_contains = [
        Need("n_neighbours", kind=Kind.PARAM, dtype=grid.n_neighbours.dtype, modes={grid.n_neighbours.mode}),
        Need("neighbour", kind=Kind.HELPER),
        Need("dx", kind=Kind.PARAM, dtype=grid.dx.dtype, modes={grid.dx.mode}),
    ]
    sumslope_downstream = mk(_sumslope_downstream_tmpl, _GRID=bag_need("_GRID", grid, contains=grid_contains))
    slope_dir = mk(_slope_dir_tmpl, _GRID=bag_need("_GRID", grid, contains=grid_contains))
    return {"sumslope_downstream": sumslope_downstream, "slope_dir": slope_dir}


# ---------------------------------------------------------------------------
# scan (Blelloch inclusive scan + flag compaction), i32 only
# ---------------------------------------------------------------------------


def next_pow2(n: int) -> int:
    """
    Smallest power of 2 >= n.

    Author: B.G (07/2026)
    """
    p = 1
    while p < n:
        p *= 2
    return p


def _make_copy_input_to_work_template(T, n, work_size):
    def copy_input_to_work_template(src: T, work: T):
        for i in range(work_size):
            if i < n:
                work[i] = src[i]
            else:
                work[i] = 0

    return copy_input_to_work_template


def _make_upsweep_template(T, work_size, stride):
    def upsweep_template(work: T):
        for i in range(work_size):
            if (i + 1) % (stride * 2) == 0:
                work[i] += work[i - stride]

    return upsweep_template


def _make_set_root_zero_template(T, root_index):
    def set_root_zero_template(work: T):
        work[root_index] = 0

    return set_root_zero_template


def _make_downsweep_template(T, work_size, stride):
    def downsweep_template(work: T):
        for i in range(work_size):
            if (i + 1) % (stride * 2) == 0:
                temp = work[i - stride]
                work[i - stride] = work[i]
                work[i] += temp

    return downsweep_template


def _make_inclusive_and_copy_template(T, n):
    def make_inclusive_and_copy_template(inp: T, work: T, out: T):
        for i in range(n):
            if i == 0:
                out[i] = inp[i]
            else:
                out[i] = work[i] + inp[i]

    return make_inclusive_and_copy_template


def _make_scatter_template(T, n):
    def scatter_template(flags: T, scan_out: T, ids: T):
        for i in range(n):
            if flags[i] == 1:
                pos = scan_out[i] - 1
                ids[pos] = i

    return scatter_template


def _make_read_count_template(T, n):
    def read_count_template(scan_out: T):
        COUNT.set_node(0, scan_out[n - 1])

    return read_count_template


def build_scan_routine(RoutineBuilderCls, KernelCls, backend: str, backend_mod, pool, n: int, work_size: int):
    """
    The Blelloch up-sweep/down-sweep routine over a size-`work_size` work
    buffer: copy-in, log2(work_size) up-sweep steps, zero the root,
    log2(work_size) down-sweep steps, convert to inclusive and copy out.

    Every step must be its own kernel launch: each step's reads depend on
    every write the previous one made, across the whole buffer, so the steps
    need a real global barrier between them. compile(fused=False) gives
    exactly that - one kernel per step, the plain launch sequence - and the
    split() after each add_kernel states the same requirement structurally,
    so the routine still compiles to one kernel per step if it is ever built
    down the fused path instead.

    Each step's stride/root-index is a plain python int closed over by the
    template factories above - never bound as a Parameter, so rebind() never
    has to reconcile one name meaning a different stride in two steps, which
    check_handles would reject outright.

    Returns the compiled Routine (data names "scan_in", "work", "scan_out",
    in that order) plus the pool handles for "scan_in"/"work"/"scan_out" it
    registered as defaults - the caller (build_scan) keeps the "work" handle
    and always overrides "scan_in"/"scan_out" at call time.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    dtype = backend_mod.i32

    scan_in_h = pool.get_data(dtype, (n,))
    work_h = pool.get_data(dtype, (work_size,))
    scan_out_h = pool.get_data(dtype, (n,))

    # No bind_bag() call: none of the KernelBuilders below bind anything at
    # all - each step's stride/root-index/n is a plain python int closed
    # over by the template factories above, never bound as a Parameter (see
    # this function's own docstring) - so there is nothing for a
    # routine-level bag to supply (see routine.py, RoutineBuilder._validate).
    # strict_needs=True is still applied to each KernelBuilder, below, for
    # consistency with every other converted build_* function.
    rb = RoutineBuilderCls()
    rb.add_data("scan_in", scan_in_h.data)
    rb.add_data("work", work_h.data)
    rb.add_data("scan_out", scan_out_h.data)

    copy_builder = KernelCls(strict_needs=True).ingest(_make_copy_input_to_work_template(T, n, work_size))
    rb.add_kernel(copy_builder, data_handle_ref=("scan_in", "work"))
    rb.split()

    stride = 1
    while stride < work_size:
        up_builder = KernelCls(strict_needs=True).ingest(_make_upsweep_template(T, work_size, stride))
        rb.add_kernel(up_builder, data_handle_ref=("work",))
        rb.split()
        stride *= 2

    zero_builder = KernelCls(strict_needs=True).ingest(_make_set_root_zero_template(T, work_size - 1))
    rb.add_kernel(zero_builder, data_handle_ref=("work",))
    rb.split()

    stride = work_size // 2
    while stride > 0:
        down_builder = KernelCls(strict_needs=True).ingest(_make_downsweep_template(T, work_size, stride))
        rb.add_kernel(down_builder, data_handle_ref=("work",))
        rb.split()
        stride //= 2

    inc_builder = KernelCls(strict_needs=True).ingest(_make_inclusive_and_copy_template(T, n))
    rb.add_kernel(inc_builder, data_handle_ref=("scan_in", "work", "scan_out"))
    rb.split()

    routine = rb.compile(fused=False)
    return routine, scan_in_h, work_h, scan_out_h


def build_count_and_scatter_kernels(KernelCls, backend: str, backend_mod, count_param, n: int):
    """
    read_count(scan_out): writes scan_out[n-1] into `count_param`, device
    side, no host sync. scatter(flags, scan_out, ids): for flags[i]==1,
    ids[scan_out[i]-1] = i - the scatter half of scan-based compaction.

    `COUNT=count_param` goes through param_need; strict_needs=True on both -
    see grid/_closure_blocks.py's build_helpers for the reference conversion.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    mkk = functools.partial(make_kernel, KernelCls, strict_needs=True)
    read_count = mkk(_make_read_count_template(T, n), COUNT=param_need("COUNT", count_param)).compile()
    scatter = mkk(_make_scatter_template(T, n)).compile()
    return read_count, scatter


# ---------------------------------------------------------------------------
# reduce (sum/min/max/argmin), i32 count / f32 value
# ---------------------------------------------------------------------------


def _make_sum_template(T, n):
    def sum_template(x: T):
        for i in range(n):
            _SUM[None] += x[i]

    return sum_template


def _make_min_template(T, n):
    def min_template(x: T):
        for i in range(n):
            _BK.atomic_min(_MIN[None], x[i])

    return min_template


def _make_max_template(T, n):
    def max_template(x: T):
        for i in range(n):
            _BK.atomic_max(_MAX[None], x[i])

    return max_template


def _make_argmin_template(T, n):
    def argmin_template(x: T):
        for i in range(n):
            packed = _PACK(x[i], i)
            _BK.atomic_min(_ARGMIN[None], packed)

    return argmin_template


def _make_argmin_unpack_template():
    def argmin_unpack_template():
        for _ in range(1):
            _ARGMIN_OUT[None] = _UNPACK_INDEX(_ARGMIN[None])

    return argmin_unpack_template


def build_reduce_kernels(
    KernelCls,
    backend: str,
    backend_mod,
    pack_helper,
    unpack_index_helper,
    sum_field,
    min_field,
    max_field,
    argmin_packed_field,
    argmin_field,
    n: int,
):
    """
    One compiled kernel per op, each accumulating atomically into its own
    raw field (bound directly, not through a Parameter's device_view, so the
    atomic ops the backend exposes on a field can be used as-is): sum by
    Taichi/Quadrants' automatic atomic `+=` reduction, min/max via
    atomic_min/atomic_max, argmin via atomic_min over the bitpack-encoded
    (value, index) pair (see _closure_blocks.build_bitpack).

    argmin gets a second, single-thread kernel that unpacks that pair and
    writes the bare index into `argmin_field`, so what Reduce exposes as
    `argmin_param` holds a plain index on every backend rather than an
    encoding a reader would have to know the backend to interpret. The
    packed accumulator stays internal.

    `_PACK`/`_UNPACK_INDEX` (HelperBuilders, from build_bitpack) go through
    helper_need; strict_needs=True on every kernel here - see
    grid/_closure_blocks.py's build_helpers for the reference conversion.
    `_SUM`/`_MIN`/`_MAX`/`_ARGMIN`/`_ARGMIN_OUT` are raw backend fields, not
    Parameters (bound directly so the backend's own atomic ops apply to them,
    per this function's own docstring above) - a plain value with no
    dtype/mode for a Need to check, same category as ops's own `SQRT2`-style
    binds elsewhere in this package; strict_needs=True binds them unchanged
    (see compile.py's `_bind_raw`). `_BK` needs no bind at all - auto-injected.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    mkk = functools.partial(make_kernel, KernelCls, strict_needs=True)
    sum_kernel = mkk(_make_sum_template(T, n), _SUM=sum_field).compile()
    min_kernel = mkk(_make_min_template(T, n), _MIN=min_field).compile()
    max_kernel = mkk(_make_max_template(T, n), _MAX=max_field).compile()
    argmin_kernel = mkk(
        _make_argmin_template(T, n),
        _PACK=helper_need("_PACK", pack_helper),
        _ARGMIN=argmin_packed_field,
    ).compile()
    argmin_unpack_kernel = mkk(
        _make_argmin_unpack_template(),
        _UNPACK_INDEX=helper_need("_UNPACK_INDEX", unpack_index_helper),
        _ARGMIN=argmin_packed_field,
        _ARGMIN_OUT=argmin_field,
    ).compile()
    return sum_kernel, min_kernel, max_kernel, argmin_kernel, argmin_unpack_kernel
