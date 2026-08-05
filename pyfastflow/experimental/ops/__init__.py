"""
Small device-op factories built on the new builder/frozen/bound stack
(../core/context/builder.py, frozen.py, bound.py; see parameter.py for
Parameter, unchanged) - bit packing, a parallel inclusive scan / stream
compaction, and plain reductions. This pass ports `bitpack`, `scan` and
`reduce` only - `math`/`elementwise`/`slope`/`block_reduce` (the old stack's
other ops members) are out of scope for this pass and stay unported; nothing
here removes them from the old stack.

Three shapes, mirroring grid/noise/visu's own split
-------------------------------------------------------
`make_bitpack_group` (structure, no Parameters - bitpack needs none) returns
a FrozenGroup, composed by name into whatever kernel/helper needs
pack/unpack_value/unpack_index (`kb.compose("bitpack", bitpack_group)`,
`ctx.bitpack.pack(f, i)` in a template) - the same two-call structure/data
split grid/__init__.py's own docstring explains, just with an empty data half
here. `make_scan`/`make_reduce` are host-orchestrated, like the old stack:
each returns a plain python object (Scan/Reduce) wrapping one or more
compiled kernels/routines plus device-side 0-d scalar Parameters for their
results - not representable as a single composable structure, since calling
them runs a short host-side sequence. Unlike `make_bitpack_group`,
`make_scan`/`make_reduce` DO take a `pool` and hand Parameters back to the
caller - deliberately, and deliberately unlike `flow/`'s factories (which
take no pool at all): see the Phase 2b brief for why this split holds.

`make_scan`'s/`make_reduce`'s returned Parameters (`Scan.count_param`,
`Reduce.{sum,min,max,argmin}_param`) are handed back bare, not wrapped in
anything - there is no Need-shaped wrapper in this stack to reach for; a
caller wanting one bound into its own KernelBuilder does so exactly like any
other Parameter (`kb.wire_param("count"); bound.bind("count", scan.
count_param)`).

Reduce's DATA-not-PARAM accumulator
-------------------------------------
PARAM access in this stack is strict get()/set_node() (compile_shared.
check_legal_accessors) - a plain, non-atomic single write, never an atomic
accumulate. sum/min/max/argmin genuinely need atomic accumulation across
threads (`ctx.bk.atomic_min(acc[None], x[i])`, Taichi's automatic atomic `+=`
for sum), which needs the raw backend field/ndarray itself, not a Parameter's
device_view - so the running total is wired as a DATA slot on the kernel
(`acc`), not a PARAM one. The Parameter objects this module hands back to its
own caller (`sum_param`, ...) still own that exact storage - `sum_p.get().
data` is literally the array bound to the "acc" DATA address - only the
device-side wiring differs from what a PARAM slot would have looked like; see
_closure_blocks.py's own module docstring for the fuller reasoning.

Scan's cupy compaction: a genuine per-step `launch=` user
-------------------------------------------------------------
`.inclusive()` on cupy stays `cp.cumsum` (CUB's own DeviceScan is already the
accelerator cupy dispatches to by default on this build). Compaction's
count-read and scatter - previously a host-side numpy slice-copy plus one
directly-launched kernel - are now a 2-step FrozenRoutine (routine_v2.py):
"read_count" (1 thread) then "scatter" (ceil(n/block) blocks), each composed
with its own `launch=` override sized to its own real thread count - see
_cupy_blocks.build_count_and_scatter_routine's own docstring. This is a
resolved design fork, flagged rather than chosen silently: the coordinating
brief calls out routine_v2.py's per-step `launch=` as needing to be
"exercised" by scan, but scan's only genuinely multi-kernel pipeline
(Blelloch up-sweep/down-sweep) lives on Taichi/Quadrants, where
compile_closure.compile_kernel takes no extra kwargs at all - there is
nothing for a non-empty `launch=` to override there. cupy is the only
backend where launch dims mean anything, and cupy's own multi-kernel
surface, before this port, was two kernels total (count-read via a numpy
slice, one scatter launch) - reading the brief as asking for BOTH "multi-
kernel" and "meaningfully different launch=" on the SAME routine would mean
reimplementing the whole Blelloch tree in CUDA text for cupy too, which
directly contradicts this package's own settled reasoning for using
`cp.cumsum` (CUB) there in the first place - exactly the kind of silent
downgrade-to-dodge-a-collision the brief warns against, just inverted (here
it would be an unrequested *addition* of reimplemented work to manufacture a
"multi-kernel" routine that already has a perfectly good accelerator). This
port instead promotes the count-read (previously a bare numpy slice) into a
real 1-thread kernel and routes both steps through a real, 2-step
FrozenRoutine with two genuinely different, non-trivial launch shapes - a
smaller-scale but literal exercise of the mechanism, without touching
`.inclusive()`'s accelerator path. One consequence: scatter now always
launches on cupy, even when `count == 0` (Taichi/Quadrants' `compact_fn`
still short-circuits before its own scatter launch, since those two kernels
are NOT bundled into one Routine call and can be skipped independently) - a
harmless extra launch in the rare all-flags-zero case, not a correctness
change. Flagged here rather than silently accepted; happy to reshape if a
different reading of "exercise it" was intended.

Author: B.G (08/2026)
"""

import numpy as np

from ..core.context.backends import backend_classes


def _blocks_for(backend: str):
    """
    The private block module implementing one of this package's factories
    for a given backend name: the closure blocks (shared by Taichi and
    Quadrants) or the cupy blocks.

    Author: B.G (08/2026)
    """
    if backend in ("taichi", "quadrants"):
        from . import _closure_blocks as blocks
    elif backend == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"ops: unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")
    return blocks


def make_bitpack_group(backend: str) -> "FrozenGroup":
    """
    pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32: pack a
    float and an int32 index into one i64 so that an atomic_min over the
    packed value behaves as a lexicographic argmin over (value, index) -
    composed by name into whatever kernel/helper needs it
    (`kb.compose("bitpack", group)`, `ctx.bitpack.pack(f, i)`). No PARAM
    slots - see this module's own docstring.

    Author: B.G (08/2026)
    """
    return _blocks_for(backend).build_bitpack_group()


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


class Scan:
    """
    Inclusive prefix-sum and flag-based stream compaction over a fixed-size
    i32 buffer, as a plain python object rather than a Routine/Sequence -
    built once by make_scan for one backend/size, reused for every call.

    `.inclusive(input_handle, output_handle)` fills `output_handle` with the
    inclusive prefix sum of `input_handle`. `.compact(flags_handle,
    ids_handle)` scatters the indices where `flags_handle` is nonzero into
    `ids_handle[0:count]` and returns `count` as a python int.

    `count_param` is a device-side 0-d scalar Parameter holding the count
    from the most recent `.compact()` call, readable by another kernel via
    `.get(0)` with no host sync; `.count()` is the syncing host equivalent.

    Author: B.G (08/2026)
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
    Build one Scan over i32 buffers of length `n`.

    Taichi/Quadrants: `.inclusive` drives an internal Blelloch FrozenRoutine
    over a `next_pow2(n)` work buffer, allocated once here - see
    _closure_blocks.build_scan_routine. cupy: `.inclusive` is `cp.cumsum`;
    `.compact`'s count-read/scatter are a 2-step FrozenRoutine - see this
    module's own docstring and _cupy_blocks.build_count_and_scatter_routine.

    Author: B.G (08/2026)
    """
    _, ParamCls, _, dtypes = backend_classes(backend)
    blocks = _blocks_for(backend)

    if backend == "cupy":
        import cupy as cp

        scan_out_h = pool.get_data(np.int32, (n,))
        count_p = ParamCls("SCAN_COUNT", dtype=dtypes["i32"], mode="scalar", value=0, pool=pool)

        routine_frozen = blocks.build_count_and_scatter_routine(n)
        bound = routine_frozen.build()
        bound.bind("read_count.scan_out", scan_out_h.data)
        bound.bind("read_count.COUNT", count_p)
        bound.bind("scatter.flags", scan_out_h.data)  # placeholder, swapped every .compact() call
        bound.bind("scatter.scan_out", scan_out_h.data)
        bound.bind("scatter.ids", scan_out_h.data)  # placeholder, swapped every .compact() call
        compiled_routine = bound.compile("cupy")

        def inclusive_fn(input_handle, output_handle):
            cp.cumsum(input_handle.data, out=output_handle.data)

        def compact_fn(flags_handle, ids_handle):
            cp.cumsum(flags_handle.data, out=scan_out_h.data)
            compiled_routine.swap("scatter.flags", flags_handle.data)
            compiled_routine.swap("scatter.ids", ids_handle.data)
            compiled_routine()
            return int(count_p.read())

        return Scan(inclusive_fn, compact_fn, count_p)

    # Taichi / Quadrants
    backend_mod, _, _, _ = backend_classes(backend)
    work_size = blocks.next_pow2(n)
    work_h = pool.get_data(dtypes["i32"], (work_size,))
    scan_out_scratch = pool.get_data(dtypes["i32"], (n,))
    count_p = ParamCls("SCAN_COUNT", dtype=dtypes["i32"], mode="scalar", value=0, pool=pool)

    routine_frozen = blocks.build_scan_routine(backend, backend_mod, n, work_size)
    bound = routine_frozen.build()
    for name in routine_frozen.order:
        bound.bind(f"{name}.work", work_h.data)
    bound.bind("copy_in.src", work_h.data)  # placeholder, swapped every call
    bound.bind("inclusive_copy.inp", work_h.data)  # placeholder, swapped every call
    bound.bind("inclusive_copy.out", work_h.data)  # placeholder, swapped every call
    compiled_routine = bound.compile(backend)

    read_count_frozen, scatter_frozen = blocks.build_count_and_scatter_kernels(backend, backend_mod, n)
    read_count_bound = read_count_frozen.build()
    read_count_bound.bind("scan_out", scan_out_scratch.data)
    read_count_bound.bind("COUNT", count_p)
    read_count_compiled = read_count_bound.compile(backend)

    scatter_bound = scatter_frozen.build()
    scatter_bound.bind("flags", scan_out_scratch.data)  # placeholder, swapped every .compact() call
    scatter_bound.bind("scan_out", scan_out_scratch.data)
    scatter_bound.bind("ids", scan_out_scratch.data)  # placeholder, swapped every .compact() call
    scatter_compiled = scatter_bound.compile(backend)

    def inclusive_fn(input_handle, output_handle):
        compiled_routine.swap("copy_in.src", input_handle.data)
        compiled_routine.swap("inclusive_copy.inp", input_handle.data)
        compiled_routine.swap("inclusive_copy.out", output_handle.data)
        compiled_routine()

    def compact_fn(flags_handle, ids_handle):
        compiled_routine.swap("copy_in.src", flags_handle.data)
        compiled_routine.swap("inclusive_copy.inp", flags_handle.data)
        compiled_routine.swap("inclusive_copy.out", scan_out_scratch.data)
        compiled_routine()
        read_count_compiled()
        count = int(count_p.read())
        if count <= 0:
            return 0
        scatter_compiled.swap("flags", flags_handle.data)
        scatter_compiled.swap("ids", ids_handle.data)
        scatter_compiled()
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
    reduction accumulates into on its way there.

    Author: B.G (08/2026)
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
    device Parameter via an async device-to-device copy. Taichi/Quadrants:
    one atomic-accumulate FrozenKernel per op (sum via Taichi/Quadrants'
    automatic atomic `+=`, min/max via `ctx.bk.atomic_min`/`atomic_max`,
    argmin via atomic_min over ops.make_bitpack_group's packed (value, index)
    i64), each preceded by writing the identity element into that op's
    Parameter from the host. See this module's own docstring for why the
    running total is a DATA argument, not a PARAM slot.

    Author: B.G (08/2026)
    """
    _, ParamCls, _, dtypes = backend_classes(backend)

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
    blocks = _blocks_for(backend)

    sum_p = ParamCls("REDUCE_SUM", dtype=dtypes["f32"], mode="scalar", value=0.0, pool=pool)
    min_p = ParamCls("REDUCE_MIN", dtype=dtypes["f32"], mode="scalar", value=float("inf"), pool=pool)
    max_p = ParamCls("REDUCE_MAX", dtype=dtypes["f32"], mode="scalar", value=float("-inf"), pool=pool)
    argmin_p = ParamCls("REDUCE_ARGMIN", dtype=backend_mod.i64, mode="scalar", value=0, pool=pool)
    # internal: the atomic_min accumulator, holding a packed (value, index)
    # pair that argmin_unpack resolves into argmin_p's bare index.
    argmin_packed_p = ParamCls("REDUCE_ARGMIN_PACKED", dtype=backend_mod.i64, mode="scalar", value=0, pool=pool)

    bitpack_group = blocks.build_bitpack_group()
    sum_frozen, min_frozen, max_frozen, argmin_frozen, argmin_unpack_frozen = blocks.build_reduce_kernels(
        backend, backend_mod, bitpack_group, n
    )

    sum_bound = sum_frozen.build()
    sum_bound.bind("acc", sum_p.get().data)
    sum_bound.bind("x", sum_p.get().data)  # placeholder, swapped every call
    sum_compiled = sum_bound.compile(backend)

    min_bound = min_frozen.build()
    min_bound.bind("acc", min_p.get().data)
    min_bound.bind("x", min_p.get().data)  # placeholder, swapped every call
    min_compiled = min_bound.compile(backend)

    max_bound = max_frozen.build()
    max_bound.bind("acc", max_p.get().data)
    max_bound.bind("x", max_p.get().data)  # placeholder, swapped every call
    max_compiled = max_bound.compile(backend)

    argmin_bound = argmin_frozen.build()
    argmin_bound.bind("acc", argmin_packed_p.get().data)
    argmin_bound.bind("x", argmin_packed_p.get().data)  # placeholder, swapped every call
    argmin_compiled = argmin_bound.compile(backend)

    argmin_unpack_bound = argmin_unpack_frozen.build()
    argmin_unpack_bound.bind("packed_acc", argmin_packed_p.get().data)
    argmin_unpack_bound.bind("out", argmin_p.get().data)
    argmin_unpack_compiled = argmin_unpack_bound.compile(backend)

    _argmin_identity = _closure_pack_identity()

    def run_sum(handle):
        sum_p.set(0.0)
        sum_compiled.swap("x", handle.data)
        sum_compiled()

    def run_min(handle):
        min_p.set(float("inf"))
        min_compiled.swap("x", handle.data)
        min_compiled()

    def run_max(handle):
        max_p.set(float("-inf"))
        max_compiled.swap("x", handle.data)
        max_compiled()

    def run_argmin(handle):
        argmin_packed_p.set(_argmin_identity)
        argmin_compiled.swap("x", handle.data)
        argmin_compiled()
        argmin_unpack_compiled()

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
