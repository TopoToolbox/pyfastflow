"""
make_receivers: the SFD (single-flow-direction) receiver Bag factory, built
on the backend-agnostic core (see ..core.context) and on a grid Bag from
..grid.

Like make_grid/make_noise there is no stateful context class - make_receivers
builds a Bag once and hands it back: a `receivers` KernelBuilder plus the
distance/slope helpers it is made of, so a caller can recombine them into its
own kernel or routine rather than being stuck with only the compiled
receivers kernel.

    grid = make_grid("taichi", pool, nx, ny, dx, topology="D8")
    recv = make_receivers("taichi", grid, mode="steepest")
    receivers_kernel = recv.receivers.compile()
    receivers_kernel(z.data, rec.data)

`mode` ("steepest"|"stochastic") and `h_aware` (False: kernel takes (z, rec)
and slopes read h as 0; True: kernel takes (z, h, rec) and slopes use
(zi-zj)+(hi-hj)) each pick one of four kernel body variants at build time -
see _closure_blocks.py/_cupy_blocks.py's build_receivers. mode="stochastic"
additionally requires `seed_p`, a scalar or const Parameter the RNG hash
mixes in alongside the node index and neighbour direction (see rand_unit in
the block modules) - the host bumps it between calls for a fresh draw.

`diagonal_partition_correction` only changes anything on a D8 grid: it swaps
the corrected distance/slope helpers in for the grid's own dist_from_k/
dist_between_nodes, dividing the diagonal-neighbour distance by sqrt(2) (see
_closure_blocks.py's build_distance_slope_helpers for exactly which k values
count as diagonal and why). Off, or on a D4 grid, dist_from_k_corrected and
dist_between_nodes_corrected are simply the grid's own helpers, unchanged.

Bag members: `receivers` (KernelBuilder), `dist_from_k_corrected`,
`dist_between_nodes_corrected`, `slope_from_values_k`, `slope_between_nodes`
(HelperBuilders), plus `rand_unit` (HelperBuilder) only when
mode="stochastic".

A node with no downslope neighbour keeps `rec[i] = i` - a self-receiver, the
same convention a can_out (base level) node uses. This is the pit convention
depression handling later depends on.

rand_unit's hash is keyed on (node, k, seed) rather than (node, seed): legacy
draws once per (node, k) candidate inside the neighbour loop
(`ti.random()`), so a node-keyed hash would scale every candidate at a node
by the same factor and weaken the randomisation rather than reproduce it.
Reproducibility itself diverges from legacy (hash-based vs ti.random()'s
counter-based PRNG) - only the selection distribution's shape is preserved.

make_accumulation: the SFD downstream-accumulation Bag factory, built on the
same core plus a `grid` Bag and a `source` Parameter (any mode - const,
scalar or field all work with no variant code, since every template reads
`source.get(i)`).

    accum = make_accumulation("taichi", grid, source_p, method="atomic")
    accum_kernel = accum.accum.compile()
    accum_kernel(rec.data, q.data)

`method`:
  - "atomic": on taichi/quadrants, one KernelBuilder ("accum", data args
    (rec, q)) - two top-level for-loops (q[i] = source.get(i), then the
    descent), which the closure backends already launch as two barrier-
    separated GPU dispatches. On cupy, two KernelBuilders ("q_init", data
    arg (q); "accum", data args (rec, q)) - a single CUDA __global__ has no
    portable grid-wide barrier the way two consecutive Taichi/Quadrants
    for-loops do, so "q_init" must be launched, and finish, before "accum".
    Either way: every node walks its receiver chain to the root, atomic-
    adding its own weight into each downstream node. Requires an acyclic
    receiver graph (run after depression handling) - a cycle degrades the
    result (the walk gives up once its guard counter reaches n_flat) rather
    than hanging.
  - "rake_compress": a RoutineBuilder (see _closure_blocks.py's
    build_rake_compress) plus its constituent KernelBuilders, keyed
    "zero_init", "reset_iteration", "bump_iteration", "decrement_iteration",
    "q_init", "receivers_to_donors", "rake_compress_accum",
    "fuse_accum_buffers". "bump_iteration" is exported for API parity but not
    part of the routine's own step list on closure backends: the increment
    it performs is folded into rake_compress_accum's own second top-level
    `for` loop instead (see build_rake_compress), which removes one
    single-thread kernel launch per rake round. Requires `iteration_p` (a
    scalar Parameter, i32) -
    it is the device-side "which ping-pong buffer holds each node's current
    data, and which round last touched it" counter the legacy kernel took as
    a plain call argument (see pyfastflow/general_algorithms/pingpong.py);
    the routine's own "reset_iteration" step zeros it every call, so the
    caller never has to remember to reset it between calls.
  - "pointer_jump_push": a RoutineBuilder (see _closure_blocks.py's
    build_pointer_jump_push) plus its constituent KernelBuilders, keyed
    "q_init", "copy_rec_to_work", and (closure backends)
    "accum_pointer_jump_push_step" or (cupy, split into two launches for a
    real barrier between the copy and the push - see _cupy_blocks.py)
    "accum_pointer_jump_push_step_copy"/"accum_pointer_jump_push_step_core".

Both RoutineBuilder methods register their data names as placeholders
(add_data(name, None)) rather than allocating anything - these factories
take no pool, per the settled design: scratch buffers are caller-supplied,
declared as template data arguments, not bound field Parameters. Every real
call to the compiled routine must therefore pass every one of its
`data_names` positionally - inspect `routine.data_names` after compiling
rather than assuming an order; do not call the compiled routine with zero
arguments; it would just launch every step against `None`.

Both RoutineBuilders are built with begin_repeat()/end_repeat(). The caller
compiles the returned RoutineBuilder itself (these factories export
builders, not compiled objects - see CLAUDE.md); fused=True and fused=False
both compile and produce bit-identical output on closure backends -
consecutive top-level `for` loops inside one compiled Taichi/Quadrants
kernel are already separate offloaded tasks launched in order (confirmed
empirically: a two-step routine where step 2 reads what step 1 wrote across
the whole buffer gives bit-identical output fused and unfused, and legacy
pyfastflow/flow/lakeflow.py's saddlesort already relies on exactly this
inside one hand-written kernel), so the choice does not affect the barrier a
round's cross-buffer dependency needs either way. These kernel templates are
nested defs closing over a per-backend Tensor annotation (`ti.template()` vs
`qd.Tensor`, picked at build time - the same idiom
../ops/_closure_blocks.py's build_elementwise/build_scan_routine use for
theirs); capture_template_meta dedents a nested def's source before parsing
it, and _fuse_group synthesizes each data argument's annotation from the
bound backend module rather than reading one out of the AST - see
_closure_backend.py's capture_template_meta/_fuse_group - so fusion is
available here on the same terms as a module-level template.

On cupy, compile with `captured=False`: captured=True's CUDA-graph replay
does not support call-time data-handle overrides at all (see
cupy_backend.py, _CapturedRoutine) and warms up/restores against the
handles registered via add_data - but this factory registers only `None`
placeholders (no pool to allocate real ones), so captured=True would crash
on its own warmup. captured=False is exactly the per-step, real-launch-
every-call semantics these buffers need anyway.

`n_flat`, if not given, is read off `grid.nx.get() * grid.ny.get()` - this
requires nx/ny to be in "const" mode (the make_grid default); pass `n_flat`
explicitly for a grid built with scalar-mode dimensions.

make_depressions: the depression-handling Bag factory, porting
../../flow/flow_reroute_kernels.py. Two orthogonal build flags:

    deps = make_depressions("taichi", grid, ndep_p, method="vanilla", reroute="carve")

`method` ("vanilla"|"optimized") picks how basins are labelled and, for
reroute="carve", how the carve itself runs; `reroute` ("carve"|"jump") picks
how a resolved basin's pit is reconnected to its outlet. All four
combinations build. `ndep_p` (a caller-allocated scalar i32 Parameter, mode
"scalar") is required - it is not built here, the same way make_accumulation
takes iteration_p rather than allocating it, since this factory takes no
pool: every scratch buffer below is a caller-supplied data arg, never a
bound field Parameter.

Every buffer is n_flat-sized, since a per-basin array is indexed by basin id
and basin id = pit index + 1 (bid/basin_saddlenode/outlet range over the
same 0..n_flat-1 index space as every per-node buffer). Required data args,
by Bag member:

  "ndep":                the `ndep_p` scalar Parameter itself (bag.ndep.read())
  "depression_counter":   closure: (rec,) - accumulates into ndep_p, bound
                          directly as a raw field. cupy: (rec, ndep) - ndep_p
                          is only ever reached through $...$ get() spans
                          there, which registers it read-only in the
                          constant block (see cupy_backend.py's
                          _SpanParser._register_ptr), so the caller instead
                          passes `ndep_p.get().data` positionally, same as
                          `rec` - see build_depression_counter in
                          _cupy_blocks.py. Either way the caller must
                          ndep_p.set(0) before each launch (mirrors
                          ops.Reduce.run_sum).
  "copy_field":           (src, dst) i32        -> dst[:] = src[:]
  "label_basins" (vanilla): Routine, data names ("rec", "bid", "rec_jump")
  "label_basins" (optimized, closure): Kernel, data args (rec, rec_jump, bid)
  "label_basins" (optimized, cupy): Routine, data names ("rec", "rec_jump", "bid")
    - see _closure_blocks.py/_cupy_blocks.py's build_basin_labelling_* for
      why cupy needs three real launches where a closure backend needs one.
  "saddlesort":           Routine (6 kernels, unchanged by `method`), data
                          names ("bid", "z", "z_prime", "is_border",
                          "basin_saddle", "basin_saddlenode", "outlet") -
                          the six constituent KernelBuilders are also
                          exposed under "saddlesort_<name>".
  "reroute" (carve, vanilla): Routine, data names ("rec", "rec_work",
                          "rec_jump", "tag", "tag_alt", "bid",
                          "basin_saddlenode", "outlet", "rerouted")
  "reroute" (carve, optimized): Kernel, data args (rec, basin_saddlenode, outlet)
  "reroute" (jump, closure): Kernel, data args (rec, outlet, rerouted)
  "reroute" (jump, cupy): Routine, data names ("rec", "outlet", "rerouted")
    - split for the same real-launch-barrier reason as label_basins above.

Every "label_basins"/"reroute" constituent KernelBuilder is also exposed
under "label_basins_<name>"/"reroute_<name>" respectively, mirroring
make_accumulation's own routine+constituent-kernels convention.

`reroute_jump`'s pit write is deliberately `rec[i - 1]`, not `rec[i]`: the
loop is over basin ids and basin id = pit index + 1 - ported exactly as
legacy has it, not "fixed" - see _closure_blocks.py's build_reroute_jump.

`ops.bitpack` (pack/unpack_value/unpack_index) replaces legacy's
f32_i32_struct module for the lexicographic (elevation, target-basin) and
(elevation, node) argmins saddlesort's atomic_min passes need; on cupy,
i64 atomic_min is a CAS loop (CUDA has no native atomicMin over signed long
long) - see _cupy_blocks.py's build_atomic_min_ll.

make_depressions builds the routines/kernels only. make_depression_solver
wraps that Bag into the outer host-driven loop the algorithm needs, as a
compiled Sequence:

    solver = make_depression_solver(
        "taichi", deps, method="vanilla", reroute="carve",
        rec=rec.data, z=z.data, bid=bid.data, ...)
    solver()
    solver.last_trip_counts   # passes the loop actually took

    depression_counter -> ndep;  ndep == 0 -> nothing to do
    loop max_times = ceil(log2(max(2, ndep))) + 2:
        <label basins>  <saddlesort>  <reroute>
        depression_counter -> ndep;  until ndep == 0

`max_times` returns 0 when the entry count is already 0, which is what
Sequence.add_loop's documented "max_times <= 0 runs the body zero times"
case is for, and matches legacy runtime.py's `if ndep == 0: return`.

Buffer naming: `rec` is the caller's authoritative receiver buffer, read at
entry and holding the resolved graph on return, in every combination. The
vanilla carve routine's own two receiver buffers are its internal working
copy (bound to `rec_scratch`) and the authoritative one (bound to `rec`) -
see build_reroute_carve_vanilla, whose last step copies the working buffer
back into the authoritative one.

make_fill_reconstruct/make_fill_reconstruct_solver: a standalone alternative
to make_depressions/make_depression_solver, not a `method` of it - grayscale
morphological reconstruction against elevation, converging `filled`/`parent`
(the receiver graph) directly with no basin ids, saddle search or outlet
routing at all. Ported from
../../../experimental/LM/fill_reconstruct_optimised.py; see that pair's own
docstrings and the block modules' section notes above build_fill_reconstruct_*
for the algorithm, the combined-buffer frontier ping-pong a Sequence's
fixed-at-compile-time data handles require, and the two closure-backend
substitutions (no 3-arg `range()`, `atomic_max` standing in for `atomicExch`)
verified before use.

Author: B.G (07/2026)
"""

import math

from ..core.context.bag import Bag
from ..core.context.backends import backend_classes
from ..core.context.routine import RoutineBuilder
from ..core.context.sequence import host_step, kernel_step, routine_step
from ..ops import make_bitpack
from ..noise import make_hash_u32

_MODES = frozenset({"steepest", "stochastic"})
_ACCUM_METHODS = frozenset({"atomic", "rake_compress", "pointer_jump_push"})


def _blocks_for(backend: str):
    """
    The private block module implementing make_receivers's device code for
    one backend name.

    Author: B.G (07/2026)
    """
    if backend in ("taichi", "quadrants"):
        from . import _closure_blocks as blocks
    elif backend == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"make_receivers: unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")
    return blocks


def _kernel_cls(backend: str):
    """
    The KernelBuilder class for `backend` - not exposed by backend_classes(),
    which only returns HelperBuilder (mirrors ../ops/__init__.py's
    _kernel_cls).

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


def make_receivers(
    backend: str,
    grid: Bag,
    *,
    mode: str = "steepest",
    seed_p=None,
    diagonal_partition_correction: bool = False,
    h_aware: bool = False,
) -> Bag:
    """
    Build one receivers Bag: the `receivers` KernelBuilder (data args
    `(z, rec)` or `(z, h, rec)` depending on `h_aware`) plus the distance/
    slope HelperBuilders it is made of, and `rand_unit` when
    mode="stochastic".

    `mode` "steepest"|"stochastic" picks the kernel body variant (see the
    module docstring). `seed_p` is required, and only used, when
    mode="stochastic". `diagonal_partition_correction` and `h_aware` are
    documented in the module docstring.

    Author: B.G (07/2026)
    """
    if mode not in _MODES:
        raise ValueError(f"make_receivers: mode must be one of {sorted(_MODES)}, got {mode!r}")
    if mode == "stochastic" and seed_p is None:
        raise ValueError("make_receivers: mode='stochastic' requires seed_p")

    backend_mod, ParamCls, HelperCls, dtypes = backend_classes(backend)
    KernelCls = _kernel_cls(backend)
    blocks = _blocks_for(backend)
    hash_u32 = make_hash_u32(backend) if mode == "stochastic" else None

    if backend in ("taichi", "quadrants"):
        helpers = blocks.build_receivers(
            KernelCls,
            HelperCls,
            backend=backend,
            backend_mod=backend_mod,
            grid=grid,
            hash_u32=hash_u32,
            mode=mode,
            seed_p=seed_p,
            diagonal_partition_correction=diagonal_partition_correction,
            h_aware=h_aware,
        )
    else:
        helpers = blocks.build_receivers(
            KernelCls,
            HelperCls,
            grid=grid,
            hash_u32=hash_u32,
            mode=mode,
            seed_p=seed_p,
            diagonal_partition_correction=diagonal_partition_correction,
            h_aware=h_aware,
        )

    return Bag(helpers)


def _routine_cls(backend: str):
    """
    The RoutineBuilder class for `backend` - not exposed by
    backend_classes(), mirrors ../ops/__init__.py's _routine_cls.

    Author: B.G (07/2026)
    """
    if backend == "taichi":
        from ..core.context.taichi_backend import TaichiRoutineBuilder

        return TaichiRoutineBuilder
    if backend == "quadrants":
        from ..core.context.quadrants_backend import QuadrantsRoutineBuilder

        return QuadrantsRoutineBuilder
    if backend == "cupy":
        from ..core.context.cupy_backend import CupyRoutineBuilder

        return CupyRoutineBuilder
    raise ValueError(f"unknown backend {backend!r}")


def _resolve_n_flat(grid: Bag, n_flat) -> int:
    """
    `n_flat` if given, else grid.nx.get() * grid.ny.get() - raises if that
    read does not come back as plain python ints (i.e. nx/ny are not in
    "const" mode).

    Author: B.G (07/2026)
    """
    if n_flat is not None:
        return int(n_flat)
    nx = grid.nx.get()
    ny = grid.ny.get()
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise ValueError(
            "make_accumulation: grid.nx/grid.ny are not const-mode - pass n_flat explicitly"
        )
    return nx * ny


def _resolve_nx_ny(grid: Bag) -> tuple:
    """
    (nx, ny) as plain python ints - raises if grid.nx/grid.ny are not in
    "const" mode. Unlike _resolve_n_flat there is no override argument: the
    row-length/row-count split (not just their product) is load-bearing for
    make_fill_reconstruct's directional sweeps, so there is nothing sensible
    to fall back to when it is unavailable.

    Author: B.G (07/2026)
    """
    nx = grid.nx.get()
    ny = grid.ny.get()
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise ValueError(
            "make_fill_reconstruct: grid.nx/grid.ny are not const-mode - a fixed-shape "
            "grid is required for the directional sweep kernels"
        )
    return nx, ny


def make_accumulation(
    backend: str,
    grid: Bag,
    source,
    *,
    method: str = "rake_compress",
    n_flat: int | None = None,
    iteration_p=None,
) -> Bag:
    """
    Build one accumulation Bag for `method` "atomic"|"rake_compress"|
    "pointer_jump_push" - see the module docstring for what each returns and
    the RoutineBuilder methods' data_names/fused=False conventions.

    `source` is a Parameter in any mode (const/scalar/field). `iteration_p`
    is required, and only used, for method="rake_compress". `n_flat`
    defaults to grid.nx.get() * grid.ny.get() (see _resolve_n_flat).

    Author: B.G (07/2026)
    """
    if method not in _ACCUM_METHODS:
        raise ValueError(f"make_accumulation: method must be one of {sorted(_ACCUM_METHODS)}, got {method!r}")
    if method == "rake_compress" and iteration_p is None:
        raise ValueError("make_accumulation: method='rake_compress' requires iteration_p")

    backend_mod, ParamCls, HelperCls, dtypes = backend_classes(backend)
    KernelCls = _kernel_cls(backend)
    blocks = _blocks_for(backend)
    n_flat_resolved = _resolve_n_flat(grid, n_flat)
    closure = backend in ("taichi", "quadrants")

    if method == "atomic":
        if closure:
            kb = blocks.build_atomic(KernelCls, backend=backend, backend_mod=backend_mod, source=source, n_flat=n_flat_resolved)
            return Bag({"accum": kb})
        # cupy: two real launches, not one - see _cupy_blocks.py's
        # build_atomic. "q_init" must be launched before "accum".
        kbs = blocks.build_atomic(KernelCls, source=source, n_flat=n_flat_resolved)
        return Bag(kbs)

    logn = math.ceil(math.log2(n_flat_resolved)) + 1
    RoutineCls = _routine_cls(backend)

    if method == "rake_compress":
        if closure:
            rb, kernels = blocks.build_rake_compress(
                RoutineCls, KernelCls, HelperCls,
                backend=backend, backend_mod=backend_mod, grid=grid,
                source=source, iteration_p=iteration_p, logn=logn,
            )
        else:
            rb, kernels = blocks.build_rake_compress(
                RoutineCls, KernelCls, HelperCls,
                grid=grid, source=source, iteration_p=iteration_p, logn=logn,
                n_flat=n_flat_resolved,
            )
    else:  # pointer_jump_push
        rounds = logn + 1
        if rounds % 2 != 0:
            rounds += 1
        if closure:
            rb, kernels = blocks.build_pointer_jump_push(
                RoutineCls, KernelCls, backend=backend, backend_mod=backend_mod, source=source, rounds=rounds,
            )
        else:
            rb, kernels = blocks.build_pointer_jump_push(
                RoutineCls, KernelCls, source=source, rounds=rounds, n_flat=n_flat_resolved,
            )

    out = dict(kernels)
    out["routine"] = rb
    return Bag(out)


# ---------------------------------------------------------------------------
# depressions
# ---------------------------------------------------------------------------

_DEP_METHODS = frozenset({"vanilla", "optimized"})
_DEP_REROUTES = frozenset({"carve", "jump"})


def make_depressions(
    backend: str,
    grid: Bag,
    depression_counter_p,
    *,
    method: str = "vanilla",
    reroute: str = "carve",
    n_flat: int | None = None,
) -> Bag:
    """
    Build one depression-handling Bag for `method` "vanilla"|"optimized" x
    `reroute` "carve"|"jump" - see the module docstring for the exact Bag
    keys, their types (Kernel vs Routine) per combination/backend, and the
    data args each expects.

    `depression_counter_p` is a caller-allocated scalar Parameter (mode
    "scalar", dtype i32) - required, not built here (this factory takes no
    pool). `n_flat` defaults to grid.nx.get() * grid.ny.get(), same as
    make_accumulation.

    This builds the routines/kernels only; running them in the
    label -> saddlesort -> reroute -> recount loop the algorithm needs is
    the outer Sequence's job, not this factory's.

    Author: B.G (07/2026)
    """
    if method not in _DEP_METHODS:
        raise ValueError(f"make_depressions: method must be one of {sorted(_DEP_METHODS)}, got {method!r}")
    if reroute not in _DEP_REROUTES:
        raise ValueError(f"make_depressions: reroute must be one of {sorted(_DEP_REROUTES)}, got {reroute!r}")

    backend_mod, ParamCls, HelperCls, dtypes = backend_classes(backend)
    KernelCls = _kernel_cls(backend)
    RoutineCls = _routine_cls(backend)
    blocks = _blocks_for(backend)
    n_flat_resolved = _resolve_n_flat(grid, n_flat)
    closure = backend in ("taichi", "quadrants")
    logn = math.ceil(math.log2(n_flat_resolved)) + 1

    bitpack_bag = make_bitpack(backend)
    bitpack = {"pack": bitpack_bag.pack, "unpack_value": bitpack_bag.unpack_value, "unpack_index": bitpack_bag.unpack_index}

    out: dict = {"ndep": depression_counter_p}

    if closure:
        copy_field = blocks.build_copy_field(KernelCls, backend=backend, backend_mod=backend_mod)
        depression_counter = blocks.build_depression_counter(
            KernelCls, backend=backend, backend_mod=backend_mod, grid=grid,
            ndep_raw=depression_counter_p.get().data,
        )
    else:
        copy_field = blocks.build_copy_field(KernelCls, n_flat=n_flat_resolved)
        depression_counter = blocks.build_depression_counter(
            KernelCls, grid=grid, n_flat=n_flat_resolved,
        )
    out["copy_field"] = copy_field
    out["depression_counter"] = depression_counter

    # basin labelling
    if method == "vanilla":
        if closure:
            lb_rb, lb_kernels = blocks.build_basin_labelling_vanilla(
                RoutineCls, KernelCls, backend=backend, backend_mod=backend_mod,
                grid=grid, copy_field=copy_field, logn=logn,
            )
        else:
            lb_rb, lb_kernels = blocks.build_basin_labelling_vanilla(
                RoutineCls, KernelCls, grid=grid, copy_field=copy_field,
                n_flat=n_flat_resolved, logn=logn,
            )
        out["label_basins"] = lb_rb
        for name, kb in lb_kernels.items():
            out[f"label_basins_{name}"] = kb
    else:  # optimized
        if closure:
            out["label_basins"] = blocks.build_basin_labelling_optimized(
                KernelCls, backend=backend, backend_mod=backend_mod, grid=grid, n_flat=n_flat_resolved,
            )
        else:
            lb_rb, lb_kernels = blocks.build_basin_labelling_optimized(
                RoutineCls, KernelCls, grid=grid, n_flat=n_flat_resolved,
            )
            out["label_basins"] = lb_rb
            for name, kb in lb_kernels.items():
                out[f"label_basins_{name}"] = kb

    # saddlesort - shared, unchanged by `method`
    if closure:
        ss_rb, ss_kernels = blocks.build_saddlesort(
            RoutineCls, KernelCls, backend=backend, backend_mod=backend_mod, grid=grid, bitpack=bitpack,
        )
    else:
        ss_rb, ss_kernels = blocks.build_saddlesort(
            RoutineCls, KernelCls, HelperCls, grid=grid, bitpack=bitpack, n_flat=n_flat_resolved,
        )
    out["saddlesort"] = ss_rb
    for name, kb in ss_kernels.items():
        out[f"saddlesort_{name}"] = kb

    # reroute
    if reroute == "carve":
        if method == "vanilla":
            if closure:
                rr_rb, rr_kernels = blocks.build_reroute_carve_vanilla(
                    RoutineCls, KernelCls, backend=backend, backend_mod=backend_mod,
                    bitpack=bitpack, copy_field=copy_field, logn=logn,
                )
            else:
                rr_rb, rr_kernels = blocks.build_reroute_carve_vanilla(
                    RoutineCls, KernelCls, bitpack=bitpack, copy_field=copy_field,
                    n_flat=n_flat_resolved, logn=logn,
                )
            out["reroute"] = rr_rb
            for name, kb in rr_kernels.items():
                out[f"reroute_{name}"] = kb
        else:  # optimized
            if closure:
                out["reroute"] = blocks.build_reroute_carve_optimized(
                    KernelCls, backend=backend, backend_mod=backend_mod, bitpack=bitpack,
                )
            else:
                out["reroute"] = blocks.build_reroute_carve_optimized(
                    KernelCls, bitpack=bitpack, n_flat=n_flat_resolved,
                )
    else:  # jump
        if closure:
            out["reroute"] = blocks.build_reroute_jump(
                KernelCls, backend=backend, backend_mod=backend_mod, bitpack=bitpack,
            )
        else:
            rr_rb, rr_kernels = blocks.build_reroute_jump(
                RoutineCls, KernelCls, bitpack=bitpack, n_flat=n_flat_resolved,
            )
            out["reroute"] = rr_rb
            for name, kb in rr_kernels.items():
                out[f"reroute_{name}"] = kb

    return Bag(out)


# ---------------------------------------------------------------------------
# depressions: the outer host-driven loop
# ---------------------------------------------------------------------------


def _sequence_cls(backend: str):
    """
    The SequenceBuilder class for `backend` - not exposed by
    backend_classes(), mirrors _kernel_cls/_routine_cls above.

    Author: B.G (07/2026)
    """
    if backend == "taichi":
        from ..core.context.taichi_backend import TaichiSequenceBuilder

        return TaichiSequenceBuilder
    if backend == "quadrants":
        from ..core.context.quadrants_backend import QuadrantsSequenceBuilder

        return QuadrantsSequenceBuilder
    if backend == "cupy":
        from ..core.context.cupy_backend import CupySequenceBuilder

        return CupySequenceBuilder
    raise ValueError(f"unknown backend {backend!r}")


def _union_bag(deps: Bag, extra: dict) -> Bag:
    """
    One Bag carrying every name any KernelBuilder in `deps` binds, plus
    `extra`.

    A Sequence rebinds every block - and every step of every inner Routine -
    against a single bag, so that bag must carry every name those blocks
    bind: the grid, the backend module, the bitpack helpers, the raw
    depression-counter cell, cupy's i64 atomic_min helper. Each of those is
    reached here off the KernelBuilders make_depressions already exposes,
    so the objects put in the bag are the very ones the blocks were built
    against and rebinding is a no-op rather than a substitution. A name bound
    to two different objects across the Bag raises - that is the same
    condition check_handles enforces, caught here where the offending Bag key
    can be named.

    Author: B.G (07/2026)
    """
    members: dict = {}
    owner: dict = {}
    for key, obj in deps.items():
        bindings = getattr(obj, "bindings", None)
        if not isinstance(bindings, dict):
            continue
        for name, bound in bindings.items():
            prior = members.get(name)
            if prior is not None and prior is not bound:
                raise ValueError(
                    f"make_depression_solver: '{name}' is bound to two different objects "
                    f"across the depression Bag ('{owner[name]}' vs '{key}')"
                )
            members[name] = bound
            owner[name] = key
    members.update(extra)
    return Bag(members)


def _fill_routine_data(routine_builder, table: dict, label: str) -> None:
    """
    Give every data name a depression RoutineBuilder registered a real
    buffer, in place.

    The flow block modules register their routines' data names as
    `add_data(name, None)` placeholders, since those factories take no pool
    and the buffers are the caller's. It has to be the registered defaults
    rather than a call-time override: a Routine compiled by a Sequence on
    cupy is graph-captured, and a captured Routine both warms up against its
    registered handles and rejects call-time overrides outright (see
    cupy_backend.py, _CapturedRoutine). fill_data() is RoutineBuilder's
    sanctioned way to replace such a placeholder.

    Author: B.G (07/2026)
    """
    registered = routine_builder._data
    missing = [name for name in registered if name not in table]
    if missing:
        raise KeyError(f"make_depression_solver: {label} needs data for {sorted(missing)}")
    for name in registered:
        routine_builder.fill_data(name, table[name])


def _require(label: str, **buffers):
    """
    Raise naming every buffer this combination needs that was left None.

    Author: B.G (07/2026)
    """
    missing = sorted(name for name, buf in buffers.items() if buf is None)
    if missing:
        raise ValueError(f"make_depression_solver: {label} requires {missing}")


def make_depression_solver(
    backend: str,
    deps: Bag,
    *,
    method: str = "vanilla",
    reroute: str = "carve",
    rec=None,
    z=None,
    bid=None,
    rec_jump=None,
    z_prime=None,
    is_border=None,
    basin_saddle=None,
    basin_saddlenode=None,
    outlet=None,
    rerouted=None,
    tag=None,
    tag_alt=None,
    rec_scratch=None,
    n_flat: int | None = None,
    block_size: int = 256,
):
    """
    Compile the outer depression-resolution loop over a Bag from
    make_depressions, as a Sequence - see the module docstring for its shape.

    `method`/`reroute` must be the ones `deps` was built with; they decide
    which buffers are required and how they map onto each block's data
    arguments. Every buffer is a raw device buffer (a DataHandle's `.data`),
    n_flat-sized, caller-allocated - this factory allocates nothing.

    Required in every combination: `rec` (the authoritative receiver buffer,
    read at entry, resolved on return), `z`, `bid`, `rec_jump`, `z_prime`,
    `is_border`, `basin_saddle`, `basin_saddlenode`, `outlet`. reroute="jump"
    and method="vanilla"+reroute="carve" additionally need `rerouted`; that
    same vanilla carve additionally needs `tag`, `tag_alt` and `rec_scratch`.
    `rerouted` is zeroed by the jump reroute itself but not by the carve one -
    a caller wanting it to mean "rerouted by this call" zeroes it beforehand.

    `n_flat` is required on cupy, where it sets the launch dimensions for the
    Sequence's own kernel blocks (`block_size` threads per block); it is
    unused on taichi/quadrants, which range over the buffers themselves.

    Returns the compiled Sequence. It takes no arguments, holds the buffers
    given here for its whole life, and reports the passes it took in
    `last_trip_counts`. Destroying or repooling any of these buffers, or
    `deps.ndep`, invalidates it - rebuild rather than patch (see
    sequence.py's contract).

    Author: B.G (07/2026)
    """
    if method not in _DEP_METHODS:
        raise ValueError(f"make_depression_solver: method must be one of {sorted(_DEP_METHODS)}, got {method!r}")
    if reroute not in _DEP_REROUTES:
        raise ValueError(f"make_depression_solver: reroute must be one of {sorted(_DEP_REROUTES)}, got {reroute!r}")
    _require(
        "every combination", rec=rec, z=z, bid=bid, rec_jump=rec_jump, z_prime=z_prime,
        is_border=is_border, basin_saddle=basin_saddle, basin_saddlenode=basin_saddlenode, outlet=outlet,
    )

    closure = backend in ("taichi", "quadrants")
    if not closure and n_flat is None:
        raise ValueError("make_depression_solver: n_flat is required on cupy - it sets the launch dimensions")

    ndep_p = deps.ndep
    SequenceCls = _sequence_cls(backend)
    if closure:
        sb = SequenceCls()
    else:
        sb = SequenceCls(grid=((int(n_flat) + block_size - 1) // block_size,), block=(block_size,))

    sb.bind_bag(_union_bag(deps, {"ndep": ndep_p}))

    sb.add_data("rec", rec)
    sb.add_data("rec_jump", rec_jump)
    sb.add_data("bid", bid)
    sb.add_data("basin_saddlenode", basin_saddlenode)
    sb.add_data("outlet", outlet)
    if rerouted is not None:
        sb.add_data("rerouted", rerouted)
    if not closure:
        sb.add_data("ndep_buf", ndep_p.get().data)

    counter_refs = ("rec",) if closure else ("rec", "ndep_buf")

    def zero_ndep(bag):
        """
        Reset the depression counter before each launch of the counting
        kernel, which only ever accumulates into it.

        Author: B.G (07/2026)
        """
        bag.ndep.set(0)

    # basin labelling
    label = deps.label_basins
    if isinstance(label, RoutineBuilder):
        _fill_routine_data(label, {"rec": rec, "bid": bid, "rec_jump": rec_jump}, "label_basins")
        label_block = routine_step(label)
    else:
        label_block = kernel_step(label, ("rec", "rec_jump", "bid"))

    # saddlesort - always a Routine, both backends, both methods
    _fill_routine_data(
        deps.saddlesort,
        {
            "bid": bid, "z": z, "z_prime": z_prime, "is_border": is_border,
            "basin_saddle": basin_saddle, "basin_saddlenode": basin_saddlenode, "outlet": outlet,
        },
        "saddlesort",
    )
    saddlesort_block = routine_step(deps.saddlesort)

    # reroute
    rr = deps.reroute
    if reroute == "carve" and method == "vanilla":
        _require("method='vanilla', reroute='carve'", rerouted=rerouted, tag=tag, tag_alt=tag_alt, rec_scratch=rec_scratch)
        _fill_routine_data(
            rr,
            {
                "rec": rec_scratch, "rec_work": rec, "rec_jump": rec_jump, "tag": tag, "tag_alt": tag_alt,
                "bid": bid, "basin_saddlenode": basin_saddlenode, "outlet": outlet, "rerouted": rerouted,
            },
            "reroute",
        )
        reroute_block = routine_step(rr)
    elif reroute == "carve":
        reroute_block = kernel_step(rr, ("rec", "basin_saddlenode", "outlet"))
    else:
        _require("reroute='jump'", rerouted=rerouted)
        if isinstance(rr, RoutineBuilder):
            _fill_routine_data(rr, {"rec": rec, "outlet": outlet, "rerouted": rerouted}, "reroute")
            reroute_block = routine_step(rr)
        else:
            reroute_block = kernel_step(rr, ("rec", "outlet", "rerouted"))

    def entry_passes(bag):
        """
        ceil(log2(max(2, ndep))) + 2 passes, or none at all when the entry
        count is already zero.

        Author: B.G (07/2026)
        """
        ndep = int(bag.ndep.read())
        if ndep == 0:
            return 0
        return math.ceil(math.log2(max(2, ndep))) + 2

    def resolved(bag):
        """
        True once the device reports no unresolved depression left.

        Author: B.G (07/2026)
        """
        return int(bag.ndep.read()) == 0

    sb.add_host(zero_ndep)
    sb.add_kernel(deps.depression_counter, counter_refs)
    sb.add_loop(
        body=[
            label_block,
            saddlesort_block,
            reroute_block,
            host_step(zero_ndep),
            kernel_step(deps.depression_counter, counter_refs),
        ],
        max_times=entry_passes,
        until=resolved,
    )
    return sb.compile()


# ---------------------------------------------------------------------------
# fill by grayscale morphological reconstruction - a standalone alternative
# to make_depressions/make_depression_solver, not a variant of it: no basin
# ids, no saddle search, no outlet routing - one frontier-relaxation kernel
# converging `filled`/`parent` (the receiver graph) directly to a fixed
# point. Ported from experimental/LM/fill_reconstruct_optimised.py - see
# that file's module docstring for the algorithm's derivation and every
# optimisation round it documents, and _cupy_blocks.py's/_closure_blocks.py's
# own section notes above build_fill_reconstruct_* for what changed to make
# it fit this framework's Sequence-driven, data-args-fixed-at-compile-time
# shape.
# ---------------------------------------------------------------------------


def make_fill_reconstruct(
    backend: str,
    grid: Bag,
    pass_p,
    *,
    n_flat: int | None = None,
) -> Bag:
    """
    Build one reconstruction-fill Bag: `init_filled`, `sweep_row_lr`,
    `sweep_row_rl`, `sweep_col_tb`, `sweep_col_bt`, `frontier_init`, `relax`
    (all KernelBuilders, data args per _cupy_blocks.py's/_closure_blocks.py's
    build_fill_reconstruct_* docstrings) plus `pass_p` itself.

    `pass_p` is a caller-allocated scalar i32 Parameter (mode "scalar"),
    required - not built here, the same way make_depressions takes
    `depression_counter_p`: this factory takes no pool, every scratch buffer
    is a caller-supplied data arg. `relax` reads it every launch
    (`$P.get(0)$`/`_P.get(0)`) to index `counters[]` and to pick which half
    of the combined `frontier` buffer is this pass's input - bumping it
    between passes is make_fill_reconstruct_solver's job, not this one's.

    `n_flat` defaults to grid.nx.get() * grid.ny.get(), same as
    make_accumulation; the row-length/row-count split the sweeps need
    (`_resolve_nx_ny`) always requires const-mode grid dimensions, with no
    override.

    Author: B.G (07/2026)
    """
    backend_mod, ParamCls, HelperCls, dtypes = backend_classes(backend)
    KernelCls = _kernel_cls(backend)
    blocks = _blocks_for(backend)
    n_flat_resolved = _resolve_n_flat(grid, n_flat)
    nx, ny = _resolve_nx_ny(grid)
    closure = backend in ("taichi", "quadrants")

    if closure:
        init_filled = blocks.build_fill_reconstruct_init(KernelCls, backend=backend, backend_mod=backend_mod, grid=grid)
        sweeps = blocks.build_fill_reconstruct_sweeps(
            KernelCls, backend=backend, backend_mod=backend_mod, nx=nx, ny=ny
        )
        frontier_init = blocks.build_fill_reconstruct_frontier_init(KernelCls, backend=backend, backend_mod=backend_mod)
        relax = blocks.build_fill_reconstruct_relax(
            KernelCls, backend=backend, backend_mod=backend_mod, grid=grid, pass_p=pass_p, n_flat=n_flat_resolved,
        )
    else:
        init_filled = blocks.build_fill_reconstruct_init(KernelCls, grid=grid, n_flat=n_flat_resolved)
        sweeps = blocks.build_fill_reconstruct_sweeps(KernelCls, nx=nx, ny=ny)
        frontier_init = blocks.build_fill_reconstruct_frontier_init(KernelCls, n_flat=n_flat_resolved)
        relax = blocks.build_fill_reconstruct_relax(KernelCls, grid=grid, pass_p=pass_p, n_flat=n_flat_resolved)

    out: dict = {
        "init_filled": init_filled, "frontier_init": frontier_init, "relax": relax,
        "pass_p": pass_p, "grid": grid,
    }
    for name, kb in sweeps.items():
        out[f"sweep_{name}"] = kb
    return Bag(out)


def _read_frontier_count(backend: str, counters, p: int) -> int:
    """
    `counters[p]` as a plain python int, synchronizing first - the one
    device readback make_fill_reconstruct_solver's loop predicate needs each
    pass, mirroring Parameter.read()'s own sync-then-return contract for a
    raw buffer that isn't a Parameter.

    Author: B.G (07/2026)
    """
    if backend == "cupy":
        return int(counters[p].get())
    return int(counters[p])


def make_fill_reconstruct_solver(
    backend: str,
    deps: Bag,
    *,
    z=None,
    filled=None,
    parent=None,
    frontier=None,
    counters=None,
    queued_gen=None,
    n_flat: int | None = None,
    block_size: int = 256,
    max_passes: int | None = None,
):
    """
    Compile the reconstruction-fill outer loop over a Bag from
    make_fill_reconstruct, as a Sequence:

        init_filled; sweep_row_lr; sweep_row_rl; sweep_col_tb; sweep_col_bt;
        frontier_init -> counters[0]
        pass_p = 0
        loop max_times = max_passes:
            relax(pass_p)
            pass_p += 1
            until counters[pass_p] == 0

    Every buffer is a raw device buffer (a DataHandle's `.data`),
    caller-allocated - this factory allocates nothing, matching
    make_depression_solver. Required: `z` (n_flat,), `filled` (n_flat,),
    `parent` (n_flat,) i32 - `filled`/`parent` need no caller-side init,
    `init_filled` seeds both; `frontier` (2*n_flat,) i32 - the two ping-pong
    halves combined into one buffer (see make_fill_reconstruct's module
    note), needs no caller-side init either; `counters` (max_passes+2,) i32,
    must be **zeroed once, by the caller, before the first call** - every
    pass writes a slot it never reuses (`counters[p+1]`), so it never needs
    zeroing again, the same trick `queued_gen` uses; `queued_gen` (n_flat,)
    i32, must be **filled with -1 once, by the caller, before the first
    call** for the same reason.

    `n_flat` is required on cupy, where it sets the launch dimensions for
    every kernel in this Sequence; unused on taichi/quadrants.

    `max_passes` defaults to `4 * max(nx, ny)` (nx, ny read off `deps.grid`) -
    generous headroom over the measured ~0.6x that ratio in
    experimental/LM/fill_reconstruct_optimised.py. Reaching it without the
    frontier emptying is not raised here (a Sequence loop simply stops after
    max_times body iterations - see sequence.py); check
    `solver.last_trip_counts[-1] < max_passes` after calling if that
    matters to the caller.

    Returns the compiled Sequence. It takes no arguments and reports the
    passes it actually took in `last_trip_counts[-1]` (the loop is the only
    loop block here, so there is exactly one entry).

    Author: B.G (07/2026)
    """
    _require(
        "make_fill_reconstruct_solver", z=z, filled=filled, parent=parent,
        frontier=frontier, counters=counters, queued_gen=queued_gen,
    )
    closure = backend in ("taichi", "quadrants")
    if not closure and n_flat is None:
        raise ValueError("make_fill_reconstruct_solver: n_flat is required on cupy - it sets the launch dimensions")

    pass_p = deps.pass_p
    SequenceCls = _sequence_cls(backend)
    if closure:
        sb = SequenceCls()
    else:
        sb = SequenceCls(grid=((int(n_flat) + block_size - 1) // block_size,), block=(block_size,))

    sb.bind_bag(_union_bag(deps, {"pass_p": pass_p}))

    sb.add_data("z", z)
    sb.add_data("filled", filled)
    sb.add_data("parent", parent)
    sb.add_data("frontier", frontier)
    sb.add_data("counters", counters)
    sb.add_data("queued_gen", queued_gen)

    zpf = ("z", "filled", "parent")
    sb.add_kernel(deps.init_filled, zpf)
    sb.add_kernel(deps.sweep_row_lr, zpf)
    sb.add_kernel(deps.sweep_row_rl, zpf)
    sb.add_kernel(deps.sweep_col_tb, zpf)
    sb.add_kernel(deps.sweep_col_bt, zpf)
    sb.add_kernel(deps.frontier_init, ("z", "filled", "frontier", "counters"))

    def zero_pass(bag):
        """Reset pass_p to 0 before the loop - fill_reconstruct's pass 0."""
        bag.pass_p.set(0)

    def bump_pass(bag):
        """pass_p += 1 - relax's own next launch reads the bumped value."""
        bag.pass_p.set(int(bag.pass_p.read()) + 1)

    def frontier_empty(bag):
        """True once relax's most recent pass produced an empty frontier."""
        p = int(bag.pass_p.read())
        return _read_frontier_count(backend, counters, p) == 0

    if max_passes is None:
        nx, ny = _resolve_nx_ny(deps.grid)
        max_passes = 4 * max(nx, ny)

    sb.add_host(zero_pass)
    sb.add_loop(
        body=[
            kernel_step(deps.relax, ("z", "filled", "parent", "frontier", "counters", "queued_gen")),
            host_step(bump_pass),
        ],
        max_times=max_passes,
        until=frontier_empty,
    )
    return sb.compile()
