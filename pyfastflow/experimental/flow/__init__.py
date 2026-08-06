"""
make_receivers: the SFD (single-flow-direction) receiver factory, built on
the new builder/frozen/bound stack (../core/context/builder.py, frozen.py,
bound.py) and on a grid FrozenGroup from ../grid's make_grid_group.

Like grid/noise/ops there is no stateful context class - make_receivers
returns a dict of unbuilt structures (FrozenKernel/FrozenHelper): a
`receivers` FrozenKernel plus the distance/slope helpers it is made of, so a
caller can recombine them into its own kernel or routine rather than being
stuck with only the compiled receivers kernel. A caller `.build()`s the
member it wants, binds its PARAM/DATA addresses, `.compile()`s:

    grid = make_grid_group("taichi", topology="D8")
    recv = make_receivers("taichi", grid, topology="D8", mode="steepest")
    bound = recv["receivers"].build()
    for k, v in grid_params.items():
        bound.bind(k, v)  # NX/NY/DX/N_NEIGHBOURS - see below
    bound.bind("z", z_field)
    bound.bind("rec", rec_field)
    receivers_kernel = bound.compile("taichi")
    receivers_kernel()

`mode` ("steepest"|"stochastic") and `h_aware` (False: kernel takes (z, rec)
and slopes read h as 0; True: kernel takes (z, h, rec) and slopes use
(zi-zj)+(hi-hj)) each pick one of four kernel body variants at build time -
see _closure_receivers.py/_cupy_receivers.py's build_receivers. `topology`
("D4"|"D8") must match whatever `grid` was itself built with (see
../grid/__init__.py's own module docstring on the two-call structure/data
split) - it is not readable off `grid` itself, a bare FrozenGroup carrying no
Parameter values yet, and only matters here for
`diagonal_partition_correction` (below); the neighbour loop itself is
already parametrised by `grid.N_NEIGHBOURS`, read as ordinary device data.

mode="stochastic" additionally needs a Parameter bound to the built
receivers kernel's `rand_unit.SEED` address (`rand_unit`'s own wired PARAM
slot, any mode - see rand_unit in the block modules; not build-phase-shared
with anything, so it stays at that nested address rather than being
promoted to the kernel's own top level the way `grid`'s own PARAM names
are) after `.build()`, exactly like any other PARAM slot - there is no Need
indirection in this stack (need.py is part of the pre-rewrite core only);
the host bumps the underlying Parameter between calls for a fresh draw.

`diagonal_partition_correction` only changes anything when `topology ==
"D8"`: it adds the sqrt(2) correction inside dist_from_k_corrected/
dist_between_nodes_corrected (see _closure_receivers.py's
build_distance_slope_helpers for exactly which k values count as diagonal and
why). Off, or on a D4 grid, dist_from_k_corrected/dist_between_nodes_corrected
call straight through to `grid`'s own dist_from_k/dist_between_nodes, no
correction applied - either way these two helpers always independently
compose their own occurrence of `grid` (see _closure_receivers.py's module
docstring for why: a uniform, always-two-occurrences shape lets
`build_receivers` collapse them with `share()` unconditionally, never a
variable number of occurrences depending on the flag).

Returned dict: `receivers` (FrozenKernel, data args (z, rec) or (z, h, rec)),
`dist_from_k_corrected`, `dist_between_nodes_corrected`, `slope_from_values_k`,
`slope_between_nodes` (FrozenHelper), plus `rand_unit` (FrozenHelper) only
when mode="stochastic". `receivers`'s own top-level PARAM slots are every
name `grid` itself wires (NX/NY/DX/N_NEIGHBOURS, plus NODATA_MASK/
OUTLET_MASK if `grid` has them) - bind those bare names once on the built
receivers kernel, not once per occurrence (`grid`'s FrozenGroup is composed
twice inside `receivers`'s own tree - once directly, once nested under
`slope.dist_from_k_corrected` - and build-phase-shared, `_share_leaf`, into
one address each).

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
same core plus a `grid` Bag and a `source` Need (need.py) - a boundary
contract, not a bare Parameter: the caller builds its own Parameter (any mode
- const, scalar or field all work with no variant code, since every template
reads `source.get(i)`) in its own script, wraps it in a
`Need("source", kind=Kind.PARAM)`, `.bind()`s it to that Parameter, and hands
the Need to make_accumulation already bound - make_accumulation raises
immediately if `source` is not a Need, is not kind=Kind.PARAM, or is not yet
bound. This is not a new deferral: every existing caller already builds the
concrete Parameter before calling make_accumulation, so nothing about *when*
a Parameter is built changes - only that the slot it fills is now inspectable
(each returned Bag member's own `.unmet_needs()`) rather than swallowed
invisibly by an internal `.bind()`.

    source = Need("source", kind=Kind.PARAM)
    source.bind(source_p)
    accum = make_accumulation("taichi", grid, source, method="atomic")
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
  - "rake_compress" (ported to the new builder/frozen/bound/sequence stack,
    ../core/context/builder.py/frozen.py/bound.py/sequence_v2.py): a
    SequenceBuilder (see _closure_accum.py's/_cupy_accum.py's
    build_rake_compress) plus its constituent KernelBuilders, keyed
    "zero_init", "reset_iteration", "decrement_iteration", "q_init",
    "receivers_to_donors", "rake_compress_accum", "fuse_accum_buffers" (plus
    "bump_iteration" on cupy only - see below). Composed sequence steps:
    "zero_init" -> "reset_iteration" -> "q_init" -> "receivers_to_donors" ->
    a loop over "rake_step" (the same rake_compress_accum kernel, `max_times
    = ceil(log2(n_flat)) + 2`) -> "decrement_iteration" (undoes the loop's
    last bump) -> "fuse_accum_buffers". On closure backends, the iteration
    bump is rake_compress_accum's own second top-level `for` loop, folded in
    rather than a separate single-thread kernel (two consecutive top-level
    `for` loops inside one compiled Taichi/Quadrants kernel are already
    separate offloaded tasks launched in order); on cupy, a single CUDA
    `__global__` gives no such guarantee, so the loop body is
    `["rake_step", "bump_iteration"]`, two real launches per round - see
    _cupy_accum.py's build_rake_compress. No `source`/`iteration_p` argument
    to this factory at all - `SOURCE`/`ITER` are bare wired PARAM slots (any
    mode), bound by the caller on the built sequence after `.build()`,
    exactly like make_receivers' `rand_unit.SEED`; see build_rake_compress's
    own docstring (per backend) for the exact addresses (four independent
    `ITER` addresses after rake_compress_accum's own `share()` collapses its
    two composed ping-pong helpers' ITER occurrences into its own, one
    `SOURCE` address).
  - "pointer_jump_push" (ported the same way): a SequenceBuilder (see
    build_pointer_jump_push) plus its constituent KernelBuilders, keyed
    "q_init", "copy_rec_to_work", and (closure backends)
    "accum_pointer_jump_push_step" or (cupy, split into two launches for a
    real barrier between the copy and the push - see _cupy_accum.py)
    "accum_pointer_jump_push_step_copy"/"accum_pointer_jump_push_step_core".
    The ping-pong between rounds is two independently-bound occurrences of
    the same step kernel ("step_a"/"step_b" - closure; "step_a_copy"/
    "step_a_core"/"step_b_copy"/"step_b_core" - cupy), alternated by a
    sequence loop of `rounds // 2` iterations (`rounds`, computed here,
    already rounded up to even) - no runtime swap() needed, unlike
    routine.py's old add_swap. Same no-`source`-argument contract as
    rake_compress; see build_pointer_jump_push's own docstring for its
    addresses.

Both factories return {"sequence": SequenceBuilder, **kernel_builders} - a
Bag, not a compiled object (these factories export builders, not compiled
kernels - see CLAUDE.md). The caller `.freeze()`s (or lets `.compile()`
freeze implicitly - SequenceBuilder has no separate freeze() call exposed
here beyond what `.build()`/`.compile()` already do internally), `.build()`s,
binds every PARAM/DATA address named above and in each build_* docstring,
then `.compile(backend, grid=..., block=...)` on cupy (grid/block size the
sequence's own default launch dims; single-thread steps override their own
via `launch=` at compose() time, already baked in by the factory) or
`.compile(backend)` on closure backends (no launch dims needed). The
compiled CompiledSequence takes no arguments; call it, then read
`last_trip_counts` if wanted (always exactly one loop entry per compiled
sequence here, so `last_trip_counts[0]` is the only entry, and is always the
full requested count - unlike depression routing's own use of the same loop
machinery, nothing here ever breaks out early via `until`).

Why a SequenceBuilder loop rather than routine_v2.py's unroll-N-times idiom
(../ops/_closure_blocks.py's build_scan_routine, for its own log-depth
passes): a scan pass's kernel body differs every round (`stride` baked in as
a build-time constant), so unrolling costs nothing beyond the kernels
themselves; here the SAME kernel body runs unchanged every round, so
unrolling would only multiply the number of addresses a caller has to bind
(once per round) for no benefit - a loop keeps that count fixed regardless of
round count (`ceil(log2(n_flat))+2` rounds at 1024x1024 is ~22). This is a
design choice this project's rewrite plan did not itself settle - flagged in
the porting report rather than decided silently.

`n_flat`, if not given, is read off `grid["NX"].get() * grid["NY"].get()` -
this requires `grid` to be a `make_grid_parameters` dict (uppercase keys) with
NX/NY in "const" mode (the make_grid_parameters default); pass `n_flat`
explicitly for a grid built with scalar-mode dimensions, or when `grid` is a
bare `make_grid_group` FrozenGroup, which carries no bound Parameter values to
read at build time at all (see ../grid/__init__.py's own module docstring) -
`_resolve_n_flat` raises immediately, naming this, rather than failing on a
missing dict key. `rake_compress` also reads `grid["N_NEIGHBOURS"].get()` as a
build-time python int (sizing the fixed per-node donor arrays), so `grid` must
be the `make_grid_parameters` dict there regardless of whether `n_flat` was
given explicitly; `pointer_jump_push` does not take `grid` at all.

make_depressions: the depression-handling Bag factory, porting
../../flow/flow_reroute_kernels.py. Two orthogonal build flags:

    ndep_need = Need("depression_counter_p", kind=Kind.PARAM, dtype=i32, modes={"scalar"})
    ndep_need.bind(ndep_p)
    deps = make_depressions("taichi", grid, ndep_need, method="vanilla", reroute="carve")

`method` ("vanilla"|"optimized") picks how basins are labelled and, for
reroute="carve", how the carve itself runs; `reroute` ("carve"|"jump") picks
how a resolved basin's pit is reconnected to its outlet. All four
combinations build. `depression_counter_p` is a `Need(kind=Kind.PARAM,
dtype=i32, modes={"scalar"})`, already `.bind()`ed to a caller-allocated
scalar i32 Parameter - the same boundary contract as make_accumulation's
`source`/`iteration_p` (see `_require_param_need`); the underlying Parameter
is not built here, the same way make_accumulation takes iteration_p rather
than allocating it, since this factory takes no pool: every scratch buffer
below is a caller-supplied data arg, never a bound field Parameter.

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
                          _cupy_depressions.py. Either way the caller must
                          ndep_p.set(0) before each launch (mirrors
                          ops.Reduce.run_sum).
  "copy_field":           (src, dst) i32        -> dst[:] = src[:]
  "label_basins" (vanilla): Routine, data names ("rec", "bid", "rec_jump")
  "label_basins" (optimized, closure): Kernel, data args (rec, rec_jump, bid)
  "label_basins" (optimized, cupy): Routine, data names ("rec", "rec_jump", "bid")
    - see _closure_depressions.py/_cupy_depressions.py's build_basin_labelling_* for
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
legacy has it, not "fixed" - see _closure_depressions.py's build_reroute_jump.

`ops.bitpack` (pack/unpack_value/unpack_index) replaces legacy's
f32_i32_struct module for the lexicographic (elevation, target-basin) and
(elevation, node) argmins saddlesort's atomic_min passes need; on cupy,
i64 atomic_min is a CAS loop (CUDA has no native atomicMin over signed long
long) - see _cupy_depressions.py's build_atomic_min_ll.

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
from importlib import import_module

from ..core.context.bag import Bag
from ..core.context.backends import backend_classes
from ..core.context.host_block import HostBlockBuilder
from ..core.context.need import Kind, Need
from ..core.context.sequence_v2 import SequenceBuilder
from ..noise import make_hash_u32

_MODES = frozenset({"steepest", "stochastic"})
_ACCUM_METHODS = frozenset({"atomic", "rake_compress", "pointer_jump_push"})


def _blocks_for(backend: str, section: str):
    """
    The private block module implementing one flow section's device code
    for one backend name - e.g. ("cupy", "depressions") -> _cupy_depressions.

    `section` is one of "receivers", "accum", "depressions", "reconstruct" -
    each has its own {_cupy,_closure}_<section>.py module
    (see those modules' own docstrings for why: flow used to keep every
    algorithm's blocks in one _cupy_blocks.py/_closure_blocks.py pair, split
    apart per algorithm since none of them share device code with each
    other).

    Author: B.G (07/2026)
    """
    if backend in ("taichi", "quadrants"):
        prefix = "_closure"
    elif backend == "cupy":
        prefix = "_cupy"
    else:
        raise ValueError(f"unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")
    return import_module(f".{prefix}_{section}", __package__)


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
    grid,
    *,
    topology: str = "D8",
    mode: str = "steepest",
    diagonal_partition_correction: bool = False,
    h_aware: bool = False,
) -> dict:
    """
    Build one receivers dict: the `receivers` FrozenKernel (data args
    `(z, rec)` or `(z, h, rec)` depending on `h_aware`) plus the distance/
    slope FrozenHelpers it is made of, and `rand_unit` when
    mode="stochastic". See the module docstring for the full contract
    (`topology`, addressing, `SEED`).

    `mode` "steepest"|"stochastic" picks the kernel body variant.
    `diagonal_partition_correction`, `h_aware` and `topology` are documented
    in the module docstring.

    Author: B.G (08/2026)
    """
    if mode not in _MODES:
        raise ValueError(f"make_receivers: mode must be one of {sorted(_MODES)}, got {mode!r}")
    if topology not in ("D4", "D8"):
        raise ValueError(f"make_receivers: topology must be 'D4' or 'D8', got {topology!r}")

    blocks = _blocks_for(backend, "receivers")
    hash_u32 = make_hash_u32(backend) if mode == "stochastic" else None

    if backend in ("taichi", "quadrants"):
        backend_mod, _, _, _ = backend_classes(backend)
        return blocks.build_receivers(
            backend=backend,
            backend_mod=backend_mod,
            grid=grid,
            hash_u32=hash_u32,
            mode=mode,
            topology=topology,
            diagonal_partition_correction=diagonal_partition_correction,
            h_aware=h_aware,
        )
    return blocks.build_receivers(
        grid=grid,
        hash_u32=hash_u32,
        mode=mode,
        topology=topology,
        diagonal_partition_correction=diagonal_partition_correction,
        h_aware=h_aware,
    )


def _resolve_n_flat(grid, n_flat) -> int:
    """
    `n_flat` if given, else `grid["NX"].get() * grid["NY"].get()` - `grid`
    for this fallback must be a `make_grid_parameters` dict (uppercase keys,
    ../grid/__init__.py) with NX/NY bound const-mode; raises immediately,
    naming why, if `grid` is not that shape at all (e.g. a bare
    `make_grid_group` FrozenGroup, which carries no bound Parameter values to
    read at build time) rather than failing on a missing/wrong-shaped key.

    Author: B.G (08/2026)
    """
    if n_flat is not None:
        return int(n_flat)
    if not isinstance(grid, dict) or "NX" not in grid or "NY" not in grid:
        raise ValueError(
            "make_accumulation: n_flat must be given explicitly unless grid is a "
            "make_grid_parameters dict (uppercase NX/NY keys) - a bare make_grid_group "
            "FrozenGroup carries no bound values to read n_flat off"
        )
    nx = grid["NX"].get()
    ny = grid["NY"].get()
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise ValueError(
            "make_accumulation: grid['NX']/grid['NY'] are not const-mode - pass n_flat explicitly"
        )
    return nx * ny


def _resolve_nx_ny(grid) -> tuple:
    """
    (nx, ny) as plain python ints, read off a `make_grid_parameters` dict
    (uppercase NX/NY keys) - raises if `grid` is not that shape, or if
    NX/NY are not const-mode. Unlike _resolve_n_flat there is no override
    argument: the row-length/row-count split (not just their product) is
    load-bearing for make_fill_reconstruct's directional sweeps, so there is
    nothing sensible to fall back to when it is unavailable.

    Author: B.G (08/2026)
    """
    if not isinstance(grid, dict) or "NX" not in grid or "NY" not in grid:
        raise ValueError(
            "make_fill_reconstruct: grid must be a make_grid_parameters dict (uppercase "
            "NX/NY keys) - a bare make_grid_group FrozenGroup carries no bound values to "
            "read nx/ny off"
        )
    nx = grid["NX"].get()
    ny = grid["NY"].get()
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise ValueError(
            "make_fill_reconstruct: grid['NX']/grid['NY'] are not const-mode - a fixed-shape "
            "grid is required for the directional sweep kernels"
        )
    return nx, ny


def _require_param_need(need_obj, label: str, *, factory: str = "make_accumulation") -> None:
    """
    Raise immediately and clearly unless `need_obj` is a `Need(kind=Kind.PARAM)`
    already `.bind()`ed to a Parameter - the boundary check make_accumulation
    runs on `source`/`iteration_p`, make_receivers on `seed_p` and
    make_depressions on `depression_counter_p` before threading each into a
    block module's build_* function (see the module docstring). TypeError for
    anything that is not a Need at all (a bare Parameter passed by mistake);
    ValueError for a Need of the wrong kind or one that is not yet bound.

    Author: B.G (08/2026)
    """
    if not isinstance(need_obj, Need):
        raise TypeError(
            f"{factory}: {label} must be a Need(kind=Kind.PARAM), got "
            f"{type(need_obj).__name__} - build a Need, .bind() it to your Parameter, "
            "and pass that instead of the bare Parameter"
        )
    if need_obj.kind is not Kind.PARAM:
        raise ValueError(f"{factory}: {label} must be kind=Kind.PARAM, got kind={need_obj.kind.value}")
    if not need_obj.is_bound:
        raise ValueError(f"{factory}: {label} is not bound yet - .bind() it to a Parameter before calling")


def make_accumulation(
    backend: str,
    grid,
    source=None,
    *,
    method: str = "rake_compress",
    n_flat: int | None = None,
    n_neighbours: int | None = None,
    iteration_p=None,
    fr_stage: int = 2048,
    blocks_per_sm: int = 2,
    threads: int = 256,
) -> dict:
    """
    Build one accumulation structure for `method`
    "atomic"|"rake_compress"|"pointer_jump_push"|"persistent_mfd".

    method="atomic" (ported to the new builder/frozen/bound stack, ../core/
    context/builder.py/frozen.py/bound.py): `grid` is unused (atomic reads no
    grid at all - it only walks `rec`) and `source` is ignored; `n_flat` is
    REQUIRED (there is no bare-FrozenGroup equivalent of the old `grid.nx.
    get() * grid.ny.get()` fallback - a `make_grid_group` result carries no
    Parameter values to read at build time - see ../grid/__init__.py's own
    module docstring on the structure/data split). Returns
    {"accum": FrozenKernel} on Taichi/Quadrants (data args (rec, q); `SOURCE`
    is this kernel's own wired PARAM slot, bound after `.build()` like any
    other) or {"q_init": FrozenKernel, "accum": FrozenKernel} on cupy (see
    _cupy_accum.py's build_atomic for why cupy needs the second real
    launch): "q_init" (data arg (q,)) must be run, and finish, before
    "accum" (data args (rec, q)); both wire `SOURCE`.

    method="rake_compress"/"pointer_jump_push" (ported to the new builder/
    frozen/bound/sequence stack, ../core/context/builder.py/frozen.py/
    bound.py/sequence_v2.py): `source`/`iteration_p` are accepted here for
    call-site parity with method="persistent_mfd" but are unused and
    ignored for these two - there is no Need indirection anywhere in this
    stack; the returned SequenceBuilder wires bare `SOURCE`/`ITER` PARAM
    slots the caller binds directly, post-`.build()` - see the module
    docstring and _closure_accum.py's/_cupy_accum.py's own docstrings for
    the exact addresses. `n_flat` defaults to grid.nx.get() * grid.ny.get()
    (see _resolve_n_flat) - which requires `grid` to be an old-stack Bag
    with bound Parameters, not a make_grid_group FrozenGroup; these methods
    are not yet callable against the new grid at all.

    method="persistent_mfd" (ported to the new builder/frozen/bound stack,
    ../core/context/builder.py/frozen.py/bound.py, like atomic): `source`/
    `iteration_p` are accepted here for call-site parity with the other three
    methods but unused and ignored - there is no Need indirection in this
    stack; the returned "q_init" FrozenKernel wires a bare `SOURCE` PARAM
    slot (any mode) the caller binds directly, post-`.build()`, exactly like
    `method="atomic"`'s own "q_init".

    `method="persistent_mfd"` is cupy-only (raises for any other backend -
    see _cupy_mfd_accum.py's module docstring for why there is, and will
    never be, a closure-backend equivalent): a persistent-kernel,
    level-synchronous MFD accumulation over a caller-supplied receiver mask
    (`dirs`, u8) + dense per-direction weights (`mfd_w`, f32, this grid's
    n_neighbours values per cell) + `indegree` - this factory does not build
    MFD topology, only accumulates over one already built. Requires `n_flat`
    and `n_neighbours` explicitly (`grid` is a bare `make_grid_group`
    FrozenGroup with no bound Parameter values to read either off, same
    reasoning as `method="atomic"`'s own required `n_flat`). Returns a dict
    with "q_init" (data arg (accum,), FrozenKernel) and "accum" (data args
    (frontier0, frontier1, count, barrier, dirs, mfd_w, accum, indegree),
    FrozenKernel, composing its own `grid` occurrence for
    `ctx.grid.neighbour_raw` - launched with
    `persistent_grid_block(blocks_per_sm=blocks_per_sm, threads=threads)`'s
    fixed (grid, block), not n_flat-sized). `fr_stage`/`blocks_per_sm`/
    `threads` are only used by this method.

    Author: B.G (08/2026)
    """
    if method == "atomic":
        if n_flat is None:
            raise ValueError(
                "make_accumulation: method='atomic' requires n_flat explicitly - "
                "grid is a bare FrozenGroup with no bound values to read it off"
            )
        blocks = _blocks_for(backend, "accum")
        if backend in ("taichi", "quadrants"):
            backend_mod, _, _, _ = backend_classes(backend)
            return {"accum": blocks.build_atomic(backend=backend, backend_mod=backend_mod, n_flat=int(n_flat))}
        return blocks.build_atomic(n_flat=int(n_flat))

    if method == "persistent_mfd":
        if backend != "cupy":
            raise ValueError(
                f"make_accumulation: method='persistent_mfd' is cupy-only (got backend={backend!r}) - "
                "see _cupy_mfd_accum.py's module docstring for why there is no closure-backend equivalent"
            )
        if n_flat is None:
            raise ValueError(
                "make_accumulation: method='persistent_mfd' requires n_flat explicitly - "
                "grid is a bare FrozenGroup with no bound values to read it off"
            )
        if n_neighbours is None:
            raise ValueError(
                "make_accumulation: method='persistent_mfd' requires n_neighbours explicitly - "
                "grid is a bare FrozenGroup with no bound values to read it off"
            )
        from . import _cupy_mfd_accum

        return _cupy_mfd_accum.build_persistent_mfd(
            grid=grid, n_flat=int(n_flat), n_neighbours=int(n_neighbours), fr_stage=fr_stage,
        )

    if method not in _ACCUM_METHODS:
        raise ValueError(f"make_accumulation: method must be one of {sorted(_ACCUM_METHODS)}, got {method!r}")

    # rake_compress/pointer_jump_push (ported to the new builder/frozen/
    # bound/sequence stack - ../core/context/builder.py, frozen.py, bound.py,
    # sequence_v2.py): `source`/`iteration_p` are no longer accepted here at
    # all - there is no Need indirection in this stack (see _closure_accum.py/
    # _cupy_accum.py's module docstrings). Both factories return a
    # SequenceBuilder wiring bare `SOURCE`/`ITER` PARAM slots the caller binds
    # itself, at the addresses each module's build_rake_compress/
    # build_pointer_jump_push docstring enumerates, after `.build()` - exactly
    # like make_receivers' `rand_unit.SEED`.
    blocks = _blocks_for(backend, "accum")
    n_flat_resolved = _resolve_n_flat(grid, n_flat)
    closure = backend in ("taichi", "quadrants")

    logn = math.ceil(math.log2(n_flat_resolved)) + 1

    if method == "rake_compress":
        if closure:
            backend_mod, _, _, _ = backend_classes(backend)
            sb, kernels = blocks.build_rake_compress(backend=backend, backend_mod=backend_mod, grid=grid, logn=logn)
        else:
            sb, kernels = blocks.build_rake_compress(grid=grid, logn=logn, n_flat=n_flat_resolved)
    else:  # pointer_jump_push
        rounds = logn + 1
        if rounds % 2 != 0:
            rounds += 1
        if closure:
            backend_mod, _, _, _ = backend_classes(backend)
            sb, kernels = blocks.build_pointer_jump_push(backend=backend, backend_mod=backend_mod, rounds=rounds)
        else:
            sb, kernels = blocks.build_pointer_jump_push(rounds=rounds, n_flat=n_flat_resolved)

    out = dict(kernels)
    out["sequence"] = sb
    return Bag(out)


# ---------------------------------------------------------------------------
# depressions
# ---------------------------------------------------------------------------

_DEP_METHODS = frozenset({"vanilla", "optimized"})
_DEP_REROUTES = frozenset({"carve", "jump"})


def make_depressions(
    backend: str,
    grid,
    depression_counter_p,
    *,
    method: str = "vanilla",
    reroute: str = "carve",
    n_flat: int,
) -> dict:
    """
    Build one depression-handling dict of unbuilt FrozenKernel/FrozenRoutine
    structures for `method` "vanilla"|"optimized" x `reroute` "carve"|"jump" -
    on the new builder/frozen/bound/routine stack (../core/context/
    builder.py, frozen.py, bound.py, routine_v2.py). Keys: "ndep_p" (the
    caller's own Parameter, passed straight through), "copy_field",
    "depression_counter" (FrozenKernel, data args (rec, ndep) - `ndep` bound
    to `depression_counter_p.get().data`, reset with `.set(0)` before each
    launch), "label_basins" (FrozenKernel for method="optimized" on closure
    backends, FrozenRoutine otherwise - see _closure_depressions.py's/
    _cupy_depressions.py's own build_basin_labelling_* docstrings for the
    exact step names/addresses each combination mints), "saddlesort"
    (FrozenRoutine, unchanged by `method`), "reroute" (FrozenKernel for
    reroute="jump" on closure backends or reroute="carve"+method="optimized"
    on either backend, FrozenRoutine otherwise). Every "label_basins_<name>"/
    "saddlesort_<name>"/"reroute_<name>" constituent FrozenKernel is also
    exposed, mirroring make_accumulation's own routine+constituent-kernels
    convention.

    `depression_counter_p` is a plain Parameter (i32, mode "scalar") the
    caller builds and owns - not built here (this factory takes no pool), no
    Need wrapper (there is no Need indirection in this stack; see the module
    docstring). `grid` is the caller's `make_grid_group` FrozenGroup (device
    structure only) - every site that reaches `can_out`/`neighbour`/
    `N_NEIGHBOURS` composes its OWN independent occurrence of it (build-phase
    `share()` only collapses occurrences within one KernelBuilder's own
    composed subtree, never across sibling routine/sequence steps - see
    _closure_depressions.py's module docstring), so a caller binds the same
    grid Parameter object at every one of those addresses after `.build()`-
    ing whatever contains them - `make_depression_solver` does this
    automatically by leaf name (`_bind_grid_everywhere`), below.

    `n_flat` is required, explicit - this factory takes no pool and reads no
    Parameter for it, and a bare `make_grid_group` FrozenGroup carries no
    bound values to read it off (same reasoning as make_accumulation's
    method="atomic").

    This builds the routines/kernels only; running them in the
    label -> saddlesort -> reroute -> recount loop the algorithm needs is
    make_depression_solver's job, not this one's.

    Author: B.G (08/2026)
    """
    if method not in _DEP_METHODS:
        raise ValueError(f"make_depressions: method must be one of {sorted(_DEP_METHODS)}, got {method!r}")
    if reroute not in _DEP_REROUTES:
        raise ValueError(f"make_depressions: reroute must be one of {sorted(_DEP_REROUTES)}, got {reroute!r}")

    backend_mod, _, _, _ = backend_classes(backend)
    blocks = _blocks_for(backend, "depressions")
    n_flat_resolved = int(n_flat)
    closure = backend in ("taichi", "quadrants")
    logn = math.ceil(math.log2(n_flat_resolved)) + 1

    from ..ops import make_bitpack_group

    bitpack = make_bitpack_group(backend)

    out: dict = {"ndep_p": depression_counter_p}

    if closure:
        copy_field = blocks.build_copy_field(backend=backend, backend_mod=backend_mod)
        depression_counter = blocks.build_depression_counter(backend=backend, backend_mod=backend_mod, grid=grid)
    else:
        copy_field = blocks.build_copy_field(n_flat=n_flat_resolved)
        depression_counter = blocks.build_depression_counter(grid=grid, n_flat=n_flat_resolved)
    out["copy_field"] = copy_field
    out["depression_counter"] = depression_counter

    # basin labelling
    if method == "vanilla":
        if closure:
            lb_rb, lb_kernels = blocks.build_basin_labelling_vanilla(
                backend=backend, backend_mod=backend_mod, grid=grid, copy_field=copy_field, logn=logn,
            )
        else:
            lb_rb, lb_kernels = blocks.build_basin_labelling_vanilla(
                grid=grid, copy_field=copy_field, n_flat=n_flat_resolved, logn=logn,
            )
        out["label_basins"] = lb_rb.freeze()
        for name, kb in lb_kernels.items():
            out[f"label_basins_{name}"] = kb
    else:  # optimized
        if closure:
            out["label_basins"] = blocks.build_basin_labelling_optimized(
                backend=backend, backend_mod=backend_mod, grid=grid, n_flat=n_flat_resolved,
            )
        else:
            lb_rb, lb_kernels = blocks.build_basin_labelling_optimized(grid=grid, n_flat=n_flat_resolved)
            out["label_basins"] = lb_rb.freeze()
            for name, kb in lb_kernels.items():
                out[f"label_basins_{name}"] = kb

    # saddlesort - shared, unchanged by `method`
    if closure:
        ss_rb, ss_kernels = blocks.build_saddlesort(backend=backend, backend_mod=backend_mod, grid=grid, bitpack=bitpack)
    else:
        ss_rb, ss_kernels = blocks.build_saddlesort(grid=grid, bitpack=bitpack, n_flat=n_flat_resolved)
    out["saddlesort"] = ss_rb.freeze()
    for name, kb in ss_kernels.items():
        out[f"saddlesort_{name}"] = kb

    # reroute
    if reroute == "carve":
        if method == "vanilla":
            if closure:
                rr_rb, rr_kernels = blocks.build_reroute_carve_vanilla(
                    backend=backend, backend_mod=backend_mod, bitpack=bitpack, copy_field=copy_field, logn=logn,
                )
            else:
                rr_rb, rr_kernels = blocks.build_reroute_carve_vanilla(
                    bitpack=bitpack, copy_field=copy_field, n_flat=n_flat_resolved, logn=logn,
                )
            out["reroute"] = rr_rb.freeze()
            for name, kb in rr_kernels.items():
                out[f"reroute_{name}"] = kb
        else:  # optimized
            if closure:
                out["reroute"] = blocks.build_reroute_carve_optimized(backend=backend, backend_mod=backend_mod, bitpack=bitpack)
            else:
                out["reroute"] = blocks.build_reroute_carve_optimized(bitpack=bitpack, n_flat=n_flat_resolved)
    else:  # jump
        if closure:
            out["reroute"] = blocks.build_reroute_jump(backend=backend, backend_mod=backend_mod, bitpack=bitpack)
        else:
            rr_rb, rr_kernels = blocks.build_reroute_jump(bitpack=bitpack, n_flat=n_flat_resolved)
            out["reroute"] = rr_rb.freeze()
            for name, kb in rr_kernels.items():
                out[f"reroute_{name}"] = kb

    return out


# ---------------------------------------------------------------------------
# depressions: the outer host-driven loop
# ---------------------------------------------------------------------------


def _bind_by_leaf(bound, prefix: tuple, mapping: dict) -> None:
    """
    Bind every address in `bound` (a BoundSequence) whose path starts with
    `prefix` and whose last segment (leaf name) is a key of `mapping`, to
    that key's value - see make_depression_solver's own docstring for why a
    leaf-name match, rather than an exact per-step address list, is what
    actually copes with saddlesort/reroute/label_basins minting a different
    number of routine steps per backend (cupy splits several closure-backend
    single-kernel passes into multiple real launches - see
    _cupy_depressions.py's module docstring) while every step's own data
    argument names stay identical regardless.

    Author: B.G (08/2026)
    """
    plen = len(prefix)
    for addr in bound.addresses():
        if addr[:plen] == prefix and addr[-1] in mapping:
            bound.bind(addr, mapping[addr[-1]])


def _bind_if_present(bound, addr: tuple, value) -> None:
    """Bind `addr` on `bound` iff it is one of its minted addresses - a no-op otherwise (this method/reroute combination never mints it)."""
    if addr in bound.addresses():
        bound.bind(addr, value)


def _bind_grid_everywhere(bound, grid_params: dict) -> None:
    """
    Bind every occurrence of a grid PARAM leaf (any address ending in
    `(..., "grid", <NAME>)` for `<NAME>` a `grid_params` key) to the matching
    Parameter - see make_depressions' own docstring for why grid is composed
    independently at every site that needs it, never build-phase-collapsed
    across routine/sequence steps, and so needs binding at every one of those
    addresses; this is the generic equivalent of make_accumulation's own
    multi-address `ITER`/`SOURCE` binding, applied by introspection rather
    than a hand-typed address list (the exact set of "grid" occurrences
    varies by `method`/`reroute`/backend - see _closure_depressions.py's/
    _cupy_depressions.py's own build_* docstrings).

    Author: B.G (08/2026)
    """
    for addr in bound.addresses():
        if len(addr) >= 2 and addr[-2] == "grid" and addr[-1] in grid_params:
            bound.bind(addr, grid_params[addr[-1]])


def _require(label: str, **buffers):
    """Raise naming every buffer this combination needs that was left None."""
    missing = sorted(name for name, buf in buffers.items() if buf is None)
    if missing:
        raise ValueError(f"make_depression_solver: {label} requires {missing}")


def make_depression_solver(
    backend: str,
    deps: dict,
    grid_params: dict,
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
    n_flat: int,
    block_size: int = 256,
):
    """
    Compile the outer depression-resolution loop over a dict from
    make_depressions, as a compiled Sequence (sequence_v2.py) - see the
    module docstring for its shape:

        zero_ndep(); depression_counter()
        loop max_times = entry_passes(ndep), until = resolved:
            label_basins; saddlesort; reroute; zero_ndep(); depression_counter()

    `method`/`reroute` must be the ones `deps` was built with; they decide
    which buffers are required and how they map onto each step's DATA
    addresses (`_bind_by_leaf`, above - a data buffer is bound wherever its
    argument NAME occurs anywhere in the composed tree, regardless of how
    many real launches that step split into per backend). Every buffer is a
    raw device buffer (a DataHandle's `.data`), n_flat-sized,
    caller-allocated - this factory allocates nothing.

    `grid_params` is the `make_grid_parameters` dict (../grid/__init__.py)
    backing the same grid `make_depressions` composed - every "grid.<NAME>"
    occurrence anywhere in this sequence's composed tree is bound to
    `grid_params[<NAME>]` (`_bind_grid_everywhere`).

    Required in every combination: `rec` (the authoritative receiver buffer,
    read at entry, resolved on return), `z`, `bid`, `rec_jump`, `z_prime`,
    `is_border`, `basin_saddle`, `basin_saddlenode`, `outlet`. reroute="jump"
    and method="vanilla"+reroute="carve" additionally need `rerouted`; that
    same vanilla carve additionally needs `tag`, `tag_alt` and `rec_scratch`.
    `rerouted` is zeroed by the jump reroute itself but not by the carve one -
    a caller wanting it to mean "rerouted by this call" zeroes it beforehand.

    `n_flat` is required (it sets cupy's launch dimensions and is otherwise
    unused - taichi/quadrants range over the buffers themselves, but every
    KernelBuilder in `deps` was itself already built against this same
    `n_flat`, see make_depressions).

    Returns the compiled Sequence. It takes no arguments, holds the buffers
    given here for its whole life, and reports the passes it took in
    `last_trip_counts[0]` (the loop is this sequence's only loop entry).

    Author: B.G (08/2026)
    """
    if method not in _DEP_METHODS:
        raise ValueError(f"make_depression_solver: method must be one of {sorted(_DEP_METHODS)}, got {method!r}")
    if reroute not in _DEP_REROUTES:
        raise ValueError(f"make_depression_solver: reroute must be one of {sorted(_DEP_REROUTES)}, got {reroute!r}")
    _require(
        "every combination", rec=rec, z=z, bid=bid, rec_jump=rec_jump, z_prime=z_prime,
        is_border=is_border, basin_saddle=basin_saddle, basin_saddlenode=basin_saddlenode, outlet=outlet,
    )
    if reroute == "jump" or (reroute == "carve" and method == "vanilla"):
        _require("reroute='jump' or method='vanilla'+reroute='carve'", rerouted=rerouted)
    if reroute == "carve" and method == "vanilla":
        _require("method='vanilla', reroute='carve'", tag=tag, tag_alt=tag_alt, rec_scratch=rec_scratch)

    ndep_p = deps["ndep_p"]

    def _zero_ndep_tmpl(ctx):
        ctx.NDEP.set(0)

    def _entry_passes_tmpl(ctx):
        ndep = int(ctx.NDEP.read())
        if ndep == 0:
            return 0
        return math.ceil(math.log2(max(2, ndep))) + 2

    def _resolved_tmpl(ctx):
        return int(ctx.NDEP.read()) == 0

    zero_ndep_hb = HostBlockBuilder().wire_param("NDEP").ingest(_zero_ndep_tmpl)
    entry_passes_hb = HostBlockBuilder().wire_param("NDEP").ingest(_entry_passes_tmpl)
    resolved_hb = HostBlockBuilder().wire_param("NDEP").ingest(_resolved_tmpl)

    sb = SequenceBuilder()
    sb.compose("zero_ndep", zero_ndep_hb)
    sb.compose("depression_counter", deps["depression_counter"])
    sb.compose("label_basins", deps["label_basins"])
    sb.compose("saddlesort", deps["saddlesort"])
    sb.compose("reroute", deps["reroute"])
    sb.compose("entry_passes", entry_passes_hb)
    sb.compose("resolved", resolved_hb)

    sb.step("zero_ndep")
    sb.step("depression_counter")
    sb.loop(
        body=["label_basins", "saddlesort", "reroute", "zero_ndep", "depression_counter"],
        max_times="entry_passes",
        until="resolved",
    )

    frozen = sb.freeze()
    bound = frozen.build()

    for name in ("zero_ndep", "entry_passes", "resolved"):
        bound.bind((name, "NDEP"), ndep_p)
    bound.bind(("depression_counter", "rec"), rec)
    bound.bind(("depression_counter", "ndep"), ndep_p.get().data)

    # label_basins: leaf-name binding copes with "vanilla" (a FrozenRoutine,
    # basin_id_init/propagate_iter_K/propagate_basin_final) and "optimized"
    # (a bare FrozenKernel on closure, a 3-step FrozenRoutine on cupy)
    # uniformly. copy_field's own generic "src"/"dst" leaves are ambiguous
    # across occurrences, so that one path is bound explicitly.
    _bind_by_leaf(bound, ("label_basins",), {"rec": rec, "rec_jump": rec_jump, "bid": bid})
    _bind_if_present(bound, ("label_basins", "copy_rec_to_recjump", "src"), rec)
    _bind_if_present(bound, ("label_basins", "copy_rec_to_recjump", "dst"), rec_jump)

    _bind_by_leaf(
        bound, ("saddlesort",),
        {
            "bid": bid, "z": z, "z_prime": z_prime, "is_border": is_border,
            "basin_saddle": basin_saddle, "basin_saddlenode": basin_saddlenode, "outlet": outlet,
        },
    )

    if reroute == "carve" and method == "vanilla":
        # NOTE the naming: build_reroute_carve_vanilla's own "rec" data name is
        # the routine's actively pointer-jumped internal chain - bound to the
        # caller's rec_scratch, not the caller's real rec - and its own
        # "rec_work" is free mid-routine scratch space, bound to the caller's
        # REAL rec buffer (safe: it holds no meaningful graph until the very
        # last copy_field step writes the finalised result back into it from
        # rec_scratch). See make_depressions' own docstring, "Buffer naming".
        leaf_map = {
            "tag": tag, "tag_alt": tag_alt, "rec": rec_scratch, "rec_work": rec, "bid": bid,
            "saddlenode": basin_saddlenode, "outlet": outlet, "rerouted": rerouted, "rec_orig": rec_jump,
        }
        _bind_by_leaf(bound, ("reroute",), leaf_map)
        _bind_if_present(bound, ("reroute", "copy_recwork_to_rec", "src"), rec)
        _bind_if_present(bound, ("reroute", "copy_recwork_to_rec", "dst"), rec_scratch)
        _bind_if_present(bound, ("reroute", "copy_recwork_to_recjump", "src"), rec)
        _bind_if_present(bound, ("reroute", "copy_recwork_to_recjump", "dst"), rec_jump)
        _bind_if_present(bound, ("reroute", "copy_rec_to_recwork", "src"), rec_scratch)
        _bind_if_present(bound, ("reroute", "copy_rec_to_recwork", "dst"), rec)
    elif reroute == "carve":  # optimized
        _bind_by_leaf(bound, ("reroute",), {"rec": rec, "basin_saddlenode": basin_saddlenode, "outlet": outlet})
    else:  # jump
        _bind_by_leaf(bound, ("reroute",), {"rec": rec, "outlet": outlet, "rerouted": rerouted})

    _bind_grid_everywhere(bound, grid_params)

    grid_dims = ((int(n_flat) + block_size - 1) // block_size,), (block_size,)
    if backend == "cupy":
        compiled = bound.compile(backend, grid=grid_dims[0], block=grid_dims[1])
    else:
        compiled = bound.compile(backend)
    return compiled


# ---------------------------------------------------------------------------
# fill by grayscale morphological reconstruction - a standalone alternative
# to make_depressions/make_depression_solver, not a variant of it: no basin
# ids, no saddle search, no outlet routing - one frontier-relaxation kernel
# converging `filled`/`parent` (the receiver graph) directly to a fixed
# point. Ported from experimental/LM/fill_reconstruct_optimised.py - see
# that file's module docstring for the algorithm's derivation and every
# optimisation round it documents, and _cupy_reconstruct.py's/
# _closure_reconstruct.py's own section notes above build_fill_reconstruct_*
# for what changed to make it fit this framework's Sequence-driven,
# data-args-fixed-at-compile-time shape.
# ---------------------------------------------------------------------------


def make_fill_reconstruct(
    backend: str,
    grid,
    *,
    nx: int,
    ny: int,
) -> dict:
    """
    Build one reconstruction-fill dict: `init_filled`, `sweep_row_lr`,
    `sweep_row_rl`, `sweep_col_tb`, `sweep_col_bt`, `frontier_init`, `relax`
    (all FrozenKernels, data args per _cupy_reconstruct.py's/
    _closure_reconstruct.py's build_fill_reconstruct_* docstrings).

    `relax` wires its own `P` PARAM slot (any mode, though the solver needs
    "scalar" since it bumps it between passes via a host block) - a caller
    binds a Parameter there after `.build()`, exactly like make_accumulation's
    `ITER`; there is no Need indirection anywhere in this stack. `grid` is
    the caller's `make_grid_group` FrozenGroup, composed independently by
    `init_filled`/`relax` (see make_depressions' own docstring for why -
    identical reasoning).

    `nx`/`ny` are explicit, required build-time python ints - the row-
    length/row-count split the sweep kernels need is not derivable from a
    bare FrozenGroup (which carries no bound values); n_flat is `nx * ny`.

    Author: B.G (08/2026)
    """
    backend_mod, _, _, _ = backend_classes(backend)
    blocks = _blocks_for(backend, "reconstruct")
    n_flat_resolved = int(nx) * int(ny)
    closure = backend in ("taichi", "quadrants")

    if closure:
        init_filled = blocks.build_fill_reconstruct_init(backend=backend, backend_mod=backend_mod, grid=grid)
        sweeps = blocks.build_fill_reconstruct_sweeps(backend=backend, backend_mod=backend_mod, nx=nx, ny=ny)
        frontier_init = blocks.build_fill_reconstruct_frontier_init(backend=backend, backend_mod=backend_mod)
        relax = blocks.build_fill_reconstruct_relax(
            backend=backend, backend_mod=backend_mod, grid=grid, n_flat=n_flat_resolved,
        )
    else:
        init_filled = blocks.build_fill_reconstruct_init(grid=grid, n_flat=n_flat_resolved)
        sweeps = blocks.build_fill_reconstruct_sweeps(nx=nx, ny=ny)
        frontier_init = blocks.build_fill_reconstruct_frontier_init(n_flat=n_flat_resolved)
        relax = blocks.build_fill_reconstruct_relax(grid=grid, n_flat=n_flat_resolved)

    out: dict = {"init_filled": init_filled, "frontier_init": frontier_init, "relax": relax}
    for name, kb in sweeps.items():
        out[f"sweep_{name}"] = kb
    return out


def make_fill_reconstruct_solver(
    backend: str,
    deps: dict,
    grid_params: dict,
    *,
    z=None,
    filled=None,
    parent=None,
    frontier=None,
    counters=None,
    queued_gen=None,
    pass_p=None,
    active_p=None,
    n_flat: int,
    nx: int,
    ny: int,
    block_size: int = 256,
    max_passes: int | None = None,
):
    """
    Compile the reconstruction-fill outer loop over a dict from
    make_fill_reconstruct, as a compiled Sequence (sequence_v2.py):

        init_filled; sweep_row_lr; sweep_row_rl; sweep_col_tb; sweep_col_bt;
        frontier_init -> counters[0]
        zero_pass(): pass_p = 0
        loop max_times = max_passes, until = converged:
            zero_active(): active_p = 0
            relax(pass_p) -> active_p += 1 per push
            bump_pass(): pass_p += 1

    Early stop, mirroring make_depression_solver's `ndep_p`/`resolved`
    pattern: `active_p` is a caller-allocated scalar i32 Parameter, zeroed by
    a host block before each `relax` call, and `relax` itself atomic-adds
    into its raw backing buffer for every node it pushes into the next
    pass's frontier (`_closure_reconstruct.py`/`_cupy_reconstruct.py`'s
    `build_fill_reconstruct_relax`) - the same "concurrently mutated is DATA
    by definition" classification `ndep_p`/`depression_counter` already use.
    A `converged` host block reads it back with `.read()` after each
    iteration; the loop stops once a pass pushes nothing.

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
    call** for the same reason. `pass_p` is a caller-allocated scalar i32
    Parameter (mode "scalar") - bound to `relax`'s own wired `P` PARAM slot
    and bumped here by a host block between passes. `active_p` is a second
    caller-allocated scalar i32 Parameter (mode "scalar") - no caller-side
    init needed, `zero_active` resets it every iteration before `relax` runs.

    `n_flat`/`nx`/`ny` are required - `n_flat` sets cupy's launch dimensions
    (unused on taichi/quadrants); `nx`/`ny` are only used, if `max_passes` is
    not given, for the `4 * max(nx, ny)` default below.

    `max_passes` defaults to `4 * max(nx, ny)` - generous headroom over the
    measured ~0.6x that ratio in
    experimental/LM/fill_reconstruct_optimised.py.

    `grid_params` is unused here (relax's own `grid` PARAM addresses are
    bound directly below via `_bind_grid_everywhere`) - accepted for call-site
    parity with make_depression_solver.

    Returns the compiled Sequence. It takes no arguments.

    Author: B.G (08/2026)
    """
    _require(
        "make_fill_reconstruct_solver", z=z, filled=filled, parent=parent,
        frontier=frontier, counters=counters, queued_gen=queued_gen, pass_p=pass_p, active_p=active_p,
    )
    if max_passes is None:
        max_passes = 4 * max(int(nx), int(ny))

    def _zero_pass_tmpl(ctx):
        ctx.P.set(0)

    def _bump_pass_tmpl(ctx):
        ctx.P.set(int(ctx.P.read()) + 1)

    def _zero_active_tmpl(ctx):
        ctx.ACTIVE.set(0)

    def _converged_tmpl(ctx):
        return int(ctx.ACTIVE.read()) == 0

    zero_pass_hb = HostBlockBuilder().wire_param("P").ingest(_zero_pass_tmpl)
    bump_pass_hb = HostBlockBuilder().wire_param("P").ingest(_bump_pass_tmpl)
    zero_active_hb = HostBlockBuilder().wire_param("ACTIVE").ingest(_zero_active_tmpl)
    converged_hb = HostBlockBuilder().wire_param("ACTIVE").ingest(_converged_tmpl)

    sb = SequenceBuilder()
    sb.compose("init_filled", deps["init_filled"])
    sb.compose("sweep_row_lr", deps["sweep_row_lr"])
    sb.compose("sweep_row_rl", deps["sweep_row_rl"])
    sb.compose("sweep_col_tb", deps["sweep_col_tb"])
    sb.compose("sweep_col_bt", deps["sweep_col_bt"])
    sb.compose("frontier_init", deps["frontier_init"])
    sb.compose("zero_pass", zero_pass_hb)
    sb.compose("relax", deps["relax"])
    sb.compose("bump_pass", bump_pass_hb)
    sb.compose("zero_active", zero_active_hb)
    sb.compose("converged", converged_hb)

    sb.step("init_filled")
    sb.step("sweep_row_lr")
    sb.step("sweep_row_rl")
    sb.step("sweep_col_tb")
    sb.step("sweep_col_bt")
    sb.step("frontier_init")
    sb.step("zero_pass")
    sb.loop(body=["zero_active", "relax", "bump_pass"], max_times=int(max_passes), until="converged")

    frozen = sb.freeze()
    bound = frozen.build()

    zpf = {"z": z, "filled": filled, "parent": parent}
    for step in ("init_filled", "sweep_row_lr", "sweep_row_rl", "sweep_col_tb", "sweep_col_bt"):
        _bind_by_leaf(bound, (step,), zpf)
    _bind_by_leaf(bound, ("frontier_init",), {"z": z, "filled": filled, "frontier": frontier, "counters": counters})
    _bind_by_leaf(
        bound, ("relax",),
        {"z": z, "filled": filled, "parent": parent, "frontier": frontier, "counters": counters, "queued_gen": queued_gen},
    )
    bound.bind(("relax", "active"), active_p.get().data)
    bound.bind(("relax", "P"), pass_p)
    bound.bind(("zero_pass", "P"), pass_p)
    bound.bind(("bump_pass", "P"), pass_p)
    bound.bind(("zero_active", "ACTIVE"), active_p)
    bound.bind(("converged", "ACTIVE"), active_p)
    _bind_grid_everywhere(bound, grid_params)

    grid_dims = ((int(n_flat) + block_size - 1) // block_size,), (block_size,)
    if backend == "cupy":
        return bound.compile(backend, grid=grid_dims[0], block=grid_dims[1])
    return bound.compile(backend)
