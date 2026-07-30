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

Author: B.G (07/2026)
"""

import math

from ..core.context.bag import Bag
from ..core.context.backends import backend_classes
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
