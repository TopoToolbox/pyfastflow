"""
Taichi/Quadrants (closure) block templates behind make_fill_reconstruct/
make_fill_reconstruct_solver - see _cupy_reconstruct.py's own module
docstring for the algorithm (ported from
experimental/LM/fill_reconstruct_optimised.py) and for why frontier_a/
frontier_b become one combined (2*n_flat,) buffer here, addressed by a
`p % 2` parity computed from the bound `P` Parameter - identical reasoning
on this backend, since a Sequence's kernel_step binds data once at compile
time on every backend, not just cupy.

Split out of a single _closure_blocks.py that used to hold every flow
algorithm - see _closure_receivers.py/_closure_accum.py/
_closure_depressions.py for the others.

Two closure-specific substitutions from the cupy version, both verified
directly against this Taichi/Quadrants install before use here:

- No 3-argument `range()` (reverse step) inside a kernel - confirmed
  Taichi rejects it ("Range should have 1 or 2 arguments"). The two
  right-to-left/bottom-to-top sweeps below instead drive a forward-
  counting loop variable and compute the descending index from it
  (`c = NX - 2 - cc`); still a single serial nested loop per thread, same
  execution order as a real reverse range.
- No `atomicExch` on Taichi (only Quadrants has `atomic_exchange`) - both
  backends use `_BK.atomic_max(queued_gen[j], p)` instead, whose returned
  old value gives the identical "first writer this pass wins" dedup
  `atomicExch` does, because `p` only ever increases across passes: the
  first thread to touch queued_gen[j] this pass raises it from some
  earlier (smaller) value to p and gets that smaller value back; every
  later thread doing the same atomic_max this pass finds it already at p,
  contributes no change, and gets p back - confirmed empirically (a fresh
  -1-filled field, one atomic_max(..., p) per candidate, only the winner's
  returned old value differs from p) before relying on it here.

Author: B.G (07/2026)
"""

from ..core.context.need import Kind, Need
from ._closure_shared import _tensor_annotation

_POS_SENTINEL = 1.0e9


def build_fill_reconstruct_init(KernelCls, *, backend, backend_mod, grid):
    """
    init_filled KernelBuilder, data args (z, filled, parent) - see
    _cupy_blocks.build_fill_reconstruct_init.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def init_filled_template(z: T, filled: T, parent: T):
        for i in z:
            if _GRID.can_out(i):
                filled[i] = z[i]
                parent[i] = i
            else:
                filled[i] = _POS_SENTINEL
                parent[i] = -1

    return KernelCls().bind("_GRID", grid).ingest(init_filled_template)


def build_fill_reconstruct_sweeps(KernelCls, *, backend, backend_mod, nx: int, ny: int):
    """
    Four KernelBuilders, each data args (z, filled, parent) - see
    _cupy_blocks.build_fill_reconstruct_sweeps. Keyed "row_lr", "row_rl",
    "col_tb", "col_bt".

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NX = nx
    NY = ny

    def sweep_row_lr_template(z: T, filled: T, parent: T):
        for r in range(NY):
            base = r * NX
            for c in range(1, NX):
                i = base + c
                left = i - 1
                cand = z[i] if z[i] > filled[left] else filled[left]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = left

    def sweep_row_rl_template(z: T, filled: T, parent: T):
        for r in range(NY):
            base = r * NX
            for cc in range(NX - 1):
                c = NX - 2 - cc
                i = base + c
                right = i + 1
                cand = z[i] if z[i] > filled[right] else filled[right]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = right

    def sweep_col_tb_template(z: T, filled: T, parent: T):
        for c in range(NX):
            for r in range(1, NY):
                i = r * NX + c
                up = i - NX
                cand = z[i] if z[i] > filled[up] else filled[up]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = up

    def sweep_col_bt_template(z: T, filled: T, parent: T):
        for c in range(NX):
            for rr in range(NY - 1):
                r = NY - 2 - rr
                i = r * NX + c
                down = i + NX
                cand = z[i] if z[i] > filled[down] else filled[down]
                if cand < filled[i]:
                    filled[i] = cand
                    parent[i] = down

    return {
        "row_lr": KernelCls().ingest(sweep_row_lr_template),
        "row_rl": KernelCls().ingest(sweep_row_rl_template),
        "col_tb": KernelCls().ingest(sweep_col_tb_template),
        "col_bt": KernelCls().ingest(sweep_col_bt_template),
    }


def build_fill_reconstruct_frontier_init(KernelCls, *, backend, backend_mod):
    """
    frontier_init KernelBuilder, data args (z, filled, frontier, counters) -
    see _cupy_blocks.build_fill_reconstruct_frontier_init.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)

    def frontier_init_template(z: T, filled: T, frontier: T, counters: T):
        for i in z:
            if filled[i] > z[i]:
                pos = _BK.atomic_add(counters[0], 1)
                frontier[pos] = i

    return KernelCls().bind("_BK", backend_mod).ingest(frontier_init_template)


def build_fill_reconstruct_relax(KernelCls, *, backend, backend_mod, grid, pass_p: Need, n_flat: int):
    """
    relax KernelBuilder, data args (z, filled, parent, frontier, counters,
    queued_gen) - see _cupy_blocks.build_fill_reconstruct_relax; the same
    push-gate and combined-buffer-parity logic, without the cupy version's
    neighbour-value caching (see that function's docstring for why dropping
    it does not change correctness) - a top-level `for idx in range(count)`
    with `count` read from `counters[p]` at kernel entry, confirmed to
    compile and execute correctly with a runtime (not compile-time) bound on
    this Taichi/Quadrants install before relying on it here.

    `pass_p` is the caller's already-bound `Need("pass_p", kind=Kind.PARAM)`
    (see make_fill_reconstruct) - a fresh, internally-named
    `Need("_P", ...)`, matching this template's own `_P.get(0)` reference, is
    bound here to the same underlying Parameter and declared on this
    KernelBuilder via `.need()`. Read only here; bumping it between passes is
    the caller's job, the same division of labour `iteration_p` has for
    rake_compress.

    Author: B.G (07/2026)
    """
    T = _tensor_annotation(backend_mod, backend)
    NFLAT = n_flat

    p_need = Need("_P", kind=Kind.PARAM, dtype=pass_p.dtype, modes=pass_p.modes)
    p_need.bind(pass_p.value)

    def relax_template(z: T, filled: T, parent: T, frontier: T, counters: T, queued_gen: T):
        p = _P.get(0)
        par = p % 2
        in_base = par * NFLAT
        out_base = (1 - par) * NFLAT
        count = counters[p]
        for idx in range(count):
            i = frontier[in_base + idx]
            nk = _GRID.n_neighbours.get(0)

            best = _POS_SENTINEL
            best_j = -1
            for k in range(nk):
                j = _GRID.neighbour(i, k)
                if j != -1:
                    v = filled[j]
                    if v < best:
                        best = v
                        best_j = j
            candidate = z[i] if z[i] > best else best

            if candidate < filled[i]:
                filled[i] = candidate
                parent[i] = best_j
                for k in range(nk):
                    j = _GRID.neighbour(i, k)
                    if j != -1:
                        cand_j = z[j] if z[j] > candidate else candidate
                        if cand_j < filled[j]:
                            old = _BK.atomic_max(queued_gen[j], p)
                            if old != p:
                                pos = _BK.atomic_add(counters[p + 1], 1)
                                frontier[out_base + pos] = j

    return (
        KernelCls()
        .bind("_GRID", grid)
        .bind("_BK", backend_mod)
        .need(p_need)
        .ingest(relax_template)
    )
