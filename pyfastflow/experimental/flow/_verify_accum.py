"""
Standalone, re-runnable verification of make_accumulation at a grid scale
large enough for the three accumulation methods to visibly disagree at f32
precision.

Why scale matters: with `source` = 1.0 per cell, `q` is literally drainage
area in cells, so the outlet accumulates a value close to n_flat. f32 has an
ulp of ~1.2e-4 at 1e3 and ~6e-2 at 1e6 - three methods that sum the same
values in genuinely different orders (a strict serial walk for "atomic", a
donor-tree rake-and-compress for "rake_compress", a power-of-two pointer-
jump doubling for "pointer_jump_push") are expected to disagree by a
noticeable fraction of an ulp at that magnitude, not agree to 1e-6 the way a
small, lightly-loaded grid would make them appear to. This script exists
because an earlier, smaller-scale (1920-node, random small `source`) run of
this same check produced deviations that looked deceptively clean (~1e-6
absolute against a maximum accumulated value of order 10) and was not
actually stressing the three summation orders apart.

What it checks, per backend (taichi/quadrants/cupy) and per method (atomic/
rake_compress/pointer_jump_push): builds a 1024x1024 D8 grid (or the largest
power-of-two-side grid that fits in free GPU memory - see MAX_SIDE_OVERRIDE
below), runs make_receivers(mode="steepest") to get a real receiver graph,
computes a numpy topological reference (summing along the same receiver
chains, in float64), runs each accumulation method, and reports max absolute
deviation, max relative deviation, AND the max accumulated value itself, so
the deviations can be read against the scale they occur on. It also reports
the receiver graph's maximum chain depth (root distance), computed by
iterative path compression, so it's clear whether the graph exercised was
actually deep or just wide.

On taichi/quadrants, a "fuse-check" mode also compiles rake_compress's and
pointer_jump_push's RoutineBuilder both fused=True and fused=False and diffs
the two outputs directly - the nested-def templates these two routines are
built from fuse via core/context/_closure_backend.py's
capture_template_meta/_fuse_group. cupy has no fuse path and is skipped.

`source` mode coverage (const/scalar/field) is exercised on taichi only, to
prove mode-agnosticism without tripling the run time; quadrants/cupy run
field mode only. A further "noninteger" pass, run on every backend, repeats
the same receiver graph with a non-integer per-cell source: source=1.0
makes every intermediate partial sum an exact integer, and f32 represents
every integer up to 2**24 exactly regardless of summation order, so the
const/scalar/field passes are expected to show exactly 0.0 deviation
whenever the accumulated total stays under 2**24 - agreement there is a
mathematical fact about IEEE-754, not evidence the three summation orders
are equivalent in general. The noninteger pass is the actual check of that.

Run:
    python -m pyfastflow.experimental.flow._verify_accum taichi
    python -m pyfastflow.experimental.flow._verify_accum quadrants
    python -m pyfastflow.experimental.flow._verify_accum cupy

Author: B.G (07/2026)
"""

import math
import sys
from collections import deque

import numpy as np
from scipy.ndimage import gaussian_filter

DX = 1.0
SEED = 2024
# Grid side length; make_accumulation's own n_flat default (grid.nx*grid.ny)
# is used throughout. Override to a smaller power of two if this exhausts
# free GPU memory - the rake-compress donor buffers are n_flat*n_neighbours
# int32 each (two of them), the dominant cost.
SIDE = 1024


def numpy_topological_accum(rec: np.ndarray, source: np.ndarray) -> np.ndarray:
    """
    Reference accumulation: sum `source` along the receiver forest `rec`
    defines, in float64, via one indegree-driven topological pass (leaves
    first). O(n).

    Author: B.G (07/2026)
    """
    n = rec.shape[0]
    q = source.astype(np.float64).copy()
    indeg = np.zeros(n, dtype=np.int64)
    is_root = rec == np.arange(n)
    np.add.at(indeg, rec[~is_root], 1)
    dq = deque(int(i) for i in range(n) if indeg[i] == 0)
    while dq:
        i = dq.popleft()
        r = rec[i]
        if r != i:
            q[r] += q[i]
            indeg[r] -= 1
            if indeg[r] == 0:
                dq.append(int(r))
    return q


def make_smooth_terrain(nx: int, ny: int, seed: int) -> np.ndarray:
    """
    Spatially-correlated elevation: i.i.d. uniform noise passed through a
    Gaussian blur (sigma ~ 1% of the grid side). Plain i.i.d. per-cell noise
    makes ~1/(n_neighbours+1) of the interior nodes local minima independent
    of their surroundings, which packs pits so densely that every receiver
    chain is only a few hops long and no basin accumulates more than a
    handful of cells - not a real test of f32 summation order at scale. A
    Gaussian blur introduces the spatial correlation real topography has, so
    local minima are basin-scale rather than cell-scale and receiver chains
    actually run the width of a basin.

    Author: B.G (07/2026)
    """
    rng = np.random.default_rng(seed)
    raw = rng.random((ny, nx))
    sigma = max(nx, ny) * 0.02
    smooth = gaussian_filter(raw, sigma=sigma, mode="reflect")
    return smooth.astype(np.float32).ravel()


def max_chain_depth(rec: np.ndarray) -> int:
    """
    Longest root distance (number of receiver hops to reach a self-receiver)
    over the whole graph, via iterative path compression - O(n) amortized,
    no python recursion (which would blow the recursion limit on a
    million-node chain).

    Author: B.G (07/2026)
    """
    n = rec.shape[0]
    depth = np.full(n, -1, dtype=np.int64)
    depth[rec == np.arange(n)] = 0
    for i in range(n):
        if depth[i] != -1:
            continue
        path = []
        j = i
        while depth[j] == -1:
            path.append(j)
            j = rec[j]
        base = int(depth[j])
        for node in reversed(path):
            base += 1
            depth[node] = base
    return int(depth.max())


def run(backend: str):
    if backend == "taichi":
        import taichi as ti
        ti.init(arch=ti.gpu)
        from pyfastflow.experimental.core.context.taichi_backend import TaichiParameter
        from pyfastflow.experimental.core.pool.taichi_pool import TaichiPool
        Param, Pool, i32, f32 = TaichiParameter, TaichiPool, ti.i32, ti.f32
    elif backend == "quadrants":
        import quadrants as qd
        qd.init(arch=qd.gpu)
        from pyfastflow.experimental.core.context.quadrants_backend import QuadrantsParameter
        from pyfastflow.experimental.core.pool.quadrants_pool import QuadrantsPool
        Param, Pool, i32, f32 = QuadrantsParameter, QuadrantsPool, qd.i32, qd.f32
    elif backend == "cupy":
        from pyfastflow.experimental.core.context.cupy_backend import CupyParameter
        from pyfastflow.experimental.core.pool.cupy_pool import CupyPool
        Param, Pool, i32, f32 = CupyParameter, CupyPool, np.int32, np.float32
    else:
        raise ValueError(f"unknown backend {backend!r}")

    from pyfastflow.experimental.grid import make_grid
    from pyfastflow.experimental.flow import make_receivers, make_accumulation

    nx = ny = SIDE
    n = nx * ny
    nn = 8  # D8

    pool = Pool()
    grid = make_grid(backend, pool, nx, ny, DX, topology="D8", boundary="normal", outlet="edge")

    z_np = make_smooth_terrain(nx, ny, SEED)
    source_np = np.ones(n, dtype=np.float32)

    z = pool.get_data(f32, (n,))
    rec = pool.get_data(i32, (n,))
    if backend == "cupy":
        z.data.set(z_np)
        launch_grid, launch_block = ((n + 255) // 256,), (256,)
    else:
        z.data.from_numpy(z_np)

    recv = make_receivers(backend, grid)
    recv_kernel = recv.receivers.compile()
    if backend == "cupy":
        recv_kernel(z.data, rec.data, grid=launch_grid, block=launch_block)
        rec_np = rec.data.get().astype(np.int64)
    else:
        recv_kernel(z.data, rec.data)
        rec_np = rec.data.to_numpy().astype(np.int64)

    depth = max_chain_depth(rec_np)
    ref = numpy_topological_accum(rec_np, source_np)
    max_q_ref = float(ref.max())

    def get_source_param(mode):
        if mode in ("const", "scalar"):
            return Param("SRC", dtype=f32, mode=mode, value=1.0, pool=pool)
        p = Param("SRC", dtype=f32, mode="field", value=source_np, pool=pool, n_flat=n)
        return p

    def run_atomic(source_p):
        accum = make_accumulation(backend, grid, source_p, method="atomic")
        q = pool.get_data(f32, (n,))
        if backend == "cupy":
            q_init_k = accum.q_init.compile()
            accum_k = accum.accum.compile()
            q_init_k(q.data, grid=launch_grid, block=launch_block)
            accum_k(rec.data, q.data, grid=launch_grid, block=launch_block)
            got = q.data.get().astype(np.float64)
        else:
            k = accum.accum.compile()
            k(rec.data, q.data)
            got = q.data.to_numpy().astype(np.float64)
        pool.release_data(q)
        return got

    def run_rake_compress(source_p, fused=False):
        iteration_p = Param("ITER", dtype=i32, mode="scalar", value=0, pool=pool)
        accum = make_accumulation(backend, grid, source_p, method="rake_compress", n_flat=n, iteration_p=iteration_p)
        q = pool.get_data(f32, (n,))
        donors = pool.get_data(i32, (n * nn,))
        ndonors = pool.get_data(i32, (n,))
        donors_alt = pool.get_data(i32, (n * nn,))
        ndonors_alt = pool.get_data(i32, (n,))
        q_alt = pool.get_data(f32, (n,))
        src = pool.get_data(i32, (n,))

        routine = accum.routine.compile(captured=False) if backend == "cupy" else accum.routine.compile(fused=fused)

        names = routine.data_names
        handles = {
            "rec": rec.data, "q": q.data, "donors": donors.data, "ndonors": ndonors.data,
            "donors_alt": donors_alt.data, "ndonors_alt": ndonors_alt.data, "q_alt": q_alt.data, "src": src.data,
        }
        args = tuple(handles[nm] for nm in names)
        routine(*args)

        got = q.data.get().astype(np.float64) if backend == "cupy" else q.data.to_numpy().astype(np.float64)

        for h in (q, donors, ndonors, donors_alt, ndonors_alt, q_alt, src):
            pool.release_data(h)
        iteration_p.destroy()
        return got

    def run_pointer_jump_push(source_p, fused=False):
        accum = make_accumulation(backend, grid, source_p, method="pointer_jump_push", n_flat=n)
        q = pool.get_data(f32, (n,))
        work = pool.get_data(i32, (n,))
        work2 = pool.get_data(i32, (n,))
        q_work = pool.get_data(f32, (n,))

        routine = accum.routine.compile(captured=False) if backend == "cupy" else accum.routine.compile(fused=fused)

        names = routine.data_names
        handles = {"rec": rec.data, "work": work.data, "work2": work2.data, "q": q.data, "q_work": q_work.data}
        args = tuple(handles[nm] for nm in names)
        routine(*args)

        got = q.data.get().astype(np.float64) if backend == "cupy" else q.data.to_numpy().astype(np.float64)

        for h in (q, work, work2, q_work):
            pool.release_data(h)
        return got

    modes = ["const", "scalar", "field"] if backend == "taichi" else ["field"]
    rows = []
    for mode in modes:
        source_p = get_source_param(mode)
        got_atomic = run_atomic(source_p)
        got_rake = run_rake_compress(source_p)
        got_pjp = run_pointer_jump_push(source_p)
        source_p.destroy()

        for name, got in (("atomic", got_atomic), ("rake_compress", got_rake), ("pointer_jump_push", got_pjp)):
            abs_diff = float(np.max(np.abs(got - ref)))
            rel_diff = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)))
            rows.append((mode, name, abs_diff, rel_diff, float(got.max())))

        rows.append((mode, "atomic_vs_rake", float(np.max(np.abs(got_atomic - got_rake))),
                     float(np.max(np.abs(got_atomic - got_rake) / np.maximum(np.abs(got_atomic), 1.0))), None))
        rows.append((mode, "atomic_vs_pjp", float(np.max(np.abs(got_atomic - got_pjp))),
                     float(np.max(np.abs(got_atomic - got_pjp) / np.maximum(np.abs(got_atomic), 1.0))), None))

    # Nested-def templates fusing (core/context/_closure_backend.py's
    # capture_template_meta/_fuse_group): compile() each RoutineBuilder both
    # ways and diff element-wise. cupy has no fuse path (RoutineBuilder.compile
    # raises on fused=True there), so this only runs on the two closure
    # backends.
    if backend != "cupy":
        source_p = get_source_param("field")
        got_rake_unfused = run_rake_compress(source_p, fused=False)
        got_rake_fused = run_rake_compress(source_p, fused=True)
        got_pjp_unfused = run_pointer_jump_push(source_p, fused=False)
        got_pjp_fused = run_pointer_jump_push(source_p, fused=True)
        source_p.destroy()
        rows.append(("fuse-check", "rake_compress_fused_vs_unfused",
                     float(np.max(np.abs(got_rake_fused - got_rake_unfused))),
                     float(np.max(np.abs(got_rake_fused - got_rake_unfused) / np.maximum(np.abs(got_rake_unfused), 1.0))),
                     None))
        rows.append(("fuse-check", "pointer_jump_push_fused_vs_unfused",
                     float(np.max(np.abs(got_pjp_fused - got_pjp_unfused))),
                     float(np.max(np.abs(got_pjp_fused - got_pjp_unfused) / np.maximum(np.abs(got_pjp_unfused), 1.0))),
                     None))

    # source=1.0 makes every partial sum an exact integer, and f32 represents
    # every integer up to 2**24 exactly regardless of summation order - so
    # the block above is expected to show 0.0 deviation everywhere as long as
    # max_q_reference stays under 2**24, no matter how deep or wide the
    # receiver graph is. This second, non-integer source pass is the direct
    # check of that: same receiver graph, weights that cannot all land on
    # exact binary fractions, so the three summation orders' rounding should
    # actually separate.
    rng2 = np.random.default_rng(SEED + 1)
    source_noninteger_np = (rng2.random(n).astype(np.float32) * 2.0 + 0.1)
    ref_noninteger = numpy_topological_accum(rec_np, source_noninteger_np)
    source_p = Param("SRC", dtype=f32, mode="field", value=source_noninteger_np, pool=pool, n_flat=n)
    got_atomic = run_atomic(source_p)
    got_rake = run_rake_compress(source_p)
    got_pjp = run_pointer_jump_push(source_p)
    source_p.destroy()
    for name, got in (("atomic", got_atomic), ("rake_compress", got_rake), ("pointer_jump_push", got_pjp)):
        abs_diff = float(np.max(np.abs(got - ref_noninteger)))
        rel_diff = float(np.max(np.abs(got - ref_noninteger) / np.maximum(np.abs(ref_noninteger), 1.0)))
        rows.append(("noninteger", name, abs_diff, rel_diff, float(got.max())))
    rows.append(("noninteger", "atomic_vs_rake", float(np.max(np.abs(got_atomic - got_rake))),
                 float(np.max(np.abs(got_atomic - got_rake) / np.maximum(np.abs(got_atomic), 1.0))), None))
    rows.append(("noninteger", "atomic_vs_pjp", float(np.max(np.abs(got_atomic - got_pjp))),
                 float(np.max(np.abs(got_atomic - got_pjp) / np.maximum(np.abs(got_atomic), 1.0))), None))

    pool.release_data(z)
    pool.release_data(rec)
    return n, depth, max_q_ref, rows


# ---------------------------------------------------------------------------
# persistent_mfd (cupy-only) - method="persistent_mfd" of make_accumulation.
#
# There is no MFD topology anywhere in this codebase yet (CLAUDE.md: "MFD is
# entirely unported") - persistent_mfd only accumulates over a caller-built
# receiver mask/weight pair, so this section fabricates one directly in
# numpy rather than deriving it from a receivers factory. `_MFD_D8_DR`/
# `_MFD_D8_DC` are the exact same 8 offsets, same k order, as grid's own
# `_delta` table (_cupy_blocks.py's build_helpers) - the persistent kernel
# calls `neighbour_raw(u, k)` on the real grid to resolve a set mask bit,
# so the reference topology's own row/col arithmetic has to agree with the
# grid's or the two would silently walk to different cells. Restricting
# every candidate direction to a strictly-greater flat index (row-major)
# makes the fabricated graph acyclic by construction - no priority-flood /
# depression handling involved, this is a synthetic DAG, not a terrain.
# ---------------------------------------------------------------------------

_MFD_D8_DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
_MFD_D8_DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
MFD_SIDE = 256
MFD_KEEP_PROB = 0.6


def make_synthetic_mfd_topology(nx: int, ny: int, seed: int):
    """
    Fabricate an acyclic MFD receiver mask (u8, one bitmask/cell) + dense
    per-direction weights (f32, 8/cell, unset directions left 0.0) over an
    nx*ny D8 grid with "normal" (non-wrapping) boundaries - the same
    boundary convention _verify_accum.py's `make_grid(..., boundary="normal")`
    calls build with elsewhere in this file.

    Every direction whose neighbour would fall outside [0, nx)x[0, ny) is
    dropped (mirrors what a real MFD topology builder's own bounds check
    would do); every surviving candidate is independently kept with
    probability MFD_KEEP_PROB and restricted to a strictly greater flat
    index (row-major) - guaranteeing the whole graph is a DAG regardless of
    which candidates survive. A cell with no surviving candidate is a sink
    (mask 0) - fine, it just never propagates further, same as a
    self-receiver in the SFD convention.

    Returns (dirs: u8[n], mfd_w: f32[n*8], indegree: i64[n]) - all plain
    numpy, in this grid's own flat/row-major indexing.

    Author: B.G (08/2026)
    """
    n = nx * ny
    rng = np.random.default_rng(seed)
    rows = np.arange(n) // nx
    cols = np.arange(n) % nx

    dirs = np.zeros(n, dtype=np.uint8)
    mfd_w = np.zeros(n * 8, dtype=np.float32)
    indegree = np.zeros(n, dtype=np.int64)

    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        candidates = []
        for k in range(8):
            nr, nc = r + int(_MFD_D8_DR[k]), c + int(_MFD_D8_DC[k])
            if nr < 0 or nr >= ny or nc < 0 or nc >= nx:
                continue
            j = nr * nx + nc
            if j <= i:
                continue
            if rng.random() < MFD_KEEP_PROB:
                candidates.append((k, j))
        if not candidates:
            continue
        weights = rng.random(len(candidates)).astype(np.float64) + 0.1
        weights /= weights.sum()
        mask = 0
        for (k, j), w in zip(candidates, weights):
            mask |= (1 << k)
            mfd_w[i * 8 + k] = np.float32(w)
            indegree[j] += 1
        dirs[i] = np.uint8(mask)

    return dirs, mfd_w, indegree


def numpy_kahn_mfd_accum(dirs: np.ndarray, mfd_w: np.ndarray, indegree: np.ndarray, source: np.ndarray) -> np.ndarray:
    """
    Reference MFD accumulation: Kahn-queue topological pass (leaves first,
    i.e. indegree==0 first) over the fabricated (dirs, mfd_w, indegree)
    graph, in float64. O(n * 8).

    Author: B.G (08/2026)
    """
    n = dirs.shape[0]
    accum = source.astype(np.float64).copy()
    indeg = indegree.copy()
    dq = deque(int(i) for i in range(n) if indeg[i] == 0)
    while dq:
        u = dq.popleft()
        au = accum[u]
        mask = int(dirs[u])
        base = u * 8
        # neighbour indices are recomputed the same way the persistent
        # kernel's neighbour_raw does (row/col + D8 offset), since this
        # graph representation stores only the mask + weights, not the
        # resolved neighbour flat indices themselves.
        row, col = divmod(u, _MFD_NX[0])
        for k in range(8):
            if not (mask & (1 << k)):
                continue
            nr, nc = row + int(_MFD_D8_DR[k]), col + int(_MFD_D8_DC[k])
            j = nr * _MFD_NX[0] + nc
            accum[j] += au * mfd_w[base + k]
            indeg[j] -= 1
            if indeg[j] == 0:
                dq.append(int(j))
    return accum


# mutable 1-element box so numpy_kahn_mfd_accum (defined above run_mfd_cupy)
# can read the grid width without threading nx through every call in this
# already-fixed reference-function signature; set once in run_mfd_cupy.
_MFD_NX = [0]


def run_mfd_cupy():
    """
    cupy-only verification of method="persistent_mfd": builds a
    MFD_SIDE x MFD_SIDE synthetic DAG (make_synthetic_mfd_topology), runs
    the persistent-kernel accumulation with a unit source, and compares
    against numpy_kahn_mfd_accum. Returns (n_flat, max_abs, max_rel, max_got,
    max_ref).

    Author: B.G (08/2026)
    """
    from pyfastflow.experimental.core.context.cupy_backend import CupyParameter
    from pyfastflow.experimental.core.pool.cupy_pool import CupyPool
    from pyfastflow.experimental.grid import make_grid
    from pyfastflow.experimental.flow import make_accumulation
    from pyfastflow.experimental.flow._cupy_mfd_accum import persistent_grid_block, init_frontier_mfd

    Param, Pool, i32, f32 = CupyParameter, CupyPool, np.int32, np.float32
    nx = ny = MFD_SIDE
    n = nx * ny
    _MFD_NX[0] = nx

    dirs_np, mfd_w_np, indegree_np = make_synthetic_mfd_topology(nx, ny, SEED + 2)
    source_np = np.ones(n, dtype=np.float32)
    ref = numpy_kahn_mfd_accum(dirs_np, mfd_w_np, indegree_np, source_np)

    pool = CupyPool()
    grid = make_grid("cupy", pool, nx, ny, DX, topology="D8", boundary="normal", outlet="edge")
    source_p = Param("SRC", dtype=f32, mode="const", value=1.0, pool=pool)

    dirs = pool.get_data(np.dtype(np.uint8), (n,))
    mfd_w = pool.get_data(f32, (n * 8,))
    indegree = pool.get_data(i32, (n,))
    accum_h = pool.get_data(f32, (n,))
    frontier0 = pool.get_data(i32, (n,))
    frontier1 = pool.get_data(i32, (n,))
    count = pool.get_data(i32, (2,))
    barrier = pool.get_data(np.dtype(np.uint32), (1,))

    dirs.data.set(dirs_np)
    mfd_w.data.set(mfd_w_np)
    indegree.data.set(indegree_np.astype(np.int32))

    accum = make_accumulation("cupy", grid, source_p, method="persistent_mfd", n_flat=n)
    q_init_k = accum.q_init.compile()
    accum_k = accum.accum.compile()

    launch_grid, launch_block = ((n + 255) // 256,), (256,)
    q_init_k(accum_h.data, grid=launch_grid, block=launch_block)

    n0 = init_frontier_mfd(indegree.data, frontier0.data)
    count.data[0] = n0
    count.data[1] = 0
    barrier.data[0] = 0

    p_grid, p_block = persistent_grid_block()
    accum_k(
        frontier0.data, frontier1.data, count.data, barrier.data,
        dirs.data, mfd_w.data, accum_h.data, indegree.data,
        grid=p_grid, block=p_block,
    )

    got = accum_h.data.get().astype(np.float64)
    n_stuck = int((indegree.data.get() > 0).sum())

    max_abs = float(np.max(np.abs(got - ref)))
    max_rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)))
    max_got = float(got.max())
    max_ref = float(ref.max())

    for h in (dirs, mfd_w, indegree, accum_h, frontier0, frontier1, count, barrier):
        pool.release_data(h)
    source_p.destroy()

    return n, max_abs, max_rel, max_got, max_ref, n_stuck


if __name__ == "__main__":
    backend_arg = sys.argv[1]
    n_flat, max_depth, max_q, rows = run(backend_arg)
    print(f"{backend_arg}: n_flat={n_flat} max_chain_depth={max_depth} max_q_reference={max_q:.6e}")
    for mode, name, abs_diff, rel_diff, max_got in rows:
        max_str = f" max_got={max_got:.6e}" if max_got is not None else ""
        print(f"{backend_arg} mode={mode:6s} {name:20s} max_abs={abs_diff:.6e} max_rel={rel_diff:.6e}{max_str}")

    if backend_arg == "cupy":
        mfd_n, mfd_abs, mfd_rel, mfd_got, mfd_ref, mfd_stuck = run_mfd_cupy()
        print(
            f"{backend_arg} persistent_mfd: n_flat={mfd_n} max_abs={mfd_abs:.6e} max_rel={mfd_rel:.6e} "
            f"max_got={mfd_got:.6e} max_ref={mfd_ref:.6e} n_stuck(indegree>0)={mfd_stuck}"
        )
