"""
Experimental: Level-based SFD accumulation strategies (pooled fields) benchmark.

Three methods:
- Brute force (fused): scan all nodes, process those with donors_remaining==0.
- Atomic compaction: compact ready IDs then process.
- Scan compaction: flags + inclusive scan + scatter + process.

Includes inclusive scan micro-benchmark, optional CPU top-down baseline, rake-compress timing,
and log10 comparison plots saved to experimental/level_sfd_comparison.png.
"""

import time
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

import pyfastflow as pf
from pyfastflow import pool
from pyfastflow.flow import level_sfd as sfd


def bench_stats(times):
    times = np.asarray(times, dtype=np.float64)
    return dict(mean=times.mean(), std=times.std(ddof=1) if len(times) > 1 else 0.0, min=times.min(), max=times.max())


def build_test_grid(nx=256, ny=192, dx=10.0, seed=42):
    dem = pf.noise.perlin_noise(nx, ny, frequency=16.0, octaves=5, persistence=0.5, amplitude=100.0, seed=seed)
    dem = dem - dem.min() + 10.0
    grid = pf.grid.Grid(nx, ny, dx, dem.astype(np.float32), boundary_mode="normal")
    router = pf.flow.FlowRouter(grid, lakeflow=True)
    router.compute_receivers()
    router.fill_z(epsilon=1e-3)
    # Recompute receivers in case fill modified topology
    router.compute_receivers()
    return grid, router


def alloc_fields(N, with_scan=False):
    Q = pool.taipool.get_tpfield(ti.f32, (N,))
    state = pool.taipool.get_tpfield(ti.u8, (N,))
    donors = pool.taipool.get_tpfield(ti.i32, (N * 4,))
    donors_remaining = pool.taipool.get_tpfield(ti.i32, (N,))
    ids = pool.taipool.get_tpfield(ti.i32, (N,))
    count = pool.taipool.get_tpfield(ti.i32, ())
    flags = scan_out = work = None
    if with_scan:
        flags = pool.taipool.get_tpfield(ti.i32, (N,))
        scan_out = pool.taipool.get_tpfield(ti.i32, (N,))
        next_pow2 = 1
        while next_pow2 < N:
            next_pow2 *= 2
        work = pool.taipool.get_tpfield(ti.i32, (next_pow2,))
    return dict(Q=Q, state=state, donors=donors, donors_remaining=donors_remaining, ids=ids, count=count, flags=flags, scan_out=scan_out, work=work)


def release_fields(f):
    for v in f.values():
        if v is not None:
            v.release()


def init_problem(grid, f):
    f["Q"].field.fill(pf.constants.DX * pf.constants.DX)
    sfd.reset_states(f["state"].field)
    sfd.init_donors_from_receivers(grid.router_receivers.field, f["donors"].field, f["donors_remaining"].field)


def run_once_bruteforce(grid, f) -> int:
    it = 0
    while True:
        it += 1
        changed = sfd.iteration_bruteforce(f["Q"].field, grid.router_receivers.field, f["donors_remaining"].field, f["state"].field)
        if not changed:
            break
    return it


def run_once_atomic(grid, f) -> int:
    it = 0
    while True:
        it += 1
        processed = sfd.iteration_atomic(
            f["Q"].field, grid.router_receivers.field, f["donors_remaining"].field, f["state"].field, f["ids"].field, f["count"].field
        )
        if processed == 0:
            break
    return it


def run_once_scan(grid, f) -> int:
    N = grid.nx * grid.ny
    it = 0
    while True:
        it += 1
        processed = sfd.iteration_scan(
            f["Q"].field,
            grid.router_receivers.field,
            f["donors_remaining"].field,
            f["state"].field,
            f["flags"].field,
            f["scan_out"].field,
            f["work"].field,
            f["ids"].field,
            N,
            f["count"].field,
        )
        if processed == 0:
            break
    return it


def main():
    ti.init(arch=ti.gpu, offline_cache=False)

    # Inclusive scan micro-benchmark
    print("Benchmarking inclusive scan (Taichi vs NumPy)...")
    Nscan = 1 << 20
    flags = ti.field(dtype=ti.i32, shape=(Nscan,))
    out = ti.field(dtype=ti.i32, shape=(Nscan,))
    next_pow2 = 1
    while next_pow2 < Nscan:
        next_pow2 *= 2
    work = ti.field(dtype=ti.i32, shape=(next_pow2,))
    rng = np.random.default_rng(123)
    arr = rng.integers(0, 2, size=Nscan, dtype=np.int32)
    from pyfastflow.general_algorithms import inclusive_scan
    flags.from_numpy(arr)
    inclusive_scan(flags, out, work, Nscan)
    ti.sync()
    t_ti = []; t_np = []
    for _ in range(10):
        flags.from_numpy(arr)
        t0 = time.perf_counter(); inclusive_scan(flags, out, work, Nscan); ti.sync(); t_ti.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); _ = np.cumsum(arr, dtype=np.int64); t_np.append(time.perf_counter() - t0)
    s_ti = bench_stats(t_ti); s_np = bench_stats(t_np)
    print(f"Taichi scan: mean={s_ti['mean']:.6f}s std={s_ti['std']:.6f}s min={s_ti['min']:.6f}s max={s_ti['max']:.6f}s")
    print(f"NumPy cumsum: mean={s_np['mean']:.6f}s std={s_np['std']:.6f}s min={s_np['min']:.6f}s max={s_np['max']:.6f}s")

    # Build grid and cache receivers as field
    grid, router = build_test_grid(nx=512, ny=512)
    # Cache receivers into a Taichi field to avoid TPField wrapper in kernels
    grid.router_receivers = pool.taipool.get_tpfield(ti.i32, (grid.nx * grid.ny,))
    grid.router_receivers.field.copy_from(router.receivers.field)

    print("\nRunning SFD strategies (10 trials, pooled fields, grouped timing)...")
    N = grid.nx * grid.ny
    f_bf = alloc_fields(N)
    f_at = alloc_fields(N)
    f_sc = alloc_fields(N, with_scan=True)

    # Warm-up
    init_problem(grid, f_bf); _ = run_once_bruteforce(grid, f_bf)
    init_problem(grid, f_at); _ = run_once_atomic(grid, f_at)
    init_problem(grid, f_sc); _ = run_once_scan(grid, f_sc)
    ti.sync()

    # Brute-force group timing
    ia = []
    t0 = time.perf_counter()
    for _ in range(10):
        init_problem(grid, f_bf)
        ia.append(run_once_bruteforce(grid, f_bf))
    ti.sync(); tA = time.perf_counter() - t0
    Qa = f_bf["Q"].field.to_numpy().reshape(grid.rshp)

    # Atomic group timing
    ib = []
    t0 = time.perf_counter()
    for _ in range(10):
        init_problem(grid, f_at)
        ib.append(run_once_atomic(grid, f_at))
    ti.sync(); tB = time.perf_counter() - t0
    Qb = f_at["Q"].field.to_numpy().reshape(grid.rshp)

    # Scan group timing
    ic = []
    t0 = time.perf_counter()
    for _ in range(10):
        init_problem(grid, f_sc)
        ic.append(run_once_scan(grid, f_sc))
    ti.sync(); tC = time.perf_counter() - t0
    Qc = f_sc["Q"].field.to_numpy().reshape(grid.rshp)

    total_it_a = int(np.sum(ia))
    total_it_b = int(np.sum(ib))
    total_it_c = int(np.sum(ic))
    per_it_a = (tA / total_it_a) if total_it_a > 0 else float('nan')
    per_it_b = (tB / total_it_b) if total_it_b > 0 else float('nan')
    per_it_c = (tC / total_it_c) if total_it_c > 0 else float('nan')

    print(f"Brute force (group): {tA:.3f}s  iters≈{int(np.mean(ia))}  per-iter≈{per_it_a*1e3:.3f} ms")
    print(f"Atomic list (group): {tB:.3f}s  iters≈{int(np.mean(ib))}  per-iter≈{per_it_b*1e3:.3f} ms")
    print(f"Scan list   (group): {tC:.3f}s  iters≈{int(np.mean(ic))}  per-iter≈{per_it_c*1e3:.3f} ms")

    # Rake-and-compress timing (FlowRouter)
    print("\nRunning rake-and-compress (FlowRouter.accumulate_constant_Q) timing...")
    router.accumulate_constant_Q(1.0, area=True)
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(10):
        router.accumulate_constant_Q(1.0, area=True)
    ti.sync()
    tRC = time.perf_counter() - t0
    print(f"Rake-compress (group): {tRC:.3f}s  mean per run={tRC/10:.3f}s")
    Qrc = router.get_Q()

    # CPU baseline (Numba): topological order + accumulation in one function
    print("\nRunning CPU SFD (Numba Kahn + accum) 10 trials...")
    rcv = router.get_receivers().ravel().astype(np.int64)
    try:
        import numba as nb

        @nb.njit(cache=True)
        def cpu_sfd_order_and_accum(rcv, dx):
            N = rcv.size
            indeg = np.zeros(N, np.int32)
            for i in range(N):
                r = rcv[i]
                if r != i and r >= 0:
                    indeg[r] += 1
            queue = np.empty(N, np.int64)
            head = 0
            tail = 0
            for i in range(N):
                if indeg[i] == 0:
                    queue[tail] = i
                    tail += 1
            Q = np.empty(N, np.float32)
            cell = np.float32(dx * dx)
            for i in range(N):
                Q[i] = cell
            order_count = 0
            while head < tail:
                u = queue[head]
                head += 1
                order_count += 1
                r = rcv[u]
                if r != u and r >= 0:
                    Q[r] += Q[u]
                    indeg[r] -= 1
                    if indeg[r] == 0:
                        queue[tail] = r
                        tail += 1
            return Q, order_count

        # Warm-up
        _Qtmp, _ = cpu_sfd_order_and_accum(rcv, np.float32(grid.dx))
        t_runs = []
        Qcpu_last = None
        for _ in range(10):
            t0 = time.perf_counter()
            Qcpu_last, _ = cpu_sfd_order_and_accum(rcv, np.float32(grid.dx))
            t_runs.append(time.perf_counter() - t0)
        print(f"CPU SFD (group): {sum(t_runs):.3f}s  mean={np.mean(t_runs):.3f}s std={np.std(t_runs):.3f}s")
    except Exception as e:
        print(f"CPU SFD (Numba) unavailable or failed: {e}")

    # Visualization: log10 comparison
    try:
        plots = [("GPU Brute", Qa), ("GPU Atomic", Qb), ("GPU Scan", Qc), ("Rake-Compress", Qrc)]
        n = len(plots)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, (title, arr) in zip(axes, plots):
            img = ax.imshow(np.log10(np.maximum(arr, 1e-12)), cmap='viridis')
            ax.set_title(title)
            ax.axis('off')
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        out_path = "experimental/level_sfd_comparison.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved comparison figure to {out_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")

    release_fields(f_bf); release_fields(f_at); release_fields(f_sc)
    # Keep router_receivers allocated if reused; else release
    grid.router_receivers.release()


if __name__ == "__main__":
    main()
