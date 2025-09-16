"""
Experimental: Level-based MFD accumulation strategies benchmarking (pooled fields).

Design goals:
- Allocate Taichi fields once using the PyFastFlow pool and reuse them.
- Keep all compute on GPU; only read minimal scalars per-iteration for loop control.
- Group timings: measure a block of N runs per method with a single sync at the end.
"""

import time
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt

import pyfastflow as pf
from pyfastflow.flow import level_based as lb
from pyfastflow import pool


def bench_stats(times):
    times = np.asarray(times, dtype=np.float64)
    return dict(mean=times.mean(), std=times.std(ddof=1) if len(times) > 1 else 0.0, min=times.min(), max=times.max())


def build_test_grid(nx=256, ny=192, dx=10.0, seed=42):
    dem = pf.noise.perlin_noise(nx, ny, frequency=16.0, octaves=5, persistence=0.5, amplitude=100.0, seed=seed)
    dem = dem - dem.min() + 10.0
    grid = pf.grid.Grid(nx, ny, dx, dem.astype(np.float32), boundary_mode="normal")
    router = pf.flow.FlowRouter(grid, lakeflow=True)
    router.compute_receivers()
    router.reroute_flow()
    router.fill_z(epsilon=1e-2)
    # plt.imshow(grid.z.to_numpy().reshape(ny,nx))
    # plt.show()
    return grid, router


def alloc_fields(N, with_scan=False):
    Q = pool.taipool.get_tpfield(ti.f32, (N,))
    state = pool.taipool.get_tpfield(ti.u8, (N,))
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
    return dict(Q=Q, state=state, ids=ids, count=count, flags=flags, scan_out=scan_out, work=work)


def release_fields(f):
    for v in f.values():
        if v is not None:
            v.release()


def init_problem(grid, f):
    f["Q"].field.fill(pf.constants.DX * pf.constants.DX)
    lb.reset_states(f["state"].field)


def run_once_bruteforce(grid, f, dn) -> int:
    it = 0
    while True:
        it += 1
        changed = lb.mfd_iteration_bruteforce(
            f["Q"].field,
            f["state"].field,
            dn["dn_idx"].field,
            dn["w"].field,
            dn["wsum"].field,
            dn["up_idx"].field,
        )
        if not changed:
            break
    return it


def run_once_atomic(grid, f, dn) -> int:
    it = 0
    while True:
        it += 1
        processed = lb.mfd_iteration_atomic(
            f["Q"].field,
            grid.z.field,
            f["state"].field,
            f["ids"].field,
            f["count"].field,
            dn["dn_idx"].field,
            dn["w"].field,
            dn["wsum"].field,
            dn["up_idx"].field,
        )
        if processed == 0:
            break
    return it


def run_once_scan(grid, f, dn) -> int:
    N = grid.nx * grid.ny
    it = 0
    while True:
        it += 1
        processed = lb.mfd_iteration_scan(
            f["Q"].field,
            grid.z.field,
            f["state"].field,
            f["flags"].field,
            f["scan_out"].field,
            f["work"].field,
            f["ids"].field,
            N,
            f["count"].field,
            dn["dn_idx"].field,
            dn["w"].field,
            dn["wsum"].field,
            dn["up_idx"].field,
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

    # Build grid
    grid, router = build_test_grid(nx=1024, ny=1024)

    print("\nRunning level-based MFD strategies (10 trials, pooled fields, grouped timing)...")
    N = grid.nx * grid.ny
    f_bf = alloc_fields(N)
    f_at = alloc_fields(N)
    f_sc = alloc_fields(N, with_scan=True)
    # Precompute downstream once (shared)
    dn = {
        "dn_idx": pool.taipool.get_tpfield(ti.i32, (N * 4,)),
        "w": pool.taipool.get_tpfield(ti.f32, (N * 4,)),
        "wsum": pool.taipool.get_tpfield(ti.f32, (N,)),
        "up_idx": pool.taipool.get_tpfield(ti.i32, (N * 4,)),
    }
    lb.precompute_downstream(grid.z.field, dn["dn_idx"].field, dn["w"].field, dn["wsum"].field)
    lb.precompute_upstream(grid.z.field, dn["up_idx"].field)

    # Warmup (one per method)
    init_problem(grid, f_bf); _ = run_once_bruteforce(grid, f_bf, dn)
    init_problem(grid, f_at); _ = run_once_atomic(grid, f_at, dn)
    init_problem(grid, f_sc); _ = run_once_scan(grid, f_sc, dn)
    ti.sync()

    # Brute-force group timing
    ia = []
    t0 = time.perf_counter()
    for _ in range(10):
        init_problem(grid, f_bf)
        ia.append(run_once_bruteforce(grid, f_bf, dn))
    ti.sync(); tA = time.perf_counter() - t0
    Qa = f_bf["Q"].field.to_numpy().reshape(grid.rshp)

    # Atomic group timing
    ib = []
    t0 = time.perf_counter()
    for _ in range(10):
        init_problem(grid, f_at)
        ib.append(run_once_atomic(grid, f_at, dn))
    ti.sync(); tB = time.perf_counter() - t0
    Qb = f_at["Q"].field.to_numpy().reshape(grid.rshp)

    # Scan group timing
    ic = []
    t0 = time.perf_counter()
    for _ in range(10):
        init_problem(grid, f_sc)
        ic.append(run_once_scan(grid, f_sc, dn))
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

    # Validate last results
    err_ab = np.max(np.abs(Qa - Qb))
    err_ac = np.max(np.abs(Qa - Qc))
    print(f"Max |Qa-Qb| (last) = {err_ab:.6e}")
    print(f"Max |Qa-Qc| (last) = {err_ac:.6e}")

    # Rake-and-compress (single accumulation) timing
    print("\nRunning rake-and-compress (FlowRouter.accumulate_constant_Q) timing...")
    # Warm-up
    router.compute_receivers()
    router.accumulate_constant_Q(1.0, area=True)
    ti.sync()
    # Group timing for 10 runs
    t0 = time.perf_counter()
    for _ in range(10):
        # router.compute_receivers()
        router.accumulate_constant_Q(1.0, area=True)
    ti.sync()
    tRC = time.perf_counter() - t0
    print(f"Rake-compress (group): {tRC:.3f}s  mean per run={tRC/10:.3f}s")
    Qrc = router.get_Q()

    # CPU baseline using topological order (Numba if available)
    print("\nRunning CPU top-down MFD (topological order, 10 trials with warm-up)...")
    z_np = grid.z.field.to_numpy().reshape(grid.rshp)
    cpu_time = None
    try:
        import numba as nb

        @nb.njit(cache=True)
        def _mfd_topdown_core(z, dx, order):
            ny, nx = z.shape
            N = ny * nx
            Q = np.empty(N, np.float32)
            cell = np.float32(dx * dx)
            for t in range(N):
                Q[t] = cell
            for t in range(N):
                idx = order[t]
                j = idx // nx
                i = idx % nx
                zi = z[j, i]
                s = np.zeros(4, np.float32)
                ni = np.zeros(4, np.int64)
                nj = np.zeros(4, np.int64)
                # up, left, right, down
                ni[0] = i;     nj[0] = j - 1
                ni[1] = i - 1; nj[1] = j
                ni[2] = i + 1; nj[2] = j
                ni[3] = i;     nj[3] = j + 1
                sums = 0.0
                for k in range(4):
                    ii = ni[k]; jj = nj[k]
                    if ii >= 0 and ii < nx and jj >= 0 and jj < ny:
                        zj = z[jj, ii]
                        if zj < zi:
                            val = (zi - zj) / dx
                            s[k] = val
                            sums += val
                if sums > 0.0:
                    Qi = Q[idx]
                    for k in range(4):
                        if s[k] > 0.0:
                            ii = ni[k]; jj = nj[k]
                            dst = jj * nx + ii
                            Q[dst] += Qi * (s[k] / sums)
            return Q

        # Precompute topological order once
        order = np.argsort(z_np.ravel()).astype(np.int64)[::-1]

        # Warm-up JIT compile
        _ = _mfd_topdown_core(z_np.astype(np.float32), np.float32(grid.dx), order)

        # Timed trials
        t_cpu_runs = []
        Qcpu_flat_last = None
        for _trial in range(10):
            t0 = time.perf_counter()
            order = np.argsort(z_np.ravel()).astype(np.int64)[::-1]
            Qcpu_flat_last = _mfd_topdown_core(z_np.astype(np.float32), np.float32(grid.dx), order)
            t_cpu_runs.append(time.perf_counter() - t0)
        cpu_time = sum(t_cpu_runs)
        stats_cpu = bench_stats(t_cpu_runs)
        print(f"CPU top-down (group): {cpu_time:.3f}s  mean={stats_cpu['mean']:.3f}s std={stats_cpu['std']:.3f}s")

        Qcpu = Qcpu_flat_last.reshape(grid.rshp)
        err_a_cpu = np.max(np.abs(Qa - Qcpu))
        print(f"Max |Qa-Qcpu| (last) = {err_a_cpu:.6e}")
    except Exception as e:
        print(f"CPU top-down (Numba) unavailable or failed: {e}")

    # Visualization: log10 comparison plots
    try:
        plots = [("GPU Brute", Qa), ("GPU Atomic", Qb), ("GPU Scan", Qc)]
        if 'Qcpu' in locals():
            plots.append(("CPU TopDown", Qcpu))
        if 'Qrc' in locals():
            plots.append(("Rake-Compress", Qrc))

        n = len(plots)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, (title, arr) in zip(axes, plots):
            img = ax.imshow(np.log10(np.maximum(arr, 1e-12)), cmap='viridis')
            ax.set_title(title)
            ax.axis('off')
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        out_path = "experimental/level_based_comparison.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved comparison figure to {out_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")

    release_fields(f_bf)
    release_fields(f_at)
    release_fields(f_sc)
    release_fields(dn)

    return {
        "scan_ti": s_ti,
        "scan_np": s_np,
        "bruteforce_group_s": tA,
        "atomic_group_s": tB,
        "scan_group_s": tC,
        "iters": (ia, ib, ic),
        "err_last": (err_ab, err_ac),
        "cpu_top_down_s": cpu_time,
    }


if __name__ == "__main__":
    main()
