"""
Hillslope landscape evolution as one Routine, exercising every part of the
core in a single model. Cupy backend; same model as lem_routine_taichi.py.

The physics is deliberately small - linear hillslope diffusion against a
spatially variable uplift field, with the domain edges pinned to base level:

    dz/dt = D * laplacian(z) + U(x, y)

What the file is here to show is how the pieces fit together when a model
needs all of them at once. The other examples each isolate one thing; this
one carries the lot:

  Parameter modes   N and DT are const Parameters bound flat, arriving as
                    #defines and read bare in the source. DX is a const bound
                    through a Bag, read as $grid.dx.get(0)$ - a const reached
                    through a span, rather than a top-level #define, always
                    goes through .get(0). D and SEA_LEVEL are scalars the
                    host retunes between frames. UPLIFT is a field, one rate
                    per node.
  Helpers           clampi binds a const; laplacian binds a bag and calls
                    clampi, so a helper reaches another helper; uplift_at
                    binds the UPLIFT *field* directly, which is what lets the
                    uplift kernel body stay a one-liner. Every scalar/field
                    parameter these reach lands in the module's __constant__
                    block, so a helper reads one exactly as its caller does.
  Bags              grid is nested (grid.n, grid.dx), hill is mixed - a
                    scalar Parameter, a helper and a const under one name -
                    and the two noise seeds arrive flat through bind_bag.
  Routine           three kernels, two of them inside the routine, with the
                    z0/z1 ping-pong unrolled twice so the swaps compose to
                    the identity and the routine can be called repeatedly.

Buffers are flat here rather than 2D, since a CUDA template indexes its own
data: a kernel takes one thread per node and recovers (i, j) itself.

The step the routine runs is diffuse then uplift-and-clamp, so uplift is
applied to what diffusion just wrote. Two of those, plus the two swaps, make
one routine call - and the result always lands back in z0.

The routine is captured into a CUDA graph (CupyRoutineBuilder.compile's
default), so a call replays recorded launches rather than re-issuing them.
D and SEA_LEVEL are retuned between calls, never between the steps inside
one: a write to a scalar Parameter goes through the same storage the graph
holds, so replay sees it, while there is no python between a routine's own
steps to run it in anyway (see routine.py, "Contract: no set()/destroy()
mid-routine").

Author: B.G (07/2026)
"""

import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from pyfastflow.experimental.core.context.bag import Bag, merge
from pyfastflow.experimental.core.context.cupy_backend import (
    CupyHelperBuilder,
    CupyKernelBuilder,
    CupyParameter,
    CupyRoutineBuilder,
)
from pyfastflow.experimental.core.pool.cupy_pool import CupyPool

# ---------------------------------------------------------------------------
# host-side constants (grid size, launch config, timing)
# ---------------------------------------------------------------------------
GRID_N = 2048
NN = GRID_N * GRID_N
DX_M = 100.0
STEPS_PER_FRAME = 200  # two routine substeps per call - see the loop below

BLOCK = 256
GRID = (NN + BLOCK - 1) // BLOCK

D_VAL = 1.0e-2  # hillslope diffusivity, m2/yr
UPLIFT_MAX = 1.0e-6  # m/yr at the range crest
CFL_SAFETY = 0.2
DT_VAL = CFL_SAFETY * DX_M**2 / (4.0 * D_VAL)

pool = CupyPool()

# ---------------------------------------------------------------------------
# parameters - one of every mode
# ---------------------------------------------------------------------------
# const Parameters, bound flat at top level: emitted as #defines and read
# bare in the source.
n_p = CupyParameter("N", dtype=np.int32, mode="const", value=GRID_N, pool=pool)
dt_p = CupyParameter("DT", dtype=np.float32, mode="const", value=DT_VAL, pool=pool)
seed_a_p = CupyParameter("SEED_A", dtype=np.float32, mode="const", value=12.9898, pool=pool)
seed_b_p = CupyParameter("SEED_B", dtype=np.float32, mode="const", value=78.233, pool=pool)

# fixed at compile time like the ones above, but reached through a span
# (bound inside a Bag) rather than as a top-level #define
dx_p = CupyParameter("DX", dtype=np.float32, mode="const", value=DX_M, pool=pool)

# scalars: one cell each, retuned from the host between routine calls
d_p = CupyParameter("D", dtype=np.float32, mode="scalar", value=D_VAL, pool=pool)
sea_p = CupyParameter("SEA_LEVEL", dtype=np.float32, mode="scalar", value=0.0, pool=pool)

# field: one value per node, filled from the host below
uplift_p = CupyParameter("UPLIFT", dtype=np.float32, mode="field", value=np.zeros(NN), pool=pool, n_flat=NN)

# a north-south uplift ridge, tapering to zero at the north and south edges
_yy = np.arange(GRID_N, dtype=np.float32)[:, None] * np.ones((1, GRID_N), np.float32)
_ridge = np.sin(np.pi * _yy / (GRID_N - 1)) ** 2
uplift_p.set((UPLIFT_MAX * _ridge).ravel())

# ---------------------------------------------------------------------------
# bags
# ---------------------------------------------------------------------------
# nested: both grid.n and grid.dx are const Parameters, reached through a
# span (.get(0)) rather than a top-level #define - members resolve on their
# own type, not the bag's
grid = Bag({"n": n_p, "dx": dx_p})

# flat, for bind_bag: the kernel that uses these reads them as bare names
noise_seeds = Bag({"SEED_A": seed_a_p, "SEED_B": seed_b_p})

# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------
clampi_fn = (
    CupyHelperBuilder()
    .bind("N", n_p)
    .ingest("__device__ int clampi(int i) { return i < 0 ? 0 : (i >= N ? N - 1 : i); }")
)

laplacian_fn = (
    CupyHelperBuilder()
    .bind("clampi", clampi_fn)
    .bind("N", n_p)
    .bind("grid", grid)
    .ingest(
        r"""
__device__ float laplacian(const float* f, int i, int j) {
    // calls another helper, and reads a const Parameter out of a bound bag
    int ip = $clampi(i + 1)$;
    int im = $clampi(i - 1)$;
    int jp = $clampi(j + 1)$;
    int jm = $clampi(j - 1)$;
    float acc = f[ip * N + j] + f[im * N + j] + f[i * N + jp] + f[i * N + jm] - 4.0f * f[i * N + j];
    float dx = $grid.dx.get(0)$;
    return acc / (dx * dx);
}
"""
    )
)

uplift_at_fn = (
    CupyHelperBuilder()
    .bind("UPLIFT", uplift_p)
    .ingest(
        r"""
__device__ float uplift_at(int idx) {
    // binds the UPLIFT *field* itself: a helper reads a non-const Parameter
    // exactly the way a kernel does, so the caller passes only the index
    return $UPLIFT.get(idx)$;
}
"""
    )
)

# ---------------------------------------------------------------------------
# one-shot setup kernel (runs once, outside the routine)
# ---------------------------------------------------------------------------
init_topo_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind_bag(noise_seeds)
    .ingest(
        r"""
extern "C" __global__ void init_topo(float* z) {
    // bind_bag put SEED_A / SEED_B in flat, so they read as bare names here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N, j = idx % N;
    float x = (float)i * SEED_A + (float)j * SEED_B;
    float s = sinf(x) * 43758.5453f;
    z[idx] = (s - floorf(s)) * 2.0f;
}
"""
    )
    .compile()
)

# ---------------------------------------------------------------------------
# routine kernels
# ---------------------------------------------------------------------------

# mixed bag: a scalar Parameter, a helper and a const Parameter under one
# name. hill.d, hill.dt are span reads (.get(0)), hill.lap a spliced call.
hill = Bag({"d": d_p, "lap": laplacian_fn, "dt": dt_p})

diffuse_builder = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind("hill", hill)
    .ingest(
        r"""
extern "C" __global__ void diffuse(float* z_out, const float* z_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N, j = idx % N;
    z_out[idx] = z_in[idx] + $hill.dt.get(0)$ * $hill.d.get(0)$ * $hill.lap(z_in, i, j)$;
}
"""
    )
)

uplift_builder = (
    CupyKernelBuilder()
    .bind("up", uplift_at_fn)
    .bind("DT", dt_p)
    .bind("grid", grid)
    .bind("SEA", sea_p)
    .ingest(
        r"""
extern "C" __global__ void uplift_bc(float* z) {
    // grid.n is a const Parameter reached through the bag, so it splices in
    // as a device accessor - .get(0) bakes to a literal just as a top-level
    // #define would
    int n = $grid.n.get(0)$;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int j = idx % n;
    z[idx] += $up(idx)$ * DT;
    // base level: pin the east and west edges, so the ridge drains outward
    if (j == 0 || j == n - 1) z[idx] = $SEA.get(0)$;
}
"""
    )
)

# ---------------------------------------------------------------------------
# buffers (pooled - two for ping-pong)
# ---------------------------------------------------------------------------
z0 = pool.get_data(np.float32, (NN,))
z1 = pool.get_data(np.float32, (NN,))

init_topo_kernel(z0.data, grid=GRID, block=BLOCK)

# ---------------------------------------------------------------------------
# the routine
# ---------------------------------------------------------------------------
# One bag for the whole routine, merged from what each builder already binds.
# Both builders reach `grid` and `N` - the same objects, so the same uids,
# which is what lets merge() accept the collision instead of raising on it.
# grid/block are set once here and inherited by every step.
evolve = (
    CupyRoutineBuilder(grid=GRID, block=BLOCK)
    .bind_bag(merge(diffuse_builder.as_bag(), uplift_builder.as_bag()))
    .add_data("z0", z0.data)
    .add_data("z1", z1.data)
    .add_kernel(diffuse_builder, data_handle_ref=("z1", "z0"))
    .add_kernel(uplift_builder, data_handle_ref=("z1",))
    .add_swap("z0", "z1")
    .add_kernel(diffuse_builder, data_handle_ref=("z1", "z0"))
    .add_kernel(uplift_builder, data_handle_ref=("z1",))
    .add_swap("z0", "z1")
    .compile()
)

# ---------------------------------------------------------------------------
# live view
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
im = ax.imshow(z0.to_numpy().reshape(GRID_N, GRID_N), cmap="terrain", vmin=0.0, vmax=150.0)
fig.colorbar(im, ax=ax, label="Elevation (m)")
ax.set_title("Hillslope LEM (Cupy backend, Routine)")
time_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
    color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.4, pad=2),
)
fig.show()

sim_time = 0.0
try:
    while True:
        t_start = time.perf_counter()
        for _ in range(STEPS_PER_FRAME // 2):
            evolve()  # two substeps, result lands back in z0
            sim_time += 2.0 * DT_VAL

        cp.cuda.Device().synchronize()  # GPU is async; sync before stopping the timer
        frame_ms = (time.perf_counter() - t_start) * 1e3
        print(f"{STEPS_PER_FRAME} steps: {frame_ms:8.1f} ms  ({frame_ms / STEPS_PER_FRAME * 1e3:6.1f} us/step)")

        time_text.set_text(f"t = {sim_time / 1e6:.2f} Myr")
        im.set_data(z0.to_numpy().reshape(GRID_N, GRID_N))
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.1)
except KeyboardInterrupt:
    pass

# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------
d_p.destroy()
sea_p.destroy()
uplift_p.destroy()
pool.release_data(z0)
pool.release_data(z1)
