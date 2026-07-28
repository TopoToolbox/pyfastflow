"""
Heat diffusion through a procedurally-generated floor plan (air + walls),
heated by a single stove, built on pyfastflow's backend-agnostic core
(Parameter/Helper/Kernel/Pool), Cupy (cp.RawKernel) backend.

Same model as heat_diffusion_taichi.py, authored as CUDA source strings. The
grid is stored flat (N*N); kernels launch one thread per cell. Params/helpers
are referenced through `$...$` spans, uniform with the closure backends:

  $wall.get(idx)$          read a field param
  $alpha.set_node(idx, v)$ device-side write of a field param
  $stove.temp.get(0)$      read a scalar param through a bound Bag
  $whash(a, b)$            call a bound __device__ helper (source auto-spliced)

For scalar/field params the parser auto-generates the matching pointer
argument into the __global__ signature and appends the launch array - the
source never declares them. Top-level const params (N, ROOM, ...) become
#defines and are written bare; a const inside a Bag does not, and is reached
through a span like any other member. Spans do not nest, so a helper's param argument is read into a temp
first (see diffuse: laplacian takes the T_in pointer directly as a data arg).

Binding styles, all three visible in one file:
  - flat, one bind() per object (most kernels here);
  - a Bag bound whole and reached by dotted path - `stove` in apply_source,
    which nests a sub-bag for the position, and `heat` in diffuse, which mixes
    a Parameter, a device helper and two consts under one name;
  - bind_bag(), merging a bag's members in flat under their own names, so the
    kernel still sees plain names - `alpha_seeds` in set_alpha.

Author: B.G (07/2026)
"""

import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from pyfastflow.experimental.core.context.base import Bag
from pyfastflow.experimental.core.context.cupy_backend import (
    CupyHelperBuilder,
    CupyKernelBuilder,
    CupyParameter,
)
from pyfastflow.experimental.core.pool.cupy_pool import CupyPool

# ---------------------------------------------------------------------------
# host-side constants
# ---------------------------------------------------------------------------
GRID_N = 512
NN = GRID_N * GRID_N
STEPS_PER_FRAME = 10000
PULSE_FREQ = 0.0  # stove temperature oscillation speed, rad/s (0 = steady stove)
BLOCK = 256
GRID = (NN + BLOCK - 1) // BLOCK

# Physical grounding (see the taichi demo for the reasoning).
ROOM_M = 3.0
DX_M = ROOM_M / (GRID_N // 4)
ALPHA_AIR_VAL = 0.015  # m^2/s, effective convective air diffusivity
ALPHA_WALL_VAL = 1.0e-6  # m^2/s, real solid diffusivity
CFL_SAFETY = 0.4
DT_VAL = CFL_SAFETY * DX_M**2 / (4.0 * ALPHA_AIR_VAL)

pool = CupyPool()

# Structural constants -> #define, used bare in the CUDA source.
n_p = CupyParameter("N", dtype=np.int32, mode="const", value=GRID_N, pool=pool)
room_p = CupyParameter("ROOM", dtype=np.int32, mode="const", value=GRID_N // 4, pool=pool)
wall_thick_p = CupyParameter("WALL_THICK", dtype=np.int32, mode="const", value=8, pool=pool)
door_p = CupyParameter("DOOR", dtype=np.int32, mode="const", value=6, pool=pool)
seed_p = CupyParameter("SEED", dtype=np.float32, mode="const", value=17.0, pool=pool)

dt_p = CupyParameter("DT", dtype=np.float32, mode="const", value=DT_VAL, pool=pool)
dx2_p = CupyParameter("DX2", dtype=np.float32, mode="const", value=DX_M**2, pool=pool)

alpha_air_seed_p = CupyParameter("ALPHA_AIR_SEED", dtype=np.float32, mode="const", value=ALPHA_AIR_VAL, pool=pool)
alpha_wall_seed_p = CupyParameter("ALPHA_WALL_SEED", dtype=np.float32, mode="const", value=ALPHA_WALL_VAL, pool=pool)
t_bg_p = CupyParameter("T_BG", dtype=np.float32, mode="const", value=15.0, pool=pool)

src_i_p = CupyParameter("SRC_I", dtype=np.int32, mode="const", value=GRID_N // 4 + GRID_N // 8, pool=pool)
src_j_p = CupyParameter("SRC_J", dtype=np.int32, mode="const", value=GRID_N // 4 + GRID_N // 8, pool=pool)
src_r_p = CupyParameter("SRC_R", dtype=np.int32, mode="const", value=10, pool=pool)

# scalar mode: host-set each frame -> pulsing stove temperature
OG_stove = 70
stove_p = CupyParameter("STOVE_T", dtype=np.float32, mode="scalar", value=OG_stove, pool=pool)

# field mode: per-cell wall/air mask and per-cell diffusivity
wall_p = CupyParameter("WALL", dtype=np.int32, mode="field", value=np.zeros(NN), pool=pool, n_flat=NN)
alpha_p = CupyParameter("ALPHA", dtype=np.float32, mode="field", value=np.zeros(NN), pool=pool, n_flat=NN)

# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------
clamp_fn = (
    CupyHelperBuilder()
    .bind("N", n_p)
    .ingest("__device__ int clampi(int i) { return i < 0 ? 0 : (i >= N ? N - 1 : i); }")
)

laplacian_fn = (
    CupyHelperBuilder()
    .bind("N", n_p)
    .bind("clampi", clamp_fn)
    .ingest(
        r"""
__device__ float laplacian(const float* f, int i, int j) {
    int ip = $clampi(i + 1)$;
    int im = $clampi(i - 1)$;
    int jp = $clampi(j + 1)$;
    int jm = $clampi(j - 1)$;
    return f[ip * N + j] + f[im * N + j] + f[i * N + jp] + f[i * N + jm] - 4.0f * f[i * N + j];
}
"""
    )
)

whash_fn = (
    CupyHelperBuilder()
    .bind("SEED", seed_p)
    .ingest(
        r"""
__device__ float whash(int a, int b) {
    float x = (float)a * 12.9898f + (float)b * 78.233f + SEED;
    float s = sinf(x) * 43758.5453f;
    return s - floorf(s);
}
"""
    )
)

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
generate_walls_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind("ROOM", room_p)
    .bind("WALL_THICK", wall_thick_p)
    .bind("DOOR", door_p)
    .bind("wall", wall_p)
    .bind("whash", whash_fn)
    .ingest(
        r"""
__global__ void generate_walls() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N;
    int j = idx % N;
    int is_wall = 0;
    if (i < WALL_THICK || i >= N - WALL_THICK || j < WALL_THICK || j >= N - WALL_THICK) {
        is_wall = 1;
    } else if (i % ROOM < WALL_THICK) {
        int door = (int)($whash(i / ROOM, j / ROOM)$ * ROOM);
        int r = j % ROOM;
        if (!(r >= door && r < door + DOOR)) is_wall = 1;
    } else if (j % ROOM < WALL_THICK) {
        int door = (int)($whash(j / ROOM + 7919, i / ROOM)$ * ROOM);
        int r = i % ROOM;
        if (!(r >= door && r < door + DOOR)) is_wall = 1;
    }
    $wall.set_node(idx, is_wall)$;
}
"""
    )
    .compile()
)

# The two seed values are grouped on the host for tidiness, then merged in with
# bind_bag() - which binds each member flat, under its own name. The source
# below is unaware: the seeds stay top-level consts, so they still arrive as
# #defines and are written bare inside the spans. bind() the bag whole instead
# when you want a dotted path.
alpha_seeds = Bag({"ALPHA_WALL_SEED": alpha_wall_seed_p, "ALPHA_AIR_SEED": alpha_air_seed_p})

set_alpha_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind_bag(alpha_seeds)
    .bind("wall", wall_p)
    .bind("alpha", alpha_p)
    .ingest(
        r"""
__global__ void set_alpha() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    if ($wall.get(idx)$ == 1) {
        $alpha.set_node(idx, ALPHA_WALL_SEED)$;
    } else {
        $alpha.set_node(idx, ALPHA_AIR_SEED)$;
    }
}
"""
    )
    .compile()
)

init_temperature_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind("T_BG", t_bg_p)
    .ingest(
        r"""
__global__ void init_temperature(float* T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    T[idx] = T_BG;
}
"""
    )
    .compile()
)

# The stove travels as ONE nested Bag rather than four flat binds: its position
# is grouped into an `at` sub-bag, and the span parser walks the dotted path
# through both levels - const members expand to CUDA literals, the scalar
# Parameter to its generated pointer arg. Everything else here still binds
# flat, so the two styles sit side by side in one file.
stove = Bag(
    {
        "at": Bag({"i": src_i_p, "j": src_j_p}),
        "r": src_r_p,
        "temp": stove_p,
    }
)

apply_source_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind("stove", stove)
    .ingest(
        r"""
__global__ void apply_source(float* T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int dx = idx / N - $stove.at.i.get(0)$;
    int dy = idx % N - $stove.at.j.get(0)$;
    if (dx * dx + dy * dy <= $stove.r.get(0)$ * $stove.r.get(0)$) {
        T[idx] = $stove.temp.get(0)$;
    }
}
"""
    )
    .compile()
)

# A MIXED Bag: everything the diffusion step needs, whatever kind it is - a
# field Parameter, a device helper, two const Parameters - under one name. A
# bag has no member type; each member is resolved on its own when the spans
# expand.
# Note the consts are reached through spans here rather than written bare: only
# top-level const params become #defines, members of a bag do not.
heat = Bag({"alpha": alpha_p, "lap": laplacian_fn, "dt": dt_p, "dx2": dx2_p})

diffuse_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind("heat", heat)
    .ingest(
        r"""
__global__ void diffuse(float* T_out, const float* T_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N;
    int j = idx % N;
    float a = $heat.alpha.get(idx)$;
    float lap = $heat.lap(T_in, i, j)$ / $heat.dx2.get(0)$;
    T_out[idx] = T_in[idx] + $heat.dt.get(0)$ * a * lap;
}
"""
    )
    .compile()
)

# ---------------------------------------------------------------------------
# fields (pooled - two flat buffers for ping-pong)
# ---------------------------------------------------------------------------
T0 = pool.get_data(np.float32, (NN,))
T1 = pool.get_data(np.float32, (NN,))

generate_walls_kernel(grid=GRID, block=BLOCK)
set_alpha_kernel(grid=GRID, block=BLOCK)
init_temperature_kernel(T0.data, grid=GRID, block=BLOCK)
apply_source_kernel(T0.data, grid=GRID, block=BLOCK)

# ---------------------------------------------------------------------------
# live view
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
im = ax.imshow(T0.to_numpy().reshape(GRID_N, GRID_N), cmap="inferno", vmin=20.0, vmax=OG_stove)
fig.colorbar(im, ax=ax, label="Temperature (deg C)")

wall_mask = wall_p.get().to_numpy().reshape(GRID_N, GRID_N)
wall_overlay = np.where(wall_mask == 1, 1.0, np.nan)
ax.imshow(wall_overlay, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.35)

ax.set_title("Heat diffusion in a floor plan (Cupy backend)")
time_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
    color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.4, pad=2),
)
fig.show()

clock = 0.0
sim_time = 0.0
try:
    while True:
        t_start = time.perf_counter()
        for _ in range(STEPS_PER_FRAME):
            clock += PULSE_FREQ * DT_VAL
            stove_p.set(OG_stove + 20.0 * math.sin(clock))

            diffuse_kernel(T1.data, T0.data, grid=GRID, block=BLOCK)
            apply_source_kernel(T1.data, grid=GRID, block=BLOCK)
            T0, T1 = T1, T0
            sim_time += DT_VAL

        cp.cuda.Device().synchronize()  # GPU is async; sync before stopping the timer
        frame_ms = (time.perf_counter() - t_start) * 1e3
        print(f"{STEPS_PER_FRAME} steps: {frame_ms:8.1f} ms  ({frame_ms / STEPS_PER_FRAME * 1e3:6.1f} us/step)")

        time_text.set_text(f"t = {sim_time:.0f} s")
        im.set_data(T0.to_numpy().reshape(GRID_N, GRID_N))
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.1)
except KeyboardInterrupt:
    pass

# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------
# destroy() hands a Parameter's storage back to the pool; it is a no-op on a
# const, which owns none. Safe only because nothing will launch again - the
# pool may reissue these buffers, while the compiled kernels above still point
# at them (see base.py, "Lifetime of a compiled object").
for param in (stove_p, wall_p, alpha_p):
    param.destroy()
pool.release_data(T0)
pool.release_data(T1)
print("pooled storage released")
