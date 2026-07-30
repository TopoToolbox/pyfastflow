"""
Same model and setup as heat_diffusion_cupy.py, with the per-substep
ping-pong expressed as a Routine instead of a hand-written python loop.

heat_diffusion_cupy.py alternates `diffuse_kernel(T1, T0, ...)`,
`apply_source_kernel(T1, ...)`, then swaps the T0/T1 python names each
iteration. A Routine has no python between its steps to do that swap in, so
the two iterations that one swap-pair covers are unrolled into one routine
with a repeat block, add_swap standing in for the python-level
`T0, T1 = T1, T0`:

    begin_repeat(times=2)
        diffuse(T1, T0); apply_source(T1); swap(T0, T1)
    end_repeat()

which records the body once and replays it twice, giving the same six-step
sequence as writing it out by hand:

    diffuse(T1, T0); apply_source(T1); swap(T0, T1)
    diffuse(T1, T0); apply_source(T1); swap(T0, T1)

Two swaps compose to the identity, which is exactly what compile() checks
for - so the compiled routine can be called over and over, each call
advancing the simulation by two substeps, and the result always ends up back
in the T0 buffer, matching two iterations of the manual loop.

diffuse_builder and apply_source_builder are ordinary KernelBuilders, built
exactly as in heat_diffusion_cupy.py; apply_source_builder is also compiled
once on its own to seed T0 before the loop starts, same as that file does -
compile() does not consume a builder, so the same builder is later handed to
add_kernel() unchanged. The routine's one shared bag is the merge of what
each builder already binds, so nothing about either CUDA source template
changes.

cupy has no auto-ranging launch the way Taichi/Quadrants derive one from the
template, so grid/block are set once on CupyRoutineBuilder's constructor and
apply to every step that does not override them - see cupy_backend.py,
CupyRoutineBuilder.

The stove's pulse is only updated between routine() calls, not between the
two substeps a single call unrolls: set() on the stove's scalar Parameter is
safe between calls (see routine.py, "Contract: no set()/destroy()
mid-routine"), but doing it *inside* a routine's steps is exactly what that
contract forbids, since there is no python between steps to run it in. With
PULSE_FREQ=0.0 by default the stove is steady anyway and this has no visible
effect.

Author: B.G (07/2026)
"""

import math
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
# host-side constants
# ---------------------------------------------------------------------------
GRID_N = 512
NN = GRID_N * GRID_N
STEPS_PER_FRAME = 10000  # two routine substeps per call - see the loop below
PULSE_FREQ = 0.0  # stove temperature oscillation speed, rad/s (0 = steady stove)
BLOCK = 256
GRID = (NN + BLOCK - 1) // BLOCK

ROOM_M = 3.0
DX_M = ROOM_M / (GRID_N // 4)
ALPHA_AIR_VAL = 0.015
ALPHA_WALL_VAL = 1.0e-6
CFL_SAFETY = 0.4
DT_VAL = CFL_SAFETY * DX_M**2 / (4.0 * ALPHA_AIR_VAL)

pool = CupyPool()

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

OG_stove = 70
stove_p = CupyParameter("STOVE_T", dtype=np.float32, mode="scalar", value=OG_stove, pool=pool)

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
# one-shot setup kernels (run once, outside the routine, exactly as in the
# manual-loop example)
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

stove = Bag(
    {
        "at": Bag({"i": src_i_p, "j": src_j_p}),
        "r": src_r_p,
        "temp": stove_p,
    }
)

# Kept as a builder, not just a compiled Kernel: compile() below seeds T0
# once, standalone, and the very same builder is later handed to the
# routine's add_kernel() - compile() does not consume it (see compile.py).
apply_source_builder = (
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
)
apply_source_kernel = apply_source_builder.compile()

heat = Bag({"alpha": alpha_p, "lap": laplacian_fn, "dt": dt_p, "dx2": dx2_p})

diffuse_builder = (
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
# the routine: two unrolled substeps, T0/T1 swapped back to their starting
# roles by the end, so it can be called over and over. grid/block are set
# once on the builder and apply to both steps.
# ---------------------------------------------------------------------------
routine_bag = merge(diffuse_builder.as_bag(), apply_source_builder.as_bag())

diffusion_routine = (
    CupyRoutineBuilder(grid=GRID, block=BLOCK)
    .add_data("T0", T0.data)
    .add_data("T1", T1.data)
    .bind_bag(routine_bag)
    .begin_repeat(times=2)
    .add_kernel(diffuse_builder, data_handle_ref=("T1", "T0"))
    .add_kernel(apply_source_builder, data_handle_ref=("T1",))
    .add_swap("T0", "T1")
    .end_repeat()
    .compile()
)

# ---------------------------------------------------------------------------
# live view
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
im = ax.imshow(T0.to_numpy().reshape(GRID_N, GRID_N), cmap="inferno", vmin=20.0, vmax=OG_stove)
fig.colorbar(im, ax=ax, label="Temperature (deg C)")

wall_mask = wall_p.get().to_numpy().reshape(GRID_N, GRID_N)
wall_overlay = np.where(wall_mask == 1, 1.0, np.nan)
ax.imshow(wall_overlay, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.35)

ax.set_title("Heat diffusion in a floor plan (Cupy backend, Routine)")
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
        for _ in range(STEPS_PER_FRAME // 2):
            clock += 2.0 * PULSE_FREQ * DT_VAL
            stove_p.set(OG_stove + 20.0 * math.sin(clock))

            diffusion_routine()  # two substeps, result lands back in T0
            sim_time += 2.0 * DT_VAL

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
for param in (stove_p, wall_p, alpha_p):
    param.destroy()
pool.release_data(T0)
pool.release_data(T1)
print("pooled storage released")
