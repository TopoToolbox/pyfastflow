"""
Same model and setup as heat_diffusion_quadrants.py, with the per-substep
ping-pong expressed as a Routine instead of a hand-written python loop.

heat_diffusion_quadrants.py alternates `diffuse_kernel(T1, T0)`,
`apply_source_kernel(T1)`, then swaps the T0/T1 python names each iteration.
A Routine has no python between its steps to do that swap in, so the two
iterations that one swap-pair covers are unrolled into one routine, with
add_swap standing in for the python-level `T0, T1 = T1, T0`:

    diffuse(T1, T0); apply_source(T1); swap(T0, T1)
    diffuse(T1, T0); apply_source(T1); swap(T0, T1)

Two swaps compose to the identity, which is exactly what compile() checks
for - so the compiled routine can be called over and over, each call
advancing the simulation by two substeps, and the result always ends up back
in the T0 buffer, matching two iterations of the manual loop.

diffuse_builder and apply_source_builder are ordinary KernelBuilders, built
exactly as in heat_diffusion_quadrants.py; apply_source_builder is also
compiled once on its own to seed T0 before the loop starts, same as that file
does - compile() does not consume a builder, so the same builder is later
handed to add_kernel() unchanged. The routine's one shared bag is the merge
of what each builder already binds, so nothing about diffuse_template or
apply_source_template's own bodies changes.

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

import matplotlib.pyplot as plt
import numpy as np
import quadrants as qd

from pyfastflow.experimental.core.context.bag import Bag, merge
from pyfastflow.experimental.core.context.quadrants_backend import (
    QuadrantsHelperBuilder,
    QuadrantsKernelBuilder,
    QuadrantsParameter,
    QuadrantsRoutineBuilder,
)
from pyfastflow.experimental.core.pool.quadrants_pool import QuadrantsPool

qd.init(arch=qd.gpu)

# ---------------------------------------------------------------------------
# host-side constants (grid size, loop/timing counts - never used as kernel globals)
# ---------------------------------------------------------------------------
GRID_N = 512
STEPS_PER_FRAME = 10000  # two routine substeps per call - see the loop below
PULSE_FREQ = 0.0  # stove temperature oscillation speed, rad/s (0 = steady stove)

ROOM_M = 3.0
DX_M = ROOM_M / (GRID_N // 4)
ALPHA_AIR_VAL = 0.015
ALPHA_WALL_VAL = 1.0e-6
CFL_SAFETY = 0.4
DT_VAL = CFL_SAFETY * DX_M**2 / (4.0 * ALPHA_AIR_VAL)

pool = QuadrantsPool()

n_p = QuadrantsParameter("N", dtype=qd.i32, mode="const", value=GRID_N, pool=pool)
room_p = QuadrantsParameter("ROOM", dtype=qd.i32, mode="const", value=GRID_N // 4, pool=pool)
wall_thick_p = QuadrantsParameter("WALL_THICK", dtype=qd.i32, mode="const", value=8, pool=pool)
door_p = QuadrantsParameter("DOOR", dtype=qd.i32, mode="const", value=6, pool=pool)
seed_p = QuadrantsParameter("SEED", dtype=qd.f32, mode="const", value=17.0, pool=pool)

dt_p = QuadrantsParameter("DT", dtype=qd.f32, mode="const", value=DT_VAL, pool=pool)
dx2_p = QuadrantsParameter("DX2", dtype=qd.f32, mode="const", value=DX_M**2, pool=pool)

alpha_air_seed_p = QuadrantsParameter("ALPHA_AIR_SEED", dtype=qd.f32, mode="const", value=ALPHA_AIR_VAL, pool=pool)
alpha_wall_seed_p = QuadrantsParameter("ALPHA_WALL_SEED", dtype=qd.f32, mode="const", value=ALPHA_WALL_VAL, pool=pool)
t_bg_p = QuadrantsParameter("T_BG", dtype=qd.f32, mode="const", value=15.0, pool=pool)

src_i_p = QuadrantsParameter("SRC_I", dtype=qd.i32, mode="const", value=GRID_N // 4 + GRID_N // 8, pool=pool)
src_j_p = QuadrantsParameter("SRC_J", dtype=qd.i32, mode="const", value=GRID_N // 4 + GRID_N // 8, pool=pool)
src_r_p = QuadrantsParameter("SRC_R", dtype=qd.i32, mode="const", value=10, pool=pool)

OG_stove = 70
stove_p = QuadrantsParameter("STOVE_T", dtype=qd.f32, mode="scalar", value=OG_stove, pool=pool)

wall_p = QuadrantsParameter("WALL", dtype=qd.i32, mode="field", value=np.zeros(GRID_N * GRID_N), pool=pool, n_flat=GRID_N * GRID_N)
alpha_p = QuadrantsParameter("ALPHA", dtype=qd.f32, mode="field", value=np.zeros(GRID_N * GRID_N), pool=pool, n_flat=GRID_N * GRID_N)

# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------


def clamp(i):
    return min(max(i, 0), N.get(0) - 1)


clamp_fn = QuadrantsHelperBuilder().bind("N", n_p).ingest(clamp)


def laplacian(field_, i, j):
    ip = clamp(i + 1)
    im = clamp(i - 1)
    jp = clamp(j + 1)
    jm = clamp(j - 1)
    return field_[ip, j] + field_[im, j] + field_[i, jp] + field_[i, jm] - 4.0 * field_[i, j]


laplacian_fn = QuadrantsHelperBuilder().bind("clamp", clamp_fn).ingest(laplacian)


def whash(a, b):
    """Deterministic pseudo-random value in [0, 1) for two integer indices."""
    x = qd.cast(a, qd.f32) * 12.9898 + qd.cast(b, qd.f32) * 78.233 + SEED.get(0)
    s = qd.sin(x) * 43758.5453
    return s - qd.floor(s)


whash_fn = QuadrantsHelperBuilder().bind("SEED", seed_p).ingest(whash)

# ---------------------------------------------------------------------------
# one-shot setup kernels (run once, outside the routine, exactly as in the
# manual-loop example)
# ---------------------------------------------------------------------------


def generate_walls_template():
    for i, j in qd.ndrange(N.get(0), N.get(0)):
        is_wall = 0
        if i < WALL_THICK.get(0) or i >= N.get(0) - WALL_THICK.get(0) or j < WALL_THICK.get(0) or j >= N.get(0) - WALL_THICK.get(0):
            is_wall = 1
        elif i % ROOM.get(0) < WALL_THICK.get(0):
            vline = i // ROOM.get(0)
            seg = j // ROOM.get(0)
            door = qd.cast(whash(vline, seg) * ROOM.get(0), qd.i32)
            gap = (j % ROOM.get(0)) >= door and (j % ROOM.get(0)) < door + DOOR.get(0)
            if not gap:
                is_wall = 1
        elif j % ROOM.get(0) < WALL_THICK.get(0):
            hline = j // ROOM.get(0)
            seg = i // ROOM.get(0)
            door = qd.cast(whash(hline + 7919, seg) * ROOM.get(0), qd.i32)
            gap = (i % ROOM.get(0)) >= door and (i % ROOM.get(0)) < door + DOOR.get(0)
            if not gap:
                is_wall = 1
        wall.set_node(i * N.get(0) + j, is_wall)


generate_walls_kernel = (
    QuadrantsKernelBuilder()
    .bind("N", n_p)
    .bind("ROOM", room_p)
    .bind("WALL_THICK", wall_thick_p)
    .bind("DOOR", door_p)
    .bind("wall", wall_p)
    .bind("whash", whash_fn)
    .ingest(generate_walls_template)
    .compile()
)


def set_alpha_template():
    for i, j in qd.ndrange(N.get(0), N.get(0)):
        idx = i * N.get(0) + j
        if wall.get(idx) == 1:
            alpha.set_node(idx, ALPHA_WALL_SEED.get(0))
        else:
            alpha.set_node(idx, ALPHA_AIR_SEED.get(0))


alpha_seeds = Bag({"ALPHA_WALL_SEED": alpha_wall_seed_p, "ALPHA_AIR_SEED": alpha_air_seed_p})

set_alpha_kernel = (
    QuadrantsKernelBuilder()
    .bind("N", n_p)
    .bind("wall", wall_p)
    .bind("alpha", alpha_p)
    .bind_bag(alpha_seeds)
    .ingest(set_alpha_template)
    .compile()
)


def init_temperature_template(T: qd.Tensor):
    for i, j in T:
        T[i, j] = T_BG.get(0)


init_temperature_kernel = QuadrantsKernelBuilder().bind("T_BG", t_bg_p).ingest(init_temperature_template).compile()

stove = Bag(
    {
        "at": Bag({"i": src_i_p, "j": src_j_p}),
        "r": src_r_p,
        "temp": stove_p,
    }
)


def apply_source_template(T: qd.Tensor):
    for i, j in T:
        dx = i - stove.at.i.get(0)
        dy = j - stove.at.j.get(0)
        if dx * dx + dy * dy <= stove.r.get(0) * stove.r.get(0):
            T[i, j] = stove.temp.get(0)


# Kept as a builder, not just a compiled Kernel: compile() below seeds T0
# once, standalone, and the very same builder is later handed to the
# routine's add_kernel() - compile() does not consume it (see base.py).
apply_source_builder = QuadrantsKernelBuilder().bind("stove", stove).ingest(apply_source_template)
apply_source_kernel = apply_source_builder.compile()

heat = Bag({"alpha": alpha_p, "lap": laplacian_fn, "dt": dt_p, "dx2": dx2_p})


def diffuse_template(T_out: qd.Tensor, T_in: qd.Tensor):
    for i, j in T_in:
        idx = i * N.get(0) + j
        a = heat.alpha.get(idx)
        lap = heat.lap(T_in, i, j) / heat.dx2.get(0)
        T_out[i, j] = T_in[i, j] + heat.dt.get(0) * a * lap


diffuse_builder = QuadrantsKernelBuilder().bind("N", n_p).bind("heat", heat).ingest(diffuse_template)

# ---------------------------------------------------------------------------
# fields (pooled - two buffers for ping-pong)
# ---------------------------------------------------------------------------
T0 = pool.get_data(qd.f32, (GRID_N, GRID_N))
T1 = pool.get_data(qd.f32, (GRID_N, GRID_N))

generate_walls_kernel()
set_alpha_kernel()
init_temperature_kernel(T0.data)
apply_source_kernel(T0.data)

# ---------------------------------------------------------------------------
# the routine: two unrolled substeps, T0/T1 swapped back to their starting
# roles by the end, so it can be called over and over.
# ---------------------------------------------------------------------------
routine_bag = merge(diffuse_builder.as_bag(), apply_source_builder.as_bag())

diffusion_routine = (
    QuadrantsRoutineBuilder()
    .add_data("T0", T0.data)
    .add_data("T1", T1.data)
    .bind_bag(routine_bag)
    .add_kernel(diffuse_builder, data_handle_ref=("T1", "T0"))
    .add_kernel(apply_source_builder, data_handle_ref=("T1",))
    .add_swap("T0", "T1")
    .add_kernel(diffuse_builder, data_handle_ref=("T1", "T0"))
    .add_kernel(apply_source_builder, data_handle_ref=("T1",))
    .add_swap("T0", "T1")
    .compile()
)

# ---------------------------------------------------------------------------
# live view
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
im = ax.imshow(T0.to_numpy(), cmap="inferno", vmin=20.0, vmax=OG_stove)
fig.colorbar(im, ax=ax, label="Temperature (deg C)")

wall_mask = wall_p.get().to_numpy().reshape(GRID_N, GRID_N)
wall_overlay = np.where(wall_mask == 1, 1.0, np.nan)
ax.imshow(wall_overlay, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.35)

ax.set_title("Heat diffusion in a floor plan (Quadrants backend, Routine)")
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
            sim_time += 2.0 * dt_p.get()

        qd.sync()  # GPU is async; sync before stopping the timer
        frame_ms = (time.perf_counter() - t_start) * 1e3
        print(f"{STEPS_PER_FRAME} steps: {frame_ms:8.1f} ms  ({frame_ms / STEPS_PER_FRAME * 1e3:6.1f} us/step)")

        time_text.set_text(f"t = {sim_time:.0f} s")
        im.set_data(T0.to_numpy())
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
