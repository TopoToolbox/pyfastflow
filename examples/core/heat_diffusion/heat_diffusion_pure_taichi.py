"""
Heat diffusion through a procedurally-generated floor plan (air + walls),
heated by a single stove - PURE Taichi, no pyfastflow framework.

Byte-for-byte the same model as heat_diffusion_taichi.py, written with plain
ti.field / ti.func / ti.kernel and module-global constants, so it can be timed
against the framework version as a zero-abstraction baseline. Constants are
captured as compile-time literals by Taichi directly (the framework's const
mode does the same via bindings); wall/alpha are flat fields; the stove
temperature is a 0-d field host-set each substep.

Author: B.G (07/2026)
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

# ---------------------------------------------------------------------------
# constants (baked into kernels as literals)
# ---------------------------------------------------------------------------
GRID_N = 512
STEPS_PER_FRAME = 10000
PULSE_FREQ = 0.0

ROOM_M = 3.0
DX_M = ROOM_M / (GRID_N // 4)
ALPHA_AIR_VAL = 0.015
ALPHA_WALL_VAL = 1.0e-6
CFL_SAFETY = 0.4
DT_VAL = CFL_SAFETY * DX_M**2 / (4.0 * ALPHA_AIR_VAL)
DX2_VAL = DX_M**2

N = GRID_N
ROOM = GRID_N // 4
WALL_THICK = 8
DOOR = 6
SEED = 17.0
T_BG = 15.0
SRC_I = GRID_N // 4 + GRID_N // 8
SRC_J = GRID_N // 4 + GRID_N // 8
SRC_R = 10
OG_stove = 70

# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------
T0 = ti.field(ti.f32, shape=(GRID_N, GRID_N))
T1 = ti.field(ti.f32, shape=(GRID_N, GRID_N))
wall = ti.field(ti.i32, shape=(GRID_N * GRID_N,))
alpha = ti.field(ti.f32, shape=(GRID_N * GRID_N,))
stove_t = ti.field(ti.f32, shape=())

# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------


@ti.func
def clamp(i):
    return min(max(i, 0), N - 1)


@ti.func
def laplacian(field_, i, j):
    ip = clamp(i + 1)
    im = clamp(i - 1)
    jp = clamp(j + 1)
    jm = clamp(j - 1)
    return field_[ip, j] + field_[im, j] + field_[i, jp] + field_[i, jm] - 4.0 * field_[i, j]


@ti.func
def whash(a, b):
    x = ti.cast(a, ti.f32) * 12.9898 + ti.cast(b, ti.f32) * 78.233 + SEED
    s = ti.sin(x) * 43758.5453
    return s - ti.floor(s)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@ti.kernel
def generate_walls():
    for i, j in ti.ndrange(N, N):
        is_wall = 0
        if i < WALL_THICK or i >= N - WALL_THICK or j < WALL_THICK or j >= N - WALL_THICK:
            is_wall = 1
        elif i % ROOM < WALL_THICK:
            vline = i // ROOM
            seg = j // ROOM
            door = ti.cast(whash(vline, seg) * ROOM, ti.i32)
            gap = (j % ROOM) >= door and (j % ROOM) < door + DOOR
            if not gap:
                is_wall = 1
        elif j % ROOM < WALL_THICK:
            hline = j // ROOM
            seg = i // ROOM
            door = ti.cast(whash(hline + 7919, seg) * ROOM, ti.i32)
            gap = (i % ROOM) >= door and (i % ROOM) < door + DOOR
            if not gap:
                is_wall = 1
        wall[i * N + j] = is_wall


@ti.kernel
def set_alpha():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if wall[idx] == 1:
            alpha[idx] = ALPHA_WALL_VAL
        else:
            alpha[idx] = ALPHA_AIR_VAL


@ti.kernel
def init_temperature(T: ti.template()):
    for i, j in T:
        T[i, j] = T_BG


@ti.kernel
def apply_source(T: ti.template()):
    for i, j in T:
        dx = i - SRC_I
        dy = j - SRC_J
        if dx * dx + dy * dy <= SRC_R * SRC_R:
            T[i, j] = stove_t[None]


@ti.kernel
def diffuse(T_out: ti.template(), T_in: ti.template()):
    for i, j in T_in:
        idx = i * N + j
        a = alpha[idx]
        lap = laplacian(T_in, i, j) / DX2_VAL
        T_out[i, j] = T_in[i, j] + DT_VAL * a * lap


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
stove_t[None] = OG_stove
generate_walls()
set_alpha()
init_temperature(T0)
apply_source(T0)

# ---------------------------------------------------------------------------
# live view
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
im = ax.imshow(T0.to_numpy(), cmap="inferno", vmin=20.0, vmax=OG_stove)
fig.colorbar(im, ax=ax, label="Temperature (deg C)")

wall_mask = wall.to_numpy().reshape(GRID_N, GRID_N)
wall_overlay = np.where(wall_mask == 1, 1.0, np.nan)
ax.imshow(wall_overlay, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.35)

ax.set_title("Heat diffusion in a floor plan (pure Taichi)")
time_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
    color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.4, pad=2),
)
fig.show()

clock = 0.0
sim_time = 0.0
while True:
    t_start = time.perf_counter()
    for _ in range(STEPS_PER_FRAME):
        clock += PULSE_FREQ * DT_VAL
        stove_t[None] = OG_stove + 20.0 * math.sin(clock)

        diffuse(T1, T0)
        apply_source(T1)
        T0, T1 = T1, T0
        sim_time += DT_VAL

    ti.sync()  # GPU is async; sync before stopping the timer
    frame_ms = (time.perf_counter() - t_start) * 1e3
    print(f"{STEPS_PER_FRAME} steps: {frame_ms:8.1f} ms  ({frame_ms / STEPS_PER_FRAME * 1e3:6.1f} us/step)")

    time_text.set_text(f"t = {sim_time:.0f} s")
    im.set_data(T0.to_numpy())
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.1)
