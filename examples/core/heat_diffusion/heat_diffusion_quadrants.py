"""
Heat diffusion through a procedurally-generated floor plan (air + walls),
heated by a single stove, built on pyfastflow's backend-agnostic core
(Parameter/Helper/Kernel/Pool), Quadrants backend.

Pipeline:
  - generate_walls: pointwise kernel, carves a grid of rooms with doors into
    a wall mask (field-mode Parameter `wall`, written device-side via
    wall.set_node), using a deterministic hash device helper instead of
    per-cell RNG so wall/door layout is reproducible from SEED.
  - set_alpha: seeds a per-cell diffusivity field (`alpha`, field mode) from
    `wall` - air diffuses fast, walls slow.
  - init_temperature: fills T with the background temperature.
  - apply_source: clamps a disc of cells around the stove to `stove.temp`
    (scalar-mode Parameter, updated from the host every substep -> a gently
    pulsing stove).
  - diffuse: explicit FTCS heat equation dT/dt = alpha(i,j) * lap(T), with a
    clamped (Neumann / no-flux) boundary Laplacian.

Uniform device surface: every Parameter is read with p.get(node) and written
with p.set_node(node, val) regardless of const/scalar/field mode - the
kernels never branch on mode, so re-declaring `alpha` as a single const
(uniform room, no walls) needs no kernel-body change.

Binding styles, all three visible in one file:
  - flat, one bind() per object (most kernels here);
  - a Bag bound whole and reached by dotted path - `stove` in apply_source,
    which nests a sub-bag for the position, and `heat` in diffuse, which mixes
    a Parameter, a device helper and two consts under one name;
  - bind_bag(), merging a bag's members in flat under their own names, so the
    kernel still sees plain names - `alpha_seeds` in set_alpha.

Compilation is the two-layer builder: QuadrantsKernelBuilder /
QuadrantsHelperBuilder collect bind()ed params + helper builders and one
ingest()ed template. A HelperBuilder (clamp_fn, laplacian_fn, whash_fn below)
is a recipe, not a compiled object - it has no compile() of its own. Binding
one into a kernel, flat or through a Bag, is what specializes it:
QuadrantsKernelBuilder.compile() specializes every HelperBuilder the kernel
reaches, against that kernel's own bindings, before compiling the kernel
body. Recompiling the kernel after rebinding a const the helper reads picks
up the new value without touching the helper builder itself.

Author: B.G (07/2026)
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import quadrants as qd

from pyfastflow.experimental.core.context.bag import Bag
from pyfastflow.experimental.core.context.quadrants_backend import (
    QuadrantsHelperBuilder,
    QuadrantsKernelBuilder,
    QuadrantsParameter,
)
from pyfastflow.experimental.core.pool.quadrants_pool import QuadrantsPool

qd.init(arch=qd.gpu)

# ---------------------------------------------------------------------------
# host-side constants (grid size, loop/timing counts - never used as kernel globals)
# ---------------------------------------------------------------------------
GRID_N = 512
STEPS_PER_FRAME = 10000
PULSE_FREQ = 0.0  # stove temperature oscillation speed, rad/s (0 = steady stove)

# Physical grounding: without a cell size, DT/ALPHA are just numbers tuned by
# feel - here they're derived from a real room size and real diffusivities so
# "seconds" and "m^2/s" mean what they say.
ROOM_M = 3.0  # room span, meters (rooms are GRID_N//4 cells across)
DX_M = ROOM_M / (GRID_N // 4)  # meters per cell

# Air's real molecular thermal diffusivity (~2.2e-5 m^2/s) would take DAYS to
# spread heat by pure conduction - rooms actually heat by convective mixing.
# ALPHA_AIR below is an effective/turbulent diffusivity standing in for that
# mixing, not molecular diffusion - otherwise a stove would need real hours.
ALPHA_AIR_VAL = 0.015  # m^2/s, effective convective air diffusivity
ALPHA_WALL_VAL = 1.0e-6  # m^2/s, real solid (drywall/brick-like) diffusivity

# Explicit FTCS stability limit is dt <= dx^2 / (4*alpha); stay well under it.
CFL_SAFETY = 0.4
DT_VAL = CFL_SAFETY * DX_M**2 / (4.0 * ALPHA_AIR_VAL)  # seconds

pool = QuadrantsPool()

# Structural constants: const mode, bake to compile-time literals in generated
# code even though the kernel body still reads them via .get(0).
n_p = QuadrantsParameter("N", dtype=qd.i32, mode="const", value=GRID_N, pool=pool)
room_p = QuadrantsParameter("ROOM", dtype=qd.i32, mode="const", value=GRID_N // 4, pool=pool)
wall_thick_p = QuadrantsParameter("WALL_THICK", dtype=qd.i32, mode="const", value=8, pool=pool)
door_p = QuadrantsParameter("DOOR", dtype=qd.i32, mode="const", value=6, pool=pool)
seed_p = QuadrantsParameter("SEED", dtype=qd.f32, mode="const", value=17.0, pool=pool)

dt_p = QuadrantsParameter("DT", dtype=qd.f32, mode="const", value=DT_VAL, pool=pool)  # seconds
dx2_p = QuadrantsParameter("DX2", dtype=qd.f32, mode="const", value=DX_M**2, pool=pool)  # meters^2

# Seed values for the alpha field - read via .get(0) inside set_alpha.
alpha_air_seed_p = QuadrantsParameter("ALPHA_AIR_SEED", dtype=qd.f32, mode="const", value=ALPHA_AIR_VAL, pool=pool)
alpha_wall_seed_p = QuadrantsParameter("ALPHA_WALL_SEED", dtype=qd.f32, mode="const", value=ALPHA_WALL_VAL, pool=pool)
t_bg_p = QuadrantsParameter("T_BG", dtype=qd.f32, mode="const", value=15.0, pool=pool)

src_i_p = QuadrantsParameter("SRC_I", dtype=qd.i32, mode="const", value=GRID_N // 4 + GRID_N // 8, pool=pool)
src_j_p = QuadrantsParameter("SRC_J", dtype=qd.i32, mode="const", value=GRID_N // 4 + GRID_N // 8, pool=pool)
src_r_p = QuadrantsParameter("SRC_R", dtype=qd.i32, mode="const", value=10, pool=pool)  # stove radius, cells

# scalar mode: a 0-d field, host-settable every frame -> a pulsing stove
# temperature. Reached in-kernel as stove.temp.get(0) (see the stove Bag).
OG_stove = 70
stove_p = QuadrantsParameter("STOVE_T", dtype=qd.f32, mode="scalar", value=OG_stove, pool=pool)

# field mode: per-cell wall/air mask, written device-side via wall.set_node,
# read via wall.get.
wall_p = QuadrantsParameter("WALL", dtype=qd.i32, mode="field", value=np.zeros(GRID_N * GRID_N), pool=pool, n_flat=GRID_N * GRID_N)

# field mode: per-cell thermal diffusivity, read in diffuse via alpha.get - so
# switching this Parameter to const/scalar mode later needs no kernel edits.
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
# kernels
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


# The two seed values are grouped on the host for tidiness, then merged in with
# bind_bag() - which binds each member flat, under its own name. The template
# above is unaware: it still reads ALPHA_WALL_SEED / ALPHA_AIR_SEED bare. Use
# this when a bag is a convenient way to carry things around but the kernel
# wants plain names; bind() the bag whole instead when you want a dotted path.
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


# The stove travels as ONE nested Bag rather than four flat binds: its position
# is grouped into an `at` sub-bag. Every member, whatever mode, is reached the
# same way - .get(0) - so const and scalar Parameters sit side by side under
# one name. Everything else here still binds flat, so the two styles sit side
# by side in one file.
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


apply_source_kernel = QuadrantsKernelBuilder().bind("stove", stove).ingest(apply_source_template).compile()


# A MIXED Bag: everything the diffusion step needs, whatever kind it is - a
# field Parameter, a device helper, two const Parameters - under one name. A
# bag has no member type; each is resolved on its own at compile time, so
# `heat.alpha` becomes a device accessor, `heat.lap` a compiled func, and
# `heat.dx2` a device accessor whose .get(0) bakes to a literal.
heat = Bag({"alpha": alpha_p, "lap": laplacian_fn, "dt": dt_p, "dx2": dx2_p})


def diffuse_template(T_out: qd.Tensor, T_in: qd.Tensor):
    for i, j in T_in:
        idx = i * N.get(0) + j
        a = heat.alpha.get(idx)
        lap = heat.lap(T_in, i, j) / heat.dx2.get(0)
        T_out[i, j] = T_in[i, j] + heat.dt.get(0) * a * lap

diffuse_kernel = QuadrantsKernelBuilder().bind("N", n_p).bind("heat", heat).ingest(diffuse_template).compile()

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
# live view
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
im = ax.imshow(T0.to_numpy(), cmap="inferno", vmin=20.0, vmax=OG_stove)
fig.colorbar(im, ax=ax, label="Temperature (deg C)")

wall_mask = wall_p.get().to_numpy().reshape(GRID_N, GRID_N)
wall_overlay = np.where(wall_mask == 1, 1.0, np.nan)
ax.imshow(wall_overlay, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.35)

ax.set_title("Heat diffusion in a floor plan (Quadrants backend)")
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

            diffuse_kernel(T1.data, T0.data)
            apply_source_kernel(T1.data)
            T0, T1 = T1, T0
            sim_time += dt_p.get()

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
# destroy() hands a Parameter's storage back to the pool; it is a no-op on a
# const, which owns none. Safe only because nothing will launch again - the
# pool may reissue these buffers, while the compiled kernels above still point
# at them (see base.py, "Lifetime of a compiled object").
for param in (stove_p, wall_p, alpha_p):
    param.destroy()
pool.release_data(T0)
pool.release_data(T1)
print("pooled storage released")
