"""
Shallow-water waves in a square tank, built on pyfastflow's backend-agnostic
core (Parameter/Helper/Kernel/Pool + Bags), Taichi backend.

Model: the Kass & Miller (1990) stable shallow-water update on an Arakawa-C
staggered grid. Cell-centered water column height h, x-velocity u on vertical
faces, y-velocity v on horizontal faces:
  - update_velocity: u,v accelerate down the height gradient (g * grad h),
    then light damping; domain-edge faces are pinned to 0 (reflective walls).
  - update_height: h advected by the velocity divergence with upwind face
    depths (mass-conserving, stable) -> waves, sloshing, reflection.
  - apply_drop: a disc splash raises h wherever a "stone" lands; the landing
    site + amplitude come from host-set scalar params, so stones drop live.

Bag showcase (heat_diffusion mixes flat binds, bind_bag and a nested bag; here
everything goes through whole-bag binds): the physical constants travel
as ONE `phys` Bag (g/dx/dt/damp), read in-kernel as phys.g.get(0),
phys.dx.get(0), ...; the neighbour math travels as ONE `ops` Bag
(clamp, face_depth), called as ops.clamp(i), ops.face_depth(...); and the
splash controls travel as a `drop` Bag (cx/cy/amp), read drop.cx.get(0).
Bind the whole bag once (bind("phys", phys_bag)); dotted paths in the template
resolve to each member's device view - the kernel body never names them flat.
The three bags are split by role, not by kind: a Bag has no member type, so
one could equally hold `phys` and `ops` together (see heat_diffusion's `heat`).

Structural constants (N, DROP_R, REST_DEPTH) are const Parameters, read
uniformly via .get(0) - the value still bakes to a compile-time literal in
generated code.

Author: B.G (07/2026)
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from pyfastflow.experimental.core.context.bag import Bag
from pyfastflow.experimental.core.context.taichi_backend import (
    TaichiHelperBuilder,
    TaichiKernelBuilder,
    TaichiParameter,
)
from pyfastflow.experimental.core.pool.taichi_pool import TaichiPool

ti.init(arch=ti.gpu)

# ---------------------------------------------------------------------------
# host-side constants (grid size, loop/timing counts - never kernel globals)
# ---------------------------------------------------------------------------
GRID_N = 640
STEPS_PER_FRAME = 40
DROP_EVERY = 25  # frames between automatic stone drops

# Physical grounding: a 4 m x 4 m tank holding a thin (5 cm) sheet of water.
# Shallow-water wave speed is c = sqrt(g*H); the explicit CFL limit is
# dt <= dx / (c*sqrt(2)), so dt is derived from the tank, not tuned by feel.
WORLD_M = 4.0
DX_M = WORLD_M / GRID_N  # meters per cell
G_VAL = 9.81  # m/s^2
REST_DEPTH_VAL = 0.05  # m, still-water column height
WAVE_C = (G_VAL * REST_DEPTH_VAL) ** 0.5  # m/s
CFL_SAFETY = 0.4
DT_VAL = CFL_SAFETY * DX_M / (WAVE_C * 2.0**0.5)  # seconds

# Light drag so a tank eventually settles: per-step factor 1 - rate*dt.
DAMP_RATE = 0.3  # 1/s
DAMP_VAL = 1.0 - DAMP_RATE * DT_VAL

DROP_R_VAL = 12  # splash radius, cells
DROP_AMP_VAL = 0.02  # m, height a stone adds at impact

pool = TaichiPool()

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
# Structural constants: const mode, folded into the generated code as a
# literal but still read through .get(0).
n_p = TaichiParameter("N", dtype=ti.i32, mode="const", value=GRID_N, pool=pool)
rest_depth_p = TaichiParameter("REST_DEPTH", dtype=ti.f32, mode="const", value=REST_DEPTH_VAL, pool=pool)
drop_r_p = TaichiParameter("DROP_R", dtype=ti.i32, mode="const", value=DROP_R_VAL, pool=pool)

# phys Bag: g is scalar (host-tunable live), dx/dt/damp are const - but all
# read uniformly as phys.<name>.get(0), so the kernels never branch on mode.
g_p = TaichiParameter("g", dtype=ti.f32, mode="scalar", value=G_VAL, pool=pool)
dx_p = TaichiParameter("dx", dtype=ti.f32, mode="const", value=DX_M, pool=pool)
dt_p = TaichiParameter("dt", dtype=ti.f32, mode="const", value=DT_VAL, pool=pool)
damp_p = TaichiParameter("damp", dtype=ti.f32, mode="const", value=DAMP_VAL, pool=pool)
phys = Bag({"g": g_p, "dx": dx_p, "dt": dt_p, "damp": damp_p})

# drop Bag: splash site + amplitude, host-set each time a stone falls.
drop_cx_p = TaichiParameter("cx", dtype=ti.i32, mode="scalar", value=GRID_N // 2, pool=pool)
drop_cy_p = TaichiParameter("cy", dtype=ti.i32, mode="scalar", value=GRID_N // 2, pool=pool)
drop_amp_p = TaichiParameter("amp", dtype=ti.f32, mode="scalar", value=0.0, pool=pool)
drop = Bag({"cx": drop_cx_p, "cy": drop_cy_p, "amp": drop_amp_p})

# ---------------------------------------------------------------------------
# device helpers -> ops Bag
# ---------------------------------------------------------------------------


def clamp(i):
    return min(max(i, 0), N.get(0) - 1)


clamp_fn = TaichiHelperBuilder().bind("N", n_p).ingest(clamp)


def face_depth(up, down, vel):
    """Upwind water depth at a face: the upstream cell when flow is outward."""
    d = down
    if vel > 0.0:
        d = up
    return d


face_depth_fn = TaichiHelperBuilder().ingest(face_depth)

ops = Bag({"clamp": clamp_fn, "face_depth": face_depth_fn})

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def init_height_template(h: ti.template()):
    for i, j in h:
        h[i, j] = REST_DEPTH.get(0)


init_height_kernel = TaichiKernelBuilder().bind("REST_DEPTH", rest_depth_p).ingest(init_height_template).compile()


def apply_drop_template(h: ti.template()):
    for i, j in h:
        dxr = i - drop.cx.get(0)
        dyr = j - drop.cy.get(0)
        if dxr * dxr + dyr * dyr <= DROP_R.get(0) * DROP_R.get(0):
            h[i, j] += drop.amp.get(0)


apply_drop_kernel = (
    TaichiKernelBuilder()
    .bind("drop", drop)
    .bind("DROP_R", drop_r_p)
    .ingest(apply_drop_template)
    .compile()
)


def update_velocity_template(u: ti.template(), v: ti.template(), h: ti.template()):
    for i, j in h:
        # u lives on the west face of cell (i,j); i==0 is the tank wall.
        if i > 0:
            acc = phys.g.get(0) * phys.dt.get(0) / phys.dx.get(0) * (h[i - 1, j] - h[i, j])
            u[i, j] = (u[i, j] + acc) * phys.damp.get(0)
        else:
            u[i, j] = 0.0
        # v lives on the south face of cell (i,j); j==0 is the tank wall.
        if j > 0:
            acc = phys.g.get(0) * phys.dt.get(0) / phys.dx.get(0) * (h[i, j - 1] - h[i, j])
            v[i, j] = (v[i, j] + acc) * phys.damp.get(0)
        else:
            v[i, j] = 0.0


update_velocity_kernel = TaichiKernelBuilder().bind("phys", phys).ingest(update_velocity_template).compile()


def update_height_template(h_out: ti.template(), h_in: ti.template(), u: ti.template(), v: ti.template()):
    for i, j in h_in:
        # face velocities: u[i]=west face, u[i+1]=east face (0 at the wall).
        uw = u[i, j]
        ue = 0.0
        if i < N.get(0) - 1:
            ue = u[i + 1, j]
        vs = v[i, j]
        vn = 0.0
        if j < N.get(0) - 1:
            vn = v[i, j + 1]

        ip = ops.clamp(i + 1)
        im = ops.clamp(i - 1)
        jp = ops.clamp(j + 1)
        jm = ops.clamp(j - 1)

        hw = ops.face_depth(h_in[im, j], h_in[i, j], uw)
        he = ops.face_depth(h_in[i, j], h_in[ip, j], ue)
        hs = ops.face_depth(h_in[i, jm], h_in[i, j], vs)
        hn = ops.face_depth(h_in[i, j], h_in[i, jp], vn)

        flux = (he * ue - hw * uw) + (hn * vn - hs * vs)
        h_out[i, j] = h_in[i, j] - phys.dt.get(0) / phys.dx.get(0) * flux


update_height_kernel = (
    TaichiKernelBuilder()
    .bind("N", n_p)
    .bind("phys", phys)
    .bind("ops", ops)
    .ingest(update_height_template)
    .compile()
)

# ---------------------------------------------------------------------------
# fields (pooled; h is ping-ponged, u/v updated in place - start at 0)
# ---------------------------------------------------------------------------
h0 = pool.get_data(ti.f32, (GRID_N, GRID_N))
h1 = pool.get_data(ti.f32, (GRID_N, GRID_N))
u = pool.get_data(ti.f32, (GRID_N, GRID_N))
v = pool.get_data(ti.f32, (GRID_N, GRID_N))

init_height_kernel(h0.data)

# first stone, dead center, so there is motion on frame 0
drop_cx_p.set(GRID_N // 2)
drop_cy_p.set(GRID_N // 2)
drop_amp_p.set(DROP_AMP_VAL)
apply_drop_kernel(h0.data)
drop_amp_p.set(0.0)

# ---------------------------------------------------------------------------
# live view (surface elevation h - rest depth)
# ---------------------------------------------------------------------------
elev_lim = DROP_AMP_VAL * 0.35
fig, ax = plt.subplots()
im = ax.imshow((h0.to_numpy() - REST_DEPTH_VAL).T, cmap="RdBu_r", vmin=-elev_lim, vmax=elev_lim, origin="lower")
fig.colorbar(im, ax=ax, label="surface elevation (m)")
ax.set_title("Shallow-water waves in a tank (Taichi backend)")
time_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
    color="black", fontsize=9, bbox=dict(facecolor="white", alpha=0.5, pad=2),
)
fig.show()

sim_time = 0.0
frame = 0
try:
    while True:
        frame += 1
        if frame % DROP_EVERY == 0:
            drop_cx_p.set(random.randint(DROP_R_VAL, GRID_N - 1 - DROP_R_VAL))
            drop_cy_p.set(random.randint(DROP_R_VAL, GRID_N - 1 - DROP_R_VAL))
            drop_amp_p.set(DROP_AMP_VAL)
            apply_drop_kernel(h0.data)
            drop_amp_p.set(0.0)

        t_start = time.perf_counter()
        for _ in range(STEPS_PER_FRAME):
            update_velocity_kernel(u.data, v.data, h0.data)
            update_height_kernel(h1.data, h0.data, u.data, v.data)
            h0, h1 = h1, h0
            sim_time += DT_VAL

        ti.sync()  # GPU is async; sync before stopping the timer
        frame_ms = (time.perf_counter() - t_start) * 1e3
        print(f"{STEPS_PER_FRAME} steps: {frame_ms:8.1f} ms  ({frame_ms / STEPS_PER_FRAME * 1e3:6.1f} us/step)")

        time_text.set_text(f"t = {sim_time:.1f} s")
        im.set_data((h0.to_numpy() - REST_DEPTH_VAL).T)
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
# at them (see parameter.py, "Lifetime of a compiled object").
for param in (g_p, drop_cx_p, drop_cy_p, drop_amp_p):
    param.destroy()
for buf in (h0, h1, u, v):
    pool.release_data(buf)
print("pooled storage released")
