"""
Shallow-water waves in a square tank, built on pyfastflow's backend-agnostic
core (Parameter/Helper/Kernel/Pool + Bags), Cupy (cp.RawKernel) backend.

Same model as shallow_water_taichi.py (Kass & Miller 1990 on an Arakawa-C
staggered grid), authored as CUDA source strings. The grid is stored flat
(N*N, row-major: idx = i*N + j); kernels launch one thread per cell.

Bag showcase: bags reach the CUDA source through the SAME `$...$` spans as
flat params, just with a dotted head - the span parser walks Bag members:

  $phys.dx.get(0)$    const bag member -> baked CUDA literal
  $phys.g.get(0)$     scalar bag member -> auto-generated `phys_g` pointer arg
  $drop.cx.get(0)$    host-set splash site, same mechanism
  $ops.clamp(i + 1)$  helper from a Bag -> its __device__ source spliced

so one .bind("phys", phys) / .bind("ops", ops) carries the whole group, and
the source never declares the generated pointer arguments. The three bags are
split by role, not by kind: a Bag has no member type, so one could equally hold
`phys` and `ops` together (see heat_diffusion's `heat`). Top-level const
params (N, REST_DEPTH, DROP_R) become #defines, used bare. Spans do not nest,
so span results are read into temps before being passed to a helper span.

Author: B.G (07/2026)
"""

import random
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from pyfastflow.experimental.core.context.bag import Bag
from pyfastflow.experimental.core.context.cupy_backend import (
    CupyHelperBuilder,
    CupyKernelBuilder,
    CupyParameter,
)
from pyfastflow.experimental.core.pool.cupy_pool import CupyPool

# ---------------------------------------------------------------------------
# host-side constants
# ---------------------------------------------------------------------------
GRID_N = 640
NN = GRID_N * GRID_N
STEPS_PER_FRAME = 40
DROP_EVERY = 25  # frames between automatic stone drops
BLOCK = 256
GRID = (NN + BLOCK - 1) // BLOCK

# Physical grounding (see the taichi demo for the reasoning): a 4 m x 4 m tank
# holding a thin (5 cm) sheet of water; c = sqrt(g*H), dt <= dx / (c*sqrt(2)).
WORLD_M = 4.0
DX_M = WORLD_M / GRID_N
G_VAL = 9.81
REST_DEPTH_VAL = 0.05
WAVE_C = (G_VAL * REST_DEPTH_VAL) ** 0.5
CFL_SAFETY = 0.4
DT_VAL = CFL_SAFETY * DX_M / (WAVE_C * 2.0**0.5)

DAMP_RATE = 0.3  # 1/s
DAMP_VAL = 1.0 - DAMP_RATE * DT_VAL

DROP_R_VAL = 12  # splash radius, cells
DROP_AMP_VAL = 0.02  # m, height a stone adds at impact

pool = CupyPool()

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
# Structural constants -> #define, used bare in the CUDA source.
n_p = CupyParameter("N", dtype=np.int32, mode="const", value=GRID_N, pool=pool)
rest_depth_p = CupyParameter("REST_DEPTH", dtype=np.float32, mode="const", value=REST_DEPTH_VAL, pool=pool)
drop_r_p = CupyParameter("DROP_R", dtype=np.int32, mode="const", value=DROP_R_VAL, pool=pool)

# phys Bag: g is scalar (host-tunable live), dx/dt/damp are const - all
# written the same way in the source, $phys.<name>.get(0)$.
g_p = CupyParameter("g", dtype=np.float32, mode="scalar", value=G_VAL, pool=pool)
dx_p = CupyParameter("dx", dtype=np.float32, mode="const", value=DX_M, pool=pool)
dt_p = CupyParameter("dt", dtype=np.float32, mode="const", value=DT_VAL, pool=pool)
damp_p = CupyParameter("damp", dtype=np.float32, mode="const", value=DAMP_VAL, pool=pool)
phys = Bag({"g": g_p, "dx": dx_p, "dt": dt_p, "damp": damp_p})

# drop Bag: splash site + amplitude, host-set each time a stone falls.
drop_cx_p = CupyParameter("cx", dtype=np.int32, mode="scalar", value=GRID_N // 2, pool=pool)
drop_cy_p = CupyParameter("cy", dtype=np.int32, mode="scalar", value=GRID_N // 2, pool=pool)
drop_amp_p = CupyParameter("amp", dtype=np.float32, mode="scalar", value=0.0, pool=pool)
drop = Bag({"cx": drop_cx_p, "cy": drop_cy_p, "amp": drop_amp_p})

# ---------------------------------------------------------------------------
# device helpers -> ops Bag
# ---------------------------------------------------------------------------
clamp_fn = (
    CupyHelperBuilder()
    .bind("N", n_p)
    .ingest("__device__ int clampi(int i) { return i < 0 ? 0 : (i >= N ? N - 1 : i); }")
)

face_depth_fn = (
    CupyHelperBuilder()
    .ingest(
        r"""
__device__ float face_depth(float up, float down, float vel) {
    // upwind water depth at a face: the upstream cell when flow is outward
    return vel > 0.0f ? up : down;
}
"""
    )
)

ops = Bag({"clamp": clamp_fn, "face_depth": face_depth_fn})

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
init_height_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind("REST_DEPTH", rest_depth_p)
    .ingest(
        r"""
__global__ void init_height(float* h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    h[idx] = REST_DEPTH;
}
"""
    )
    .compile()
)

apply_drop_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind("DROP_R", drop_r_p)
    .bind("drop", drop)
    .ingest(
        r"""
__global__ void apply_drop(float* h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int dxr = idx / N - $drop.cx.get(0)$;
    int dyr = idx % N - $drop.cy.get(0)$;
    if (dxr * dxr + dyr * dyr <= DROP_R * DROP_R) {
        h[idx] += $drop.amp.get(0)$;
    }
}
"""
    )
    .compile()
)

update_velocity_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind("phys", phys)
    .ingest(
        r"""
__global__ void update_velocity(float* u, float* v, const float* h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N;
    int j = idx % N;
    float gdtdx = $phys.g.get(0)$ * $phys.dt.get(0)$ / $phys.dx.get(0)$;
    float damp = $phys.damp.get(0)$;
    // u lives on the west face of cell (i,j); i==0 is the tank wall.
    if (i > 0) {
        u[idx] = (u[idx] + gdtdx * (h[idx - N] - h[idx])) * damp;
    } else {
        u[idx] = 0.0f;
    }
    // v lives on the south face of cell (i,j); j==0 is the tank wall.
    if (j > 0) {
        v[idx] = (v[idx] + gdtdx * (h[idx - 1] - h[idx])) * damp;
    } else {
        v[idx] = 0.0f;
    }
}
"""
    )
    .compile()
)

update_height_kernel = (
    CupyKernelBuilder()
    .bind("N", n_p)
    .bind("phys", phys)
    .bind("ops", ops)
    .ingest(
        r"""
__global__ void update_height(float* h_out, const float* h_in, const float* u, const float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N;
    int j = idx % N;

    // face velocities: u[i]=west face, u[i+1]=east face (0 at the wall).
    float uw = u[idx];
    float ue = 0.0f;
    if (i < N - 1) ue = u[idx + N];
    float vs = v[idx];
    float vn = 0.0f;
    if (j < N - 1) vn = v[idx + 1];

    int ip = $ops.clamp(i + 1)$;
    int im = $ops.clamp(i - 1)$;
    int jp = $ops.clamp(j + 1)$;
    int jm = $ops.clamp(j - 1)$;

    float hc = h_in[idx];
    float hxm = h_in[im * N + j];
    float hxp = h_in[ip * N + j];
    float hym = h_in[i * N + jm];
    float hyp = h_in[i * N + jp];

    float hw = $ops.face_depth(hxm, hc, uw)$;
    float he = $ops.face_depth(hc, hxp, ue)$;
    float hs = $ops.face_depth(hym, hc, vs)$;
    float hn = $ops.face_depth(hc, hyp, vn)$;

    float flux = (he * ue - hw * uw) + (hn * vn - hs * vs);
    h_out[idx] = hc - $phys.dt.get(0)$ / $phys.dx.get(0)$ * flux;
}
"""
    )
    .compile()
)

# ---------------------------------------------------------------------------
# fields (pooled flat buffers; h is ping-ponged, u/v updated in place)
# ---------------------------------------------------------------------------
h0 = pool.get_data(np.float32, (NN,))
h1 = pool.get_data(np.float32, (NN,))
u = pool.get_data(np.float32, (NN,))
v = pool.get_data(np.float32, (NN,))

u.data[...] = 0.0
v.data[...] = 0.0
init_height_kernel(h0.data, grid=GRID, block=BLOCK)

# first stone, dead center, so there is motion on frame 0
drop_cx_p.set(GRID_N // 2)
drop_cy_p.set(GRID_N // 2)
drop_amp_p.set(DROP_AMP_VAL)
apply_drop_kernel(h0.data, grid=GRID, block=BLOCK)
drop_amp_p.set(0.0)

# ---------------------------------------------------------------------------
# live view (surface elevation h - rest depth)
# ---------------------------------------------------------------------------
elev_lim = DROP_AMP_VAL * 0.35
fig, ax = plt.subplots()
im = ax.imshow(
    (h0.to_numpy().reshape(GRID_N, GRID_N) - REST_DEPTH_VAL).T,
    cmap="RdBu_r", vmin=-elev_lim, vmax=elev_lim, origin="lower",
)
fig.colorbar(im, ax=ax, label="surface elevation (m)")
ax.set_title("Shallow-water waves in a tank (Cupy backend)")
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
            apply_drop_kernel(h0.data, grid=GRID, block=BLOCK)
            drop_amp_p.set(0.0)

        t_start = time.perf_counter()
        for _ in range(STEPS_PER_FRAME):
            update_velocity_kernel(u.data, v.data, h0.data, grid=GRID, block=BLOCK)
            update_height_kernel(h1.data, h0.data, u.data, v.data, grid=GRID, block=BLOCK)
            h0, h1 = h1, h0
            sim_time += DT_VAL

        cp.cuda.Device().synchronize()  # GPU is async; sync before stopping the timer
        frame_ms = (time.perf_counter() - t_start) * 1e3
        print(f"{STEPS_PER_FRAME} steps: {frame_ms:8.1f} ms  ({frame_ms / STEPS_PER_FRAME * 1e3:6.1f} us/step)")

        time_text.set_text(f"t = {sim_time:.1f} s")
        im.set_data((h0.to_numpy().reshape(GRID_N, GRID_N) - REST_DEPTH_VAL).T)
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
