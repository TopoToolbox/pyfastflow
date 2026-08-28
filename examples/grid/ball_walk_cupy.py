"""
One kernel template, four grids: make_grid's boundary/nodata knobs made
visible, cupy backend. Same model as ball_walk_taichi.py; see that file's
docstring for the full explanation of the mechanism. This one differs only in
template syntax (CUDA source text with `$...$` spans, per cupy_backend.py)
and buffer shape (flat, since a CUDA thread indexes its own data).

Two device templates are written ONCE, below, as CUDA source strings - `walk`
(moves a point one hop) and `spread` (one Jacobi sweep of a geodesic distance
field). Each is ingested by FOUR separate CupyKernelBuilders, one per grid
config, differing only in which grid Bag gets bound under the name `grid`; the
source text itself never changes. Every scalar/field Parameter a span reaches
- CENTRE, SEED, and whatever grid.neighbour_and_distance needs internally -
lands in that compile's own module-scope constant block (see cupy_backend.py),
so the same two device functions, compiled four times against four grids, read
four independent sets of pointers.

The four grids, one per panel:
  1. boundary="normal",       nodata=False               - plain bounded grid
  2. boundary="periodic_EW",  nodata=False               - wraps east/west
  3. boundary="normal",       nodata=True + island        - an impassable disc
  4. boundary="periodic_EW",  nodata=True + island        - both at once

`walk` xorshifts a SEED Parameter (stored as int32, reinterpreted as unsigned
inside the kernel body - cupy's Parameter has no unsigned dtype mapping, see
the module's own `_CTYPE` table, so the cast happens in the template instead
of the storage), turns the top byte into a direction k, and moves CENTRE to
whatever `grid.neighbour_and_distance` reports, only if that neighbour index
is not -1. `spread` does one Jacobi relaxation sweep of the geodesic distance
field: `d_out[i] = min(d_in[i], min_k(d_in[j] + w))` for every (j, w) pair
`neighbour_and_distance` returns through its out-pointers, skipping j == -1.
Reseed the field to 0 at CENTRE and +inf elsewhere every frame, run K sweeps,
and `d < R` is the disc shown per panel - wrapping across a periodic edge or
bending around the nodata island purely because the grid's own neighbour
lookup says so, never because the kernel text does.

All four panels start from the same CENTRE but each gets its own SEED, so the
four balls are independent random walks rather than one walk replayed four
times. What the panel compares is how each configuration *confines* a ball -
edges that block, edges that wrap, an island that is never a valid target -
not four copies of one trajectory drifting apart and re-converging.

Author: B.G (07/2026)
"""

import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from pyfastflow.core.context.cupy_backend import (
    CupyKernelBuilder,
    CupyParameter,
)
from pyfastflow.grid import make_grid
from pyfastflow.core.pool.cupy_pool import CupyPool

# ---------------------------------------------------------------------------
# host-side constants
# ---------------------------------------------------------------------------
NX, NY = 256, 256
NN = NX * NY
DX = 1.0

INF = 1.0e6  # stand-in for +inf in the distance field (avoids float overflow)
R_BALL = 20.0  # display threshold: d < R_BALL is "inside the ball"
K_SWEEPS = 50  # Jacobi sweeps per frame - enough to converge for R_BALL=20
WALK_STEPS_PER_FRAME = 5

BLOCK = 256
GRID_DIM = (NN + BLOCK - 1) // BLOCK

START_ROW, START_COL = 50, 50
START_IDX = START_ROW * NX + START_COL
SEED_VALUE = 20260728
SEED_STRIDE = 0x9E3779B9  # per-panel seed offset, so each ball walks on its own

ISLAND_ROW, ISLAND_COL, ISLAND_R = NY // 2, NX // 2, 40

pool = CupyPool()

# ---------------------------------------------------------------------------
# device templates - written once, ingested by four builders below
# ---------------------------------------------------------------------------
WALK_SRC = r"""
extern "C" __global__ void walk(void) {
    // one hop of a random walk, entirely on-device. State (SEED, CENTRE) is
    // bound per-panel as scalar Parameters; this source never changes.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;
    unsigned int s = (unsigned int)$SEED.get(0)$;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    $SEED.set_node(0, (int)s)$;
    int k = (s >> 24) % $grid.n_neighbours.get(0)$;
    int c = $CENTRE.get(0)$;
    int n;
    float w;
    $grid.neighbour_and_distance(c, k, &n, &w)$;
    if (n >= 0) {
        $CENTRE.set_node(0, n)$;
    }
}
"""

SPREAD_SRC = r"""
extern "C" __global__ void spread(float* d_out, const float* d_in) {
    // one Jacobi sweep of the geodesic distance field.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $grid.nx.get(0)$ * $grid.ny.get(0)$) return;
    float best = d_in[i];
    for (int k = 0; k < $grid.n_neighbours.get(0)$; k++) {
        int j;
        float w;
        $grid.neighbour_and_distance(i, k, &j, &w)$;
        if (j != -1) {
            float cand = d_in[j] + w;
            if (cand < best) best = cand;
        }
    }
    d_out[i] = best;
}
"""

# ---------------------------------------------------------------------------
# nodata island mask (shared shape, independently allocated per grid)
# ---------------------------------------------------------------------------
_rr, _cc = np.mgrid[0:NY, 0:NX]
_island = ((_rr - ISLAND_ROW) ** 2 + (_cc - ISLAND_COL) ** 2) < ISLAND_R**2
island_mask_flat = _island.astype(np.uint8).ravel()

# ---------------------------------------------------------------------------
# build the four panels - same two templates, four different grid bindings
# ---------------------------------------------------------------------------
PANEL_CONFIGS = [
    ("normal, no nodata", "normal", False),
    ("periodic_EW, no nodata", "periodic_EW", False),
    ("normal, nodata island", "normal", True),
    ("periodic_EW, nodata island", "periodic_EW", True),
]

panels = []
for panel_idx, (title, boundary, nodata) in enumerate(PANEL_CONFIGS):
    grid_bag = make_grid(
        "cupy", pool, NX, NY, DX, topology="D8", boundary=boundary, nodata=nodata
    )
    if nodata:
        grid_bag.nodata_mask.set(island_mask_flat)

    centre_p = CupyParameter("CENTRE", dtype=np.int32, mode="scalar", value=START_IDX, pool=pool)
    panel_seed = (SEED_VALUE + panel_idx * SEED_STRIDE) & 0x7FFFFFFF
    seed_p = CupyParameter("SEED", dtype=np.int32, mode="scalar", value=panel_seed, pool=pool)

    walk_kernel = (
        CupyKernelBuilder()
        .bind("SEED", seed_p)
        .bind("CENTRE", centre_p)
        .bind("grid", grid_bag)
        .ingest(WALK_SRC)
        .compile()
    )
    spread_kernel = CupyKernelBuilder().bind("grid", grid_bag).ingest(SPREAD_SRC).compile()

    d0 = pool.get_data(np.float32, (NN,))
    d1 = pool.get_data(np.float32, (NN,))

    panels.append(
        dict(
            title=title,
            nodata=nodata,
            grid=grid_bag,
            centre_p=centre_p,
            seed_p=seed_p,
            walk_kernel=walk_kernel,
            spread_kernel=spread_kernel,
            d0=d0,
            d1=d1,
        )
    )

# ---------------------------------------------------------------------------
# live view
# ---------------------------------------------------------------------------
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("dimgray")

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
for panel, ax in zip(panels, axes.ravel()):
    im = ax.imshow(np.zeros((NY, NX)), cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    panel["im"] = im
    panel["ax"] = ax
fig.suptitle("Ball walk (Cupy) - one kernel template, four grids")
fig.tight_layout()
fig.show()

frame = 0
try:
    while True:
        t_start = time.perf_counter()
        for panel in panels:
            for _ in range(WALK_STEPS_PER_FRAME):
                panel["walk_kernel"](grid=1, block=1)

            c = int(panel["centre_p"].get().to_numpy())
            seed_arr = np.full(NN, INF, dtype=np.float32)
            seed_arr[c] = 0.0
            panel["d0"].from_numpy(seed_arr)

            d0, d1 = panel["d0"], panel["d1"]
            for _ in range(K_SWEEPS):
                panel["spread_kernel"](d1.data, d0.data, grid=GRID_DIM, block=BLOCK)
                d0, d1 = d1, d0
            panel["d0"], panel["d1"] = d0, d1

            dd = d0.to_numpy().reshape(NY, NX)
            disp = (dd < R_BALL).astype(np.float32)
            if panel["nodata"]:
                disp[_island] = np.nan
            panel["im"].set_data(disp)
            panel["ax"].set_title(f"{panel['title']}\ncentre row={c // NX} col={c % NX}", fontsize=9)

        cp.cuda.Device().synchronize()  # GPU is async; sync before stopping the timer
        frame += 1
        frame_ms = (time.perf_counter() - t_start) * 1e3
        if frame % 20 == 0:
            print(f"frame {frame}: {frame_ms:6.1f} ms")

        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.05)
except KeyboardInterrupt:
    pass

# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------
for panel in panels:
    panel["centre_p"].destroy()
    panel["seed_p"].destroy()
    panel["grid"].nx.destroy()
    panel["grid"].ny.destroy()
    panel["grid"].dx.destroy()
    panel["grid"].n_neighbours.destroy()
    if panel["nodata"]:
        panel["grid"].nodata_mask.destroy()
    pool.release_data(panel["d0"])
    pool.release_data(panel["d1"])
