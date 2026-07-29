"""
One kernel template, four grids: make_grid's boundary/nodata knobs made
visible, Quadrants backend. Same model as ball_walk_taichi.py; see that
file's docstring for the full explanation of the mechanism. This one differs
only in which backend module TaichiKernelBuilder's counterpart wraps -
QuadrantsKernelBuilder compiles to qd.func/qd.kernel instead of ti.func/
ti.kernel, through the same closure-splicing machinery
(context/_closure_backend.py).

Two device templates are written ONCE, below, as plain python defs - `walk`
(moves a point one hop) and `spread` (one Jacobi sweep of a geodesic distance
field). Each is ingested by FOUR separate QuadrantsKernelBuilders, one per
grid config, differing only in which grid Bag gets bound under the name
`grid`. Nothing in either template body changes; make_grid's own block
substitution (see grid/_closure_blocks.py, shared verbatim with Taichi) is
what makes `grid.neighbour(i, k)` and `grid.neighbour_and_distance(i, k)`
mean something different per panel.

The four grids, one per panel:
  1. boundary="normal",       nodata=False               - plain bounded grid
  2. boundary="periodic_EW",  nodata=False               - wraps east/west
  3. boundary="normal",       nodata=True + island        - an impassable disc
  4. boundary="periodic_EW",  nodata=True + island        - both at once

The "ball" is a geodesic distance field, computed by Jacobi relaxation:
`spread(d_out, d_in)` sets each node's distance to
`min(d_in[i], min_k(d_in[neighbour(i,k)] + dist_from_k(k)))`, walking the
`neighbour_and_distance` helper's -1 sentinel to skip missing/blocked
neighbours. Reseed the field to 0 at one node and +inf elsewhere, run K
sweeps, and `d < R` is a disc that has flowed outward through the grid's own
notion of adjacency - wrapping across a periodic edge, or bending around a
nodata island, without the template knowing either is happening.

The disc's centre does a random walk, also entirely on-device: `walk` xorshifts
a u32 SEED Parameter, turns the top byte into a direction k, and moves CENTRE
to `grid.neighbour(centre, k)` only if that is not -1. Since `neighbour()`
already folds in the edge gate and the nodata gate (see _valid_nodata_tmpl in
_closure_blocks.py), a lone `n >= 0` check is sufficient: normal boundaries
block the walk at the domain edge, periodic ones wrap it, and the nodata
island is simply never a valid target. CENTRE and SEED are scalar Parameters,
so the same kernel that mutates them on-device leaves the new value sitting in
their pooled storage for the host to read back (`.get().to_numpy()`) for the
panel title, no extra plumbing required.

All four panels start from the same CENTRE but each gets its own SEED, so the
four balls are independent random walks rather than one walk replayed four
times. What the panel compares is how each configuration *confines* a ball -
edges that block, edges that wrap, an island that is never a valid target -
not four copies of one trajectory drifting apart and re-converging.

Author: B.G (07/2026)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import quadrants as qd

from pyfastflow.experimental.core.context.quadrants_backend import (
    QuadrantsKernelBuilder,
    QuadrantsParameter,
)
from pyfastflow.experimental.grid import make_grid
from pyfastflow.experimental.core.pool.quadrants_pool import QuadrantsPool

qd.init(arch=qd.gpu)

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

START_ROW, START_COL = 50, 50
START_IDX = START_ROW * NX + START_COL
SEED_VALUE = 20260728
SEED_STRIDE = 0x9E3779B9  # per-panel seed offset, so each ball walks on its own

ISLAND_ROW, ISLAND_COL, ISLAND_R = NY // 2, NX // 2, 40

pool = QuadrantsPool()

# ---------------------------------------------------------------------------
# device templates - written once, ingested by four builders below
# ---------------------------------------------------------------------------


def walk_template():
    # one hop of a random walk, entirely on-device. State (SEED, CENTRE) is
    # bound per-panel as scalar Parameters; the body never changes.
    for _dummy in range(1):
        s = SEED.get(0)
        s = s ^ (s << 13)
        s = s ^ (s >> 17)
        s = s ^ (s << 5)
        SEED.set_node(0, s)
        k = (s >> 24) % grid.n_neighbours.get(0)
        c = CENTRE.get(0)
        n = grid.neighbour(c, k)
        if n >= 0:
            CENTRE.set_node(0, n)


def spread_template(d_out: qd.template(), d_in: qd.template()):
    # one Jacobi sweep of the geodesic distance field.
    for i in d_in:
        best = d_in[i]
        for k in range(grid.n_neighbours.get(0)):
            j, w = grid.neighbour_and_distance(i, k)
            if j != -1:
                cand = d_in[j] + w
                if cand < best:
                    best = cand
        d_out[i] = best


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
        "quadrants", pool, NX, NY, DX, topology="D8", boundary=boundary, nodata=nodata
    )
    if nodata:
        grid_bag.nodata_mask.set(island_mask_flat)

    centre_p = QuadrantsParameter("CENTRE", dtype=qd.i32, mode="scalar", value=START_IDX, pool=pool)
    panel_seed = (SEED_VALUE + panel_idx * SEED_STRIDE) & 0x7FFFFFFF
    seed_p = QuadrantsParameter("SEED", dtype=qd.u32, mode="scalar", value=panel_seed, pool=pool)

    walk_kernel = (
        QuadrantsKernelBuilder()
        .bind("SEED", seed_p)
        .bind("CENTRE", centre_p)
        .bind("grid", grid_bag)
        .ingest(walk_template)
        .compile()
    )
    spread_kernel = QuadrantsKernelBuilder().bind("grid", grid_bag).ingest(spread_template).compile()

    d0 = pool.get_data(qd.f32, (NN,))
    d1 = pool.get_data(qd.f32, (NN,))

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
fig.suptitle("Ball walk (Quadrants) - one kernel template, four grids")
fig.tight_layout()
fig.show()

frame = 0
try:
    while True:
        t_start = time.perf_counter()
        for panel in panels:
            for _ in range(WALK_STEPS_PER_FRAME):
                panel["walk_kernel"]()

            c = int(panel["centre_p"].get().to_numpy())
            seed_arr = np.full(NN, INF, dtype=np.float32)
            seed_arr[c] = 0.0
            panel["d0"].from_numpy(seed_arr)

            d0, d1 = panel["d0"], panel["d1"]
            for _ in range(K_SWEEPS):
                panel["spread_kernel"](d1.data, d0.data)
                d0, d1 = d1, d0
            panel["d0"], panel["d1"] = d0, d1

            dd = d0.to_numpy().reshape(NY, NX)
            disp = (dd < R_BALL).astype(np.float32)
            if panel["nodata"]:
                disp[_island] = np.nan
            panel["im"].set_data(disp)
            panel["ax"].set_title(f"{panel['title']}\ncentre row={c // NX} col={c % NX}", fontsize=9)

        qd.sync()
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
