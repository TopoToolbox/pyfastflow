"""
Perlin terrain -> fill by grayscale morphological reconstruction ->
"reconstruct_epsilon" (a parent-hop-distance-scaled epsilon added on top of
`filled`) -> persistent-kernel MFD accumulation on that surface, on cupy.

Mirrors fill_reconstruct_cupy.py, swapping its single-flow-direction
`make_accumulation(method="atomic")` step for multiple-flow-direction
accumulation over the SAME fill_reconstruct result - see
pyfastflow/graphflood/__init__.py's module docstring,
"kind='vanilla_mfd'" section, for the full algorithm this example runs by
hand (make_graphflood's own kind="vanilla_mfd" wraps exactly this pipeline
plus the graphflood-specific friction/divergence steps this example has no
use for, computing drainage area rather than water depth).

Why "reconstruct_epsilon" rather than plain `filled`: MFD topology
(pyfastflow/graphflood/_cupy_mfd_topology.py) derives
`dirs`/`mfd_w` from `slope(filled[i], filled[j]) > 0` between neighbours -
exactly 0 for every pair inside a resolved depression's flat lake bottom,
which gives every cell in there zero outgoing MFD edges and stalls
accumulation at the flat's boundary. SFD accumulation (fill_reconstruct_
cupy.py's own `accum_bound.bind("rec", parent.data)`) never has this
problem because it walks `parent` directly rather than re-deriving edges
from `filled`'s elevation values. "reconstruct_epsilon"
(pyfastflow/graphflood/_cupy_reconstruct_epsilon.py) fixes
this without touching _cupy_mfd_topology.py's own slope-based logic at
all: `filled_eps[i] = filled[i] + MFD_EPSILON * hops[i]`, where `hops[i]`
is i's distance to the outlet along `parent` (pointer-jumping, double-
buffered - see build_hops_jump's own docstring for why an earlier,
in-place version of this raced and gave wrong distances). Real slopes are
unaffected; a flat gets a small but strictly monotonic, acyclic synthetic
gradient along the direction `parent` already established.

Every buffer every step touches is allocated here - none of these
factories take a pool or allocate anything themselves.

Author: B.G (08/2026)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from pyfastflow.core.context.builder import KernelBuilder
from pyfastflow.core.context.cupy_backend import CupyParameter
from pyfastflow.core.pool.cupy_pool import CupyPool
from pyfastflow.flow import make_fill_reconstruct, make_fill_reconstruct_solver
from pyfastflow.flow._cupy_mfd_accum import build_persistent_mfd, init_frontier_mfd, persistent_grid_block
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.graphflood._cupy_mfd_topology import build_mfd_topology
from pyfastflow.graphflood._cupy_reconstruct_epsilon import build_apply_epsilon, build_hops_init, build_hops_jump
from pyfastflow.noise import make_noise_group, make_noise_parameters

N = 2048
DX = 50.0
n_flat = N * N
N_NEIGHBOURS = 8  # D8
BLOCK = 256
LAUNCH = {"grid": ((n_flat + BLOCK - 1) // BLOCK,), "block": (BLOCK,)}
MAX_PASSES = 4 * N

pool = CupyPool()
grid_group = make_grid_group("cupy", topology="D8", boundary="normal", outlet="edge")
grid_params = make_grid_parameters("cupy", pool, N, N, DX, topology="D8", outlet="edge")
noise_group = make_noise_group("cupy", kind="perlin")
noise_params = make_noise_parameters("cupy", pool, kind="perlin", amplitude=0.001, frequency=6.0, octaves=6)

z = pool.get_data(np.float32, (n_flat,))
filled = pool.get_data(np.float32, (n_flat,))
parent = pool.get_data(np.int32, (n_flat,))
frontier = pool.get_data(np.int32, (2 * n_flat,))
counters = pool.get_data(np.int32, (MAX_PASSES + 2,))
queued_gen = pool.get_data(np.int32, (n_flat,))

dist = pool.get_data(np.float32, (n_flat,))
anc = pool.get_data(np.int32, (n_flat,))
dist2 = pool.get_data(np.float32, (n_flat,))
anc2 = pool.get_data(np.int32, (n_flat,))
filled_eps = pool.get_data(np.float32, (n_flat,))

dirs = pool.get_data(np.uint8, (n_flat,))
mfd_w = pool.get_data(np.float32, (n_flat * N_NEIGHBOURS,))
indegree = pool.get_data(np.int32, (n_flat,))
frontier0 = pool.get_data(np.int32, (n_flat,))
frontier1 = pool.get_data(np.int32, (n_flat,))
count = pool.get_data(np.int32, (2,))
barrier = pool.get_data(np.uint32, (1,))
q = pool.get_data(np.float32, (n_flat,))

init_bound = (
    KernelBuilder().compose("noise", noise_group).wire_data("z").ingest(
        f"""
extern "C" __global__ void init_z(float* z) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    z[i] = $ctx.noise.at(i)$;
}}
"""
    ).build()
)
init_bound.bind_leaf(grid_params, prefix=("noise",))
init_bound.bind_leaf(noise_params, prefix=("noise",))
init_bound.bind("z", z.data)
init_kernel = init_bound.compile("cupy", **LAUNCH)

# --- fill by reconstruction --------------------------------------------------
pass_p = CupyParameter("PASS", dtype=np.int32, mode="scalar", value=0, pool=pool)
active_p = CupyParameter("ACTIVE", dtype=np.int32, mode="scalar", value=0, pool=pool)
deps = make_fill_reconstruct("cupy", grid_group, nx=N, ny=N)
solver = make_fill_reconstruct_solver(
    "cupy", deps, grid_params,
    z=z.data, filled=filled.data, parent=parent.data, frontier=frontier.data,
    counters=counters.data, queued_gen=queued_gen.data, pass_p=pass_p, active_p=active_p,
    n_flat=n_flat, nx=N, ny=N, block_size=BLOCK, max_passes=MAX_PASSES,
)

# --- reconstruct_epsilon: hops-to-outlet along parent, then filled_eps -----
hops_init_bound = build_hops_init(n_flat=n_flat).build()
hops_init_bound.bind("parent", parent.data)
hops_init_bound.bind("filled", filled.data)
hops_init_bound.bind("dist", dist.data)
hops_init_bound.bind("anc", anc.data)
hops_init = hops_init_bound.compile("cupy", **LAUNCH)

hops_jump_fk = build_hops_jump(n_flat=n_flat)
hops_jump_fwd_bound = hops_jump_fk.build()
hops_jump_fwd_bound.bind("dist_in", dist.data)
hops_jump_fwd_bound.bind("anc_in", anc.data)
hops_jump_fwd_bound.bind("dist_out", dist2.data)
hops_jump_fwd_bound.bind("anc_out", anc2.data)
hops_jump_fwd = hops_jump_fwd_bound.compile("cupy", **LAUNCH)

hops_jump_bwd_bound = hops_jump_fk.build()
hops_jump_bwd_bound.bind("dist_in", dist2.data)
hops_jump_bwd_bound.bind("anc_in", anc2.data)
hops_jump_bwd_bound.bind("dist_out", dist.data)
hops_jump_bwd_bound.bind("anc_out", anc.data)
hops_jump_bwd = hops_jump_bwd_bound.compile("cupy", **LAUNCH)

# rounded up to even so alternating fwd/bwd always ends back in dist/anc
HOPS_ROUNDS = int(np.ceil(np.log2(max(2, n_flat)))) + 1
if HOPS_ROUNDS % 2 != 0:
    HOPS_ROUNDS += 1

apply_epsilon_bound = build_apply_epsilon(n_flat=n_flat).build()
apply_epsilon_bound.bind("filled", filled.data)
apply_epsilon_bound.bind("dist", dist.data)
apply_epsilon_bound.bind("filled_eps", filled_eps.data)
apply_epsilon = apply_epsilon_bound.compile("cupy", **LAUNCH)

# --- MFD topology on filled_eps, then persistent-kernel accumulation -------
topo = build_mfd_topology(
    grid=grid_group, n_flat=n_flat, topology="D8", diagonal_partition_correction=True,
)
dirs_weights_bound = topo["dirs_weights"].build()
dirs_weights_bound.bind("filled", filled_eps.data)
dirs_weights_bound.bind("dirs", dirs.data)
dirs_weights_bound.bind("mfd_w", mfd_w.data)
dirs_weights_bound.bind_leaf(grid_params)
dirs_weights = dirs_weights_bound.compile("cupy", **LAUNCH)

indegree_reset_bound = topo["indegree_reset"].build()
indegree_reset_bound.bind("indegree", indegree.data)
indegree_reset = indegree_reset_bound.compile("cupy", **LAUNCH)

indegree_count_bound = topo["indegree_count"].build()
indegree_count_bound.bind("dirs", dirs.data)
indegree_count_bound.bind("indegree", indegree.data)
indegree_count_bound.bind_leaf(grid_params)
indegree_count = indegree_count_bound.compile("cupy", **LAUNCH)

source = CupyParameter("SOURCE", dtype=np.float32, mode="const", value=1.0, pool=pool)
persistent = build_persistent_mfd(grid=grid_group, n_flat=n_flat, n_neighbours=N_NEIGHBOURS)
q_init_bound = persistent["q_init"].build()
q_init_bound.bind("SOURCE", source)
q_init_bound.bind("accum", q.data)
q_init = q_init_bound.compile("cupy", **LAUNCH)

accum_bound = persistent["accum"].build()
accum_bound.bind("frontier0", frontier0.data)
accum_bound.bind("frontier1", frontier1.data)
accum_bound.bind("count", count.data)
accum_bound.bind("barrier", barrier.data)
accum_bound.bind("dirs", dirs.data)
accum_bound.bind("mfd_w", mfd_w.data)
accum_bound.bind("accum", q.data)
accum_bound.bind("indegree", indegree.data)
accum_bound.bind_leaf(grid_params)
persistent_grid, persistent_block = persistent_grid_block()
accum = accum_bound.compile("cupy", grid=persistent_grid, block=persistent_block)

# --- run ---------------------------------------------------------------------
init_kernel()
counters.data.fill(0)
queued_gen.data.fill(-1)
solver()

hops_init()
for _ in range(HOPS_ROUNDS // 2):
    hops_jump_fwd()
    hops_jump_bwd()
apply_epsilon()

indegree_reset()
dirs_weights()
indegree_count()

q_init()
n0 = init_frontier_mfd(indegree.data, frontier0.data)
count.data[0:1] = n0
count.data[1:2] = 0
barrier.data[0:1] = 0
accum()

print(f"reconstruction fill: passes taken = {solver.last_trip_counts}")
print(f"MFD: n0 (ready cells) = {n0}, indegree stuck > 0 after run = {int((indegree.data.get() > 0).sum())}")

zz = z.data.get().reshape(N, N)
zf = filled.data.get().reshape(N, N)
qq = q.data.get().reshape(N, N)
print(f"cells raised: {int(np.count_nonzero(zf > zz))}/{n_flat}, max raise = {float((zf - zz).max()):.4f} m")

ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(zf, vert_exag=2.0, dx=DX, dy=DX)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
axes[0].imshow(hs, cmap="gray")
im0 = axes[0].imshow(zf, cmap="terrain", alpha=0.6)
axes[0].set_title("filled DEM (m), reconstruction")
fig.colorbar(im0, ax=axes[0], shrink=0.8)

axes[1].imshow(hs, cmap="gray")
im1 = axes[1].imshow(np.log10(qq), cmap="Blues", alpha=0.7)
axes[1].set_title("log10 MFD drainage area (cells)")
fig.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
