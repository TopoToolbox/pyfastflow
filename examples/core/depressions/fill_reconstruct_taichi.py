"""
Perlin terrain -> fill by grayscale morphological reconstruction ->
drainage area, on Taichi.

The reconstruction alternative to depressions_taichi.py's carve/label/saddle
loop, on the new builder/frozen/bound stack (pyfastflow/experimental/core/
context/builder.py, frozen.py, bound.py): make_fill_reconstruct/
make_fill_reconstruct_solver converge `filled`/`parent` (the receiver graph)
directly to a fixed point - no basin ids, no saddle search, no outlet
routing. See pyfastflow/experimental/flow/__init__.py's module docstring and
experimental/LM/fill_reconstruct_optimised.py for the algorithm.

Every buffer the solver touches is allocated here: the factory takes no pool
and allocates nothing, so scratch is the caller's throughout - including the
two buffers this algorithm needs that make_depressions' solver does not:
`frontier` (2*n_flat, the ping-ponged active-cell list - see
make_fill_reconstruct's module note for why it is one combined buffer here)
and `counters` (per-pass frontier sizes), plus `queued_gen`, which - like
`counters` - needs a one-time init before the first call (`-1` and `0`
respectively) since the solver never resets them itself. `active_p` is the
solver's early-stop scalar Parameter (make_fill_reconstruct_solver's own
docstring) - no caller-side init needed, the solver zeroes it every pass.

Author: B.G (08/2026)
"""

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
from matplotlib.colors import LightSource

from pyfastflow.experimental.core.context.builder import KernelBuilder
from pyfastflow.experimental.core.context.taichi_backend import TaichiParameter
from pyfastflow.experimental.core.pool.taichi_pool import TaichiPool
from pyfastflow.experimental.flow import make_accumulation, make_fill_reconstruct, make_fill_reconstruct_solver
from pyfastflow.experimental.grid import make_grid_group, make_grid_parameters
from pyfastflow.experimental.noise import make_noise_group, make_noise_parameters

ti.init(arch=ti.gpu)

N = 512
DX = 50.0
n_flat = N * N
MAX_PASSES = 4 * N

pool = TaichiPool()
grid_group = make_grid_group("taichi", topology="D8", boundary="normal", outlet="edge")
grid_params = make_grid_parameters("taichi", pool, N, N, DX, topology="D8", outlet="edge")
noise_group = make_noise_group("taichi", kind="perlin")
noise_params = make_noise_parameters("taichi", pool, kind="perlin", amplitude=300.0, frequency=6.0, octaves=6)

z = pool.get_data(ti.f32, (n_flat,))
filled = pool.get_data(ti.f32, (n_flat,))
parent = pool.get_data(ti.i32, (n_flat,))
frontier = pool.get_data(ti.i32, (2 * n_flat,))
counters = pool.get_data(ti.i32, (MAX_PASSES + 2,))
queued_gen = pool.get_data(ti.i32, (n_flat,))
q = pool.get_data(ti.f32, (n_flat,))


def init_z_tmpl(ctx, z: ti.template()):
    for i in z:
        z[i] = ctx.noise.at(i)


init_bound = KernelBuilder().compose("noise", noise_group).wire_data("z").ingest(init_z_tmpl).build()
init_bound.bind_leaf(grid_params, prefix=("noise",))
init_bound.bind_leaf(noise_params, prefix=("noise",))
init_bound.bind("z", z.data)
init_kernel = init_bound.compile("taichi")

pass_p = TaichiParameter("PASS", dtype=ti.i32, mode="scalar", value=0, pool=pool)
active_p = TaichiParameter("ACTIVE", dtype=ti.i32, mode="scalar", value=0, pool=pool)
deps = make_fill_reconstruct("taichi", grid_group, nx=N, ny=N)
solver = make_fill_reconstruct_solver(
    "taichi", deps, grid_params,
    z=z.data, filled=filled.data, parent=parent.data, frontier=frontier.data,
    counters=counters.data, queued_gen=queued_gen.data, pass_p=pass_p, active_p=active_p,
    n_flat=n_flat, nx=N, ny=N, max_passes=MAX_PASSES,
)

source = TaichiParameter("SRC", dtype=ti.f32, mode="const", value=1.0, pool=pool)
accum_bound = make_accumulation("taichi", grid_group, method="atomic", n_flat=n_flat)["accum"].build()
accum_bound.bind("SOURCE", source)
accum_bound.bind("rec", parent.data)
accum_bound.bind("q", q.data)
accum_kernel = accum_bound.compile("taichi")

init_kernel()
counters.data.fill(0)
queued_gen.data.fill(-1)
solver()
accum_kernel()

print(f"reconstruction fill: passes taken = {solver.last_trip_counts}")

zz = z.data.to_numpy().reshape(N, N)
zf = filled.data.to_numpy().reshape(N, N)
qq = q.data.to_numpy().reshape(N, N)
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
axes[1].set_title("log10 drainage area (cells)")
fig.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
