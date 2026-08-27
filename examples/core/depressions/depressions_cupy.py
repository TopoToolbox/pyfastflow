"""
Perlin terrain -> receivers -> carve depressions -> drainage area, on cupy.

The shortest path through the flow stack, on the new builder/frozen/bound
stack (pyfastflow/experimental/core/context/builder.py, frozen.py, bound.py):
noise fills z inline in the init kernel (make_noise_group composes an `at(i)`
device helper, never a field), one make_receivers pass builds the D8
receiver graph, make_depression_solver resolves every pit by carving, and
make_accumulation sums a unit source over the resolved graph so `q` is
drainage area in cells.

Every buffer the depression solver touches is allocated here: the flow
factories take no pool and allocate nothing, so scratch is the caller's
throughout. Two things differ from the Taichi/Quadrants files: every launch
carries its own grid/block, and the atomic accumulation is two kernels
rather than one, since a single `__global__` has no grid-wide barrier
between initializing q and accumulating into it.

Author: B.G (08/2026)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from pyfastflow.experimental.core.context.builder import KernelBuilder
from pyfastflow.experimental.core.context.cupy_backend import CupyParameter
from pyfastflow.experimental.core.pool.cupy_pool import CupyPool
from pyfastflow.experimental.flow import (
    make_accumulation,
    make_depression_solver,
    make_depressions,
    make_receivers,
)
from pyfastflow.experimental.grid import make_grid_group, make_grid_parameters
from pyfastflow.experimental.noise import make_noise_group, make_noise_parameters

N = 2048
DX = 50.0
n_flat = N * N
BLOCK = 256
LAUNCH = {"grid": ((n_flat + BLOCK - 1) // BLOCK,), "block": (BLOCK,)}

pool = CupyPool()
grid_group = make_grid_group("cupy", topology="D8", boundary="normal", outlet="edge")
grid_params = make_grid_parameters("cupy", pool, N, N, DX, topology="D8", outlet="edge")
noise_group = make_noise_group("cupy", kind="perlin")
noise_params = make_noise_parameters("cupy", pool, kind="perlin", amplitude=300.0, frequency=6.0, octaves=6)

# z plus every scratch buffer the carve solver and the accumulation need
z = pool.get_data(np.float32, (n_flat,))
z_prime = pool.get_data(np.float32, (n_flat,))
q = pool.get_data(np.float32, (n_flat,))
rec = pool.get_data(np.int32, (n_flat,))
rec_jump = pool.get_data(np.int32, (n_flat,))
rec_scratch = pool.get_data(np.int32, (n_flat,))
bid = pool.get_data(np.int32, (n_flat,))
basin_saddlenode = pool.get_data(np.int32, (n_flat,))
basin_saddle = pool.get_data(np.int64, (n_flat,))
outlet = pool.get_data(np.int64, (n_flat,))
is_border = pool.get_data(np.uint8, (n_flat,))
tag = pool.get_data(np.uint8, (n_flat,))
tag_alt = pool.get_data(np.uint8, (n_flat,))
rerouted = pool.get_data(np.uint8, (n_flat,))
basin_route = pool.get_data(np.int32, (n_flat,))
b_rcv = pool.get_data(np.int32, (n_flat,))

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

recv = make_receivers("cupy", grid_group, mode="steepest")
recv_bound = recv["receivers"].build()
recv_bound.bind_leaf(grid_params)
recv_bound.bind("z", z.data)
recv_bound.bind("rec", rec.data)
receivers_kernel = recv_bound.compile("cupy", **LAUNCH)

ndep = CupyParameter("NDEP", dtype=np.int32, mode="scalar", value=0, pool=pool)
deps = make_depressions("cupy", grid_group, ndep, method="vanilla", reroute="carve", n_flat=n_flat)
solver = make_depression_solver(
    "cupy", deps, grid_params, method="vanilla", reroute="carve",
    rec=rec.data, z=z.data, bid=bid.data, rec_jump=rec_jump.data, z_prime=z_prime.data,
    is_border=is_border.data, basin_saddle=basin_saddle.data, basin_saddlenode=basin_saddlenode.data,
    outlet=outlet.data, rerouted=rerouted.data, tag=tag.data, tag_alt=tag_alt.data,
    rec_scratch=rec_scratch.data, basin_route=basin_route.data, b_rcv=b_rcv.data, n_flat=n_flat, block_size=BLOCK,
)

source = CupyParameter("SRC", dtype=np.float32, mode="const", value=1.0, pool=pool)
accumulation = make_accumulation("cupy", grid_group, method="atomic", n_flat=n_flat)
q_init_bound = accumulation["q_init"].build()
q_init_bound.bind("SOURCE", source)
q_init_bound.bind("q", q.data)
q_init = q_init_bound.compile("cupy", **LAUNCH)
accum_bound = accumulation["accum"].build()
accum_bound.bind("SOURCE", source)
accum_bound.bind("rec", rec.data)
accum_bound.bind("q", q.data)
accum = accum_bound.compile("cupy", **LAUNCH)

init_kernel()
receivers_kernel()
solver()
q_init()
accum()

print(f"depressions left: {ndep.read()}, passes taken: {solver.last_trip_counts}")

zz = z.data.get().reshape(N, N)
qq = q.data.get().reshape(N, N)

ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(zz, vert_exag=2.0, dx=DX, dy=DX)

fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
axes[0].imshow(hs, cmap="gray")
im0 = axes[0].imshow(zz, cmap="terrain", alpha=0.6)
axes[0].set_title("terrain (m)")
fig.colorbar(im0, ax=axes[0], shrink=0.8)

axes[1].imshow(hs, cmap="gray")
im1 = axes[1].imshow(np.log10(qq), cmap="Blues", alpha=0.7)
axes[1].set_title("log10 drainage area (cells)")
fig.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
