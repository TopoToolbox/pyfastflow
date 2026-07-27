"""
Hillslope landscape evolution as one Routine, exercising every part of the
core in a single model. Quadrants backend; same model as lem_routine_taichi.py.

The physics is deliberately small - linear hillslope diffusion against a
spatially variable uplift field, with the domain edges pinned to base level:

    dz/dt = D * laplacian(z) + U(x, y)

What the file is here to show is how the pieces fit together when a model
needs all of them at once. The other examples each isolate one thing; this
one carries the lot:

  Parameter modes   N and DT are solo consts, read bare as compile-time
                    literals. DX is a non-solo const, read as grid.dx.get(0).
                    D and SEA_LEVEL are scalars the host retunes between
                    frames. UPLIFT is a field, one rate per node.
  Helpers           clampi binds a const; laplacian binds a bag and calls
                    clampi, so a helper reaches another helper; uplift_at
                    binds the UPLIFT *field* directly, which is what lets the
                    uplift kernel body stay a one-liner.
  Bags              grid is nested (grid.n, grid.dx), hill is mixed - a
                    scalar Parameter, a helper and a const under one name -
                    and the two noise seeds arrive flat through bind_bag.
  Routine           three kernels, two of them inside the routine, with the
                    z0/z1 ping-pong unrolled twice so the swaps compose to
                    the identity and the routine can be called repeatedly.

The step the routine runs is diffuse then uplift-and-clamp, so uplift is
applied to what diffusion just wrote. Two of those, plus the two swaps, make
one routine call - and the result always lands back in z0.

D and SEA_LEVEL are retuned between routine calls, never between the steps
inside one: set() on a scalar Parameter is what a routine expects between
calls (see routine.py, "Contract: no set()/destroy() mid-routine"), and there
is no python between a routine's own steps to run it in anyway.

Author: B.G (07/2026)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import quadrants as qd

from pyfastflow.experimental.core.context.base import Bag, merge
from pyfastflow.experimental.core.context.quadrants_backend import (
    QuadrantsHelperBuilder,
    QuadrantsKernelBuilder,
    QuadrantsParameter,
    QuadrantsRoutineBuilder,
)
from pyfastflow.experimental.core.pool.quadrants_pool import QuadrantsPool

qd.init(arch=qd.gpu)

# ---------------------------------------------------------------------------
# host-side constants (grid size, timing - never used as kernel globals)
# ---------------------------------------------------------------------------
GRID_N = 2048
DX_M = 100.0
STEPS_PER_FRAME = 200  # two routine substeps per call - see the loop below

D_VAL = 1.0e-2  # hillslope diffusivity, m2/yr
UPLIFT_MAX = 1.0e-6  # m/yr at the range crest
CFL_SAFETY = 0.2
DT_VAL = CFL_SAFETY * DX_M**2 / (4.0 * D_VAL)

pool = QuadrantsPool()

# ---------------------------------------------------------------------------
# parameters - one of every mode
# ---------------------------------------------------------------------------
# solo consts: folded into the generated code as bare literals, read as `N`
n_p = QuadrantsParameter("N", dtype=qd.i32, mode="const", value=GRID_N, pool=pool, solo=True)
dt_p = QuadrantsParameter("DT", dtype=qd.f32, mode="const", value=DT_VAL, pool=pool, solo=True)
seed_a_p = QuadrantsParameter("SEED_A", dtype=qd.f32, mode="const", value=12.9898, pool=pool, solo=True)
seed_b_p = QuadrantsParameter("SEED_B", dtype=qd.f32, mode="const", value=78.233, pool=pool, solo=True)

# non-solo const: still fixed at compile time, but read through .get(0) like
# any other mode, so a template can be written without knowing it is const
dx_p = QuadrantsParameter("DX", dtype=qd.f32, mode="const", value=DX_M, pool=pool)

# scalars: one cell each, retuned from the host between routine calls
d_p = QuadrantsParameter("D", dtype=qd.f32, mode="scalar", value=D_VAL, pool=pool)
sea_p = QuadrantsParameter("SEA_LEVEL", dtype=qd.f32, mode="scalar", value=0.0, pool=pool)

# field: one value per node, filled from the host below
uplift_p = QuadrantsParameter(
    "UPLIFT", dtype=qd.f32, mode="field", value=np.zeros(GRID_N * GRID_N), pool=pool, n_flat=GRID_N * GRID_N
)

# a north-south uplift ridge, tapering to zero at the north and south edges
_yy = np.arange(GRID_N, dtype=np.float32)[:, None] * np.ones((1, GRID_N), np.float32)
_ridge = np.sin(np.pi * _yy / (GRID_N - 1)) ** 2
uplift_p.set((UPLIFT_MAX * _ridge).ravel())

# ---------------------------------------------------------------------------
# bags
# ---------------------------------------------------------------------------
# nested: grid.n is a solo const read bare, grid.dx a non-solo const read
# through .get(0) - members resolve on their own type, not the bag's
grid = Bag({"n": n_p, "dx": dx_p})

# flat, for bind_bag: the kernel that uses these reads them as bare names
noise_seeds = Bag({"SEED_A": seed_a_p, "SEED_B": seed_b_p})

# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------


def clampi(i):
    return min(max(i, 0), N - 1)


clampi_fn = QuadrantsHelperBuilder().bind("N", n_p).ingest(clampi)


def laplacian(field_, i, j):
    # calls another helper, and reads a non-solo const out of a bound bag
    ip = clampi(i + 1)
    im = clampi(i - 1)
    jp = clampi(j + 1)
    jm = clampi(j - 1)
    acc = field_[ip, j] + field_[im, j] + field_[i, jp] + field_[i, jm] - 4.0 * field_[i, j]
    return acc / (grid.dx.get(0) * grid.dx.get(0))


laplacian_fn = QuadrantsHelperBuilder().bind("clampi", clampi_fn).bind("grid", grid).ingest(laplacian)


def uplift_at(i, j):
    # binds the UPLIFT *field* itself: a helper reads a non-const Parameter
    # exactly the way a kernel does, so the caller passes only the indices
    return UPLIFT.get(i * N + j)


uplift_at_fn = QuadrantsHelperBuilder().bind("UPLIFT", uplift_p).bind("N", n_p).ingest(uplift_at)

# ---------------------------------------------------------------------------
# one-shot setup kernel (runs once, outside the routine)
# ---------------------------------------------------------------------------


def init_topo_template(z: qd.Tensor):
    # bind_bag put SEED_A / SEED_B in flat, so they read as bare names here
    for i, j in z:
        x = qd.cast(i, qd.f32) * SEED_A + qd.cast(j, qd.f32) * SEED_B
        s = qd.sin(x) * 43758.5453
        z[i, j] = (s - qd.floor(s)) * 2.0


init_topo_kernel = QuadrantsKernelBuilder().bind_bag(noise_seeds).ingest(init_topo_template).compile()

# ---------------------------------------------------------------------------
# routine kernels
# ---------------------------------------------------------------------------

# mixed bag: a scalar Parameter, a helper and a solo const under one name.
# hill.d is a device accessor, hill.lap a specialized func, hill.dt a literal.
hill = Bag({"d": d_p, "lap": laplacian_fn, "dt": dt_p})


def diffuse_template(z_out: qd.Tensor, z_in: qd.Tensor):
    for i, j in z_in:
        z_out[i, j] = z_in[i, j] + hill.dt * hill.d.get(0) * hill.lap(z_in, i, j)


diffuse_builder = QuadrantsKernelBuilder().bind("hill", hill).ingest(diffuse_template)


def uplift_template(z: qd.Tensor):
    for i, j in z:
        z[i, j] += up(i, j) * DT
        # base level: pin the east and west edges, so the ridge drains outward
        if j == 0 or j == grid.n - 1:
            z[i, j] = SEA.get(0)


uplift_builder = (
    QuadrantsKernelBuilder()
    .bind("up", uplift_at_fn)
    .bind("DT", dt_p)
    .bind("grid", grid)
    .bind("SEA", sea_p)
    .ingest(uplift_template)
)

# ---------------------------------------------------------------------------
# fields (pooled - two buffers for ping-pong)
# ---------------------------------------------------------------------------
z0 = pool.get_data(qd.f32, (GRID_N, GRID_N))
z1 = pool.get_data(qd.f32, (GRID_N, GRID_N))

init_topo_kernel(z0.data)

# ---------------------------------------------------------------------------
# the routine
# ---------------------------------------------------------------------------
# One bag for the whole routine, merged from what each builder already binds.
# Both builders reach `grid` - the same Bag object, so the same uid, which is
# what lets merge() accept the collision instead of raising on it.
evolve = (
    QuadrantsRoutineBuilder()
    .bind_bag(merge(diffuse_builder.as_bag(), uplift_builder.as_bag()))
    .add_data("z0", z0.data)
    .add_data("z1", z1.data)
    .add_kernel(diffuse_builder, data_handle_ref=("z1", "z0"))
    .add_kernel(uplift_builder, data_handle_ref=("z1",))
    .add_swap("z0", "z1")
    .add_kernel(diffuse_builder, data_handle_ref=("z1", "z0"))
    .add_kernel(uplift_builder, data_handle_ref=("z1",))
    .add_swap("z0", "z1")
    .compile()
)

# ---------------------------------------------------------------------------
# live view
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
im = ax.imshow(z0.to_numpy(), cmap="terrain", vmin=0.0, vmax=150.0)
fig.colorbar(im, ax=ax, label="Elevation (m)")
ax.set_title("Hillslope LEM (Quadrants backend, Routine)")
time_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
    color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.4, pad=2),
)
fig.show()

sim_time = 0.0
try:
    while True:
        t_start = time.perf_counter()
        for _ in range(STEPS_PER_FRAME // 2):
            evolve()  # two substeps, result lands back in z0
            sim_time += 2.0 * DT_VAL

        qd.sync()
        frame_ms = (time.perf_counter() - t_start) * 1e3
        print(f"{STEPS_PER_FRAME} steps: {frame_ms:8.1f} ms  ({frame_ms / STEPS_PER_FRAME * 1e3:6.1f} us/step)")

        time_text.set_text(f"t = {sim_time / 1e6:.2f} Myr")
        im.set_data(z0.to_numpy())
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.1)
except KeyboardInterrupt:
    pass

# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------
d_p.destroy()
sea_p.destroy()
uplift_p.destroy()
pool.release_data(z0)
pool.release_data(z1)
