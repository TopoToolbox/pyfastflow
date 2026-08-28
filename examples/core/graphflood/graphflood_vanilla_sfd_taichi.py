"""
GraphFlood on a real DEM (topotoolbox's "greenriver") - any backend, any
`kind` (pyfastflow.graphflood.make_graphflood's own dispatch:
"vanilla_sfd" with fill_method="jump"|"reconstruct", "unstable", or the
cupy-only "vanilla_mfd" - see make_graphflood's own module docstring,
pyfastflow/graphflood/__init__.py, for what each does).

Only the buffers the selected `kind`/`fill_method` combination actually
needs are allocated - make_graphflood takes no pool and allocates nothing
itself (every array argument is caller-supplied), so each branch below
allocates exactly its own combination's own required set, per
make_graphflood's own docstring.

fill_method="jump" pins `depression_method="vanilla"` explicitly, not the
factory's own default "optimized": that method's carve reroute kernel has
an unbounded on-device loop that hangs forever on real DEM data (confirmed
on this exact DEM, all three backends - not GraphFlood-specific, reproduces
calling ../../flow's make_depression_solver directly). "vanilla" solves
this same DEM in under a second.

Run:
    python graphflood_vanilla_sfd_taichi.py [backend] [kind] [fill_method]
        backend:     taichi (default) | quadrants | cupy
        kind:        vanilla_sfd (default) | unstable | vanilla_mfd (cupy-only)
        fill_method: jump (default) | reconstruct - only used by vanilla_sfd

Author: B.G (08/2026)
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import topotoolbox as ttb
from matplotlib.colors import LightSource

from pyfastflow.core.context.backends import backend_classes
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.graphflood import make_graphflood

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "taichi"
KIND = sys.argv[2] if len(sys.argv) > 2 else "vanilla_sfd"
FILL_METHOD = sys.argv[3] if len(sys.argv) > 3 else "jump"

if KIND == "vanilla_mfd" and BACKEND != "cupy":
    raise ValueError("kind='vanilla_mfd' is cupy-only")

if BACKEND == "taichi":
    import taichi as ti
    ti.init(arch=ti.gpu)
    from pyfastflow.core.pool.taichi_pool import TaichiPool as PoolCls
elif BACKEND == "quadrants":
    import quadrants as qd
    qd.init(arch=qd.gpu)
    from pyfastflow.core.pool.quadrants_pool import QuadrantsPool as PoolCls
elif BACKEND == "cupy":
    from pyfastflow.core.pool.cupy_pool import CupyPool as PoolCls
else:
    raise ValueError(f"unknown backend {BACKEND!r}, expected 'taichi', 'quadrants' or 'cupy'")

N_STEPS = 100
RAIN = 100e-3 / 3600.0  # 50 mm/hr, in m/s
DT = 1e-2
MANNING = 0.033
FRICTION_EXPONENT = 2.0 / 3.0

dem = ttb.load_dem("greenriver")
NX, NY, DX = dem.columns, dem.rows, dem.cellsize
n_flat = NX * NY
N_NEIGHBOURS = 8  # D8

_bk = backend_classes(BACKEND); ParamCls, dtypes = _bk.ParameterCls, _bk.dtypes
i32, i64, f32, u8 = dtypes["i32"], dtypes["i64"], dtypes["f32"], dtypes["u8"]
pool = PoolCls()

grid_group = make_grid_group(BACKEND, topology="D8", boundary="normal", outlet="edge")
grid_params = make_grid_parameters(BACKEND, pool, NX, NY, DX, topology="D8", outlet="edge")

# --- buffers/params every kind needs ---------------------------------------
z = pool.get_data(f32, (n_flat,))
h = pool.get_data(f32, (n_flat,))
Q_in = pool.get_data(f32, (n_flat,))
Qo = pool.get_data(f32, (n_flat,))
z.from_numpy(dem.z.ravel().astype(np.float32))
h.from_numpy(np.zeros(n_flat, dtype=np.float32))

# SOURCE is Q (m^3/s per cell), not a bare rate - apply_divergence computes
# (Q_in - Qo)/area*dt against Qo (m^3/s, from the friction law), so Q_in
# must be in the same units: rain rate * cell area.
source_p = ParamCls("SOURCE", dtype=f32, mode="const", value=RAIN * DX * DX, pool=pool)
manning_p = ParamCls("MANNING", dtype=f32, mode="const", value=MANNING, pool=pool)
expo_p = ParamCls("EXPO", dtype=f32, mode="const", value=FRICTION_EXPONENT, pool=pool)
dt_p = ParamCls("DT", dtype=f32, mode="const", value=DT, pool=pool)
gf_min_increment_p = ParamCls("GF_MIN_INCREMENT", dtype=f32, mode="const", value=0.0, pool=pool)
boundary_h_p = ParamCls("BOUNDARY_H", dtype=f32, mode="const", value=0.0, pool=pool)

kwargs = dict(
    n_flat=n_flat, nx=NX, ny=NY, z=z.data, h=h.data, Q_in=Q_in.data, Qo=Qo.data,
    source_p=source_p, manning_p=manning_p, friction_exponent_p=expo_p, dt_p=dt_p,
    gf_min_increment_p=gf_min_increment_p, boundary_h_p=boundary_h_p,
    outlet_behavior="fixed_h", kind=KIND,
)

# --- buffers only this kind/fill_method combination needs -------------------
if KIND == "unstable":
    Q_next = pool.get_data(f32, (n_flat,))
    kwargs["Q_next"] = Q_next.data

elif KIND == "vanilla_mfd":
    surface = pool.get_data(f32, (n_flat,))
    filled = pool.get_data(f32, (n_flat,))
    parent = pool.get_data(i32, (n_flat,))
    frontier = pool.get_data(i32, (2 * n_flat,))
    max_passes = 4 * max(NX, NY)
    counters = pool.get_data(i32, (max_passes + 2,))
    queued_gen = pool.get_data(i32, (n_flat,))
    pass_p = ParamCls("P", dtype=i32, mode="scalar", value=0, pool=pool)
    active_p = ParamCls("ACTIVE", dtype=i32, mode="scalar", value=0, pool=pool)
    dirs = pool.get_data(u8, (n_flat,))
    mfd_w = pool.get_data(f32, (n_flat * N_NEIGHBOURS,))
    indegree = pool.get_data(i32, (n_flat,))
    frontier0 = pool.get_data(i32, (n_flat,))
    frontier1 = pool.get_data(i32, (n_flat,))
    count = pool.get_data(i32, (2,))
    barrier = pool.get_data(dtypes.get("u32", i32), (1,))
    dist = pool.get_data(f32, (n_flat,))
    anc = pool.get_data(i32, (n_flat,))
    dist2 = pool.get_data(f32, (n_flat,))
    anc2 = pool.get_data(i32, (n_flat,))
    filled_eps = pool.get_data(f32, (n_flat,))
    kwargs.update(
        surface=surface.data, filled=filled.data, parent=parent.data, frontier=frontier.data,
        counters=counters.data, queued_gen=queued_gen.data, pass_p=pass_p, active_p=active_p,
        max_passes=max_passes, dirs=dirs.data, mfd_w=mfd_w.data, indegree=indegree.data,
        frontier0=frontier0.data, frontier1=frontier1.data, count=count.data, barrier=barrier.data,
        dist=dist.data, anc=anc.data, dist2=dist2.data, anc2=anc2.data, filled_eps=filled_eps.data,
    )

else:  # kind == "vanilla_sfd"
    kwargs["fill_method"] = FILL_METHOD
    if FILL_METHOD == "jump":
        # depression_method="vanilla", not the default "optimized": that
        # method's own carve reroute kernel (carve_basins_serial,
        # _closure_depressions.py) has an unbounded on-device while loop
        # that hangs forever on real DEM data (confirmed on greenriver,
        # all three backends) - see the memory note
        # depression_optimized_carve_hang.md. "vanilla"'s own carve is a
        # bounded pointer-jump sweep and solves this same DEM in <1s.
        kwargs["depression_method"] = "vanilla"
        rec = pool.get_data(i32, (n_flat,))
        bid = pool.get_data(i32, (n_flat,))
        rec_jump = pool.get_data(i32, (n_flat,))
        z_prime = pool.get_data(f32, (n_flat,))
        is_border = pool.get_data(i32, (n_flat,))
        basin_saddle = pool.get_data(i64, (n_flat,))
        basin_saddlenode = pool.get_data(i32, (n_flat,))
        outlet_h = pool.get_data(i64, (n_flat,))
        rerouted = pool.get_data(i32, (n_flat,))
        tag = pool.get_data(i32, (n_flat,))
        tag_alt = pool.get_data(i32, (n_flat,))
        rec_scratch = pool.get_data(i32, (n_flat,))
        basin_route = pool.get_data(i32, (n_flat,))
        b_rcv = pool.get_data(i32, (n_flat,))
        ndep_p = ParamCls("NDEP", dtype=i32, mode="scalar", value=0, pool=pool)
        kwargs.update(
            rec=rec.data, ndep_p=ndep_p, bid=bid.data, rec_jump=rec_jump.data, z_prime=z_prime.data,
            is_border=is_border.data, basin_saddle=basin_saddle.data, basin_saddlenode=basin_saddlenode.data,
            outlet=outlet_h.data, rerouted=rerouted.data, tag=tag.data, tag_alt=tag_alt.data,
            rec_scratch=rec_scratch.data, basin_route=basin_route.data, b_rcv=b_rcv.data,
        )
    else:  # "reconstruct"
        surface = pool.get_data(f32, (n_flat,))
        filled = pool.get_data(f32, (n_flat,))
        parent = pool.get_data(i32, (n_flat,))
        frontier = pool.get_data(i32, (2 * n_flat,))
        max_passes = 4 * max(NX, NY)
        counters = pool.get_data(i32, (max_passes + 2,))
        queued_gen = pool.get_data(i32, (n_flat,))
        pass_p = ParamCls("P", dtype=i32, mode="scalar", value=0, pool=pool)
        active_p = ParamCls("ACTIVE", dtype=i32, mode="scalar", value=0, pool=pool)
        kwargs.update(
            surface=surface.data, filled=filled.data, parent=parent.data, frontier=frontier.data,
            counters=counters.data, queued_gen=queued_gen.data, pass_p=pass_p, active_p=active_p,
            max_passes=max_passes,
        )

gf = make_graphflood(BACKEND, grid_group, grid_params, **kwargs)

for step in range(N_STEPS):
    gf.step()
    if step % 10 == 0:
        h_np = h.to_numpy()
        print(f"step {step}/{N_STEPS}  h_max={h_np.max():.4g}  h_mean={h_np.mean():.4g}")

zz = z.to_numpy().reshape(NY, NX)
hh = h.to_numpy().reshape(NY, NX)

ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(zz, vert_exag=2.0, dx=DX, dy=DX)

fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
ax.imshow(hs, cmap="gray")
im = ax.imshow(np.where(hh > 1e-3, hh, np.nan), cmap="Blues", vmin=0.0, vmax=0.5, alpha=0.8)
fig.colorbar(im, ax=ax, shrink=0.8, label="water depth h (m)")
title = f"GraphFlood {KIND}" + (f" ({FILL_METHOD})" if KIND == "vanilla_sfd" else "")
ax.set_title(f"{title}, {BACKEND}, {N_STEPS} steps, dt={DT}s")
ax.set_xticks([])
ax.set_yticks([])
plt.show()
