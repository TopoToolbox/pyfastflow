"""
GraphFlood CLI: run make_graphflood to steady state (dh/dt convergence) or
n_max on a real DEM (any file ttb.read_tif() accepts), reporting progress
and saving results.

NoData handling
------------------
If the DEM has NaNs, the grid is built with `nodata=True, outlet="mask"`
(never the plain `outlet="edge"` this package's other examples use) -
NODATA_MASK marks every NaN cell; OUTLET_MASK marks every DEM-edge cell AND
every cell that neighbours a NaN cell (not the NaN cells themselves - see
below). `z`'s NaNs are replaced with a large finite sentinel for
computation (kept as real NaN only in the array used for the hillshade
plot) - the grid's own `_move_allowed`/`_valid` machinery (../grid/
_closure_blocks.py) already makes `neighbour()` return -1 for any lookup
that touches a NoData cell on either end, so nothing ever dereferences that
sentinel's neighbours; it exists only so a NoData cell's own z value is
never a bare NaN sitting in the buffer.

A NoData cell is deliberately NOT put in OUTLET_MASK - only cells
neighbouring one are, per spec. That leaves a NoData cell with no valid
downslope neighbour at all (can never route out) and not can_out either -
harmless for h and Qo (both stay exactly 0 there, since compute_qo's own
neighbour loop only ever executes when the target list is nonempty), but
Q_in would otherwise accumulate that cell's own rain contribution forever
with nowhere for it to go, drifting h upward without bound over many steps.
SOURCE is switched from a uniform const to a field masked to 0 at every
NoData cell to prevent exactly that - a direct consequence of "NoData means
excluded from the simulation", not a change to the physics anywhere real
data exists.

Convergence metric
---------------------
The `CONVERGENCE_PERCENTILE`th-percentile (90th, not 99th - see below)
|dh/dt| check at every `--n_check` steps is computed only over currently
"wet" cells (`h > WET_H`, 1cm) - not the whole domain. Most of a real DEM
never floods, or floods far more slowly than the active front; a global
percentile is dominated by that static dry background the moment the wet
fraction is small, reporting a flat, falsely tiny value (essentially
float32 noise around 0) that never reflects whether the actually-flooding
part of the domain is still changing. Before any cell is wet, the check
reports `n_wet=0` and does not count towards convergence.

The wet-cell set itself is typically a small fraction of the domain, so its
own 99th percentile sits close to the noisy tail of a comparatively small
sample (whichever handful of wet cells happen to be most active right now,
not the bulk). 90th tracks the bulk of the wet region instead - still high
enough to be a "is the flood still changing" check, not a mean.

Run:
    python graphflood_cli.py DEM.tif [--kind mfd|sfd|unstable] [--backend cupy|taichi|quadrants]
        [--manning 0.033] [--dt 5e-3] [--rain 50.0] [--n_check 10] [--threshold 1e-5]
        [--n_max 5000] [--prefix NAME_out_]

Author: B.G (08/2026)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import topotoolbox as ttb
from matplotlib.colors import LightSource
from scipy.ndimage import binary_dilation

from pyfastflow.core.context.backends import backend_classes
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.graphflood import make_graphflood

FRICTION_EXPONENT = 2.0 / 3.0
N_NEIGHBOURS = 8  # D8
NODATA_Z_SENTINEL = 1.0e8
WET_H = 1e-2  # depth above which a cell counts as "wet" - convergence metric and plot both use this
CONVERGENCE_PERCENTILE = 95  # of |dh/dt| over wet cells - wet fraction is typically small, 99th was too
                              # close to the noisy tail of a small sample; 90th tracks the bulk instead

_KIND_MAP = {"sfd": "vanilla_sfd", "mfd": "vanilla_mfd", "unstable": "unstable"}


def parse_args():
    p = argparse.ArgumentParser(description="GraphFlood CLI runner")
    p.add_argument("dem", type=str, help="path to a DEM readable by topotoolbox.read_tif()")
    p.add_argument("--kind", choices=sorted(_KIND_MAP), default="mfd")
    p.add_argument("--backend", choices=["taichi", "quadrants", "cupy"], default="cupy")
    p.add_argument("--manning", type=float, default=0.033)
    p.add_argument("--dt", type=float, default=5e-3, help="timestep, seconds")
    p.add_argument("--rain", type=float, default=50.0, help="uniform rain rate, mm/h (converted to m/s internally)")
    p.add_argument("--n_check", type=int, default=10)
    p.add_argument("--threshold", type=float, default=1e-5)
    p.add_argument("--n_max", type=int, default=5000)
    p.add_argument("--prefix", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    kind = _KIND_MAP[args.kind]
    backend = args.backend
    if kind == "vanilla_mfd" and backend != "cupy":
        raise ValueError("--kind mfd is cupy-only")

    prefix = args.prefix
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(args.dem))[0] + "_out_"

    if backend == "taichi":
        import taichi as ti
        ti.init(arch=ti.gpu)
        from pyfastflow.core.pool.taichi_pool import TaichiPool as PoolCls
    elif backend == "quadrants":
        import quadrants as qd
        qd.init(arch=qd.gpu)
        from pyfastflow.core.pool.quadrants_pool import QuadrantsPool as PoolCls
    else:
        from pyfastflow.core.pool.cupy_pool import CupyPool as PoolCls

    dem = ttb.read_tif(args.dem)
    NX, NY, DX = dem.columns, dem.rows, dem.cellsize
    n_flat = NX * NY

    z_display = dem.z.astype(np.float32)
    nodata_np = ~np.isfinite(z_display)
    has_nodata = bool(nodata_np.any())

    z_np = z_display.copy()
    z_np[nodata_np] = NODATA_Z_SENTINEL

    outlet_np = np.zeros((NY, NX), dtype=bool)
    outlet_np[0, :] = True
    outlet_np[-1, :] = True
    outlet_np[:, 0] = True
    outlet_np[:, -1] = True
    if has_nodata:
        touches_nodata = binary_dilation(nodata_np, structure=np.ones((3, 3), dtype=bool)) & ~nodata_np
        outlet_np |= touches_nodata

    outlet_mode = "mask" if has_nodata else "edge"

    _bk = backend_classes(backend); ParamCls, dtypes = _bk.ParameterCls, _bk.dtypes
    i32, i64, f32, u8 = dtypes["i32"], dtypes["i64"], dtypes["f32"], dtypes["u8"]
    pool = PoolCls()

    grid_group = make_grid_group(backend, topology="D8", boundary="normal", nodata=has_nodata, outlet=outlet_mode)
    grid_params = make_grid_parameters(
        backend, pool, NX, NY, DX, topology="D8", nodata=has_nodata, outlet=outlet_mode,
    )
    if has_nodata:
        grid_params["NODATA_MASK"].get().from_numpy(nodata_np.ravel().astype(np.uint8))
    if outlet_mode == "mask":
        grid_params["OUTLET_MASK"].get().from_numpy(outlet_np.ravel().astype(np.uint8))

    z = pool.get_data(f32, (n_flat,))
    h = pool.get_data(f32, (n_flat,))
    Q_in = pool.get_data(f32, (n_flat,))
    Qo = pool.get_data(f32, (n_flat,))
    z.from_numpy(z_np.ravel())
    h.from_numpy(np.zeros(n_flat, dtype=np.float32))

    rain_m_s = args.rain * 1e-3 / 3600.0  # mm/h -> m/s
    # SOURCE is Q (m^3/s per cell), not a bare rate - apply_divergence computes
    # (Q_in - Qo)/area*dt against Qo (m^3/s, from the friction law), so Q_in
    # must be in the same units: rain rate * cell area.
    rain_q = rain_m_s * DX * DX
    source_np = np.where(nodata_np.ravel(), 0.0, rain_q).astype(np.float32)
    source_p = ParamCls("SOURCE", dtype=f32, mode="field", value=source_np, pool=pool, n_flat=n_flat)
    manning_p = ParamCls("MANNING", dtype=f32, mode="const", value=args.manning, pool=pool)
    expo_p = ParamCls("EXPO", dtype=f32, mode="const", value=FRICTION_EXPONENT, pool=pool)
    dt_p = ParamCls("DT", dtype=f32, mode="const", value=args.dt, pool=pool)
    gf_min_increment_p = ParamCls("GF_MIN_INCREMENT", dtype=f32, mode="const", value=0.0, pool=pool)
    boundary_h_p = ParamCls("BOUNDARY_H", dtype=f32, mode="const", value=0.0, pool=pool)

    kwargs = dict(
        n_flat=n_flat, nx=NX, ny=NY, z=z.data, h=h.data, Q_in=Q_in.data, Qo=Qo.data,
        source_p=source_p, manning_p=manning_p, friction_exponent_p=expo_p, dt_p=dt_p,
        gf_min_increment_p=gf_min_increment_p, boundary_h_p=boundary_h_p,
        outlet_behavior="fixed_h", kind=kind,
    )

    if kind == "unstable":
        Q_next = pool.get_data(f32, (n_flat,))
        kwargs["Q_next"] = Q_next.data

    elif kind == "vanilla_mfd":
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
        kwargs.update(
            surface=surface.data, filled=filled.data, parent=parent.data, frontier=frontier.data,
            counters=counters.data, queued_gen=queued_gen.data, pass_p=pass_p, active_p=active_p,
            max_passes=max_passes, dirs=dirs.data, mfd_w=mfd_w.data, indegree=indegree.data,
            frontier0=frontier0.data, frontier1=frontier1.data, count=count.data, barrier=barrier.data,
            dist=dist.data, anc=anc.data, dist2=dist2.data, anc2=anc2.data,
        )

    else:  # kind == "vanilla_sfd"
        # depression_method="vanilla", not the default "optimized" - see
        # this package's memory note depression_optimized_carve_hang.md:
        # "optimized"'s carve kernel can hang forever on real DEM data.
        kwargs["fill_method"] = "jump"
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

    gf = make_graphflood(backend, grid_group, grid_params, **kwargs)

    h_prev = h.to_numpy()
    below_streak = 0
    converged = False
    step = 0
    for step in range(1, args.n_max + 1):
        gf.step()
        if step % args.n_check == 0:
            h_now = h.to_numpy()
            wet = h_now > WET_H
            n_wet = int(wet.sum())
            if n_wet == 0:
                # nothing has flooded yet - nothing to declare converged
                print(f"step {step}/{args.n_max}  n_wet=0 (nothing flooded yet)", flush=True)
                h_prev = h_now
                below_streak = 0
                continue
            dhdt = np.abs(h_now[wet] - h_prev[wet]) / (args.n_check * args.dt)
            metric = float(np.percentile(dhdt, CONVERGENCE_PERCENTILE))
            q_in_now = Q_in.to_numpy()
            q_out_now = Qo.to_numpy()
            print(
                f"step {step}/{args.n_max}  n_wet={n_wet}  {CONVERGENCE_PERCENTILE}th pct |dh/dt| (wet only) = {metric:.6g}  "
                f"h_max={h_now.max():.6g}  Qin_max={q_in_now.max():.6g}  Qout_max={q_out_now.max():.6g}",
                flush=True,
            )
            h_prev = h_now
            if metric < args.threshold:
                below_streak += 1
                if below_streak >= 2:
                    converged = True
                    break
            else:
                below_streak = 0

    if converged:
        print(
            f"converged at step {step} ({CONVERGENCE_PERCENTILE}th pct |dh/dt| over wet cells < "
            f"{args.threshold} for 2 consecutive checks)"
        )
    else:
        print(f"n_max={args.n_max} reached without converging")

    h_np = h.to_numpy().reshape(NY, NX)
    q_in_np = Q_in.to_numpy().reshape(NY, NX)
    q_out_np = Qo.to_numpy().reshape(NY, NX)
    np.save(prefix + "h.npy", h_np)
    np.save(prefix + "Qin.npy", q_in_np)
    np.save(prefix + "Qout.npy", q_out_np)

    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(z_display, vert_exag=2.0, dx=DX, dy=DX)
    wet = h_np > WET_H
    vmax = float(np.percentile(h_np[wet], 90)) if wet.any() else 1.0
    status = " (converged)" if converged else " (n_max)"

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

    ax = axes[0]
    ax.imshow(hs, cmap="gray")
    im = ax.imshow(np.where(wet, h_np, np.nan), cmap="Blues", vmin=0.0, vmax=vmax, alpha=0.8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="water depth h (m)")
    ax.set_title(f"GraphFlood {args.kind}, {backend}, step {step}{status}")

    for ax, data, name in ((axes[1], q_in_np, "Qin"), (axes[2], q_out_np, "Qout")):
        ax.imshow(hs, cmap="gray")
        log_data = np.full_like(data, np.nan)
        np.log10(data, out=log_data, where=data > 0.0)
        im = ax.imshow(log_data, cmap="Blues", alpha=0.8)
        fig.colorbar(im, ax=ax, shrink=0.8, label=f"log10 {name} (m3/s)")
        ax.set_title(f"{name}, step {step}{status}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(prefix + "hillshade_h_Qin_Qout.png", dpi=150)
    print(f"saved {prefix}h.npy, {prefix}Qin.npy, {prefix}Qout.npy, {prefix}hillshade_h_Qin_Qout.png")


if __name__ == "__main__":
    main()
