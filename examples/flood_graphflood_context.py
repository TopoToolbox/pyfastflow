from math import ceil, log2

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import topotoolbox as ttb
import time

import pyfastflow.constants as cte
from pyfastflow import tp
from pyfastflow.flow import FlowContext
from pyfastflow.flood import FloodContext
from pyfastflow.grid import GridContext
from pyfastflow.visu import VisuContext

ti.init(arch=ti.gpu, offline_cache=False)

# ---------------------------------------------------------------------------
# Helper functions — defined after field allocation below, as closures.
# Three methods to test:
#   propagate()  — SFD Q accumulation with reroute, no h fill
#   distribute() — local slope-weighted Q redistribution to neighbours
#   core()       — Manning divergence update on h (unsafe=True skips dh buffer)
# ---------------------------------------------------------------------------

# dem = ttb.load_dem("greenriver")
dem = ttb.read_tif('/home/bgailleton/Desktop/data/Lidar_swiss/bettlach/DEM.tif')
NX, NY, DX = dem.columns, dem.rows, dem.cellsize
N = NX * NY
precrate = 100e-3 / 3600

gridctx = GridContext(NX, NY, DX, boundary_mode="normal", topology="D8")
flowctx = FlowContext(
    gridctx,
    weight_mode="const",
    weight=1.0,
    min_slope_mode="const",
    min_slope=1e-3,
    diagonal_partition_correction=True,
)
floodctx = FloodContext(
    gridctx,
    flowctx=flowctx,
    dth_mode="const",
    dth=1e-3,
    source_w_mode="const",
    source_w=precrate,
    source_w_kind="precip",
    boundary_h_mode="const",
    boundary_h=0.0,
    gf_min_increment=0.0,
)

visuctx = VisuContext(gridctx)
accum = floodctx._accum_flowctx
logn = flowctx.logn

# --- persistent fields ---
z           = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
h           = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
receivers   = tp.get_tpfield(ti.i32, N)
Q           = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
Q_next      = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
surface     = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
dh          = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
out_sum     = tp.get_tpfield(cte.FLOAT_TYPE_TI, ())

# reroute temps
bid             = tp.get_tpfield(ti.i32, N)
rec_work        = tp.get_tpfield(ti.i32, N)
rec_jump        = tp.get_tpfield(ti.i32, N)
z_prime         = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
is_border       = tp.get_tpfield(ti.u1, N)
outlet          = tp.get_tpfield(ti.i64, N)
basin_saddle    = tp.get_tpfield(ti.i64, N)
basin_saddlenode = tp.get_tpfield(ti.i32, N)
tag             = tp.get_tpfield(ti.u1, N)
tag_alt         = tp.get_tpfield(ti.u1, N)
rerouted        = tp.get_tpfield(ti.u1, N)

# sfd accumulation temps
donors      = tp.get_tpfield(ti.i32, N * gridctx.n_neighbours)
ndonors     = tp.get_tpfield(ti.i32, N)
donors_alt  = tp.get_tpfield(ti.i32, N * gridctx.n_neighbours)
ndonors_alt = tp.get_tpfield(ti.i32, N)
Q_alt       = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
src         = tp.get_tpfield(ti.i32, N)

z.field.from_numpy(dem.z.ravel().astype(np.float32))
h.field.fill(0.0)
Q.field.fill(0.0)

hs = visuctx.generate_hillshade(z.field, altitude_deg=45.0, azimuth_deg=315.0, z_factor=1.0)


def propagate():
    Q.field.fill(0.0)
    floodctx.add_source_to_Q(Q.field)
    accum.set_weight(Q.field)

    floodctx.make_surface(z.field, h.field, surface.field)
    flowctx.compute_receivers(surface.field, receivers.field)

    rec_work.field.copy_from(receivers.field)
    rerouted.field.fill(False)
    ndep = flowctx.depression_counter(receivers.field)
    if ndep > 0:
        for _ in range(ceil(log2(max(1, int(ndep)))) + 1):
            ndep_bis = flowctx.depression_counter(rec_work.field)
            flowctx.basin_id_init(bid.field)
            rec_jump.field.copy_from(rec_work.field)
            for _ in range(logn + 1):
                flowctx.propagate_basin_iter(rec_jump.field)
            flowctx.propagate_basin_final(bid.field, rec_jump.field)
            if ndep_bis == 0:
                break
            flowctx.saddlesort(bid.field, is_border.field, z_prime.field, basin_saddle.field, basin_saddlenode.field, outlet.field, surface.field)
            flowctx.init_reroute_carve(tag.field, tag_alt.field, basin_saddlenode.field)
            receivers.field.copy_from(rec_work.field)
            rec_jump.field.copy_from(rec_work.field)
            for _ in range(logn + 1):
                flowctx.iteration_reroute_carve(tag.field, tag_alt.field, receivers.field, rec_work.field, bid.field)
            flowctx.finalise_reroute_carve(receivers.field, rec_jump.field, tag.field, basin_saddlenode.field, outlet.field, rerouted.field)
            rec_work.field.copy_from(receivers.field)
        receivers.field.copy_from(rec_work.field)

    ndonors.field.fill(0)
    ndonors_alt.field.fill(0)
    src.field.fill(0)
    accum.init_weighted_source(Q.field)
    accum.receivers_to_donors(receivers.field, donors.field, ndonors.field)
    for iteration in range(logn + 1):
        accum.rake_compress_accum(donors.field, ndonors.field, Q.field, src.field, donors_alt.field, ndonors_alt.field, Q_alt.field, iteration)
    accum.fuse_accum_buffers(Q.field, src.field, Q_alt.field, logn)


def distribute():
    floodctx.distribute_flow_local(z.field, h.field, Q.field, Q_next.field)
    Q.field.copy_from(Q_next.field)


def core(unsafe=False):
    if unsafe:
        floodctx.graphflood_core_unsafe(z.field, h.field, Q.field)
    else:
        floodctx.graphflood_core(z.field, h.field, Q.field, dh.field)


# --- viz ---
_dummy = np.zeros((NY, NX), dtype=np.float32)
fig, ax = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
for a in ax:
    a.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
imh  = ax[0].imshow(_dummy, cmap="Blues",  vmin=0,    vmax=2.0,  alpha=0.6)
imQ  = ax[1].imshow(_dummy, cmap="Purples", vmin=-3,  vmax=1.0,  alpha=0.6)
imdh = ax[2].imshow(_dummy, cmap="RdBu_r", vmin=-0.01, vmax=0.01, alpha=0.6)
ax[0].set_title("hw")
ax[1].set_title("Q")
ax[2].set_title("dh")
fig.show()

# --- initial propagation ---
propagate()

# --- loop ---
while True:
    hm1 = h.field.to_numpy().reshape(NY, NX)
    st = time.perf_counter()
    # propagate()# if i % 100 == 0 else 0
    for i in range(1000):
        # for j in range(10):
        distribute()
        core(unsafe=True)
    ti.sync()
    floodctx.sum_Q_at_outlets(Q.field, out_sum.field)
    tdh = (hm1 - h.field.to_numpy().reshape(NY, NX))

    print('iteration took', time.perf_counter() - st, 's dh perc90:',np.percentile(np.abs(tdh), 90),' balance:', float(out_sum.field[None]), 'vs', precrate * NX * NY * DX**2, end='          \r')


    imQ.set_data(np.log10(Q.field.to_numpy()).reshape(NY, NX))
    imh.set_data(h.field.to_numpy().reshape(NY, NX))
    imdh.set_data(tdh)
    hm1 = h.field.to_numpy().reshape(NY, NX)
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)

# --- cleanup ---
for f in [z, h, receivers, Q, Q_next, surface, dh, out_sum,
          bid, rec_work, rec_jump, z_prime, is_border, outlet,
          basin_saddle, basin_saddlenode, tag, tag_alt, rerouted,
          donors, ndonors, donors_alt, ndonors_alt, Q_alt, src]:
    f.release()
floodctx.destroy()
flowctx.destroy()
gridctx.destroy()
