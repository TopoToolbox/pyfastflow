import matplotlib.pyplot as plt
import taichi as ti

import pyfastflow.constants as cte
from pyfastflow import tp
from pyfastflow.flow import FlowContext
from pyfastflow.flood import FloodContext
from pyfastflow.grid import GridContext
from pyfastflow.noise import NoiseContext


import topotoolbox as ttb

ti.init(arch=ti.gpu, offline_cache=False)


dem = ttb.load_dem('greenriver')
# fig,ax = plt.subplots()
# ax.imshow(dem.z,cmap = 'terrain', vmax = dem.z.mean())
# plt.show()
# plt.close('all')

NX = dem.columns
NY = dem.rows
DX = dem.cellsize
N_LOCALPASS = 10000
precrate = 1e-4


gridctx = GridContext(NX, NY, DX, boundary_mode="normal", topology="D4")
noisectx = NoiseContext(gridctx)
flowctx = FlowContext(gridctx)
floodctx = FloodContext(
    gridctx,
    flowctx=flowctx,
    dth_mode="const",
    dth=1e-3,
    source_w_mode="const",
    source_w=precrate,
    source_w_kind="precip",
    friction_coeff_mode="const",
    friction_coeff=0.033,
    friction_exponent_mode="const",
    friction_exponent=2.0 / 3.0,
    boundary_h_mode="const",
    boundary_h=0.0,
)



z = tp.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(NX * NY))
z.from_numpy(dem.z.ravel())
h = tp.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(NX * NY))
qx = tp.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(NX * NY))
qy = tp.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(NX * NY))
h.field.fill(0.0)
qx.field.fill(0.0)
qy.field.fill(0.0)

h_np = h.field.to_numpy().reshape((NY, NX))
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
im = ax.imshow(h_np, cmap="Blues", vmax = 0.6)
fig.show()

while True:
    for _ in range(N_LOCALPASS):
        floodctx.ls_add_source_to_h(h.field)
        floodctx.ls_flow_route(h.field, z.field, qx.field, qy.field)
        floodctx.ls_depth_update(h.field, z.field, qx.field, qy.field)


    h_np = h.field.to_numpy().reshape((NY, NX))
    im.set_data(h_np)
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)




z.release()
h.release()
qx.release()
qy.release()
floodctx.destroy()
flowctx.destroy()
gridctx.destroy()
