import matplotlib.pyplot as plt
import taichi as ti

from pyfastflow import tp
from pyfastflow import constants as cte
from pyfastflow.flow import FlowContext, runtime as flow_runtime
from pyfastflow.grid import GridContext
from pyfastflow.noise import NoiseContext
from pyfastflow.visu import VisuContext
import numpy as np

ti.init(arch=ti.gpu)


MODE = "sfd_reroute"
# MODE = "mfd_fill"


gridctx = GridContext(2096, 2096, 10.0, boundary_mode="periodic_NS", topology="D8")
noisectx = NoiseContext(gridctx)
flowctx = FlowContext(
    gridctx,
    weight_mode="const",
    weight=1.0,
    min_slope_mode="const",
    min_slope=0.01,
)
visuctx = VisuContext(gridctx)


z = noisectx.generate_perlin_noise(
    frequency=3.,
    octaves=6,
    persistence=1.5,
    amplitude=1000.0,
    seed=1729,
    layout="flat",
)


receivers = tp.get_tpfield(dtype=ti.i32, shape=(gridctx.nx * gridctx.ny))
acc = tp.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(gridctx.nx * gridctx.ny))


flowctx.compute_receivers(z.field, receivers.field)
rerouted = flow_runtime.reroute_flow(flowctx, z, receivers, carve=True)

if MODE == "sfd_reroute":
    flow_runtime.accumulate_sfd(flowctx, receivers, acc)
    title = "Rerouted SFD Accumulation on Perlin Terrain"
elif MODE == "mfd_fill":
    ogz = z.field.to_numpy()
    flow_runtime.fill_topography_inplace(flowctx, z, receivers)
    flow_runtime.accumulate_mfd(flowctx, z, acc, max_iterations=128, check_interval=8)
    title = "Rerouted + Filled MFD Accumulation on Perlin Terrain"
else:
    raise ValueError("MODE must be 'sfd_reroute' or 'mfd_fill'")


hillshade = visuctx.generate_multishade(z, output_layout="2d")
accumulation = acc.field.to_numpy().reshape((gridctx.ny, gridctx.nx))


fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
ax.imshow(hillshade, cmap="gray", vmin=0.0, vmax=1.0)
ax.imshow(np.log10(accumulation), cmap="Blues", alpha=1.)
ax.set_title(title)
ax.set_axis_off()

plt.show()

if MODE == 'mfd_fill':
    fig,ax = plt.subplots()
    ax.imshow(np.abs(z.field.to_numpy()-ogz).reshape(gridctx.rshp))
    plt.show()


z.release()
receivers.release()
acc.release()
rerouted.release()
flowctx.destroy()
gridctx.destroy()
