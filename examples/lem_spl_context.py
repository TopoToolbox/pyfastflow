import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from pyfastflow.erodep import LEMContext, runtime as lem_runtime
from pyfastflow.flow import FlowContext
from pyfastflow.grid import GridContext
from pyfastflow.noise import NoiseContext
from pyfastflow.visu import VisuContext


ti.init(arch=ti.gpu)


NX = 1024
NY = 1024
DX = 50.0

N_OUTER = 2000
N_INNER = 50

REROUTE = True
FILL = True
CARVE = True

UPLIFT_RATE = 1e-3
DT = 1e4
K_BEDROCK = 2e-5
M_EXP = 0.45
N_EXP = 1.0


gridctx = GridContext(NX, NY, DX, boundary_mode="periodic_EW", topology="D8")
noisectx = NoiseContext(gridctx)
flowctx = FlowContext(
    gridctx,
    weight_mode="const",
    weight=DX**2,
    min_slope_mode="const",
    min_slope=1e-3,
    diagonal_partition_correction=True,
)
visuctx = VisuContext(gridctx)
lemctx = LEMContext(
    gridctx,
    flowctx=flowctx,
    dt_mode="const",
    dt=DT,
    uplift_rate_mode="const",
    uplift_rate=UPLIFT_RATE,
    K_bedrock_mode="const",
    K_bedrock=K_BEDROCK,
    m_exp_mode="const",
    m_exp=M_EXP,
    n_exp_mode="const",
    n_exp=N_EXP,
)


z = noisectx.generate_perlin_noise(
    frequency=4.0,
    octaves=8,
    persistence=0.55,
    amplitude=150.0,
    seed=1729,
    layout="flat",
)


terrain_np = z.field.to_numpy().reshape((NY, NX))
shade_np = visuctx.generate_multishade(z, output_layout="2d")


fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
terrain_im = ax.imshow(terrain_np, cmap="terrain")
shade_im = ax.imshow(shade_np, cmap="gray", alpha=0.35, vmin=0.0, vmax=1.0)
ax.set_title("LEMContext SPL")
ax.set_axis_off()
fig.show()


for outer in range(N_OUTER):
    lem_runtime.run_spl(
        lemctx,
        z,
        n_iterations=N_INNER,
        reroute=REROUTE,
        fill=FILL,
        carve=CARVE,
    )

    terrain_np = z.field.to_numpy().reshape((NY, NX))
    shade_np = visuctx.generate_multishade(z, output_layout="2d")

    terrain_im.set_data(terrain_np)
    terrain_im.set_clim(float(np.min(terrain_np)), float(np.max(terrain_np)))
    shade_im.set_data(shade_np)
    ax.set_title(
        f"LEMContext SPL | outer={outer + 1}/{N_OUTER} | inner={N_INNER} | reroute={REROUTE} | fill={FILL}"
    )
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)


plt.show()


z.release()
lemctx.destroy()
flowctx.destroy()
gridctx.destroy()
