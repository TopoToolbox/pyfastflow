import matplotlib.pyplot as plt
import taichi as ti

from pyfastflow.grid import GridContext
from pyfastflow.noise import NoiseContext
from pyfastflow.rastermanip import RasManContext


ti.init(arch=ti.gpu, offline_cache=False)


NX, NY, DX = 1024, 1024, 20.0

gridctx = GridContext(
    nx=NX,
    ny=NY,
    dx=DX,
    boundary_mode="normal",
    topology="D8",
)
noisectx = NoiseContext(gridctx)
rasmanctx = RasManContext()


z = noisectx.generate_perlin_noise(
    frequency=6.0,
    octaves=6,
    persistence=0.55,
    amplitude=1.0,
    seed=1729,
    layout="flat",
)

z_np = z.field.to_numpy().reshape(gridctx.rshp)

z_x2_np, gridctx_x2 = rasmanctx.double_resolution(
    z,
    method="bicubic",
    output_layout="2d",
    as_numpy=True,
    return_gridctx=True,
    gridctx=gridctx,
)

z_half_mean_np, gridctx_half = rasmanctx.halve_resolution(
    z,
    method="mean",
    output_layout="2d",
    as_numpy=True,
    return_gridctx=True,
    gridctx=gridctx,
)

z_half_cubic_np = rasmanctx.halve_resolution(
    z,
    method="median",
    output_layout="2d",
    as_numpy=True,
)

z_640x300_np = rasmanctx.resize_to_dims(
    z,
    target_nx=640,
    target_ny=300,
    output_layout="2d",
    as_numpy=True,
)

z_max256_np = rasmanctx.resize_to_max_dim(
    z,
    max_dim=256,
    output_layout="2d",
    as_numpy=True,
)


fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

ax[0, 0].imshow(z_np, cmap="terrain")
ax[0, 0].set_title(f"Original ({NX} x {NY})")
ax[0, 0].set_axis_off()

ax[0, 1].imshow(z_x2_np, cmap="terrain")
ax[0, 1].set_title(f"Double x2 ({gridctx_x2.nx} x {gridctx_x2.ny})")
ax[0, 1].set_axis_off()

ax[0, 2].imshow(z_half_mean_np, cmap="terrain")
ax[0, 2].set_title(f"Half mean ({gridctx_half.nx} x {gridctx_half.ny})")
ax[0, 2].set_axis_off()

ax[1, 0].imshow(z_half_cubic_np, cmap="terrain")
ax[1, 0].set_title("Half median")
ax[1, 0].set_axis_off()

ax[1, 1].imshow(z_640x300_np, cmap="terrain")
ax[1, 1].set_title("Resize to 640 x 300")
ax[1, 1].set_axis_off()

ax[1, 2].imshow(z_max256_np, cmap="terrain")
ax[1, 2].set_title("Resize to max_dim=256")
ax[1, 2].set_axis_off()

plt.show()


z.release()
gridctx.destroy()
gridctx_half.destroy()
gridctx_x2.destroy()
