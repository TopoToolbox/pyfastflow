import matplotlib.pyplot as plt
import taichi as ti

from pyfastflow.grid import GridContext
from pyfastflow.noise import NoiseContext
from pyfastflow.visu import VisuContext


ti.init(arch=ti.gpu)


gridctx_a = GridContext(256, 192, 1.0, boundary_mode="normal", topology="D4")
gridctx_b = GridContext(384, 256, 1.0, boundary_mode="periodic_EW", topology="D8")

noisectx_a = NoiseContext(gridctx_a)
noisectx_b = NoiseContext(gridctx_b)

visuctx_a = VisuContext(gridctx_a)
visuctx_b = VisuContext(gridctx_b)


terrain_a = noisectx_a.generate_perlin_noise(
    frequency=4.0,
    octaves=5,
    persistence=0.55,
    amplitude=1.0,
    seed=1729,
)
terrain_b = noisectx_b.generate_perlin_noise(
    frequency=10.0,
    octaves=7,
    persistence=0.5,
    amplitude=1.0,
    seed=3141,
)


hillshade_a = visuctx_a.generate_hillshade(terrain_a)
multishade_a = visuctx_a.generate_multishade(terrain_a)

hillshade_b = visuctx_b.generate_hillshade(terrain_b)
multishade_b = visuctx_b.generate_multishade(terrain_b)


terrain_np_a = terrain_a.field.to_numpy()
terrain_np_b = terrain_b.field.to_numpy()


fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

ax[0, 0].imshow(terrain_np_a.reshape(gridctx_a.rshp), cmap="terrain")
ax[0, 0].imshow(hillshade_a, cmap="gray", alpha=0.38)
ax[0, 0].set_title("Grid A - Hillshade")
ax[0, 0].set_axis_off()

ax[0, 1].imshow(terrain_np_a.reshape(gridctx_a.rshp), cmap="terrain")
ax[0, 1].imshow(multishade_a, cmap="gray", alpha=0.38)
ax[0, 1].set_title("Grid A - Multishade")
ax[0, 1].set_axis_off()

ax[1, 0].imshow(terrain_np_b.reshape(gridctx_b.rshp), cmap="terrain")
ax[1, 0].imshow(hillshade_b, cmap="gray", alpha=0.38)
ax[1, 0].set_title("Grid B - Hillshade")
ax[1, 0].set_axis_off()

ax[1, 1].imshow(terrain_np_b.reshape(gridctx_b.rshp), cmap="terrain")
ax[1, 1].imshow(multishade_b, cmap="gray", alpha=0.38)
ax[1, 1].set_title("Grid B - Multishade")
ax[1, 1].set_axis_off()

plt.show()


terrain_a.release()
terrain_b.release()
gridctx_a.destroy()
gridctx_b.destroy()
