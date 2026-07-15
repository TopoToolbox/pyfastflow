import matplotlib.pyplot as plt
import taichi as ti

import pyfastflow as pf


ti.init(arch=ti.gpu, offline_cache=False)


gridctx_a = pf.grid.GridContext(512, 1048, 1.0, boundary_mode="normal", topology="D4")
gridctx_b = pf.grid.GridContext(800, 600, 1.0, boundary_mode="periodic_EW", topology="D8")

noisectx_a = pf.noise.NoiseContext(gridctx_a)
noisectx_b = pf.noise.NoiseContext(gridctx_b)

white_a_1 = noisectx_a.generate_white_noise(amplitude=0.5, seed=11)
white_a_2 = noisectx_a.generate_white_noise(amplitude=1.0, seed=23)
perlin_a_1 = noisectx_a.generate_perlin_noise(
    frequency=6.0, octaves=3, persistence=0.5, amplitude=1.0, seed=31
)
perlin_a_2 = noisectx_a.generate_perlin_noise(
    frequency=12.0, octaves=5, persistence=0.6, amplitude=1.0, seed=47
)

white_b_1 = noisectx_b.generate_white_noise(amplitude=0.5, seed=101)
white_b_2 = noisectx_b.generate_white_noise(amplitude=1.0, seed=131)
perlin_b_1 = noisectx_b.generate_perlin_noise(
    frequency=8.0, octaves=4, persistence=0.5, amplitude=1.0, seed=151
)
perlin_b_2 = noisectx_b.generate_perlin_noise(
    frequency=16.0, octaves=6, persistence=0.6, amplitude=1.0, seed=181
)

fig, ax = plt.subplots(2, 4, figsize=(14, 7))

ax[0, 0].imshow(white_a_1.to_numpy().reshape(gridctx_a.rshp), cmap="gray")
ax[0, 0].set_title("A white #1")
ax[0, 1].imshow(white_a_2.to_numpy().reshape(gridctx_a.rshp), cmap="gray")
ax[0, 1].set_title("A white #2")
ax[0, 2].imshow(perlin_a_1.to_numpy().reshape(gridctx_a.rshp), cmap="terrain")
ax[0, 2].set_title("A perlin #1")
ax[0, 3].imshow(perlin_a_2.to_numpy().reshape(gridctx_a.rshp), cmap="terrain")
ax[0, 3].set_title("A perlin #2")

ax[1, 0].imshow(white_b_1.to_numpy().reshape(gridctx_b.rshp), cmap="gray")
ax[1, 0].set_title("B white #1")
ax[1, 1].imshow(white_b_2.to_numpy().reshape(gridctx_b.rshp), cmap="gray")
ax[1, 1].set_title("B white #2")
ax[1, 2].imshow(perlin_b_1.to_numpy().reshape(gridctx_b.rshp), cmap="terrain")
ax[1, 2].set_title("B perlin #1")
ax[1, 3].imshow(perlin_b_2.to_numpy().reshape(gridctx_b.rshp), cmap="terrain")
ax[1, 3].set_title("B perlin #2")

for row in ax:
    for axis in row:
        axis.set_xticks([])
        axis.set_yticks([])

fig.tight_layout()
plt.show()

white_a_1.release()
white_a_2.release()
perlin_a_1.release()
perlin_a_2.release()
white_b_1.release()
white_b_2.release()
perlin_b_1.release()
perlin_b_2.release()
