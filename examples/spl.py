import pyfastflow as pff
import matplotlib.pyplot as plt
import taichi as ti
import numpy as np

ti.init(ti.gpu)

nx, ny = 512,512

dx = 32.

znp = pff.noise.perlin_noise(nx, ny, frequency = 4.0, octaves = 12, persistence = 0.4, amplitude = 100.0, seed = 42)

# plt.imshow(znp, cmap = 'terrain')
# plt.show()

grid = pff.grid.Grid(nx,ny,dx,znp)

router = pff.flow.FlowRouter(grid)
router.compute_receivers()
router.reroute_flow()
router.fill_z()
router.accumulate_constant_Q(1.)




# plt.imshow(np.log10(router.get_Q()), cmap = 'magma')
# plt.imshow(grid.get_z(), cmap = 'terrain')
# plt.show()

