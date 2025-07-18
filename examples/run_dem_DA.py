import taichi as ti
import numpy as np
import scabbard as scb
import fastflow_taichi as ff
import matplotlib.pyplot as plt


ti.init(ti.gpu, debug = False)

dem = scb.io.load_raster('/home/bgailleton/Desktop/data/green_river_1.tif')
router = ff.flow.FlowRouter(dem.geo.nx, dem.geo.ny, dem.geo.dx, boundary_mode = 'normal', boundaries = None, lakeflow = True, stochastic_receivers = False)
router.set_z(dem.Z.ravel())
router.compute_receivers()
router.reroute_flow()
router.accumulate_constant_Q(1., area = True)



fig,ax = plt.subplots()

im=ax.imshow(router.get_Q(), cmap = 'Blues')
plt.colorbar(im, label='Discharge')
plt.show()