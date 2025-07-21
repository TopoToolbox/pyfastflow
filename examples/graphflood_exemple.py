import taichi as ti
import numpy as np
import scabbard as scb
import pyfastflow as pf
import pyfastflow.constants as cte
import matplotlib.pyplot as plt


ti.init(ti.gpu, debug = False)

dem = scb.io.load_raster('/home/bgailleton/Desktop/data/green_river_1.tif')
# dem = scb.io.load_raster('/home/bgailleton/Desktop/data/NZ/archive/points_v2_6.tif')

nx,ny=dem.geo.nx,dem.geo.ny
router = pf.flow.FlowRouter(dem.geo.nx, dem.geo.ny, dem.geo.dx, boundary_mode = 'normal', boundaries = None, lakeflow = True, stochastic_receivers = False)
router.set_z(dem.Z.ravel())

gf = pf.flood.Flooder(router, precipitation_rates = 10e-3/3600, manning=0.033, edge_slope = 1e-2)

gf.run_graphflood(1000)

plt.imshow(gf.get_h(), cmap = 'Blues', vmax = 1.)
plt.show()