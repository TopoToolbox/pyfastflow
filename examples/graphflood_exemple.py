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
router = pf.flow.FlowRouter(dem.geo.nx, dem.geo.ny, dem.geo.dx, boundary_mode = 'normal', boundaries = None, lakeflow = True, stochastic_receivers = True)
router.set_z(dem.Z.ravel())

gf = pf.flood.Flooder(router, precipitation_rates = 50e-3/3600, manning=0.033, edge_slope = 0.01, dt_hydro = 1e-2)

# gf.run_graphflood(1000)
hw = np.zeros_like(dem.Z)
hw[500:510, 600:610] = 1.

# gf.h.from_numpy(hw.ravel())
gf.set_h(hw.ravel())

fig,ax = plt.subplots()

# im = ax.imshow(gf.get_dh(), cmap = 'RdBu_r', vmin = -1e-4,vmax = 1e-4)
# im = ax.imshow(gf.get_Q(), cmap = 'Blues', vmin = 0.,vmax = .5)
im = ax.imshow(gf.get_h(), cmap = 'Blues', vmin = 0.,vmax = 1.)
# im = ax.imshow(router.get_Z(), cmap = 'terrain', vmin=1300, vmax=1320)
# im = ax.imshow(router.get_Z(), cmap = 'RdBu_r', vmin=-1e-3, vmax=1e-3)
# im = ax.imshow(router.get_Z()-dem.Z, cmap = 'RdBu_r', vmin=-1., vmax=1.)
fig.show()
# plt.show()


while(True):

	# gf.run_graphflood(100)
	gf.run_LS(10000)
	im.set_data(gf.get_h())
	# im.set_data(gf.get_qx())
	# im.set_data((router.get_Z())-dem.Z)
	# im.set_data((router.get_Z())-gf.get_h())
	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.01)

# gf.run_LS(1000000)

# plt.imshow(gf.get_h(), cmap = 'Blues', vmax = 1.)
# plt.show()