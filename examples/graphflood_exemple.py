import taichi as ti
import numpy as np
import topotoolbox as tt3
import matplotlib.pyplot as plt
import pyfastflow as pf
import pyfastflow.constants as cte
import matplotlib.pyplot as plt

import random

def add_circle(array, X, N, center_x = None, center_y = None):
    """
    Add value X on a circle of radius N pixels at a random location in a 2D array.
    
    Parameters:
    array: 2D numpy array
    X: value to add to the circle pixels
    N: radius of the circle in pixels
    
    Returns:
    Modified array with circle added
    """
    height, width = array.shape
    
    # Choose random center point, ensuring circle fits within bounds
    if(center_x is None):
	    center_y = random.randint(N, height - N - 1)
	    center_x = random.randint(N, width - N - 1)
	    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Calculate distance from center for each pixel
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Create circle mask (pixels within radius N)
    circle_mask = distances <= N
    
    # Add X to pixels within the circle
    array[circle_mask] += X
    
    return array

@ti.kernel
def yolo():

	gf.h[500*nx+500] += 2e-2
	gf.h[800*nx+500] += 2e-2
	gf.h[800*nx+100] += 2e-2
	gf.h[800*nx+800] += 2e-2


ti.init(ti.gpu, debug = False)

# dem = scb.io.load_raster('/home/bgailleton/Desktop/data/green_river_1.tif')
# dem = scb.io.load_raster('/home/bgailleton/Desktop/data/NZ/archive/points_v2_6.tif')

dem = tt3.read_tif('/home/bgailleton/Desktop/data/green_river_1.tif')
dem.info()

nx, ny = dem.columns, dem.rows
dx = dem.cellsize

router = pf.flow.FlowRouter(nx, ny, dx, boundary_mode = 'normal', boundaries = None, lakeflow = True, stochastic_receivers = True)
router.set_z(dem.z.ravel())

gf = pf.flood.Flooder(router, precipitation_rates = 50e-3/3600, manning=0.033, edge_slope = 0.01, dt_hydro = 5e-3, dt_hydro_ls = 1e-2)

# gf.run_graphflood(1000)
# hw = np.zeros_like(dem.z)
# hw += 0.001
# radius = 30
# water = 0.1
# add_circle(hw,water,radius, 650,830)
# add_circle(hw,water,radius, 377,850)
# add_circle(hw,water,radius, 144,940)
# add_circle(hw,water,radius, 55,750)
# add_circle(hw,water,radius, 744,826)
# # gf.h.from_numpy(hw.ravel())
# gf.set_h(hw.ravel())

fig,ax = plt.subplots(figsize = (16,16))
ax.imshow(dem.hillshade(), cmap = 'gray')
th = gf.get_h()
th[th<0.1] = np.nan


# im = ax.imshow(gf.get_dh(), cmap = 'RdBu_r', vmin = -1e-4,vmax = 1e-4)
# im = ax.imshow(gf.get_Q(), cmap = 'Blues', vmin = 0.,vmax = .5)
im = ax.imshow(th, cmap = 'Blues', vmin = 0.,vmax = 0.6)
# im = ax.imshow(router.get_Z(), cmap = 'terrain', vmin=1300, vmax=1320)
# im = ax.imshow(router.get_Z(), cmap = 'RdBu_r', vmin=-1e-3, vmax=1e-3)
# im = ax.imshow(router.get_Z()-dem.z, cmap = 'RdBu_r', vmin=-1., vmax=1.)
fig.show()
# plt.show()


it = 0
while(True):
	it+=1
	if(it<0):
		ax.set_title('Graphlood'+str(it))
		gf.run_graphflood(100)
	else:
		ax.set_title('lisflood'+str(it))
		gf.run_LS(10000, input_mode = 'constant_prec')
	th = gf.get_h()
	th[th<0.01] = np.nan

	im.set_data(th)

	# im.set_data(gf.get_qx())
	# im.set_data((router.get_Z())-dem.z)
	# im.set_data((router.get_Z())-gf.get_h())
	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.01)

# gf.run_LS(1000000)

# plt.imshow(gf.get_h(), cmap = 'Blues', vmax = 1.)
# plt.show()