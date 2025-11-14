import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import pyfastflow as pff
import topotoolbox as ttb
import utils as ut
import time

ti.init(ti.cuda)

dem = ttb.read_tif(f'name_of_dem.tif')
nx,ny = dem.columns,dem.rows
dx = dem.cellsize

bc = np.ones_like(dem.z)
bc[[0,-1],:] = 0
bc[[0,-1],:] = 0
bc = bc.astype(np.uint8)

grid = pff.grid.Grid(nx,ny,dx, dem.z,boundary_mode = 'custom', boundaries = bc.ravel())
flow = pff.flow.FlowRouter(grid)
flood = pff.flood.GGF_Object(flow)

# fills a first time with full transport
flood.fill_lakes_full(compute_Qsfd = True)

# runs 10k local iterations
flood.run_graphflood(iterations=10000, dt=0.001, temporal_dumping=0.1, prec2D=100e-3/3600.)


# Avaiable flood function to get results:
# set_h -> set initial flow depth
# get_h -> get flow depth
# get_Q -> Qi
# get_Qo -> Qo
# get_sw -> hydraulic slope
# get_tau -> basal shear stress