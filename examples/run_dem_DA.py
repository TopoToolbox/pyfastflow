import taichi as ti
import numpy as np
import scabbard as scb
import pyfastflow as pf
import matplotlib.pyplot as plt
import math
import time
import fastscapelib as fs

ti.init(ti.gpu, debug = False)

dem = scb.io.load_raster('/home/bgailleton/Desktop/data/green_river_0.5.tif')
nxy = dem.geo.nx * dem.geo.ny
nx,ny = dem.geo.nx , dem.geo.ny

# RANDOM LANDSCAPE TO MAXIMIZE DEPRESSIONS
dem.Z = np.random.rand(nxy)

router = pf.flow.FlowRouter(dem.geo.nx, dem.geo.ny, dem.geo.dx, boundary_mode = 'normal', boundaries = None, lakeflow = True, stochastic_receivers = False)
router.set_z(dem.Z.ravel())

router.compute_receivers()
router.reroute_flow()
router.fill_z(epsilon=1e-3)
# router.accumulate_constant_Q(1., area = True)

st = time.time()
for i in range(10):
	# print(i)
	router.compute_receivers()
	router.reroute_flow()
	router.fill_z(epsilon=1e-3)
	router.ndonors.fill(0)
	router.Q.fill(dem.geo.dx**2)
	pf.flow.ndon_MFD(router.z, router.ndonors)
	keep = True
	it=0
	while(keep):
		it+=1
		keep = pf.flow.iteration_accumulate_flow_MFD(router.Q, router.z, router.ndonors)
	ti.sync()
print('Timing full MFD GPU:', (time.time() - st)/10)


grid = fs.RasterGrid([dem.geo.ny, dem.geo.nx], [dem.geo.dx, dem.geo.dx], fs.NodeStatus.FIXED_VALUE)
graph = fs.FlowGraph(
    # grid, [fs.MultiFlowRouter(1.0)]
    grid, [fs.MultiFlowRouter(1.0),fs.PFloodSinkResolver()]
)

st = time.time()

for i in range(10):
	# print(i)
	graph.update_routes(dem.Z)
	drainage_area = graph.accumulate(dem.geo.dx**2)
print('Timing full MDF CPU:', (time.time() - st)/10)


st = time.time()
for i in range(100):
	router.compute_receivers()
	router.reroute_flow()
	router.fill_z(epsilon=1e-3)
	router.accumulate_constant_Q(1., area = True)
	ti.sync()
print('Timing full SFD GPU:', (time.time() - st)/100)

grid = fs.RasterGrid([dem.geo.ny, dem.geo.nx], [dem.geo.dx, dem.geo.dx], fs.NodeStatus.FIXED_VALUE)
graph = fs.FlowGraph(
    # grid, [fs.MultiFlowRouter(1.0)]
    grid, [fs.SingleFlowRouter(), fs.MSTSinkResolver()]
)

st = time.time()

for i in range(10):
	# print(i)
	graph.update_routes(dem.Z)
	drainage_area = graph.accumulate(dem.geo.dx**2)
print('Timing full SFD CPU:', (time.time() - st)/10)

# no dep
st = time.time()
for i in range(10):
	router.ndonors.fill(0)
	router.Q.fill(dem.geo.dx**2)
	pf.flow.ndon_MFD(router.z, router.ndonors)
	keep = True
	it=0
	while(keep):
		it+=1
		keep = pf.flow.iteration_accumulate_flow_MFD(router.Q, router.z, router.ndonors)
	ti.sync()
print('Timing nodep MFD GPU:', (time.time() - st)/10)


grid = fs.RasterGrid([dem.geo.ny, dem.geo.nx], [dem.geo.dx, dem.geo.dx], fs.NodeStatus.FIXED_VALUE)
graph = fs.FlowGraph(
    # grid, [fs.MultiFlowRouter(1.0)]
    grid, [fs.MultiFlowRouter(1.0)]
)

st = time.time()

for i in range(10):
	# print(i)
	graph.update_routes(dem.Z)
	drainage_area = graph.accumulate(dem.geo.dx**2)
print('Timing nodep MDF CPU:', (time.time() - st)/10)


st = time.time()
for i in range(100):
	router.compute_receivers()
	router.accumulate_constant_Q(1., area = True)
	ti.sync()
print('Timing nodep SFD GPU:', (time.time() - st)/100)

grid = fs.RasterGrid([dem.geo.ny, dem.geo.nx], [dem.geo.dx, dem.geo.dx], fs.NodeStatus.FIXED_VALUE)
graph = fs.FlowGraph(
    # grid, [fs.MultiFlowRouter(1.0)]
    grid, [fs.SingleFlowRouter()]
)

st = time.time()

for i in range(10):
	# print(i)
	graph.update_routes(dem.Z)
	drainage_area = graph.accumulate(dem.geo.dx**2)
print('Timing nodep SFD CPU:', (time.time() - st)/10)


fig,ax = plt.subplots()

# im=ax.imshow(router.ndonors.to_numpy().reshape(dem.rshp), cmap = 'Blues')
im=ax.imshow(router.get_Q(), cmap = 'Blues')
# im=ax.imshow(drainage_area, cmap = 'Blues')
plt.colorbar(im, label='Discharge')
plt.show()