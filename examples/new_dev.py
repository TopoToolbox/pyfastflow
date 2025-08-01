import pyfastflow as pf
import matplotlib.pyplot as plt
import numpy as np
import time
import taichi as ti

ti.init(ti.gpu)


grid = pf.io.raster_to_grid('/home/bgailleton/Desktop/data/green_river_1.tif')


ff = pf.flow.FlowRouter(grid)

st = time.time()
for i in range(100):
	ff.compute_receivers()
	ff.reroute_flow()
	ff.accumulate_constant_Q(1.)
ti.sync()
print(f"100 iterations of rec computation + local minima handling + flow acc Done in {time.time() - st} s")

gf = pf.flood.Flooder(ff, precipitation_rates = 1e-4, dt_hydro = 5e-3)

st = time.time()
gf.run_graphflood(N=200)
print(f"200 iterations of graphflood (all flow + extras) done in {time.time() - st} s")
st = time.time()
gf.run_LS(N=200)
print(f"200 iterations of lisflood (simpler) done in {time.time() - st} s")


Q = ff.get_Q()
h = gf.get_h()
h[h<1e-2] = np.nan

hs = grid.hillshade(multidirectional = True)

plt.imshow(hs, cmap = "gray")
plt.imshow(h, cmap = 'Blues', vmax = 1.)

plt.show()