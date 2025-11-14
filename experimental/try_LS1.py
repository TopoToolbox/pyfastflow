import LS
import matplotlib.pyplot as plt
import numpy as np
import scabbard as scb
import taichi as ti
import time
ti.init(ti.gpu, debug=False)

dem = scb.io.load_raster("/home/bgailleton/Desktop/data/green_river_1.tif")
# dem = scb.io.load_raster('/home/bgailleton/Desktop/data/NZ/archive/points_v2_6.tif')

nx, ny = dem.geo.nx, dem.geo.ny
LS.NX = nx
LS.NY = ny

thw = np.zeros((ny, nx), dtype=np.float32)
# thw[800:820,100:120] = 1.
# thw[500:510, 600:610] = 1.0
thw[400:450,250:300] += .1
# thw += np.random.rand(ny,nx) * 1e-1

elev = ti.field(ti.f32, shape=(ny, nx))
elev.from_numpy(dem.Z)
qx = ti.field(ti.f32, shape=(ny, nx))
qy = ti.field(ti.f32, shape=(ny, nx))
hw = ti.field(ti.f32, shape=(ny, nx))
qx.fill(0.0)
qy.fill(0.0)
hw.from_numpy(thw)

flow_timestep = 1e-3

fig, ax = plt.subplots()
im = ax.imshow(hw.to_numpy(), cmap="Blues", vmax=1.0)
fig.show()

it = 0
st = time.perf_counter()
while True:
    it += 1
    LS.flow_route(hw, elev, qy, qx, flow_timestep)
    LS.depth_update(hw, elev, qy, qx, flow_timestep)

    if it % 10000 == 0:
        print('took', time.perf_counter() - st)
        im.set_data(hw.to_numpy())
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.01)
        st = time.perf_counter()
