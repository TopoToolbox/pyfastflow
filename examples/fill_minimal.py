import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import topotoolbox as ttb

import pyfastflow.constants as cte
from pyfastflow import tp
from pyfastflow.flow import FlowContext
from pyfastflow.flow.runtime import fill_h_epsilon_inplace, fill_topography_inplace
from pyfastflow.grid import GridContext

ti.init(arch=ti.gpu)

def monitor_lm():
    n = flowctx.monitor_lm_z(z.field)
    print('local minima (z):', n)

dem = ttb.read_tif('/home/bgailleton/Desktop/data/Lidar_swiss/bettlach/DEM.tif')
NX, NY, DX = dem.columns, dem.rows, dem.cellsize
N = NX * NY

gridctx = GridContext(NX, NY, DX, boundary_mode="normal", topology="D8")
flowctx = FlowContext(
    gridctx,
    min_slope_mode="const",
    min_slope=1e-2,
    diagonal_partition_correction=True,
)

z = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
receivers = tp.get_tpfield(ti.i32, N)

z.field.from_numpy(dem.z.ravel().astype(np.float32))
z_orig = z.field.to_numpy().copy()

flowctx.compute_receivers(z.field, receivers.field)
fill_topography_inplace(flowctx, z.field, receivers.field)

monitor_lm()

z_filled = z.field.to_numpy()
diff = (z_filled - z_orig).reshape(NY, NX)

print('max diff', diff.max(), 'mean diff', diff.mean(), 'n>0', (diff > 0).sum())

plt.imshow(dem.hillshade(), cmap = 'gray')
plt.imshow(diff, cmap='viridis', alpha = 0.45)
plt.colorbar(label='z_filled - z_orig')
plt.title('fill_topography_step diff')
plt.show()

# --- same fill, but via h on top of untouched z ---
z2 = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
h = tp.get_tpfield(cte.FLOAT_TYPE_TI, N)
receivers_h = tp.get_tpfield(ti.i32, N)

z2.field.from_numpy(z_orig)
h.field.fill(0.0)
flowctx.compute_receivers(z2.field, receivers_h.field)
fill_h_epsilon_inplace(flowctx, z2.field, h.field, receivers_h.field)

diff_h = h.field.to_numpy().reshape(NY, NX)

print('max diff_h', diff_h.max(), 'mean diff_h', diff_h.mean(), 'n>0', (diff_h > 0).sum())
print('max |diff - diff_h|', np.abs(diff - diff_h).max())

fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
for a in ax:
    a.imshow(dem.hillshade(), cmap='gray')
im0 = ax[0].imshow(diff, cmap='viridis', alpha=0.45)
ax[0].set_title('z fill')
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(diff_h, cmap='viridis', alpha=0.45)
ax[1].set_title('h fill')
plt.colorbar(im1, ax=ax[1])
im2 = ax[2].imshow(diff - diff_h, cmap='RdBu_r', alpha=0.45)
ax[2].set_title('z fill - h fill')
plt.colorbar(im2, ax=ax[2])
plt.show()
