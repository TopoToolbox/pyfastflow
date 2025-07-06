#!/usr/bin/env python3
"""
Straightforward comparison: CUDA vs Taichi drainage accumulation
Test both before and after depression routing
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import taichi as ti
import dg2
from fastflow.unified_fields import UnifiedFlowFields
from fastflow.flow_accumulation import compute_drainage_accumulation
from fastflow.lakeflow import lakeflow
from fastflow.compute_receivers import compute_receivers
from fastflow.fillinig_topo import fill_dem
from fastflow.kernels.rcv import make_rcv
from fastflow.kernels.tree_accum_up import rcv2donor, rake_compress_accum, fuse
from fastflow.kernels.lakeflow_bg import basin_identification, _reroute_flow, unpack_full_float_index, pack_full_float_index

ti.init(arch=ti.gpu, debug = False)  # Using default ti.i32 to suppress warnings




# --- Main execution ---
nx, ny = 512, 512
N = nx * ny

print("Generating terrain...")
noise = dg2.PerlinNoiseF32(frequency=0.01, amplitude=1.0, octaves=6)
z = noise.create_noise_grid(nx, ny, 0, 0, 100, 100).as_numpy()
z = (z - z.min()) / (z.max() - z.min()) * 1000

# Boundary conditions
z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = 0.15
z = np.maximum(z, 0.01)
z[3:6,3:6] -= 5000

# plt.imshow(z)
# plt.show()



unified_fields = UnifiedFlowFields(N)
unified_fields.load_terrain(z)
unified_fields.set_boundary_edges()

make_rcv(unified_fields.z, unified_fields.res, unified_fields.N, 
                    unified_fields.boundary, unified_fields.rcv, unified_fields.W)

rcv_before_flow = unified_fields.get_receivers_2d()



is_border = ti.field(ti.u1, shape = (N))
active_basin = ti.field(ti.u1, shape = (N))

bid = ti.field(ti.i32, shape = (N))
basin_saddlenode = ti.field(ti.i32, shape = (N))
rec_2 = ti.field(ti.i32, shape = (N))
rec_3 = ti.field(ti.i32, shape = (N))

basin_saddle = ti.field(ti.i64, shape = (N))
outlet = ti.field(ti.i64, shape = (N))

saddlez = ti.field(ti.f32, shape = (N))
border_z = ti.field(ti.f32, shape = (N))
z_prime = ti.field(ti.f32, shape = (N))

debug_i = ti.field(ti.i32, shape = (N))
debug_f = ti.field(ti.f32, shape = (N))

_reroute_flow(bid, unified_fields.rcv, rec_2, rec_3, 
    unified_fields.boundary, active_basin, unified_fields.z, 
    z_prime, is_border, outlet, basin_saddle,basin_saddlenode,
     nx, ny)

# debug_f.fill(25.)
# debug_i.fill(41)
# pack_full_float_index(basin_saddle, debug_f, debug_i)
# unpack_full_float_index(outlet, debug_f, debug_i)

# fig,ax = plt.subplots(1,2)
# ax[0].imshow(unified_fields.rcv.to_numpy().reshape(ny,nx), vmax = 100)
# ax[1].imshow(rcv_before_flow.reshape(ny,nx), vmax = 100)
# plt.show()




# quit()








# Phase 2: Initialize accumulation values to unity
# Each cell starts with drainage value of 1.0 (representing itself)
unified_fields.reset_accumulation()

# Phase 3: Tree accumulation using rake-compress algorithm
# Calculate the number of iterations needed: ceil(log₂(N))
logn = int(math.ceil(math.log2(unified_fields.N)))

# Initialize working arrays for donor construction
unified_fields.ndnr.fill(0)  # Number of donors per cell
unified_fields.src.fill(0)   # Source tracking for rake-compress

# Phase 4: Build donor relationships (inverse of receiver graph)
# For each cell, record which cells drain into it
rcv2donor(unified_fields.rcv, unified_fields.dnr, unified_fields.ndnr, 
          unified_fields.N, unified_fields.res)

# Phase 5: Rake-compress iterations for tree accumulation
# Each iteration doubles the effective path length being compressed
for i in range(logn):
    rake_compress_accum(unified_fields.dnr, unified_fields.ndnr, unified_fields.p, unified_fields.src,
                       unified_fields.dnr_, unified_fields.ndnr_, unified_fields.p_, 
                       unified_fields.N, i)

# Phase 6: Final fuse step to consolidate results
# Merge accumulated values from working arrays
fuse(unified_fields.p, unified_fields.src, unified_fields.p_, unified_fields.N, logn + 1)

# Phase 7: Copy result to external output array
# Convert from internal 1D storage to external 2D format
drain = unified_fields.get_drainage_2d()

# plt.imshow(bid.to_numpy().reshape(ny,nx))
# # plt.imshow(drain)
# plt.show()

# quit()

fig,ax = plt.subplots(1,5)
ax[0].imshow(rcv_before_flow)
ax[1].imshow(unified_fields.get_receivers_2d())
ax[2].imshow(unified_fields.get_receivers_2d() - rcv_before_flow)
ax[3].imshow(drain, cmap = 'Blues')
ax[4].imshow(bid.to_numpy().reshape(ny,nx), cmap = 'jet')


# print(np.unique(unified_fields.b.to_numpy()))
plt.show()
# fill_dem(unified_fields.z, nx, ny, N_iterations=None)
# plt.imshow(unified_fields.z.to_numpy().reshape(ny,nx), cmap = 'terrain')
# plt.show()


# drain = np.zeros_like(z)
# compute_drainage_accumulation(unified_fields, drain, method='deterministic')


# plt.imshow(np.log10(drain), cmap = 'Blues')
# plt.show()

# # # Depression routing and drainage  
# # print("  Running depression routing...")
# # rcv_carved_taichi = processor.route_depressions(method='carve')
# # drain_taichi_carved = processor.accumulate_flow_upward()

# # plt.imshow(drain_taichi_carved, cmap = 'Blues')
# # plt.show()
