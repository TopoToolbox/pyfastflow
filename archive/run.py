#!/usr/bin/env python3
"""
Straightforward comparison: CUDA vs Taichi drainage accumulation
Test both before and after depression routing
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import taichi as ti
import dg2
from fastflow.unified_fields import UnifiedFlowFields
from fastflow.flow_accumulation import compute_drainage_accumulation
from fastflow.lakeflow import lakeflow
from fastflow.compute_receivers import compute_receivers
from fastflow.fillinig_topo import fill_dem

ti.init(arch=ti.gpu, debug = False)  # Using default ti.i32 to suppress warnings




# --- Main execution ---
nx, ny = 2048, 2048
N = nx * ny

print("Generating terrain...")
noise = dg2.PerlinNoiseF32(frequency=0.01, amplitude=1.0, octaves=6)
z = noise.create_noise_grid(nx, ny, 0, 0, 100, 100).as_numpy()
z = (z - z.min()) / (z.max() - z.min()) * 1000

# Boundary conditions
z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = 0.15
z = np.maximum(z, 0.01)



unified_fields = UnifiedFlowFields(N)
unified_fields.load_terrain(z)
unified_fields.set_boundary_edges()

fill_dem(unified_fields.z, nx, ny, N_iterations=None)
plt.imshow(unified_fields.z.to_numpy().reshape(ny,nx), cmap = 'terrain')
plt.show()


drain = np.zeros_like(z)
compute_drainage_accumulation(unified_fields, drain, method='deterministic')


plt.imshow(np.log10(drain), cmap = 'Blues')
plt.show()

# # Depression routing and drainage  
# print("  Running depression routing...")
# rcv_carved_taichi = processor.route_depressions(method='carve')
# drain_taichi_carved = processor.accumulate_flow_upward()

# plt.imshow(drain_taichi_carved, cmap = 'Blues')
# plt.show()
