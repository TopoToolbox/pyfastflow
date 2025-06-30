#!/usr/bin/env python3
"""
Straightforward comparison: CUDA vs Taichi drainage accumulation
Test both before and after depression routing
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import taichi as ti
from fastflow.flow_api import FastFlowProcessor

ti.init(arch=ti.gpu, debug = False)  # Using default ti.i32 to suppress warnings



# Main terrain
# Create test terrain with depression
res = 512
np.random.seed(42)
x = np.linspace(-2, 2, res)
y = np.linspace(-2, 2, res)
X, Y = np.meshgrid(x, y)

z = np.zeros_like(X, dtype=np.float32)
z += 2.0 * np.exp(-(X**2 + Y**2) * 0.4)  # Central hill
z -= 0.8 * np.exp(-((X-0.5)**2 + (Y+0.8)**2) * 4.0)  # Depression
z += 0.3 * (X * 0.3 + Y * 0.4)  # Regional slope
z += 0.1 * np.random.randn(res, res)  # Noise

# Boundary conditions
z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = 0.15
z = np.maximum(z, 0.01)

plt.imshow(z, cmap = 'terrain')
plt.show()

processor = FastFlowProcessor(res)
processor.setup_terrain(z)

# Basic receivers and drainage
rcv_taichi = processor.compute_receivers_deterministic()
rcv_original = rcv_taichi.copy()  # Capture BEFORE accumulate_flow_upward
drain_taichi_basic = processor.accumulate_flow_upward()

plt.imshow(drain_taichi_basic, cmap = 'Blues')
plt.show()

# Depression routing and drainage  
print("  Running depression routing...")
rcv_carved_taichi = processor.route_depressions(method='carve')
drain_taichi_carved = processor.accumulate_flow_upward()

plt.imshow(drain_taichi_carved, cmap = 'Blues')
plt.show()
