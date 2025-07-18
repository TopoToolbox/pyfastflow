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
import time
import fastscapelib as fs
import fastflow_taichi as ff
import dg2

ti.init(arch=ti.gpu, cpu_max_num_threads = 1, debug = False)  # Using default ti.i32 to suppress warnings


# --- Main execution ---
nx, ny = 1024, 1024
N = nx * ny
dx = 100.
# np.random.seed(42)

print("Generating terrain...")
noise =  dg2.PerlinNoiseF32(frequency=0.01, amplitude=1.0, octaves=6)
z_np     =  noise.create_noise_grid(nx, ny, 0, 0, 100, 100).as_numpy()
z_np    +=  np.random.rand(ny,nx)
z_np = (z_np - z_np.min()) / (z_np.max() - z_np.min()) * 1000

# Boundary conditions
z_np[0, :] = z_np[-1, :] = z_np[:, 0] = z_np[:, -1] = 0.15
z_np = np.maximum(z_np, 0.01)
# z_np[:,[0,-1]] -= 50
z_np    +=  np.random.rand(ny,nx)
# z_np[200:300,200:400] = 5


# z_np    =  np.random.rand(ny,nx)


# grid = fs.RasterGrid([ny, nx], [dx,dx], [fs.NodeStatus.FIXED_VALUE, fs.NodeStatus.FIXED_VALUE,fs.NodeStatus.FIXED_VALUE,fs.NodeStatus.FIXED_VALUE])
# flow_graph = fs.FlowGraph(grid, [fs.SingleFlowRouter(), fs.MSTSinkResolver()])

# print("Fastscapelib routing:")
# st = time.time() 
# for i in range(100):
#     flow_graph.update_routes(z_np.ravel())
# print(f'took {(time.time() - st)/100} - average of 100')


#Setting up constants
ff.flow.constants.NX = nx
ff.flow.constants.NY = ny
ff.flow.constants.DX = dx
ff.flow.constants.RAND_RCV = False
ff.flow.constants.BOUND_MODE = 0

if ff.flow.constants.BOUND_MODE == 3:
    BCs = np.ones((ny,nx), dtype=np.uint8)
    BCs[-1,256] = 3
    ff.flow.init_custom_boundaries(BCs.ravel())

ff.flow.initialise()

logn = math.ceil(math.log2(nx*ny))+1


# Fields for all
z = ti.field(ti.f32, shape = (nx*ny))
z.from_numpy(z_np.ravel())
Q = ti.field(ti.f32, shape = (nx*ny))
Q.fill(1.)
Q_ = ti.field(ti.f32, shape = (nx*ny))
Q_.fill(0.)
gradient = ti.field(ti.f32, shape = (nx*ny))
gradient.fill(0.)

receivers = ti.field(ti.i32, shape = (nx*ny))
src = ti.field(ti.i32, shape = (nx*ny))
ndonors = ti.field(ti.i32, shape = (nx*ny))
ndonors_ = ti.field(ti.i32, shape = (nx*ny))
donors = ti.field(ti.i32, shape = (nx*ny*4))
donors_ = ti.field(ti.i32, shape = (nx*ny*4))


# Fields for lake flow
z_prime = ti.field(ti.f32, shape = (nx*ny))

receivers_ = ti.field(ti.i32, shape = (nx*ny))
bid = ti.field(ti.i32, shape = (nx*ny))
basin_saddlenode = ti.field(ti.i32, shape = (nx*ny))

outlet = ti.field(ti.i64, shape = (nx*ny))
basin_saddle = ti.field(ti.i64, shape = (nx*ny))
receivers__ = ti.field(ti.i32, shape = (nx*ny))

is_border = ti.field(ti.u1, shape = (nx*ny))
tag = ti.field(ti.u1, shape = (nx*ny))
tag_ = ti.field(ti.u1, shape = (nx*ny))

change = ti.field(ti.u1, shape = ())

# edges = ti.field(ti.u8, shape = (nx*ny))
# Acc = ti.field(ti.f32, shape = (nx*ny))
# Acc.fill(0)
# ff.flow.flow_out_nodes(edges)
# plt.imshow(edges.to_numpy().reshape(ny,nx))
# plt.show()


# @ti.kernel
# def accumulate(A:ti.template(), receivers:ti.template()):

#     for i in A:
#         tnode = i
#         while tnode != receivers[tnode]:
#             ti.atomic_add(A[tnode], 1)
#             tnode = receivers[tnode]


print("fastflow taichi routing:")
NRUN= 100000
st = time.time() 
for i in range(NRUN):
    print(i)
    ff.flow.compute_receivers(z, receivers, gradient)
    ff.flow.reroute_flow(bid, receivers, receivers_, receivers__,
        z, z_prime, is_border, outlet, basin_saddle, 
        basin_saddlenode, tag, tag_, change, carve = True)



    # donors.fill(0)
    ndonors.fill(0)
    src.fill(0)
    Q.fill(dx*dx)
    # Q_.fill(1.)
    ff.flow.rcv2donor(receivers, donors, ndonors)

    # Phase 5: Rake-compress iterations for tree accumulation
    # Each iteration doubles the effective path length being compressed
    for i in range(logn+1):
        ff.flow.rake_compress_accum(donors, ndonors, Q, src,
                           donors_, ndonors_, Q_, i)

    # Phase 6: Final fuse step to consolidate results
    # Merge accumulated values from working arrays
    
    ff.flow.fuse(Q, src, Q_, logn)

print(f'took {(time.time() - st)/NRUN} - average of {NRUN}')

recs = receivers.to_numpy().reshape(ny,nx)
recs = np.mod(recs, nx)

plt.imshow(np.log10(Q.to_numpy()).reshape(ny,nx))
 # plt.imshow(np.log10(Acc.to_numpy()).reshape(ny,nx))
# plt.imshow(bid.to_numpy().reshape(ny,nx))
plt.show()
