"""
Wavefront-based Multiple Flow Direction (MFD) accumulation.

This module implements a sparse wavefront (topological sort) algorithm 
for flow accumulation using MFD. It mirrors the efficiency of CUDA 
queue-based implementations using Taichi.

Author: B.G. / Sub-agent
"""

import taichi as ti
import pyfastflow.grid.neighbourer_flat as nei
from .. import constants as cte

@ti.kernel
def compute_indegree_mfd(z: ti.template(), indegree: ti.template()):
    """
    Compute how many uphill neighbors each cell has.
    """
    for i in z:
        if nei.nodata(i):
            continue
        for k in ti.static(range(4)):
            j = nei.neighbour(i, k)
            if j != -1 and not nei.nodata(j):
                # If neighbor j is higher than i, it's a donor to i
                if z[j] > z[i]:
                    indegree[i] += 1

@ti.kernel
def wavefront_init(indegree: ti.template(), q: ti.template(), q_count: ti.template()):
    """
    Find all source nodes (indegree == 0) and add them to the initial queue.
    """
    for i in indegree:
        if nei.nodata(i):
            continue
        if indegree[i] == 0:
            pos = ti.atomic_add(q_count[None], 1)
            q[pos] = i

@ti.kernel
def wavefront_step(
    z: ti.template(),
    Q: ti.template(),
    indegree: ti.template(),
    q_curr: ti.template(),
    q_curr_size: ti.i32,
    q_next: ti.template(),
    q_next_count: ti.template()
):
    """
    Process one wavefront of nodes.
    """
    for i in range(q_curr_size):
        idx = q_curr[i]
        
        # 1. Compute total downhill slope for MFD weighting
        sum_slope = 0.0
        for k in ti.static(range(4)):
            j = nei.neighbour(idx, k)
            if j != -1 and not nei.nodata(j):
                slope = (z[idx] - z[j]) / cte.DX
                if slope > 0:
                    sum_slope += slope
        
        # 2. Distribute flow to downhill neighbors
        if sum_slope > 0:
            for k in ti.static(range(4)):
                j = nei.neighbour(idx, k)
                if j != -1 and not nei.nodata(j):
                    slope = (z[idx] - z[j]) / cte.DX
                    if slope > 0:
                        weight = slope / sum_slope
                        ti.atomic_add(Q[j], Q[idx] * weight)
                        
                        # 3. Unlock child: decrement indegree
                        # Use atomic_add with -1 because Taichi atomic_sub is for ints/floats 
                        # but often simplified to atomic_add
                        prev_deg = ti.atomic_add(indegree[j], -1)
                        if prev_deg == 1:
                            # Child has no more pending donors, add to next wavefront
                            pos = ti.atomic_add(q_next_count[None], 1)
                            q_next[pos] = j
        # If sum_slope == 0, it's a sink; flow stops here (or handled by lake routing)

def accumulate_wavefront_mfd(z_field, Q_field, nx, ny):
    """
    Host-side orchestration of the wavefront MFD accumulation.
    """
    import pyfastflow as pf
    
    # Allocate temporary fields
    indegree = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(nx * ny))
    q1 = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(nx * ny))
    q2 = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(nx * ny))
    q_count1 = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=())
    q_count2 = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=())
    
    indegree.field.fill(0)
    q_count1.field.fill(0)
    
    # 1. Initial indegree pass
    compute_indegree_mfd(z_field, indegree.field)
    
    # 2. Initialize queue
    wavefront_init(indegree.field, q1.field, q_count1.field)
    
    curr_q = q1
    next_q = q2
    curr_count = q_count1
    next_count = q_count2
    
    # 3. Loop until queue is empty
    iters = 0
    max_iters = nx * ny # Safety limit
    
    while iters < max_iters:
        next_count.field.fill(0)
        
        wavefront_step(
            z_field,
            Q_field,
            indegree.field,
            curr_q.field,
            curr_count.field[None],
            next_q.field,
            next_count.field
        )
        
        # Swap queues
        curr_q, next_q = next_q, curr_q
        curr_count, next_count = next_count, curr_count
        iters += 1

        if iters % 100 == 0:
            if curr_count.field[None] <= 0:
                break
        
    # Cleanup
    indegree.release()
    q1.release()
    q2.release()
    q_count1.release()
    q_count2.release()
