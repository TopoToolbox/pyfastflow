"""
MFD Flow Accumulation via Power Iteration (Jacobi Method) with Convergence Check.

Solves Q = S + WQ by iteratively computing Q_{k+1} = S + W * Q_k.
Includes a light-weight GPU-based convergence check.

Author: B.G. / Sub-agent
"""

import taichi as ti
import pyfastflow.grid.neighbourer_flat as nei
from .. import constants as cte

@ti.kernel
def compute_initial_weights(z: ti.template(), w: ti.template(), wsum: ti.template()):
    """
    Compute initial MFD weights to 4 neighbors.
    w[i, k] is the fraction of flow i sends to neighbor k.
    Directions: 0:top, 1:left, 2:right, 3:bottom
    """
    for i in z:
        if nei.nodata(i):
            continue
        
        sum_s = 0.0
        zi = z[i]
        for k in ti.static(range(4)):
            j = nei.neighbour(i, k)
            if j != -1 and not nei.nodata(j):
                slope = (zi - z[j]) / cte.DX
                if slope > 0:
                    w[i, k] = slope
                    sum_s += slope
                else:
                    w[i, k] = 0.0
            else:
                w[i, k] = 0.0
        
        wsum[i] = sum_s
        if sum_s > 0:
            for k in ti.static(range(4)):
                w[i, k] /= sum_s

@ti.kernel
def mfd_power_iteration(
    S: ti.template(),
    Q: ti.template(),
    W: ti.template(),
    Q_next: ti.template()
):
    """
    Jacobi Step: Q_next = S + W * Q_curr
    """
    for i in S:
        if nei.nodata(i):
            continue
            
        acc = S[i]
        for k in ti.static(range(4)):
            j = nei.neighbour(i, k) 
            if j != -1 and not nei.nodata(j):
                opp_k = 3 - k 
                wj = W[j, opp_k]
                if wj > 0:
                    acc += wj * Q[j]
        Q_next[i] = acc

@ti.kernel
def check_convergence(Q1: ti.template(), Q2: ti.template(), eps: ti.template()):
    """
    Compute max absolute difference between two flow fields on GPU.
    """
    eps[None] = 0.0
    for i in Q1:
        if nei.nodata(i):
            continue
        diff = ti.abs(Q1[i] - Q2[i])
        ti.atomic_max(eps[None], diff)

def accumulate_power_iteration_mfd(z_field, Q_field, nx, ny, max_iterations=2000, tol=1e-6, check_interval=20):
    """
    MFD accumulation using Jacobi Power Iteration with convergence check.
    """
    import pyfastflow as pf
    
    weights = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(nx * ny, 4))
    wsum = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(nx * ny))
    compute_initial_weights(z_field, weights.field, wsum.field)
    
    S = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(nx * ny))
    S.field.copy_from(Q_field)
    
    Q_tmp = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(nx * ny))
    Q_tmp.field.copy_from(Q_field)
    
    eps = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=())
    
    converged = False
    for i in range(max_iterations):
        mfd_power_iteration(S.field, Q_tmp.field, weights.field, Q_field)
        
        # Periodic convergence check
        if i > 0 and i % check_interval == 0:
            check_convergence(Q_field, Q_tmp.field, eps.field)
            if eps.field[None] < tol:
                converged = True
                break
        
        Q_tmp.field.copy_from(Q_field)
        
    weights.release()
    wsum.release()
    S.release()
    Q_tmp.release()
    eps.release()
    return converged
