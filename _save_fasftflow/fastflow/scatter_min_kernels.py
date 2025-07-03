"""
Scatter-min atomic operation kernels - 1:1 CUDA port with proper Taichi structure
All kernels declared globally with template() arguments
"""
import taichi as ti
import math


# ============================================================================
# SCATTER-MIN ATOMIC KERNELS (1:1 CUDA PORT)
# ============================================================================

@ti.kernel
def fill_array_with_value(array: ti.template(), value: ti.i64, n: int):
    """1:1 port of CUDA array filling kernel"""
    for tid in range(n):
        array[tid] = value


@ti.kernel
def scatter_min_atomic(z: ti.template(), basin: ti.template(), minh: ti.template(), n: int):
    """1:1 port of CUDA scatter_min atomic operation kernel"""
    max_tid_processed: ti.i64 = 0
    for tid in range(n):
        if tid < n and basin[tid] > 0:
            basin_idx = basin[tid]
            ti.atomic_min(minh[basin_idx], z[tid])
            if tid > max_tid_processed:
                max_tid_processed = tid
        if tid == n - 1:  # Last iteration
            print(f"  DEBUG: scatter_min processed up to tid={max_tid_processed}, n={n}")


@ti.kernel
def scatter_argmin_atomic(z: ti.template(), basin: ti.template(), minh: ti.template(), 
                         argminh: ti.template(), argbasin: ti.template(), nbasin: ti.template(), n: int):
    """1:1 port of CUDA scatter_argmin atomic operation kernel"""
    max_tid_processed: ti.i64 = 0
    for tid in range(n):
        basin_idx = basin[tid]
        if tid < n and z[tid] <= minh[basin_idx] and nbasin[tid] == argbasin[basin_idx]:
            argminh[basin_idx] = tid
            if tid > max_tid_processed:
                max_tid_processed = tid
        if tid == n - 1:  # Last iteration
            print(f"  DEBUG: scatter_argmin processed up to tid={max_tid_processed}, n={n}")


@ti.kernel
def scatter_argbasin_atomic(zz: ti.template(), z: ti.template(), basin: ti.template(), minh: ti.template(), 
                           argbasin: ti.template(), nbasin: ti.template(), res: int, n: int):
    """1:1 port of CUDA scatter_argbasin atomic operation kernel"""
    for tid in range(n):
        basin_idx = basin[tid]
        if tid < n and basin_idx > 0 and z[tid] <= minh[basin_idx]:
            bn = 1000000000
            bt = basin_idx

            # EXACT CUDA BEHAVIOR: bounds check to match CUDA's actual memory access
            if tid - res >= 0:
                neighbor_basin = basin[tid-res]
                if neighbor_basin != bt and z[tid-res] <= minh[basin_idx] and neighbor_basin < bn:
                    bn = neighbor_basin
            if tid + res < n:
                neighbor_basin = basin[tid+res]
                if neighbor_basin != bt and z[tid+res] <= minh[basin_idx] and neighbor_basin < bn:
                    bn = neighbor_basin
            if tid - 1 >= 0:
                neighbor_basin = basin[tid-1]
                if neighbor_basin != bt and z[tid-1] <= minh[basin_idx] and neighbor_basin < bn:
                    bn = neighbor_basin
            if tid + 1 < n:
                neighbor_basin = basin[tid+1]
                if neighbor_basin != bt and z[tid+1] <= minh[basin_idx] and neighbor_basin < bn:
                    bn = neighbor_basin

            nbasin[tid] = bn
            ti.atomic_min(argbasin[basin_idx], bn)


# ============================================================================
# SCATTER-MIN INTERFACE FUNCTION
# ============================================================================

def run_scatter_min_atomic(zz: ti.template(), z: ti.template(), basin: ti.template(), rcv: ti.template(), 
                          minh: ti.template(), argminh: ti.template(), argbasin_: ti.template(), nbasin_: ti.template()):
    """
    Run complete scatter-min atomic operation sequence
    1:1 port of CUDA scatter-min with proper Taichi memory management
    """
    n = z.shape[0]
    m = minh.shape[0]
    res = int(math.sqrt(float(n)))
    
    # Initialize argbasin array
    fill_array_with_value(argbasin_, 1000000000, m)

    # Execute scatter-min sequence
    scatter_min_atomic(z, basin, minh, n)
    scatter_argbasin_atomic(zz, z, basin, minh, argbasin_, nbasin_, res, n)
    scatter_argmin_atomic(z, basin, minh, argminh, argbasin_, nbasin_, n)

    return argminh