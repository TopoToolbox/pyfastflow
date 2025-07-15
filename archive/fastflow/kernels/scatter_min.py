"""
1:1 port of src/cuda/core/scatter_min.cu
"""
import taichi as ti

@ti.kernel
def fillArray(array: ti.template(), value: ti.i64, n: int):
    """1:1 port of fillArray CUDA kernel"""
    for tid in range(n):
        array[tid] = value


@ti.kernel
def scatter_min_atomic(z: ti.template(), basin: ti.template(), minh: ti.template(), n: int):
    """1:1 port of scatter_min_atomic CUDA kernel"""
    for tid in range(n):
        if tid < n and basin[tid] > 0:
            ti.atomic_min(minh[ti.i32(basin[tid])], z[tid])


@ti.kernel
def scatter_argmin_atomic(z: ti.template(), basin: ti.template(), minh: ti.template(),
                         argminh: ti.template(), argbasin: ti.template(), nbasin: ti.template(), n: int):
    """1:1 port of scatter_argmin_atomic CUDA kernel"""
    for tid in range(n):
        if tid < n and z[tid] <= minh[ti.i32(basin[tid])] and nbasin[tid] == argbasin[ti.i32(basin[tid])]:
            argminh[ti.i32(basin[tid])] = tid

@ti.kernel
def scatter_argbasin_atomic(zz: ti.template(), z: ti.template(), basin: ti.template(), 
                            minh: ti.template(), argbasin: ti.template(), nbasin: ti.template(),
                            res: int, n: int):
    for tid in range(n):
        b = ti.i32(basin[tid])
        # b = ti.i32(b)
        if b > 0 and z[tid] <= minh[b]:
            bn = ti.i64(10000000000)
            
            # Top
            if tid >= res:
                bt = basin[tid - res]
                if bt != b and z[tid - res] <= minh[b] and bt < bn:
                    bn = bt
            # Bottom
            if tid + res < n:
                bt = basin[tid + res]
                if bt != b and z[tid + res] <= minh[b] and bt < bn:
                    bn = bt
            # Left
            if tid % res > 0:
                bt = basin[tid - 1]
                if bt != b and z[tid - 1] <= minh[b] and bt < bn:
                    bn = bt
            # Right
            if tid % res < res - 1:
                bt = basin[tid + 1]
                if bt != b and z[tid + 1] <= minh[b] and bt < bn:
                    bn = bt

            nbasin[tid] = bn
            ti.atomic_min(argbasin[b], bn)
