import taichi as ti
import math

@ti.kernel
def fillArray(array: ti.template(), value: ti.i64, n: int):
    for tid in range(n):
        array[tid] = value


@ti.kernel
def scatter_min_atomic(z: ti.template(), basin: ti.template(), minh: ti.template(), n: int):
    for tid in range(n):
        if tid < n and basin[tid] > 0:
            ti.atomic_min(minh[basin[tid]], z[tid])


@ti.kernel
def scatter_argmin_atomic(z: ti.template(), basin: ti.template(), minh: ti.template(), argminh: ti.template(), argbasin: ti.template(), nbasin: ti.template(), n: int):
    for tid in range(n):
        if tid < n and z[tid] <= minh[basin[tid]] and nbasin[tid] == argbasin[basin[tid]]:
            argminh[basin[tid]] = tid


@ti.kernel
def scatter_argbasin_atomic(zz: ti.template(), z: ti.template(), basin: ti.template(), minh: ti.template(), argbasin: ti.template(), nbasin: ti.template(), res: int, n: int):
    for tid in range(n):
        if tid < n and basin[tid] > 0 and z[tid] <= minh[basin[tid]]:
            bn = ti.i64(1000000000)  # EXACT CUDA: matches the 1 billion value used elsewhere
            bt = basin[tid]

            # EXACT CUDA BEHAVIOR: bounds check to match CUDA's actual memory access
            if tid - res >= 0 and basin[tid-res] != bt and z[tid-res] <= minh[basin[tid]] and basin[tid-res] < bn:
                bn = basin[tid-res]
            if tid + res < n and basin[tid+res] != bt and z[tid+res] <= minh[basin[tid]] and basin[tid+res] < bn:
                bn = basin[tid+res]
            if tid - 1 >= 0 and basin[tid-1] != bt and z[tid-1] <= minh[basin[tid]] and basin[tid-1] < bn:
                bn = basin[tid-1]
            if tid + 1 < n and basin[tid+1] != bt and z[tid+1] <= minh[basin[tid]] and basin[tid+1] < bn:
                bn = basin[tid+1]

            nbasin[tid] = bn
            ti.atomic_min(argbasin[basin[tid]], bn)


def declare_flow_cuda_scatter_min_atomic(n: int, m: int):
    argbasin_ = ti.field(ti.i64, shape=m)  # EXACT CUDA: use i64
    nbasin_ = ti.field(ti.i64, shape=n)    # EXACT CUDA: use i64
    return argbasin_, nbasin_

def run_flow_cuda_scatter_min_atomic(zz: ti.template(), z: ti.template(), basin: ti.template(), rcv: ti.template(), minh: ti.template(), argminh: ti.template(), argbasin_: ti.template(), nbasin_: ti.template()):
    n = z.shape[0]
    m = minh.shape[0]
    res = int(math.sqrt(float(n)))
    
    fillArray(argbasin_, 1000000000, m)

    scatter_min_atomic(z, basin, minh, n)
    scatter_argbasin_atomic(zz, z, basin, minh, argbasin_, nbasin_, res, n)
    scatter_argmin_atomic(z, basin, minh, argminh, argbasin_, nbasin_, n)

    return argminh