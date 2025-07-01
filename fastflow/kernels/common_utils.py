"""
Common utility kernels for array operations
"""
import taichi as ti

@ti.kernel
def swap_arrays(array1: ti.template(), array2: ti.template(), N: int):
    """
    Swap contents of two arrays of the same type and size
    After this operation: array1 contains original array2, array2 contains original array1
    """
    for i in range(ti.i32(N)):
        temp = array1[i]
        array1[i] = array2[i]
        array2[i] = temp

@ti.kernel
def copy_array(src: ti.template(), dst: ti.template(), N: int):
    """
    Copy contents from src to dst
    """
    for i in range(ti.i32(N)):
        dst[i] = src[i]

@ti.kernel
def init_lakeflow_iteration(N: int, rcv: ti.template(), rcv_: ti.template(), 
                           W: ti.template(), W_: ti.template(),
                           basin_edgez: ti.template(), bignum: float, 
                           reverse_path: ti.template(), minh: ti.template()):
    """
    Initialize arrays for each lakeflow iteration (equivalent to CUDA init kernel)
    """
    for i in rcv:
        rcv_[i] = rcv[i]
        W_[i] = W[i]
        basin_edgez[i] = bignum
        reverse_path[i] = 0.0
        minh[i] = 1e10