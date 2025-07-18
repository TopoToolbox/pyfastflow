import taichi as ti

@ti.func
def copy_data(from_arr: ti.template(), to_arr: ti.template(), n: int):
    """
    Copy data from one array to another
    """
    for tid in range(n):
        to_arr[tid] = from_arr[tid]

@ti.kernel
def copy_data_kernel(from_arr: ti.template(), to_arr: ti.template(), n: int):
    """
    Kernel wrapper for copy_data that can be called from Python scope
    """
    copy_data(from_arr, to_arr, n)