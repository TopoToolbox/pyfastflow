"""
Efficient GPU parallel inclusive scan for Taichi
Work-efficient up-sweep/down-sweep algorithm (Blelloch scan)
O(n) work complexity, O(log n) depth
"""
import taichi as ti
import math

@ti.kernel
def upsweep_step(data: ti.template(), n: int, stride: int):
    """Up-sweep phase step"""
    for i in range(n):
        if (i + 1) % (stride * 2) == 0:
            data[i] += data[i - stride]

@ti.kernel
def downsweep_step(data: ti.template(), n: int, stride: int):
    """Down-sweep phase step"""
    for i in range(n):
        if (i + 1) % (stride * 2) == 0:
            temp = data[i - stride]
            data[i - stride] = data[i]
            data[i] += temp

@ti.kernel
def copy_input_to_work(src: ti.template(), dst: ti.template(), n: int, work_size: int):
    """Copy input to working array and pad with zeros"""
    for i in range(work_size):
        if i < n:
            dst[i] = src[i]
        else:
            dst[i] = 0

@ti.kernel
def set_zero(data: ti.template(), index: int):
    """Set specific index to zero"""
    data[index] = 0

@ti.kernel
def make_inclusive_and_copy(input_arr: ti.template(), work_data: ti.template(), output_arr: ti.template(), n: int):
    """Convert exclusive scan to inclusive and copy result"""
    for i in range(n):
        if i == 0:
            output_arr[i] = input_arr[i]
        else:
            output_arr[i] = work_data[i] + input_arr[i]

def inclusive_scan(input_arr: ti.template(), output_arr: ti.template(), work_arr: ti.template(), n: int):
    """
    Work-efficient parallel inclusive scan
    Requires work_arr to be at least next_power_of_2(n) in size
    """
    # Find next power of 2
    next_pow2 = 1
    while next_pow2 < n:
        next_pow2 *= 2
    
    # Copy input data to work array
    copy_input_to_work(input_arr, work_arr, n, next_pow2)
    
    # Up-sweep phase (build sum tree)
    stride = 1
    while stride < next_pow2:
        upsweep_step(work_arr, next_pow2, stride)
        stride *= 2
    
    # Set root to zero for exclusive scan base
    set_zero(work_arr, next_pow2 - 1)
    
    # Down-sweep phase (traverse down tree)
    stride = next_pow2 // 2
    while stride > 0:
        downsweep_step(work_arr, next_pow2, stride)
        stride //= 2
    
    # Convert to inclusive scan and copy result
    make_inclusive_and_copy(input_arr, work_arr, output_arr, n)