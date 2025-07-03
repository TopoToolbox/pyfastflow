import taichi as ti
import math
import numpy as np

@ti.kernel
def copy_array(src: ti.template(), dst: ti.template(), n: int):
    for i in range(n):
        dst[i] = src[i]

@ti.kernel
def kogge_stone_step(input_arr: ti.template(), output_arr: ti.template(), n: int, offset: int):
    for i in range(n):
        if i >= offset:
            output_arr[i] = input_arr[i] + input_arr[i - offset]
        else:
            output_arr[i] = input_arr[i]

def kogge_stone_scan(input_arr: ti.template(), output_arr: ti.template(), temp_arr: ti.template(), n: int):
    """
    Kogge-Stone parallel scan - uses pre-allocated temp buffer from data class
    """
    # Copy input to output
    copy_array(input_arr, output_arr, n)
    
    offset = 1
    while offset < n:
        # Use ping-pong buffers
        if (offset // 1) % 2 == 1:
            kogge_stone_step(output_arr, temp_arr, n, offset)
            copy_array(temp_arr, output_arr, n)
        else:
            kogge_stone_step(output_arr, temp_arr, n, offset)
            copy_array(temp_arr, output_arr, n)
        offset *= 2

# Pre-declare kernels to avoid compilation in loops
@ti.kernel  
def scan_step_1(output_arr: ti.template(), n: int):
    for i in range(1, n):
        output_arr[i] = output_arr[i] + output_arr[i - 1]

@ti.kernel
def scan_step_2(output_arr: ti.template(), n: int):
    for i in range(2, n):
        output_arr[i] = output_arr[i] + output_arr[i - 2]

@ti.kernel 
def scan_step_4(output_arr: ti.template(), n: int):
    for i in range(4, n):
        output_arr[i] = output_arr[i] + output_arr[i - 4]

@ti.kernel
def scan_step_8(output_arr: ti.template(), n: int):
    for i in range(8, n):
        output_arr[i] = output_arr[i] + output_arr[i - 8]

@ti.kernel
def scan_step_16(output_arr: ti.template(), n: int):
    for i in range(16, n):
        output_arr[i] = output_arr[i] + output_arr[i - 16]

@ti.kernel
def scan_step_32(output_arr: ti.template(), n: int):
    for i in range(32, n):
        output_arr[i] = output_arr[i] + output_arr[i - 32]

@ti.kernel
def scan_step_64(output_arr: ti.template(), n: int):
    for i in range(64, n):
        output_arr[i] = output_arr[i] + output_arr[i - 64]

@ti.kernel
def scan_step_128(output_arr: ti.template(), n: int):
    for i in range(128, n):
        output_arr[i] = output_arr[i] + output_arr[i - 128]

@ti.kernel
def scan_step_256(output_arr: ti.template(), n: int):
    for i in range(256, n):
        output_arr[i] = output_arr[i] + output_arr[i - 256]

@ti.kernel
def scan_step_512(output_arr: ti.template(), n: int):
    for i in range(512, n):
        output_arr[i] = output_arr[i] + output_arr[i - 512]

@ti.kernel
def scan_step_1024(output_arr: ti.template(), n: int):
    for i in range(1024, n):
        output_arr[i] = output_arr[i] + output_arr[i - 1024]

@ti.kernel
def scan_step_2048(output_arr: ti.template(), n: int):
    for i in range(2048, n):
        output_arr[i] = output_arr[i] + output_arr[i - 2048]

@ti.kernel
def scan_step_4096(output_arr: ti.template(), n: int):
    for i in range(4096, n):
        output_arr[i] = output_arr[i] + output_arr[i - 4096]

@ti.kernel
def scan_step_8192(output_arr: ti.template(), n: int):
    for i in range(8192, n):
        output_arr[i] = output_arr[i] + output_arr[i - 8192]

@ti.kernel
def scan_step_16384(output_arr: ti.template(), n: int):
    for i in range(16384, n):
        output_arr[i] = output_arr[i] + output_arr[i - 16384]

@ti.kernel
def scan_step_32768(output_arr: ti.template(), n: int):
    for i in range(32768, n):
        output_arr[i] = output_arr[i] + output_arr[i - 32768]

@ti.kernel
def scan_step_65536(output_arr: ti.template(), n: int):
    for i in range(65536, n):
        output_arr[i] = output_arr[i] + output_arr[i - 65536]

@ti.kernel
def scan_step_131072(output_arr: ti.template(), n: int):
    for i in range(131072, n):
        output_arr[i] = output_arr[i] + output_arr[i - 131072]

@ti.kernel
def scan_step_262144(output_arr: ti.template(), n: int):
    for i in range(262144, n):
        output_arr[i] = output_arr[i] + output_arr[i - 262144]

def sequential_scan(input_arr: ti.template(), output_arr: ti.template(), n: int):
    """Sequential inclusive scan - executed on CPU for correctness"""
    # Copy to CPU, compute scan, copy back
    input_np = input_arr.to_numpy()
    scan_result = np.cumsum(input_np)
    
    # Handle size mismatch: output might be larger than input
    output_size = output_arr.shape[0]
    if output_size >= n:
        # Copy scan result to the first n positions
        for i in range(n):
            output_arr[i] = int(scan_result[i])
        # Fill remaining positions with the final cumulative value if needed
        if output_size > n:
            final_val = int(scan_result[-1])
            for i in range(n, output_size):
                output_arr[i] = final_val
    else:
        # Output is smaller than input - truncate
        truncated_result = scan_result[:output_size].astype(np.int64)
        output_arr.from_numpy(truncated_result)

# Export the main function
def inclusive_scan(input_arr: ti.template(), output_arr: ti.template(), n: int):
    """
    Reliable inclusive scan implementation
    For now using sequential scan to ensure correctness
    """
    sequential_scan(input_arr, output_arr, n)