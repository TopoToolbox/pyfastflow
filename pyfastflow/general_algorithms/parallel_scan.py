"""
Parallel Scan Implementation

This module implements the work-efficient parallel inclusive scan algorithm
(also known as parallel prefix sum) using the Blelloch scan approach.
Optimized for GPU execution with Taichi.

Algorithm Details:
    - Based on Blelloch (1990) work-efficient scan
    - Two-phase approach: up-sweep (reduce) + down-sweep (distribute)
    - O(n) work complexity, O(log n) depth complexity
    - Requires power-of-2 sized working buffer for optimal performance

Mathematical Operation:
    Given input array [a₀, a₁, a₂, ..., aₙ₋₁], produces output:
    [a₀, a₀+a₁, a₀+a₁+a₂, ..., a₀+a₁+...+aₙ₋₁]

Applications:
    - Stream compaction and filtering
    - Parallel allocation and indexing  
    - Graph algorithms (BFS, connected components)
    - Computational geometry (mesh processing)

Author: B. Gailleton
Reference: Blelloch, G. E. (1990). "Prefix sums and their applications"
"""
import taichi as ti
import math

@ti.kernel
def upsweep_step(data: ti.template(), n: int, stride: int):
    """
    Execute one step of the up-sweep phase (reduce phase) of parallel scan.
    
    This phase builds a binary sum tree by combining pairs of elements at
    increasing stride distances. Each thread processes elements separated
    by the current stride, accumulating partial sums upward in the tree.
    
    Args:
        data: Working array (modified in-place)
        n: Size of the working array (must be power of 2)
        stride: Current stride distance for this up-sweep step
        
    Mathematical Operation:
        For stride s: data[i] += data[i-s] where (i+1) % (2s) == 0
        
    Time Complexity: O(n/stride) work per step
    """
    for i in range(n):
        if (i + 1) % (stride * 2) == 0:
            data[i] += data[i - stride]

@ti.kernel
def downsweep_step(data: ti.template(), n: int, stride: int):
    """
    Execute one step of the down-sweep phase (distribute phase) of parallel scan.
    
    This phase distributes partial sums down the binary tree, computing
    the final exclusive scan values. Each step processes elements at
    decreasing stride distances, propagating sums from parent to child nodes.
    
    Args:
        data: Working array (modified in-place)
        n: Size of the working array (must be power of 2)
        stride: Current stride distance for this down-sweep step
        
    Mathematical Operation:
        For stride s and indices where (i+1) % (2s) == 0:
        temp = data[i-s]; data[i-s] = data[i]; data[i] += temp
        
    Time Complexity: O(n/stride) work per step
    """
    for i in range(n):
        if (i + 1) % (stride * 2) == 0:
            temp = data[i - stride]
            data[i - stride] = data[i]
            data[i] += temp

@ti.kernel
def copy_input_to_work(src: ti.template(), dst: ti.template(), n: int, work_size: int):
    """
    Copy input data to working buffer and pad to power-of-2 size.
    
    The parallel scan algorithm requires a power-of-2 sized working buffer
    for optimal tree-based processing. This kernel copies the original data
    and zero-pads the remaining elements to avoid affecting the scan results.
    
    Args:
        src: Input array with original data
        dst: Working buffer (size must be >= next_power_of_2(n))
        n: Number of valid elements in input
        work_size: Total size of working buffer (power of 2)
        
    Ensures: dst[i] = src[i] for i < n, dst[i] = 0 for i >= n
    """
    for i in range(work_size):
        if i < n:
            dst[i] = src[i]
        else:
            dst[i] = 0

@ti.kernel
def set_zero(data: ti.template(), index: int):
    """
    Set a specific array element to zero.
    
    Used to initialize the root of the sum tree to zero before the
    down-sweep phase, which is required to convert the sum tree into
    an exclusive scan representation.
    
    Args:
        data: Array to modify
        index: Index to set to zero (typically the last element)
        
    Note: This operation is critical for correct exclusive scan computation
    """
    data[index] = 0

@ti.kernel
def make_inclusive_and_copy(input_arr: ti.template(), work_data: ti.template(), output_arr: ti.template(), n: int):
    """
    Convert exclusive scan result to inclusive scan and copy to output.
    
    The Blelloch algorithm naturally produces an exclusive scan (prefix sums
    not including the current element). This kernel converts it to inclusive
    scan by adding the original input values back to the scan results.
    
    Args:
        input_arr: Original input data
        work_data: Exclusive scan results from algorithm
        output_arr: Final inclusive scan output
        n: Number of valid elements to process
        
    Mathematical Operation:
        output[0] = input[0] (first element unchanged)
        output[i] = work_data[i] + input[i] for i > 0
    """
    for i in range(n):
        if i == 0:
            output_arr[i] = input_arr[i]
        else:
            output_arr[i] = work_data[i] + input_arr[i]

def inclusive_scan(input_arr: ti.template(), output_arr: ti.template(), work_arr: ti.template(), n: int):
    """
    Compute parallel inclusive scan (prefix sum) using work-efficient algorithm.
    
    Implements the Blelloch scan algorithm with O(n) work complexity and
    O(log n) depth complexity. The algorithm consists of two phases:
    1. Up-sweep: Build binary sum tree (parallel reduce)
    2. Down-sweep: Distribute partial sums (parallel distribute)
    
    Args:
        input_arr: Input data array to scan
        output_arr: Output array for inclusive scan results  
        work_arr: Working buffer (size >= next_power_of_2(n))
        n: Number of elements to scan
        
    Requirements:
        - work_arr must be at least next_power_of_2(n) elements
        - All arrays must be compatible Taichi field types
        - Input and output arrays must be at least n elements
        
    Example:
        Input:  [3, 1, 7, 0, 4, 1, 6, 3]
        Output: [3, 4, 11, 11, 15, 16, 22, 25]
        
    Time Complexity: O(n) work, O(log n) depth
    Space Complexity: O(next_power_of_2(n)) working space
    
    Author: B. Gailleton
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