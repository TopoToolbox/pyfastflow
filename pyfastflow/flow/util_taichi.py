"""
Utility functions for Taichi array operations.

Provides common operations like swapping arrays and other utility functions
for efficient GPU-accelerated computations.

Author: B.G.
"""

import taichi as ti


#########################################
###### SWAP, COPY AND STUFF #############
#########################################


@ti.kernel
def swap_arrays(array1: ti.template(), array2: ti.template()):
    """
    Swap contents of two arrays element-wise in parallel.
    
    Args:
        array1: First array to swap
        array2: Second array to swap (must have same shape and type as array1)
        
    Note:
        After this operation: array1 contains original array2, array2 contains original array1
        
    Author: B.G.
    """
    for I in ti.grouped(array1):  # Parallel iteration over all elements
        temp = array1[I]  # Store original value from array1
        array1[I] = array2[I]  # Copy array2 value to array1
        array2[I] = temp  # Copy original array1 value to array2

@ti.kernel
def add_B_to_A(array1: ti.template(), array2: ti.template()):

    for i in array1:
        array1[i] += array2[i]

@ti.kernel
def add_B_to_weighted_A(array1: ti.template(), array2: ti.template(), weight:ti.f32):

    for i in array1:
        array1[i] += array2[i]*weight

@ti.kernel
def weighted_mean_B_in_A(array1: ti.template(), array2: ti.template(), weight:ti.f32):

    for i in array1:
        array1[i] = array2[i]*weight + array1[i]*(1-weight)

@ti.kernel
def init_arange(array: ti.template()):
    """
    """
    for i in array:
        array[i] = i
