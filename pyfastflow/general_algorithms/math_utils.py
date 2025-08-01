"""
Mathematical Utility Functions for Taichi

This module provides mathematical functions that are missing or not directly
available in Taichi's math module. These functions are implemented using
available Taichi operations to ensure GPU compatibility.

Available Functions:
    - atan: Arctangent function implemented using atan2 for Taichi compatibility

Author: B. Gailleton
"""

import taichi as ti


@ti.func
def atan(x: ti.f32) -> ti.f32:
    """
    Compute arctangent of x using atan2 for Taichi compatibility.
    
    Since ti.math.atan is not available in Taichi, this function provides
    the arctangent functionality using ti.math.atan2(y, x) where y = x and x = 1.
    
    The mathematical relationship is: atan(x) = atan2(x, 1)
    
    Args:
        x (ti.f32): The input value for which to compute arctangent
        
    Returns:
        ti.f32: The arctangent of x in radians, in the range [-π/2, π/2]
        
    Note:
        This implementation handles all cases including:
        - Positive values: returns positive angles
        - Negative values: returns negative angles  
        - Zero: returns 0
        - The result is always in the range [-π/2, π/2] as expected for atan
        
    Usage:
        ```python
        import taichi as ti
        from pyfastflow.general_algorithms.math_utils import atan
        
        @ti.kernel
        def compute_slope():
            slope_rad = atan(gradient_magnitude)
        ```
        
    Author: B. Gailleton
    """
    return ti.math.atan2(x, 1.0)