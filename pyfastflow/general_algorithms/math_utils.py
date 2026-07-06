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
from .. import constants as cte


@ti.func
def atan(x: cte.FLOAT_TYPE_TI) -> cte.FLOAT_TYPE_TI:
    """
    Compute arctangent of x using atan2 for Taichi compatibility.

    Since ti.math.atan is not available in Taichi, this function provides
    the arctangent functionality using ti.math.atan2(y, x) where y = x and x = 1.

    The mathematical relationship is: atan(x) = atan2(x, 1)

    Args:
        x (cte.FLOAT_TYPE_TI): The input value for which to compute arctangent

    Returns:
        cte.FLOAT_TYPE_TI: The arctangent of x in radians, in the range [-π/2, π/2]

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


@ti.func
def nextafter(x: cte.FLOAT_TYPE_TI, y: cte.FLOAT_TYPE_TI) -> cte.FLOAT_TYPE_TI:
    """
    Return the next representable float after ``x`` in the direction of ``y``.

    IEEE-754 bit-twiddling implementation (no libm nextafter on GPU): bumps
    the raw bit pattern by one ULP towards ``y``. NaN/Inf are not handled --
    only meant for finite physical field values.

    Author: B.G (07/2026)
    """
    result = y
    if x != y:
        if ti.static(cte.FLOAT_TYPE_TI == ti.f32):
            sign_mask = ti.bit_cast(ti.cast(-0.0, ti.f32), ti.u32)
            ix = ti.bit_cast(x, ti.u32)
            if x == 0.0:
                ix = (ti.bit_cast(y, ti.u32) & sign_mask) | ti.cast(1, ti.u32)
            elif (x > 0.0) == (y > x):
                ix += ti.cast(1, ti.u32)
            else:
                ix -= ti.cast(1, ti.u32)
            result = ti.bit_cast(ix, ti.f32)
        else:
            sign_mask = ti.bit_cast(ti.cast(-0.0, ti.f64), ti.u64)
            ix = ti.bit_cast(x, ti.u64)
            if x == 0.0:
                ix = (ti.bit_cast(y, ti.u64) & sign_mask) | ti.cast(1, ti.u64)
            elif (x > 0.0) == (y > x):
                ix += ti.cast(1, ti.u64)
            else:
                ix -= ti.cast(1, ti.u64)
            result = ti.bit_cast(ix, ti.f64)
    return result
