"""
General Algorithms Module

This module provides GPU-accelerated implementations of fundamental computer science
algorithms optimized for Taichi. These algorithms are designed to be reusable
building blocks for computational geometry, parallel processing, and numerical
methods.

Available Algorithms:
    - parallel_scan: Work-efficient parallel prefix sum (Blelloch scan)
    - pingpong: Utilities for ping-pong buffer management in iterative algorithms
    - math_utils: Mathematical functions for Taichi compatibility (atan, etc.)

These algorithms are particularly useful for:
    - Stream processing and data reduction
    - Graph algorithms requiring parallel traversal
    - Iterative solvers with double buffering
    - Computational geometry operations
    - Mathematical computations requiring missing Taichi functions

Example Usage:
    ```python
    from pyfastflow.general_algorithms import inclusive_scan, getSrc, updateSrc, atan
    import taichi as ti

    # Parallel prefix sum
    input_data = ti.field(ti.f32, shape=1024)
    output_data = ti.field(ti.f32, shape=1024)
    work_buffer = ti.field(ti.f32, shape=2048)
    inclusive_scan(input_data, output_data, work_buffer, 1024)

    # Ping-pong buffer management
    state_buffer = ti.field(ti.i32, shape=100)
    for iteration in range(10):
        flip = getSrc(state_buffer, thread_id, iteration)
        # Process data...
        updateSrc(state_buffer, thread_id, iteration, flip)

    # Mathematical utilities in Taichi kernels
    @ti.kernel
    def compute_slope():
        gradient = ti.field(ti.f32, shape=100)
        slope = atan(gradient[0])  # Arctangent compatible with Taichi
    ```

Author: B. Gailleton
"""

from .math_utils import atan
from .parallel_scan import inclusive_scan
from .pingpong import fuse, getSrc, updateSrc

__all__ = ["inclusive_scan", "getSrc", "updateSrc", "fuse", "atan"]
