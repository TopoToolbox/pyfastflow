"""
General Algorithms Module

This module provides GPU-accelerated implementations of fundamental computer science
algorithms optimized for Taichi. These algorithms are designed to be reusable
building blocks for computational geometry, parallel processing, and numerical
methods.

Available Algorithms:
    - parallel_scan: Work-efficient parallel prefix sum (Blelloch scan)
    - pingpong: Utilities for ping-pong buffer management in iterative algorithms

These algorithms are particularly useful for:
    - Stream processing and data reduction
    - Graph algorithms requiring parallel traversal
    - Iterative solvers with double buffering
    - Computational geometry operations

Example Usage:
    ```python
    from pyfastflow.general_algorithms import inclusive_scan, getSrc, updateSrc
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
    ```

Author: B. Gailleton
"""

from .parallel_scan import inclusive_scan
from .pingpong import getSrc, updateSrc, fuse
from .slope_tools import *

__all__ = [
    'inclusive_scan',
    'getSrc', 
    'updateSrc',
    'fuse'
]