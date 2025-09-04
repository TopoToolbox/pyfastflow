"""Raster manipulation module for PyFastFlow.

Provides GPU-accelerated raster upscaling, downscaling, and arbitrary resizing
functions for 2D grid data. All functions support both Taichi fields and NumPy
arrays as input/output, with efficient memory pool integration for GPU field
management.

Author: B.G.
"""

from .upscaling import double_resolution, double_resolution_kernel
from .downscaling import (
    halve_resolution,
    halve_resolution_kernel_max,
    halve_resolution_kernel_min,
    halve_resolution_kernel_mean,
    halve_resolution_kernel_cubic,
)
from .resizing import resize_raster, resize_kernel
from .resizing import resize_to_max_dim

__all__ = [
    "double_resolution",
    "double_resolution_kernel",
    "halve_resolution",
    "halve_resolution_kernel_max",
    "halve_resolution_kernel_min",
    "halve_resolution_kernel_mean",
    "halve_resolution_kernel_cubic",
    "resize_raster",
    "resize_kernel",
    "resize_to_max_dim",
]
