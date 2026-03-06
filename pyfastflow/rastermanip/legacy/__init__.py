"""
Legacy raster manipulation module exports.

These functions are kept for backward compatibility during migration to the
context-driven API.

Author: B.G (02/2026)
"""

from .upscaling import double_resolution, double_resolution_kernel
from .downscaling import (
    halve_resolution,
    halve_resolution_kernel_max,
    halve_resolution_kernel_min,
    halve_resolution_kernel_mean,
    halve_resolution_kernel_cubic,
)
from .resizing import resize_raster, resize_kernel, resize_to_dims, resize_to_max_dim

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
    "resize_to_dims",
    "resize_to_max_dim",
]
