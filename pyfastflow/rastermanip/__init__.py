"""
Raster manipulation module for PyFastflow.

Provides GPU-accelerated raster upscaling and downscaling functions for 2D grid data.
All functions support both Taichi fields and numpy arrays as input/output, with efficient
memory pool integration for GPU field management.

Core Operations:
- Upscaling: Double resolution with slope-preserving interpolation and optional noise
- Downscaling: Halve resolution with multiple aggregation methods (max, min, mean, cubic)

Features:
- Row-major vectorized grid operations
- Boundary-aware neighbor access using grid neighbourer system
- Slope-preserving interpolation for upscaling 
- Multiple downscaling algorithms
- Optional noise injection for realistic upscaling
- Memory pool integration for efficient field management

Usage:
    import pyfastflow as pf
    
    # Double resolution with slope-preserving interpolation
    upscaled = pf.rastermanip.double_resolution(grid_data, noise_amplitude=0.1)
    
    # Halve resolution using mean aggregation
    downscaled = pf.rastermanip.halve_resolution(grid_data, method='mean')
    
    # Multiple downscaling methods available
    max_downscaled = pf.rastermanip.halve_resolution(grid_data, method='max')
    cubic_downscaled = pf.rastermanip.halve_resolution(grid_data, method='cubic')

Author: B.G.
"""

from .upscaling import double_resolution, double_resolution_kernel
from .downscaling import halve_resolution, halve_resolution_kernel_max, halve_resolution_kernel_min, halve_resolution_kernel_mean, halve_resolution_kernel_cubic

# Export all raster manipulation functions
__all__ = [
    "double_resolution", "double_resolution_kernel",
    "halve_resolution", "halve_resolution_kernel_max", 
    "halve_resolution_kernel_min", "halve_resolution_kernel_mean",
    "halve_resolution_kernel_cubic"
]