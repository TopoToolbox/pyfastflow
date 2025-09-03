"""
Command Line Interface for PyFastFlow

This module provides command line utilities for PyFastFlow, enabling
easy access to common operations from the terminal without writing Python scripts.

Available Commands:
- raster2npy: Convert raster files to numpy arrays
- raster-upscale: Double raster resolution using rastermanip utilities
- raster-downscale: Halve raster resolution using rastermanip utilities

Author: B.G.
"""

from .raster_commands import raster2npy
from .rastermanip_commands import raster_downscale, raster_upscale
from .grid_commands import boundary_gui

# Export public API
__all__ = ["raster2npy", "raster_upscale", "raster_downscale", "boundary_gui"]
