"""
Command Line Interface for PyFastFlow

This module provides command line utilities for PyFastFlow, enabling
easy access to common operations from the terminal without writing Python scripts.

Available Commands:
- raster2npy: Convert raster files to numpy arrays

Author: B.G.
"""

from .raster_commands import raster2npy

# Export public API
__all__ = ["raster2npy"]
