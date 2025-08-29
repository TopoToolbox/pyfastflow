"""
Miscellaneous Raster Utilities for PyFastFlow

This module provides utility functions for raster data processing and conversion,
particularly for workflows that need to convert raster files to numpy arrays
and save them for later use in PyFastFlow simulations.

Key Features:
- Load raster files and save as numpy arrays
- Simple conversion utilities
- Integration with PyFastFlow's io module

Dependencies:
- numpy: For array operations and file I/O
- topotoolbox (via pyfastflow.io): For raster file reading

Author: B.G.
"""

import numpy as np
import pyfastflow as pf


def load_raster_save_numpy(raster_path, output_path):
    """
    Load a raster file and save its values as a .npy file.
    
    This function provides a simple workflow for converting raster files
    (GeoTIFF, ASCII grid, etc.) to numpy arrays that can be easily loaded
    for PyFastFlow simulations. Uses TopoToolbox via pyfastflow.io as the
    backend for raster file reading.
    
    Args:
        raster_path (str): Path to input raster file
        output_path (str): Path for output .npy file
        
    Raises:
        ImportError: If topotoolbox is not installed
        FileNotFoundError: If input raster file doesn't exist
        ValueError: If raster file cannot be read
        OSError: If output file cannot be written
        
    Example:
        import pyfastflow as pf
        
        # Convert raster to numpy format
        pf.misc.load_raster_save_numpy('elevation.tif', 'elevation.npy')
        
        # Later, load for PyFastFlow simulation
        elevation = np.load('elevation.npy')
        grid = pf.grid.Grid(nx, ny, dx, elevation.ravel())
        router = pf.flow.FlowRouter(grid)
        
    Note:
        - Output .npy file preserves the 2D structure of the raster
        - Coordinate system and metadata information is not preserved
        - For full metadata preservation, use pf.io.raster_to_grid() instead
        
    Author: B.G.
    """
    # Load raster using pyfastflow.io
    elevation_array = pf.io.raster_to_numpy(raster_path)
    
    # Save as numpy array
    try:
        np.save(output_path, elevation_array)
    except Exception as e:
        raise OSError(f"Failed to save numpy array to '{output_path}': {e}")
    
    print(f"Successfully converted raster '{raster_path}' to numpy array '{output_path}'")
    print(f"Array shape: {elevation_array.shape}")
    print(f"Value range: [{elevation_array.min():.2f}, {elevation_array.max():.2f}]")