"""
Visualization and terrain analysis submodule for PyFastFlow.

This submodule provides GPU-accelerated visualization tools and terrain analysis
algorithms for flow routing, flood modeling, and landscape evolution simulations.
Integrates with pool-based memory management for efficient GPU field operations.

**Recommended Usage**: For most hillshading needs, use the built-in `grid.hillshade()` 
method which provides a clean interface with automatic memory management. The functions
in this module are for advanced use cases or when working directly with NumPy arrays.

Core Modules:
- live: Real-time 3D visualization and animation utilities using Taichi GUI
- hillshading: GPU-accelerated terrain hillshading and shaded relief algorithms

Key Features:
- Real-time 3D surface visualization with interactive controls
- GPU-accelerated hillshading with multiple illumination models
- FlowRouter integration for seamless terrain analysis
- Support for NumPy arrays and Taichi fields
- Multiple hillshading algorithms (standard, multidirectional)
- Vectorized and 2D array processing modes
- Boundary condition and masking support
- Animation and time-series visualization capabilities

Hillshading Algorithms:
- Standard hillshading: Single light source illumination
- Multidirectional: Multiple azimuth angles for enhanced terrain detail
- Vectorized processing: Direct computation on flattened field data
- 2D array processing: Standard NumPy array workflow with spatial derivatives
- Pool-based computation: Efficient GPU memory management for temporary fields

Available Functions:
- hillshade_numpy: Standard hillshading for NumPy elevation arrays
- hillshade_multidirectional_numpy: Multi-azimuth hillshading for NumPy arrays
- hillshade_grid: GPU hillshading using Grid elevation data
- hillshade_multidirectional_grid: Multi-azimuth hillshading for Grid
- hillshade_vectorized: Low-level GPU kernel for vectorized hillshading (advanced use)
- hillshade_2d: 2D array hillshading kernel (advanced use)
- SurfaceViewer: Interactive 3D terrain visualization class

Usage:
    import pyfastflow as pf
    import taichi as ti
    import numpy as np
    
    # Initialize Taichi for GPU acceleration
    ti.init(ti.gpu)
    
    # Create terrain data
    nx, ny, dx = 256, 256, 30.0
    elevation = np.random.rand(ny, nx) * 1000
    
    # Standard NumPy hillshading
    hillshade = pf.visu.hillshade_numpy(
        elevation, 
        altitude_deg=45,   # Sun elevation angle
        azimuth_deg=315,   # Sun azimuth angle (NW)
        dx=dx              # Grid spacing
    )
    
    # Multidirectional hillshading for enhanced detail
    multi_hs = pf.visu.hillshade_multidirectional_numpy(
        elevation,
        altitude_deg=45,
        dx=dx,
        azimuths_deg=[315, 45, 135, 225]  # Four cardinal directions
    )
    
    # Integration with Grid (uses pool-based GPU fields)
    grid = pf.grid.Grid(nx, ny, dx, elevation)
    router = pf.flow.FlowRouter(grid)
    
    # Built-in Grid hillshading (recommended)
    hillshade_builtin = grid.hillshade(altitude_deg=30, azimuth_deg=270)
    
    # Or use visu functions directly
    hillshade_gpu = pf.visu.hillshade_grid(
        grid, 
        altitude_deg=30,
        azimuth_deg=270    # West illumination
    )
    
    # Real-time 3D visualization
    viewer = pf.visu.SurfaceViewer(elevation)
    viewer.run()  # Interactive window with mouse controls
    
    # Animation example with simulation integration
    for timestep in range(1000):
        # Run simulation step
        router.compute_receivers()
        pf.erodep.SPL(router, alpha_, alpha__)
        
        # Update visualization
        new_terrain = router.get_Z()
        viewer.update_surface(new_terrain)
        
        # Render frame (returns False if window closed)
        if not viewer.render_frame():
            break

Scientific Applications:
Hillshading algorithms follow standard illumination models used in cartography
and digital terrain analysis. The multidirectional approach combines multiple
light sources to enhance terrain feature visibility, particularly useful for 
analyzing flow patterns and geomorphic features in elevation models.

Author: B.G.
"""

from .live import *
from .hillshading import (
    hillshade_vectorized,
    hillshade_2d,
    hillshade_numpy,
    hillshade_multidirectional_numpy,
    hillshade_grid,
    hillshade_multidirectional_grid
)

# Export all modules and classes
__all__ = [
    # Modules
    "live",
    "hillshading",
    
    # Classes
    "SurfaceViewer",
    
    # Hillshading Functions
    "hillshade_vectorized",
    "hillshade_2d",
    "hillshade_numpy",
    "hillshade_multidirectional_numpy",
    "hillshade_grid",
    "hillshade_multidirectional_grid"
]