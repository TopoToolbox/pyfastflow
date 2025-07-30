"""
Visualization utilities submodule for FastFlow Taichi.

This module provides tools for real-time visualization and analysis of
flow routing, flood modeling, and landscape evolution simulations.

Modules:
- live: Real-time plotting and animation utilities
- hillshading: GPU-accelerated terrain hillshading and shaded relief

Key Features:
- Live plotting during simulation
- Flow field visualization
- Topographic and hydrographic display
- Animation and time-series plotting
- Hillshading with multiple illumination models
- Support for boundary conditions and masking

Usage:
    import pyfastflow as pf
    import numpy as np
    
    # Create sample terrain data
    terrain = np.random.rand(100, 100) * 1000
    
    # Interactive 3D visualization
    viewer = pf.visu.SurfaceViewer(terrain)
    viewer.run()
    
    # Hillshading for terrain visualization
    hillshade = pf.visu.hillshade_numpy(terrain, altitude_deg=45, azimuth_deg=315)
    multidirectional_hs = pf.visu.hillshade_multidirectional_numpy(terrain)
    
    # With FlowRouter integration
    ff = pf.flow.FlowRouter(...)
    hillshade_fr = pf.visu.hillshade_flowrouter(ff, altitude_deg=30)
    
    # Or integrate with simulation
    for step in range(1000):
        new_terrain = simulate_step()
        viewer.update_surface(new_terrain)
        if not viewer.render_frame():
            break

Author: B.G.
"""

from .live import *
from .hillshading import (
    hillshade_flowrouter,
    hillshade_multidirectional_flowrouter,
    hillshade_numpy,
    hillshade_multidirectional_numpy
)

# Export all modules and classes
__all__ = [
    # Modules
    "live",
    "hillshading",
    
    # Classes
    "SurfaceViewer",
    
    # Hillshading Functions
    "hillshade_flowrouter",
    "hillshade_multidirectional_flowrouter", 
    "hillshade_numpy",
    "hillshade_multidirectional_numpy"
]