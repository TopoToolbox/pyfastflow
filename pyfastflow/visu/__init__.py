"""
Visualization utilities submodule for FastFlow Taichi.

This module provides tools for real-time visualization and analysis of
flow routing, flood modeling, and landscape evolution simulations.

Modules:
- live: Real-time plotting and animation utilities

Key Features:
- Live plotting during simulation
- Flow field visualization
- Topographic and hydrographic display
- Animation and time-series plotting

Usage:
    import pyfastflow as pf
    import numpy as np
    
    # Create sample terrain data
    terrain = np.random.rand(100, 100) * 1000
    
    # Interactive 3D visualization
    viewer = pf.visu.SurfaceViewer(terrain)
    viewer.run()
    
    # Or integrate with simulation
    for step in range(1000):
        new_terrain = simulate_step()
        viewer.update_surface(new_terrain)
        if not viewer.render_frame():
            break

Author: B.G.
"""

from .live import *

# Export all modules and classes
__all__ = [
    # Modules
    "live",
    
    # Classes
    "SurfaceViewer"
]