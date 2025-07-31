"""
FastFlow Taichi - GPU-accelerated flow routing algorithms.

A high-performance Python package for hydrological flow routing on digital elevation models
using Taichi for GPU acceleration. Implements parallel algorithms for:

- Steepest descent flow routing
- Lake/depression handling (priority flood, carving)
- Flow accumulation using rake-and-compress algorithms
- Multiple boundary condition modes (open, periodic, custom)
- 2D shallow water flow modeling (Flood)
- Stream Power Law erosion and landscape evolution (Erodep)
- Real-time 3D visualization and analysis tools (Visu)
- General-purpose GPU algorithms (parallel scan, ping-pong buffers)

Usage:
    import pyfastflow as ff
    
    # Access flow functions through the flow submodule
    ff.flow.lakeflow.depression_counter(...)
    ff.flow.receivers.compute_receivers(...)
    
    # Access constants directly
    ff.constants.NX
    
    # Access Flood for 2D shallow water modeling
    flooder = ff.flood.Flooder(router, precipitation_rates=10e-3/3600)
    
    # Access Erodep for landscape evolution
    ff.erodep.SPL(router, alpha_, alpha__, Kr=1e-5)
    
    # Access Visu for 3D visualization
    viewer = ff.visu.SurfaceViewer(terrain_data)
    viewer.run()
    
    # Access general algorithms for parallel computing
    ff.general_algorithms.inclusive_scan(input_arr, output_arr, work_arr, n)

Author: B.G.
"""

__version__ = "0.1.0"
__author__ = "B.G."

# Import all submodules in alphabetical order
from . import constants
from . import erodep
from . import flood
from . import flow
from . import general_algorithms
from . import visu
from . import pool

# Export all submodules
__all__ = [
    "constants",
    "erodep", 
    "flood",
    "flow",
    "general_algorithms",
    "visu"
]