"""
PyFastFlow - GPU-accelerated geomorphological and hydraulic flow modeling.

A high-performance Python package for geomorphological and hydraulic flow routing on digital 
elevation models using Taichi for GPU acceleration. Built for large-scale simulations with
efficient memory management through field pooling.

Key Features:
- GPU-accelerated flow routing (steepest descent, stochastic)
- Advanced depression filling (priority flood, carving algorithms)
- Flow accumulation using rake-and-compress parallel algorithms
- Multiple boundary condition modes (normal, periodic EW/NS, custom per-node)
- 2D shallow water flow modeling (LisFlood, GraphFlood)
- Stream Power Law erosion and landscape evolution with transport
- Real-time 3D visualization and hillshading
- Efficient GPU memory management through field pooling system
- General-purpose parallel algorithms (parallel scan, ping-pong buffers)

Core Components:
- flow: Flow routing, receivers computation, flow accumulation
- flood: 2D shallow water flow (LisFlood/GraphFlood implementations)
- erodep: Stream Power Law erosion, sediment transport, landscape evolution
- visu: Real-time 3D visualization, hillshading, terrain analysis
- pool: GPU memory management and field pooling for performance
- general_algorithms: Fundamental parallel algorithms (scan, ping-pong)
- constants: Global simulation parameters and configuration

Basic Usage:
    import pyfastflow as pf
    import taichi as ti
    import numpy as np
    
    # Initialize Taichi GPU backend
    ti.init(ti.gpu)
    
    # Create grid and flow router
    nx, ny, dx = 512, 512, 10.0
    elevation = np.random.rand(ny, nx) * 100
    
    grid = pf.grid.Grid(nx, ny, dx, elevation)
    router = pf.flow.FlowRouter(grid)
    
    # Flow routing and accumulation
    router.compute_receivers()
    router.reroute_flow()
    router.accumulate_constant_Q(1.0)
    
    # 2D flood modeling
    flooder = pf.flood.Flooder(router, precipitation_rates=10e-3/3600)
    flooder.run_LS(N=1000)
    
    # Landscape evolution
    alpha_ = ti.field(ti.f32, shape=(nx*ny,))
    alpha__ = ti.field(ti.f32, shape=(nx*ny,))
    alpha_.fill(1e-5)
    alpha__.fill(1e-5)
    pf.erodep.SPL(router, alpha_, alpha__)
    
    # Visualization
    hillshade = pf.visu.hillshade_numpy(elevation)
    viewer = pf.visu.SurfaceViewer(elevation)
    
    # Memory management with field pooling
    with pf.pool.temp_field(ti.f32, (nx*ny,)) as temp:
        # Use temporary field efficiently
        temp.fill(0.0)

Optional Dependencies:
PyFastFlow supports optional integration with additional packages for enhanced functionality:

- TopoToolbox: For advanced DEM processing and terrain analysis
  Install with: pip install pyfastflow[topotoolbox]
  Or separately: pip install topotoolbox

Scientific Background:
The algorithms follow recent advances in GPU geomorphological modeling,
particularly Jain et al. (2024) for fast flow routines, Bates et al. (2010)
for shallow water flow, Gailleton et al. (2024) for GraphFlood, and standard 
Stream Power Law formulations for landscape evolution.

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
from . import grid
from . import io
from . import pool
from . import visu

# Export all submodules
__all__ = [
    "constants",
    "erodep", 
    "flood",
    "flow",
    "general_algorithms",
    "grid",
    "io",
    "pool",
    "visu"
]