"""
Flow routing algorithms submodule for PyFastFlow.

This submodule implements GPU-accelerated flow routing algorithms for hydrological
modeling on digital elevation models. All algorithms use pool-based memory management
for efficient GPU field allocation and reuse.

Core Modules:
- neighbourer_flat: Vectorized grid navigation with boundary condition handling
- receivers: Steepest descent and stochastic receiver computation algorithms  
- downstream_propag: Parallel flow accumulation using rake-and-compress
- lakeflow: Depression filling, carving, and closed basin handling
- flowfields: FlowRouter class with pool-based field management
- fill_topo: Topographic filling and depression removal utilities
- level_acc: Level-set flow accumulation algorithms
- f32_i32_struct: Utility structures for atomic operations
- util_taichi: General Taichi utility functions and kernels
- environment: System environment detection and configuration

Key Features:
- Multiple boundary conditions (normal, periodic EW/NS, custom per-node)
- Efficient parallel flow accumulation with O(log N) complexity
- Lake and depression handling with priority flood algorithms
- Stochastic flow routing for uncertainty quantification
- Pool-based memory management for optimal GPU performance
- Support for large grids (millions of nodes) with scalable algorithms

Usage:
    import pyfastflow as pf
    import taichi as ti
    import numpy as np
    
    # Initialize Taichi and create grid
    ti.init(ti.gpu)
    nx, ny, dx = 512, 512, 30.0
    elevation = np.random.rand(ny, nx) * 100
    
    # Create grid and flow router with pool management
    grid = pf.grid.Grid(nx, ny, dx, elevation)
    router = pf.flow.FlowRouter(grid)
    
    # Complete flow routing workflow
    router.compute_receivers()        # Steepest descent routing
    router.reroute_flow()            # Handle depressions and lakes
    router.accumulate_constant_Q(1.0) # Flow accumulation
    
    # Get results
    drainage_area = router.get_Q() * dx * dx
    receivers = router.get_receivers()
    
    # Advanced usage with boundary conditions
    boundaries = np.ones((ny, nx), dtype=np.uint8)
    boundaries[0, :] = 3  # Top can drain
    boundaries[-1, :] = 3 # Bottom can drain
    grid_custom = pf.grid.Grid(nx, ny, dx, elevation, boundary_mode='custom', boundaries=boundaries)

Scientific Background:
Flow routing algorithms follow O'Callaghan & Mark (1984) for steepest descent,
with parallel accumulation based on Jain et al. (2024). Depression handling
uses priority flood algorithms with efficient GPU implementation. GraphFlood
shallow water flow follows Gailleton et al. (2024) ESurf.

Author: B.G.
"""

# Import all flow modules - accessible as ff.flow.module_name
from ..constants import *
from ..grid.neighbourer_flat import *
from .receivers import *
from .downstream_propag import *
from .lakeflow import *
from .f32_i32_struct import *
from ..general_algorithms.util_taichi import *
from .flowfields import *
from .fill_topo import *
from .level_acc import *

# Export all modules
__all__ = [
    "neighbourer_flat", 
    "receivers",
    "downstream_propag",
    "lakeflow",
    "f32_i32_struct",
    "util_taichi",
    "flowfields",
    "fill_topo",
    "environment"
]