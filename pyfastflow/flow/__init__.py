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

# Import key classes and functions for direct API access
# Import modules for access as ff.flow.module_name
from .. import constants
from ..general_algorithms import util_taichi
from ..grid import neighbourer_flat
from ..grid.gridfields import Grid as GridField
from ..grid.neighbourer_flat import (
    bottom,
    bottom_custom,
    bottom_n,
    bottom_pew,
    bottom_pns,
    can_leave_domain,
    can_leave_domain_custom,
    can_leave_domain_n,
    can_leave_domain_pew,
    can_leave_domain_pns,
    compile_neighbourer,
    fill_edges,
    flow_out_nodes,
    i_from_rc,
    is_on_edge,
    left,
    left_custom,
    left_n,
    left_pew,
    left_pns,
    neighbour,
    neighbour_custom,
    neighbour_n,
    neighbour_pew,
    neighbour_pns,
    rc_from_i,
    right,
    right_custom,
    right_n,
    right_pew,
    right_pns,
    top,
    top_custom,
    top_n,
    top_pew,
    top_pns,
    validate_link,
    validate_link_custom,
    validate_link_n,
    validate_link_pew,
    validate_link_pns,
    which_edge,
)
from . import (
    downstream_propag,
    f32_i32_struct,
    fill_topo,
    flowfields,
    lakeflow,
    level_acc,
    receivers,
)
from .fill_topo import fill_z_add_delta, topofill
from .flowfields import FlowRouter

# Note: environment module referenced in __all__ but doesn't exist

# Export all modules
__all__ = [
    # Main API classes and functions
    "FlowRouter",
    "GridField",
    "fill_z_add_delta",
    "topofill",
    # Neighbour functions
    "neighbour", "can_leave_domain", "is_on_edge", "which_edge", "validate_link",
    "top", "bottom", "left", "right", "neighbour_custom", "can_leave_domain_custom",
    "validate_link_custom", "top_custom", "bottom_custom", "left_custom", "right_custom",
    "neighbour_n", "can_leave_domain_n", "validate_link_n", "top_n", "bottom_n", "left_n", "right_n",
    "neighbour_pew", "can_leave_domain_pew", "validate_link_pew", "top_pew", "bottom_pew", "left_pew", "right_pew",
    "neighbour_pns", "can_leave_domain_pns", "validate_link_pns", "top_pns", "bottom_pns", "left_pns", "right_pns",
    "compile_neighbourer", "fill_edges", "flow_out_nodes", "i_from_rc", "rc_from_i",
    # Modules
    "neighbourer_flat",
    "receivers",
    "downstream_propag",
    "lakeflow",
    "level_acc",
    "f32_i32_struct",
    "util_taichi",
    "flowfields",
    "fill_topo",
    "constants",
]
