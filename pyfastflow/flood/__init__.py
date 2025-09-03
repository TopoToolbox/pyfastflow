"""
2D shallow water flow modeling submodule for PyFastFlow.

This submodule implements GPU-accelerated 2D shallow water flow algorithms for flood
modeling and hydrodynamic simulations. Uses pool-based memory management for efficient
GPU field allocation and provides both explicit (LisFlood) and implicit (GraphFlood)
numerical schemes.

Core Modules:
- gf_fields: Flooder class with pool-based field management for flood variables
- gf_hydrodynamics: Core hydrodynamic computation kernels with Manning's friction
- gf_ls: LisFlood explicit finite difference implementation (Bates et al. 2010)

Key Features:
- LisFlood: Explicit 2D shallow water equations with inertial simplification
- GraphFlood: Implicit flow routing with diffusion-based shallow water flow
- Manning's friction: Configurable roughness coefficients for flow resistance
- Pool-based memory management: Efficient GPU field allocation and reuse
- Precipitation input: Time-varying rainfall and boundary conditions
- Integration with flow routing: Use drainage networks as initial conditions
- Boundary conditions: Configurable edge slopes and flow outlets
- Hydrodynamic timestep adaptation: Stable CFL-limited time stepping

Algorithm Implementations:
- LisFlood: Local inertial approximation following Bates et al. (2010)
- GraphFlood: Graph-based implicit routing following Gailleton et al. (2024)
- Manning's equation: Flow resistance based on surface roughness
- CFL timestep limiting: Automatic timestep adaptation for stability

Usage:
    import pyfastflow as pf
    import taichi as ti
    import numpy as np

    # Initialize Taichi and create flow routing
    ti.init(ti.gpu)
    nx, ny, dx = 256, 256, 30.0
    elevation = np.random.rand(ny, nx) * 50

    grid = pf.flow.GridField(nx, ny, dx)
    grid.set_z(elevation)
    router = pf.flow.FlowRouter(grid)
    router.compute_receivers()

    # Create flood model with pool-based field management
    flooder = pf.flood.Flooder(
        router,
        precipitation_rates=10e-3/3600,  # 10 mm/hr in m/s
        manning=0.033,                   # Manning's n roughness
        dt_hydro=1e-3,                   # Hydrodynamic timestep (s)
        edge_slope=1e-2                  # Boundary slope
    )

    # Run LisFlood simulation (explicit scheme)
    flooder.run_LS(N=1000)  # 1000 time steps

    # Run GraphFlood simulation (implicit scheme)
    flooder.run_graphflood(
        N=10,              # Major iterations
        N_stochastic=4,    # Stochastic flow paths
        N_diffuse=2,       # Diffusion steps
        temporal_filtering=0.1  # Temporal smoothing
    )

    # Get flood simulation results
    water_depth = flooder.get_h()     # Water depth (m)
    discharge_x = flooder.get_qx()    # x-direction unit discharge (m²/s)
    discharge_y = flooder.get_qy()    # y-direction unit discharge (m²/s)
    velocity_x = discharge_x / (water_depth + 1e-6)  # x-velocity (m/s)

Scientific Background:
LisFlood follows the local inertial approximation of Bates et al. (2010) for explicit
2D shallow water flow. GraphFlood implements the implicit flow routing approach of
Gailleton et al. (2024) ESurf with diffusion-based shallow water dynamics.

Author: B.G.
"""

# Import specific classes and functions from each module
# Import modules themselves for module-level access
from . import gf_fields, gf_hydrodynamics, gf_ls
from .precipitation_gui import precipitation_gui
from .gf_fields import Flooder
from .gf_hydrodynamics import diffuse_Q_constant_prec, graphflood_core_cte_mannings
from .gf_ls import (
    depth_update,
    flow_route,
    init_LS_on_hw_from_constant_effective_prec,
    init_LS_on_hw_from_variable_effective_prec,
)

# Export all modules
__all__ = [
    # Core classes
    "Flooder",
    # LisFlood kernels
    "flow_route",
    "depth_update",
    "init_LS_on_hw_from_constant_effective_prec",
    "init_LS_on_hw_from_variable_effective_prec",
    # GraphFlood kernels
    "diffuse_Q_constant_prec",
    "graphflood_core_cte_mannings",
    # Module names
    "gf_fields",
    "gf_hydrodynamics",
    "gf_ls",
    "precipitation_gui",
]
