"""
Landscape evolution and erosion-deposition modeling submodule for PyFastFlow.

This submodule implements GPU-accelerated landscape evolution models including bedrock
erosion, sediment transport, and deposition processes. Uses pool-based memory management
for efficient GPU field allocation and provides both detachment-limited and transport-
limited erosion models for long-term landscape evolution simulations.

Core Modules:
- SPL: Stream Power Law erosion implementation with implicit numerical schemes
- deposition: Sediment transport and deposition algorithms

Key Algorithms:
- Stream Power Law (SPL): E = K * A^m * S^n bedrock erosion model
- Implicit finite difference: Stable numerical schemes for large time steps
- Transport-limited erosion: Coupled erosion-transport-deposition system
- Tectonic uplift: Block and spatially-varying uplift implementations
- Pool-based computation: Efficient GPU memory management for temporary fields

Erosion Models:
- Detachment-limited: Simple bedrock erosion following SPL
- Transport-limited: Coupled erosion-sediment transport with capacity limits
- Implicit solver: Stable iteration scheme for erosion propagation
- Receiver-chain traversal: Efficient parallel processing along flow paths

Available Functions:
- block_uplift: Apply uniform tectonic uplift to interior nodes
- ext_uplift_nobl: Apply spatially-variable uplift without boundary checking
- ext_uplift_bl: Apply spatially-variable uplift with boundary checking
- SPL: Execute detachment-limited SPL erosion for one time step
- SPL_transport: Execute transport-limited SPL with erosion-deposition coupling
- init_erode_SPL: Initialize implicit SPL computation with erosion coefficients
- iteration_erode_SPL: Perform iterative SPL solver using receiver chains
- erosion_to_source: Convert erosion volumes to sediment source terms

Usage:
    import pyfastflow as pf
    import taichi as ti
    import numpy as np

    # Initialize Taichi and create landscape
    ti.init(ti.gpu)
    nx, ny, dx = 512, 512, 100.0
    elevation = np.random.rand(ny, nx) * 1000 + np.linspace(1000, 0, ny)[:, None]

    # Setup flow router with pool-based field management
    grid = pf.flow.GridField(nx, ny, dx)
    grid.set_z(elevation)
    router = pf.flow.FlowRouter(grid)

    # Create erosion coefficient fields for implicit solver
    alpha_ = ti.field(ti.f32, shape=(nx*ny,))
    alpha__ = ti.field(ti.f32, shape=(nx*ny,))
    alpha_.fill(1e-5)  # Erodibility coefficient
    alpha__.fill(1e-5)

    # Run landscape evolution simulation
    dt_years = 1000.0  # Time step in years
    for timestep in range(1000):
        # Flow routing
        router.compute_receivers()
        router.reroute_flow()
        router.accumulate_constant_Q(1.0)

        # Tectonic processes
        pf.erodep.block_uplift(router.grid.z, rate=1e-3)  # 1 mm/yr uplift

        # Erosion (detachment-limited)
        pf.erodep.SPL(router, alpha_, alpha__)

        # Get evolved topography
        new_elevation = router.get_Z()

    # Transport-limited erosion example
    Qs = ti.field(ti.f32, shape=(nx*ny,))  # Sediment flux field
    pf.erodep.SPL_transport(router, alpha_, alpha__, Qs, Nit=5)

Physical Background:
The Stream Power Law describes bedrock erosion rate as E = K * A^m * S^n where
A is drainage area, S is local slope, and K is erodibility. The implicit
solver allows stable computation with large time steps by solving the
nonlinear erosion equation iteratively along receiver chains.

Author: B.G.
"""

# Import specific functions from each module
from .archive.SPL import (
    SPL,
    SPL_transport,
    erosion_to_source,
    init_erode_SPL,
    iteration_erode_SPL,
)
from .fluvial_deposition import iterate_deposition_ptr_jmp_cte_kd as iterate_deposition
from .uplift import block_uplift, ext_uplift_bl, ext_uplift_nobl

# Export all functions and classes
__all__ = [
    # Uplift functions
    "block_uplift",
    "ext_uplift_nobl",
    "ext_uplift_bl",
    # SPL erosion functions
    "SPL",
    "SPL_transport",
    "init_erode_SPL",
    "iteration_erode_SPL",
    "erosion_to_source",
    "iterate_deposition",
]
