"""
Erosion and deposition algorithms submodule for FastFlow Taichi.

This module implements landscape evolution models including erosion, sediment
transport, and deposition processes. The algorithms are designed for long-term
landscape evolution simulations and integrate with FastFlow's routing system.

Key Features:
- Stream Power Law (SPL) erosion models
- Implicit finite difference schemes for numerical stability
- Block uplift and tectonic processes
- GPU-accelerated computation using Taichi
- Integration with flow routing for erosion patterns
- Support for landscape evolution modeling

Modules:
- SPL: Stream Power Law erosion implementation with implicit schemes

Available Functions:
- block_uplift: Apply uniform tectonic uplift
- ext_uplift_nobl: Apply spatially variable uplift without boundary limits
- ext_uplift_bl: Apply spatially variable uplift with boundary limits
- SPL: Execute Stream Power Law erosion for one time step
- SPL_transport: Execute transport-limited SPL with erosion and deposition
- init_erode_SPL: Initialize implicit SPL computation
- iteration_erode_SPL: Iterate SPL erosion solver
- erosion_to_source: Convert erosion to sediment source terms
- iterate_deposition: Apply sediment deposition based on transport capacity

Usage:
    import pyfastflow as pf
    
    # Setup flow router for erosion
    router = pf.flow.FlowRouter(nx=512, ny=512, dx=100.0)
    
    # Create erosion coefficient fields
    alpha_ = ti.field(ti.f32, shape=(nx*ny,))
    alpha__ = ti.field(ti.f32, shape=(nx*ny,))
    
    # Run landscape evolution
    for timestep in range(1000):
        router.compute_receivers()
        router.accumulate_constant_Q(1.0)
        pf.erodep.block_uplift(router.z, uplift_rate=1e-3)
        pf.erodep.SPL(router, alpha_, alpha__, Kr=1e-5)

Author: B.G.
"""

from .SPL import *

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

