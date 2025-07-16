"""
FastFlow Taichi - GPU-accelerated flow routing algorithms.

A high-performance Python package for hydrological flow routing on digital elevation models
using Taichi for GPU acceleration. Implements parallel algorithms for:

- Steepest descent flow routing
- Lake/depression handling (priority flood, carving)
- Flow accumulation using rake-and-compress algorithms
- Multiple boundary condition modes (open, periodic, custom)

Usage:
    import fastflow_taichi as ff
    
    # Access flow functions through the flow submodule
    ff.flow.lakeflow.depression_counter(...)
    ff.flow.receivers.compute_receivers(...)
    ff.flow.constants.NX

Author: B.G.
"""

__version__ = "0.1.0"
__author__ = "B.G."

# Import the flow submodule - users access functions as ff.flow.module.function()
from . import flow

# Only export the flow submodule
__all__ = ["flow"]