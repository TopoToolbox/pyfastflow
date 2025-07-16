"""
Flow algorithms submodule for FastFlow Taichi.

Contains all flow routing algorithms and utilities:

- constants: Compile-time parameters and grid configuration
- neighbourer_flat: Vectorized grid navigation with boundary handling
- receivers: Steepest descent receiver computation
- downstream_propag: Parallel flow accumulation algorithms
- lakeflow: Depression filling and carving algorithms
- f32_i32_struct: Utility for lexicographic atomic operations
- util_taichi: General Taichi utility functions

Usage:
    import fastflow_taichi as ff
    
    # Access individual modules
    ff.flow.constants.NX
    ff.flow.lakeflow.depression_counter(rec)
    ff.flow.receivers.compute_receivers(z, rec, grad)

Author: B.G.
"""

# Import all flow modules - accessible as ff.flow.module_name
from .constants import *
from .neighbourer_flat import *
from .receivers import *
from .downstream_propag import *
from .lakeflow import *
from .f32_i32_struct import *
from .util_taichi import *

# Export all modules
__all__ = [
    "constants",
    "neighbourer_flat", 
    "receivers",
    "downstream_propag",
    "lakeflow",
    "f32_i32_struct",
    "util_taichi"
]