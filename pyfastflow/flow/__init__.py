"""
Flow algorithms submodule for FastFlow Taichi.

Contains all flow routing algorithms and utilities:

- neighbourer_flat: Vectorized grid navigation with boundary handling
- receivers: Steepest descent receiver computation
- downstream_propag: Parallel flow accumulation algorithms
- lakeflow: Depression filling and carving algorithms
- flowfields: FlowRouter class and field management
- fill_topo: Topography filling and manipulation utilities
- f32_i32_struct: Utility for lexicographic atomic operations
- util_taichi: General Taichi utility functions
- environment: System environment utilities

Usage:
    import pyfastflow as pf
    
    # Create flow router
    router = pf.flow.FlowRouter(nx=512, ny=512, dx=100.0)
    
    # Access individual modules
    pf.flow.lakeflow.depression_counter(rec)
    pf.flow.receivers.compute_receivers(z, rec, grad)

Author: B.G.
"""

# Import all flow modules - accessible as ff.flow.module_name
from ..constants import *
from .neighbourer_flat import *
from .receivers import *
from .downstream_propag import *
from .lakeflow import *
from .f32_i32_struct import *
from .util_taichi import *
from .flowfields import *
from .fill_topo import *
from .environment import *
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