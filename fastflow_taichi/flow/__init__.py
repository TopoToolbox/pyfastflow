"""
FastFlow Taichi - GPU-accelerated flow routing algorithms.

A high-performance Python package for hydrological flow routing on digital elevation models
using Taichi for GPU acceleration. Implements parallel algorithms for:

- Steepest descent flow routing
- Lake/depression handling (priority flood, carving)
- Flow accumulation using rake-and-compress algorithms
- Multiple boundary condition modes (open, periodic, custom)

Key modules:
- constants: Compile-time parameters and grid configuration
- neighbourer_flat: Vectorized grid navigation with boundary handling
- receivers: Steepest descent receiver computation
- downstream_propag: Parallel flow accumulation algorithms
- lakeflow: Depression filling and carving algorithms
- f32_i32_struct: Utility for lexicographic atomic operations
- util_taichi: General Taichi utility functions

Author: B.G.
"""

__version__ = "0.1.0"
__author__ = "B.G."

# Import main modules for easier access
from . import constants
from . import neighbourer_flat
from . import receivers
from . import downstream_propag
from . import lakeflow
from . import f32_i32_struct
from . import util_taichi

__all__ = [
    "constants",
    "neighbourer_flat", 
    "receivers",
    "downstream_propag",
    "lakeflow",
    "f32_i32_struct",
    "util_taichi"
]