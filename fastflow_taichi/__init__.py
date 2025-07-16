"""
FastFlow Taichi - GPU-accelerated flow routing algorithms.

A high-performance Python package for hydrological flow routing on digital elevation models
using Taichi for GPU acceleration. Implements parallel algorithms for:

- Steepest descent flow routing
- Lake/depression handling (priority flood, carving)
- Flow accumulation using rake-and-compress algorithms
- Multiple boundary condition modes (open, periodic, custom)

Author: B.G.
"""

__version__ = "0.1.0"
__author__ = "B.G."

# Import all flow modules from the flow subfolder
from .flow import constants
from .flow import neighbourer_flat
from .flow import receivers
from .flow import downstream_propag
from .flow import lakeflow
from .flow import f32_i32_struct
from .flow import util_taichi

# Export all flow modules for easy access
__all__ = [
    "constants",
    "neighbourer_flat", 
    "receivers",
    "downstream_propag",
    "lakeflow",
    "f32_i32_struct",
    "util_taichi"
]