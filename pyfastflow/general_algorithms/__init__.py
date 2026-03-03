"""
General algorithm tools for PyFastFlow.

The package root now exposes the context-driven ``GAContext``. Older standalone
helpers remain available in their modules but are intentionally not re-exported
from here.

Author: B.G (02/2026)
"""

from .gacontext import GAContext

__all__ = ["GAContext"]

# LEGACY:
# from .math_utils import atan
# from .parallel_scan import inclusive_scan
# from .pingpong import fuse, getSrc, updateSrc
