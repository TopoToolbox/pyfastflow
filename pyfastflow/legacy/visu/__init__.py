"""
Visualization tools for PyFastFlow.

The package root exposes:
- ``live`` as the current interactive viewer module
- ``hillshading`` as the standalone ndarray-kernel module
- ``VisuContext`` as the streamlined grid-bound hillshading interface

Author: B.G (02/2026)
"""

from . import hillshading
from . import live
from .live import SurfaceViewer
from .visuctx import VisuContext

__all__ = ["live", "hillshading", "SurfaceViewer", "VisuContext"]
