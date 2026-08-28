"""
Raster manipulation tools for PyFastFlow.

The package root now exports only the reworked context API.
Legacy resizing/upscaling/downscaling helpers live under
``pyfastflow.rastermanip.legacy``.

Author: B.G (02/2026)
"""

from .rasmancontext import RasManContext

__all__ = ["RasManContext"]

# LEGACY (moved):
# - pyfastflow.rastermanip.legacy.upscaling
# - pyfastflow.rastermanip.legacy.downscaling
# - pyfastflow.rastermanip.legacy.resizing
