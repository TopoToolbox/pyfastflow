"""
Flood modeling contexts for PyFastFlow.

The package root now exports only the reworked context API.
Legacy flood modules are still available under ``pyfastflow.flood.legacy``.

Author: B.G (02/2026)
"""

from .floodcontext import FloodContext

__all__ = ["FloodContext"]

# LEGACY (moved):
# - pyfastflow.flood.legacy.gf_fields
# - pyfastflow.flood.legacy.gf_hydrodynamics
# - pyfastflow.flood.legacy.gf_ls
# - pyfastflow.flood.legacy.gf_part
# - pyfastflow.flood.legacy.ggf_object
# - pyfastflow.flood.legacy.precipitation_gui
