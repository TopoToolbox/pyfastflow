"""FullSWOF 2D shallow-water solver as a CuPy Program.

Modules: ``_common`` (precision, boundary codes, device layouts),
``_limiters``, ``_riemann``, ``_friction``, ``_infiltration``,
``_boundary`` (ghost states), ``_reconstruction`` (face states),
``_kernels`` (kernels and step routine), ``program`` (FuswProgram).

Author: B.G (10/2026)
"""

from ._common import BOUNDARY_CODES
from .program import FuswProgram, build_fusw_program

__all__ = ["BOUNDARY_CODES", "FuswProgram", "build_fusw_program"]
