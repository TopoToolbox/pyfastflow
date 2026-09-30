"""Experimental CuPy landscape-evolution programs.

``GolemNoSedProgram`` is the sediment-free foundation of GOLEM.  It couples
SFD drainage, implicit detachment-limited stream-power incision, uplift, and
implicit linear hillslope diffusion on the shared PyFastFlow grid.

``GolemSedProgram`` adds a SPACE-style sediment cover, SFD sediment routing,
transport-length deposition, and optional implicit surface diffusion.
"""

from .program import GolemNoSedProgram, build_golem_nosed_program
from .sed import GolemSedProgram, build_golem_sed_program

__all__ = [
    "GolemNoSedProgram", "build_golem_nosed_program",
    "GolemSedProgram", "build_golem_sed_program",
]
