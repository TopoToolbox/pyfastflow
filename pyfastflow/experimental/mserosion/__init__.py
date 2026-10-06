"""Multi-scale erosion (Schott et al. 2024) as a CuPy post-process.

``MultiScaleErosionProgram`` holds the three ported shader steps (erosion,
thermal, deposition) on one grid; ``amplify`` runs the release's preset
sequence on a heightfield: per stage, erosion, then thermal, then
deposition steps, with a x2 bilinear upsampling between stages.
"""

from .amplify import PRESET, amplify, upsample_x2
from .program import MultiScaleErosionProgram, build_mserosion_program

__all__ = ["MultiScaleErosionProgram", "PRESET", "amplify",
           "build_mserosion_program", "upsample_x2"]
