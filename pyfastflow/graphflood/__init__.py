"""GraphFlood programs: Vanilla (iterative), Relax (analytical relaxation)
and Particles (live particle processors), on one shared recipe (``_base``).
"""

from ._convergence import ConvergenceRule
from .particles import GraphFloodParticles, build_graphflood_particles
from .relax import GraphFloodRelax, build_graphflood_relax
from .vanilla import GraphFloodVanilla, build_graphflood_vanilla

__all__ = [
    "ConvergenceRule",
    "GraphFloodParticles", "GraphFloodRelax", "GraphFloodVanilla",
    "build_graphflood_particles", "build_graphflood_relax",
    "build_graphflood_vanilla",
]
