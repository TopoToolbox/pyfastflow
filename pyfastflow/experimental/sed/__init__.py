"""Experimental CuPy sediment-transport programs.

``GolemParcelSedProgram``: GOLEM stream-power erosion with the eroded volume
released as stochastic sediment parcels deposited by a transport length.

``ShearParcelSedProgram``: shear-stress (excess Shields) erosion from a flow
depth field, with parcels walking stochastically on the water surface.

``ParticleLEMProgram``: a landscape evolution model run by stochastic
particles that update MFD drainage area, erode and deposit as they walk.

``MFDParcelSedProgram``: stream-power erosion on the MFD drainage area,
with parcels walking the MFD graph and depositing over a transport distance.

``CellularSedProgram``: local erosion/deposition laws (davy_lague, cover,
lift_impact) advanced by explicit steps with receivers drawn on the current
surface and the sediment flux moving one cell per step.

``PrecipitonSedProgram``: the same laws applied by precipitons, water
volumes carrying sediment over the current surface.

Author: B.G (10/2026)
"""

from .cellular import CellularSedProgram, build_cellular_sed_program
from .mfd_parcels import MFDParcelSedProgram, build_mfd_parcel_sed_program
from .particle_lem import ParticleLEMProgram, build_particle_lem_program
from .precipiton import PrecipitonSedProgram, build_precipiton_sed_program
from .program import GolemParcelSedProgram, build_golem_parcel_sed_program
from .shear import ShearParcelSedProgram, build_shear_parcel_sed_program

__all__ = [
    "GolemParcelSedProgram", "build_golem_parcel_sed_program",
    "ShearParcelSedProgram", "build_shear_parcel_sed_program",
    "ParticleLEMProgram", "build_particle_lem_program",
    "MFDParcelSedProgram", "build_mfd_parcel_sed_program",
    "CellularSedProgram", "build_cellular_sed_program",
    "PrecipitonSedProgram", "build_precipiton_sed_program",
]
