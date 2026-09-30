"""CuPy-only shallow-water flood programs."""

from .inertial import InertialFloodProgram, build_inertial_flood_program

__all__ = ["InertialFloodProgram", "build_inertial_flood_program"]
