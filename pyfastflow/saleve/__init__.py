"""Analytical terrain generation."""

from ._speed import HILLSLOPE_MODELS
from .program import SLOPE_CORRECTIONS, VALLEY_MODELS, SaleveProgram

__all__ = ["HILLSLOPE_MODELS", "SLOPE_CORRECTIONS", "SaleveProgram", "VALLEY_MODELS"]
