"""
Internal hillshading wrapper for Grid class.

This module contains the implementation details for hillshading functionality,
keeping the Grid class interface clean while providing comprehensive hillshading
capabilities with automatic pool-based memory management.

This is an internal module - users should call grid.hillshade() instead of
importing functions from this module directly.

Author: B.G.
"""

import math

import numpy as np

import pyfastflow as pf
from pyfastflow import constants as cte


def compute_hillshade(
    grid,
    altitude_deg=45.0,
    azimuth_deg=315.0,
    z_factor=1.0,
    multidirectional=False,
    azimuths_deg=None,
    weights=None,
    style=None,
):
    """
    Internal function to compute hillshading with comprehensive options.

    This function handles all hillshading computation logic, including single-direction,
    multidirectional, and styled hillshading. Uses pool-based memory management for
    all temporary GPU fields.

    Args:
        grid: Grid object containing elevation data and parameters
        altitude_deg (float): Sun altitude angle in degrees (0-90°)
        azimuth_deg (float): Sun azimuth angle in degrees (0-360°)
        z_factor (float): Vertical exaggeration factor
        multidirectional (bool): If True, use multiple light directions
        azimuths_deg (list): List of azimuth angles for multidirectional
        weights (list): Weights for each azimuth direction
        style (str): Predefined style name for quick configuration

    Returns:
        np.ndarray: 2D hillshade array with shape (ny, nx) and values in [0, 1]

    Author: B.G.
    """

    # Handle predefined styles first
    if style is not None:
        return _apply_style(grid, style)

    # Handle multidirectional hillshading
    if multidirectional or azimuths_deg is not None:
        return _compute_multidirectional(
            grid, altitude_deg, z_factor, azimuths_deg, weights
        )

    # Single-direction hillshading (default case)
    return _compute_single_direction(grid, altitude_deg, azimuth_deg, z_factor)


def _compute_single_direction(grid, altitude_deg, azimuth_deg, z_factor):
    """Compute single-direction hillshading."""
    import taichi as ti

    import pyfastflow.visu.hillshading as hs

    # Convert angles to radians
    zenith_rad = math.radians(90.0 - altitude_deg)
    azimuth_rad = math.radians(azimuth_deg)

    # Get temporary hillshade field from pool
    with pf.pool.temp_field(cte.FLOAT_TYPE_TI, (grid.nx * grid.ny,)) as hillshade_field:
        # Compute hillshading using vectorized kernel
        hs.hillshade_vectorized(
            grid.z.field, hillshade_field.field, zenith_rad, azimuth_rad, z_factor
        )

        # Convert to 2D numpy array and return
        return hillshade_field.field.to_numpy().reshape(grid.rshp)


def _compute_multidirectional(grid, altitude_deg, z_factor, azimuths_deg, weights):
    """Compute multidirectional hillshading with multiple light sources."""
    import taichi as ti

    import pyfastflow.visu.hillshading as hs

    # Set default azimuths if not provided (2 cardinal directions)
    if azimuths_deg is None:
        azimuths_deg = [315.0, 45.0]  # NW, NE

    # Set equal weights if not provided
    if weights is None:
        weights = [1.0] * len(azimuths_deg)

    # Validate inputs
    if len(weights) != len(azimuths_deg):
        raise ValueError(
            f"weights length ({len(weights)}) must match azimuths_deg length ({len(azimuths_deg)})"
        )

    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Sum of weights must be positive")
    weights = [w / total_weight for w in weights]

    # Convert altitude to zenith in radians
    zenith_rad = math.radians(90.0 - altitude_deg)

    # Initialize result array
    result = np.zeros(grid.rshp, dtype=cte.FLOAT_TYPE_NP)

    # Get temporary hillshade field from pool for each direction
    with pf.pool.temp_field(cte.FLOAT_TYPE_TI, (grid.nx * grid.ny,)) as hillshade_field:
        # Compute hillshading for each azimuth direction
        for azimuth_deg, weight in zip(azimuths_deg, weights):
            azimuth_rad = math.radians(azimuth_deg)

            # Compute hillshading for this direction
            hs.hillshade_vectorized(
                grid.z.field, hillshade_field.field, zenith_rad, azimuth_rad, z_factor
            )

            # Add weighted contribution to result
            direction_result = hillshade_field.field.to_numpy().reshape(grid.rshp)
            result += weight * direction_result

    # Ensure values stay in [0, 1] range
    return np.clip(result, 0.0, 1.0)


def _apply_style(grid, style):
    """Apply predefined hillshading style configurations."""

    # Predefined style configurations
    styles = {
        "default": {"altitude_deg": 45.0, "azimuth_deg": 315.0, "z_factor": 1.0},
        "dramatic": {"altitude_deg": 30.0, "azimuth_deg": 315.0, "z_factor": 2.0},
        "subtle": {"altitude_deg": 60.0, "azimuth_deg": 315.0, "z_factor": 0.5},
        "east": {"altitude_deg": 45.0, "azimuth_deg": 90.0, "z_factor": 1.0},
        "south": {"altitude_deg": 45.0, "azimuth_deg": 180.0, "z_factor": 1.0},
        "multi": {
            "altitude_deg": 45.0,
            "z_factor": 1.0,
            "azimuths_deg": [315, 45, 135, 225],
            "multidirectional": True,
        },
        "multi_smooth": {
            "altitude_deg": 45.0,
            "z_factor": 1.0,
            "azimuths_deg": [0, 45, 90, 135, 180, 225, 270, 315],
            "multidirectional": True,
        },
    }

    if style not in styles:
        available = ", ".join(f"'{s}'" for s in styles.keys())
        raise ValueError(f"Unknown style '{style}'. Available styles: {available}")

    config = styles[style]

    # Use multidirectional for multi styles
    if config.get("multidirectional", False):
        return _compute_multidirectional(
            grid,
            config["altitude_deg"],
            config["z_factor"],
            config["azimuths_deg"],
            None,  # Use equal weights
        )
    else:
        # Use single-direction hillshading
        return _compute_single_direction(
            grid, config["altitude_deg"], config["azimuth_deg"], config["z_factor"]
        )
