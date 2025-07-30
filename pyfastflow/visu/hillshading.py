"""
Hillshading and Terrain Visualization

This module provides GPU-accelerated hillshading functions for digital elevation models.
Supports both vectorized operations using the flow routing neighboring system and
generic 2D array operations for external numpy arrays.

Hillshading Algorithm:
    Computes shaded relief based on illumination angle and terrain surface normals.
    Uses central difference gradients with boundary-aware neighboring for robust
    edge handling across different boundary conditions.

Key Features:
    - Single direction and multidirectional hillshading
    - Integration with FlowRouter boundary conditions
    - Support for external numpy arrays with optional masking
    - GPU-accelerated computation with Taichi
    - Handles no-data values and periodic boundaries

Mathematical Basis:
    Hillshade = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect)
    Where:
    - slope: terrain slope angle
    - aspect: terrain aspect (direction of steepest descent)
    - zenith: sun zenith angle (90° - altitude)
    - azimuth: sun azimuth angle

Author: B. Gailleton
"""

import taichi as ti
import numpy as np
import math
from .. import constants as cte
from ..flow import neighbourer_flat as nei


@ti.kernel
def hillshade_vectorized(z: ti.template(), hillshade: ti.template(), 
                        zenith_rad: ti.f32, azimuth_rad: ti.f32, z_factor: ti.f32):
    """
    Compute hillshading for vectorized elevation data using neighboring system.
    
    Uses the flow routing neighboring abstraction to handle boundary conditions,
    no-data values, and periodic boundaries automatically. Computes surface
    normals using central difference gradients with boundary-aware neighbors.
    
    Args:
        z: Input elevation field (1D vectorized)
        hillshade: Output hillshade values (1D vectorized)
        zenith_rad: Sun zenith angle in radians (90° - altitude)
        azimuth_rad: Sun azimuth angle in radians (0° = North, clockwise)
        z_factor: Vertical exaggeration factor for slope calculation
        
    Mathematical Implementation:
        1. Compute dz/dx and dz/dy using central differences
        2. Calculate slope and aspect from gradients
        3. Apply hillshading formula with illumination angles
        
    Boundary Handling:
        - Uses nei.neighbour() for boundary-aware neighbor access
        - Handles -1 return values (blocked/invalid neighbors)
        - Maintains gradient continuity across periodic boundaries
        
    Author: B. Gailleton
    """
    for i in z:
        # Get neighbors using boundary-aware system
        left_idx = nei.neighbour(i, 1)   # Left neighbor (direction 1)  
        right_idx = nei.neighbour(i, 2)  # Right neighbor (direction 2)
        top_idx = nei.neighbour(i, 0)    # Top neighbor (direction 0)
        bottom_idx = nei.neighbour(i, 3) # Bottom neighbor (direction 3)
        
        # Compute gradients using central differences with boundary handling
        dz_dx = 0.0
        dz_dy = 0.0
        
        # X-gradient (East-West)
        if left_idx != -1 and right_idx != -1:
            # Central difference (preferred)
            dz_dx = (z[right_idx] - z[left_idx]) / (2.0 * cte.DX)
        elif left_idx != -1:
            # Forward difference
            dz_dx = (z[i] - z[left_idx]) / cte.DX
        elif right_idx != -1:
            # Backward difference  
            dz_dx = (z[right_idx] - z[i]) / cte.DX
        # else: dz_dx remains 0 (no valid neighbors)
        
        # Y-gradient (North-South) - note: top is negative Y direction
        if top_idx != -1 and bottom_idx != -1:
            # Central difference (preferred)
            dz_dy = (z[bottom_idx] - z[top_idx]) / (2.0 * cte.DX)
        elif top_idx != -1:
            # Forward difference
            dz_dy = (z[i] - z[top_idx]) / cte.DX
        elif bottom_idx != -1:
            # Backward difference
            dz_dy = (z[bottom_idx] - z[i]) / cte.DX
        # else: dz_dy remains 0 (no valid neighbors)
        
        # Apply vertical exaggeration
        dz_dx *= z_factor
        dz_dy *= z_factor
        
        # Calculate slope and aspect
        slope_rad = ti.math.atan(ti.math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))
        
        # Aspect calculation (direction of steepest descent)
        aspect_rad = 0.0
        if dz_dx != 0.0 or dz_dy != 0.0:
            # atan2 gives aspect in mathematical convention (CCW from East)
            # Convert to geographic convention (CW from North)
            aspect_rad = ti.math.pi/2.0 - ti.math.atan2(dz_dy, dz_dx)
            if aspect_rad < 0.0:
                aspect_rad += 2.0 * ti.math.pi
        
        # Hillshading calculation
        # Standard formula: cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth-aspect)
        cos_slope = ti.math.cos(slope_rad)
        sin_slope = ti.math.sin(slope_rad)
        cos_zenith = ti.math.cos(zenith_rad)
        sin_zenith = ti.math.sin(zenith_rad)
        cos_azimuth_aspect = ti.math.cos(azimuth_rad - aspect_rad)
        
        hillshade_value = cos_zenith * cos_slope + sin_zenith * sin_slope * cos_azimuth_aspect
        
        # Clamp to [0, 1] range and store
        hillshade[i] = ti.math.max(0.0, ti.math.min(1.0, hillshade_value))


@ti.kernel  
def hillshade_2d(z: ti.types.ndarray(dtype=ti.f32, ndim=2),
                hillshade: ti.types.ndarray(dtype=ti.f32, ndim=2),
                zenith_rad: ti.f32, azimuth_rad: ti.f32, z_factor: ti.f32, dx: ti.f32):
    """
    Compute hillshading for 2D numpy arrays using generic neighboring.
    
    Operates directly on 2D numpy arrays without boundary condition abstraction.
    Uses simple edge clamping for boundary handling. Suitable for external
    elevation data that doesn't use the flow routing system.
    
    Args:
        z: Input elevation array (2D numpy array)
        hillshade: Output hillshade array (2D numpy array, same shape as z)
        zenith_rad: Sun zenith angle in radians (90° - altitude)
        azimuth_rad: Sun azimuth angle in radians (0° = North, clockwise)
        z_factor: Vertical exaggeration factor for slope calculation
        dx: Grid cell size for gradient calculation
        
    Boundary Handling:
        - Uses array bounds checking with edge clamping
        - No periodic boundary support
        - Forward/backward differences at edges
        
    Performance:
        - Direct array indexing for maximum speed
        - No boundary condition abstraction overhead
        - Suitable for large external datasets
        
    Author: B. Gailleton
    """
    ny, nx = z.shape
    
    for i, j in ti.ndrange(ny, nx):
        # Compute gradients using central differences with boundary clamping
        dz_dx = 0.0
        dz_dy = 0.0
        
        # X-gradient (East-West)
        if j > 0 and j < nx - 1:
            # Central difference (preferred)
            dz_dx = (z[i, j + 1] - z[i, j - 1]) / (2.0 * dx)
        elif j > 0:
            # Backward difference (right edge)
            dz_dx = (z[i, j] - z[i, j - 1]) / dx
        elif j < nx - 1:
            # Forward difference (left edge)
            dz_dx = (z[i, j + 1] - z[i, j]) / dx
        # else: dz_dx remains 0 (single column case)
        
        # Y-gradient (North-South)
        if i > 0 and i < ny - 1:
            # Central difference (preferred)
            dz_dy = (z[i + 1, j] - z[i - 1, j]) / (2.0 * dx)
        elif i > 0:
            # Backward difference (bottom edge)
            dz_dy = (z[i, j] - z[i - 1, j]) / dx
        elif i < ny - 1:
            # Forward difference (top edge)
            dz_dy = (z[i + 1, j] - z[i, j]) / dx
        # else: dz_dy remains 0 (single row case)
        
        # Apply vertical exaggeration
        dz_dx *= z_factor
        dz_dy *= z_factor
        
        # Calculate slope and aspect
        slope_rad = ti.math.atan(ti.math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))
        
        # Aspect calculation (direction of steepest descent)
        aspect_rad = 0.0
        if dz_dx != 0.0 or dz_dy != 0.0:
            # atan2 gives aspect in mathematical convention (CCW from East)
            # Convert to geographic convention (CW from North)
            aspect_rad = ti.math.pi/2.0 - ti.math.atan2(dz_dy, dz_dx)
            if aspect_rad < 0.0:
                aspect_rad += 2.0 * ti.math.pi
        
        # Hillshading calculation
        cos_slope = ti.math.cos(slope_rad)
        sin_slope = ti.math.sin(slope_rad)
        cos_zenith = ti.math.cos(zenith_rad)
        sin_zenith = ti.math.sin(zenith_rad)
        cos_azimuth_aspect = ti.math.cos(azimuth_rad - aspect_rad)
        
        hillshade_value = cos_zenith * cos_slope + sin_zenith * sin_slope * cos_azimuth_aspect
        
        # Clamp to [0, 1] range and store
        hillshade[i, j] = ti.math.max(0.0, ti.math.min(1.0, hillshade_value))


def hillshade_flowrouter(flowrouter, altitude_deg=45.0, azimuth_deg=315.0, z_factor=1.0):
    """
    Compute hillshading for FlowRouter elevation data.
    
    Uses the FlowRouter's elevation field and boundary conditions to compute
    hillshaded relief. Leverages the neighboring system for proper boundary
    handling including periodic boundaries and no-data regions.
    
    Args:
        flowrouter: FlowRouter object with elevation data and boundary settings
        altitude_deg: Sun altitude angle in degrees (0-90°, default: 45°)
        azimuth_deg: Sun azimuth angle in degrees (0° = North, CW, default: 315°)
        z_factor: Vertical exaggeration factor (default: 1.0)
        
    Returns:
        numpy.ndarray: 2D hillshade array with values in [0, 1] range
        
    Implementation:
        - Uses flowrouter.z_ as temporary storage (no new fields created)
        - Respects flowrouter boundary conditions via neighboring system
        - Converts between angular units (degrees to radians)
        - Reshapes vectorized output to 2D array
        
    Boundary Conditions:
        - Inherits from flowrouter.boundary_mode setting
        - Supports normal, periodic EW/NS, and custom boundaries
        - Handles no-data values automatically via neighboring system
        
    Author: B. Gailleton
    """
    # Convert angles to radians
    zenith_rad = math.radians(90.0 - altitude_deg)  # Zenith = 90° - altitude
    azimuth_rad = math.radians(azimuth_deg)
    
    # Use flowrouter.z_ as temporary storage for hillshade results
    hillshade_vectorized(flowrouter.z, flowrouter.z_, zenith_rad, azimuth_rad, z_factor)
    
    # Convert vectorized result to 2D numpy array
    hillshade_np = flowrouter.z_.to_numpy().reshape((cte.NY, cte.NX))
    
    return hillshade_np


def hillshade_multidirectional_flowrouter(flowrouter, altitude_deg=45.0, z_factor=1.0, 
                                        azimuths_deg=None):
    """
    Compute multidirectional hillshading for FlowRouter elevation data.
    
    Combines hillshading from multiple light sources to create more balanced
    illumination that reduces shadowing artifacts. Uses multiple azimuth
    angles while maintaining consistent altitude.
    
    Args:
        flowrouter: FlowRouter object with elevation data and boundary settings
        altitude_deg: Sun altitude angle in degrees (0-90°, default: 45°)
        z_factor: Vertical exaggeration factor (default: 1.0)
        azimuths_deg: List of azimuth angles in degrees (default: [315, 45, 135, 225])
        
    Returns:
        numpy.ndarray: 2D multidirectional hillshade array with values in [0, 1]
        
    Algorithm:
        1. Compute hillshade for each azimuth direction
        2. Combine results using averaging for balanced illumination
        3. Store intermediate results in numpy arrays (no new fields)
        
    Default Azimuths:
        - 315° (NW): Classic GIS default
        - 45° (NE): Opposite diagonal  
        - 135° (SE): Perpendicular directions
        - 225° (SW): Complete 90° coverage
        
    Memory Usage:
        - Uses flowrouter.z_ for each computation
        - Stores intermediates in numpy arrays
        - No permanent field allocation
        
    Author: B. Gailleton
    """
    if azimuths_deg is None:
        azimuths_deg = [315.0, 45.0, 135.0, 225.0]  # Four cardinal diagonal directions
    
    # Convert altitude to zenith
    zenith_rad = math.radians(90.0 - altitude_deg)
    
    # Initialize result array
    result_shape = (cte.NY, cte.NX)
    multidirectional_hs = np.zeros(result_shape, dtype=np.float32)
    
    # Compute and accumulate hillshade for each azimuth
    for azimuth_deg in azimuths_deg:
        azimuth_rad = math.radians(azimuth_deg)
        
        # Compute hillshade for this azimuth using flowrouter.z_ as temp storage
        hillshade_vectorized(flowrouter.z, flowrouter.z_, zenith_rad, azimuth_rad, z_factor)
        
        # Convert to 2D and accumulate
        single_hs = flowrouter.z_.to_numpy().reshape(result_shape)
        multidirectional_hs += single_hs
    
    # Average the accumulated hillshades
    multidirectional_hs /= len(azimuths_deg)
    
    # Ensure values remain in [0, 1] range
    multidirectional_hs = np.clip(multidirectional_hs, 0.0, 1.0)
    
    return multidirectional_hs


def hillshade_numpy(elevation_array, altitude_deg=45.0, azimuth_deg=315.0, 
                   z_factor=1.0, dx=1.0, mask=None):
    """
    Compute hillshading for external 2D numpy elevation arrays.
    
    Operates on arbitrary 2D elevation data without FlowRouter integration.
    Uses simple boundary handling with edge clamping. Supports optional
    masking to set specific regions to NaN values.
    
    Args:
        elevation_array: Input 2D elevation data (numpy array)
        altitude_deg: Sun altitude angle in degrees (0-90°, default: 45°)
        azimuth_deg: Sun azimuth angle in degrees (0° = North, CW, default: 315°)
        z_factor: Vertical exaggeration factor (default: 1.0)
        dx: Grid cell size for gradient calculation (default: 1.0)
        mask: Optional boolean mask array, True values set to NaN (default: None)
        
    Returns:
        numpy.ndarray: 2D hillshade array with values in [0, 1] or NaN
        
    Boundary Handling:
        - Uses simple edge clamping (no periodic boundaries)
        - Forward/backward differences at array edges
        - No special no-data value handling
        
    Masking:
        - If mask provided, True values in mask → NaN in output
        - Useful for water bodies, no-data regions, etc.
        - Applied after hillshade computation
        
    Example Usage:
        ```python
        # Basic usage
        hs = hillshade_numpy(dem_array, altitude_deg=30, azimuth_deg=270)
        
        # With masking for water bodies
        water_mask = dem_array < sea_level
        hs = hillshade_numpy(dem_array, mask=water_mask)
        ```
        
    Author: B. Gailleton
    """
    # Validate input
    if elevation_array.ndim != 2:
        raise ValueError("elevation_array must be 2D")
    
    # Convert angles to radians
    zenith_rad = math.radians(90.0 - altitude_deg)
    azimuth_rad = math.radians(azimuth_deg)
    
    # Create output array
    hillshade_array = np.zeros_like(elevation_array, dtype=np.float32)
    
    # Compute hillshade using 2D kernel
    hillshade_2d(elevation_array.astype(np.float32), hillshade_array, 
                zenith_rad, azimuth_rad, z_factor, dx)
    
    # Apply mask if provided
    if mask is not None:
        if mask.shape != elevation_array.shape:
            raise ValueError("mask shape must match elevation_array shape")
        hillshade_array[mask] = np.nan
    
    return hillshade_array


def hillshade_multidirectional_numpy(elevation_array, altitude_deg=45.0, z_factor=1.0,
                                   dx=1.0, azimuths_deg=None, mask=None):
    """
    Compute multidirectional hillshading for external 2D numpy elevation arrays.
    
    Combines hillshading from multiple light sources for more balanced
    illumination. Operates on arbitrary 2D elevation data with optional masking.
    
    Args:
        elevation_array: Input 2D elevation data (numpy array)
        altitude_deg: Sun altitude angle in degrees (0-90°, default: 45°)
        z_factor: Vertical exaggeration factor (default: 1.0)
        dx: Grid cell size for gradient calculation (default: 1.0)
        azimuths_deg: List of azimuth angles in degrees (default: [315, 45, 135, 225])
        mask: Optional boolean mask array, True values set to NaN (default: None)
        
    Returns:
        numpy.ndarray: 2D multidirectional hillshade array with values in [0, 1] or NaN
        
    Algorithm:
        1. Compute hillshade for each azimuth using hillshade_2d kernel
        2. Average results for balanced illumination
        3. Apply optional masking
        
    Memory Efficiency:
        - Accumulates results in single output array
        - Temporary arrays automatically managed
        - Suitable for large datasets
        
    Example Usage:
        ```python
        # Standard multidirectional hillshade
        hs = hillshade_multidirectional_numpy(dem_array)
        
        # Custom light sources with masking
        custom_azimuths = [0, 90, 180, 270]  # Cardinal directions
        water_mask = dem_array < 0
        hs = hillshade_multidirectional_numpy(dem_array, azimuths_deg=custom_azimuths, 
                                            mask=water_mask)
        ```
        
    Author: B. Gailleton
    """
    # Validate input
    if elevation_array.ndim != 2:
        raise ValueError("elevation_array must be 2D")
    
    if azimuths_deg is None:
        azimuths_deg = [315.0, 45.0, 135.0, 225.0]  # Four cardinal diagonal directions
    
    # Convert altitude to zenith
    zenith_rad = math.radians(90.0 - altitude_deg)
    
    # Initialize result array
    multidirectional_hs = np.zeros_like(elevation_array, dtype=np.float32)
    temp_hs = np.zeros_like(elevation_array, dtype=np.float32)
    
    # Compute and accumulate hillshade for each azimuth
    for azimuth_deg in azimuths_deg:
        azimuth_rad = math.radians(azimuth_deg)
        
        # Compute hillshade for this azimuth
        hillshade_2d(elevation_array.astype(np.float32), temp_hs, 
                    zenith_rad, azimuth_rad, z_factor, dx)
        
        # Accumulate results
        multidirectional_hs += temp_hs
    
    # Average the accumulated hillshades
    multidirectional_hs /= len(azimuths_deg)
    
    # Ensure values remain in [0, 1] range
    multidirectional_hs = np.clip(multidirectional_hs, 0.0, 1.0)
    
    # Apply mask if provided
    if mask is not None:
        if mask.shape != elevation_array.shape:
            raise ValueError("mask shape must match elevation_array shape")
        multidirectional_hs[mask] = np.nan
    
    return multidirectional_hs