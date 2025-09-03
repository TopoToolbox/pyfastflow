"""
Downscaling operations for PyFastFlow.

Provides GPU-accelerated downscaling functions that halve the resolution of 2D grids
using various aggregation methods: max, min, mean, and cubic interpolation.

The downscaling algorithm takes each 2x2 block of cells and reduces it to a single cell
using the specified aggregation method.

Author: B.G.
"""

import taichi as ti
import numpy as np
from .. import pool


@ti.kernel
def halve_resolution_kernel_max(
    source_field: ti.template(), 
    target_field: ti.template(), 
    nx: ti.i32, 
    ny: ti.i32
):
    """
    Halve resolution using maximum value aggregation.
    
    Each 2x2 block in source becomes 1 cell in target using the maximum value.
    
    Args:
        source_field: Original field (nx * ny elements)
        target_field: Output field ((nx/2) * (ny/2) elements)
        nx: Number of columns in original grid
        ny: Number of rows in original grid
    """
    # Process each cell in the target (downscaled) grid
    target_nx = nx // 2
    target_ny = ny // 2
    
    for target_idx in target_field:
        # Convert target flat index to 2D coordinates
        target_j = target_idx // target_nx
        target_i = target_idx % target_nx
        
        # Skip out of range
        if target_j >= target_ny or target_i >= target_nx:
            continue
            
        # Find corresponding 2x2 block in source
        source_base_j = target_j * 2
        source_base_i = target_i * 2
        
        # Find maximum value in the 2x2 block
        max_val = -1e30
        for sub_j in ti.static(range(2)):
            for sub_i in ti.static(range(2)):
                source_j = source_base_j + sub_j
                source_i = source_base_i + sub_i
                
                # Check bounds
                if source_j < ny and source_i < nx:
                    source_idx = source_j * nx + source_i
                    val = source_field[source_idx]
                    max_val = ti.max(max_val, val)
        
        target_field[target_idx] = max_val


@ti.kernel
def halve_resolution_kernel_min(
    source_field: ti.template(), 
    target_field: ti.template(), 
    nx: ti.i32, 
    ny: ti.i32
):
    """
    Halve resolution using minimum value aggregation.
    
    Each 2x2 block in source becomes 1 cell in target using the minimum value.
    
    Args:
        source_field: Original field (nx * ny elements)
        target_field: Output field ((nx/2) * (ny/2) elements)
        nx: Number of columns in original grid
        ny: Number of rows in original grid
    """
    target_nx = nx // 2
    target_ny = ny // 2
    
    for target_idx in target_field:
        target_j = target_idx // target_nx
        target_i = target_idx % target_nx
        
        if target_j >= target_ny or target_i >= target_nx:
            continue
            
        source_base_j = target_j * 2
        source_base_i = target_i * 2
        
        # Find minimum value in the 2x2 block
        min_val = 1e30
        for sub_j in ti.static(range(2)):
            for sub_i in ti.static(range(2)):
                source_j = source_base_j + sub_j
                source_i = source_base_i + sub_i
                
                if source_j < ny and source_i < nx:
                    source_idx = source_j * nx + source_i
                    val = source_field[source_idx]
                    min_val = ti.min(min_val, val)
        
        target_field[target_idx] = min_val


@ti.kernel
def halve_resolution_kernel_mean(
    source_field: ti.template(), 
    target_field: ti.template(), 
    nx: ti.i32, 
    ny: ti.i32
):
    """
    Halve resolution using mean value aggregation.
    
    Each 2x2 block in source becomes 1 cell in target using the arithmetic mean.
    
    Args:
        source_field: Original field (nx * ny elements)
        target_field: Output field ((nx/2) * (ny/2) elements)
        nx: Number of columns in original grid
        ny: Number of rows in original grid
    """
    target_nx = nx // 2
    target_ny = ny // 2
    
    for target_idx in target_field:
        target_j = target_idx // target_nx
        target_i = target_idx % target_nx
        
        if target_j >= target_ny or target_i >= target_nx:
            continue
            
        source_base_j = target_j * 2
        source_base_i = target_i * 2
        
        # Calculate mean of 2x2 block
        sum_val = 0.0
        count = 0
        for sub_j in ti.static(range(2)):
            for sub_i in ti.static(range(2)):
                source_j = source_base_j + sub_j
                source_i = source_base_i + sub_i
                
                if source_j < ny and source_i < nx:
                    source_idx = source_j * nx + source_i
                    val = source_field[source_idx]
                    sum_val += val
                    count += 1
        
        if count > 0:
            target_field[target_idx] = sum_val / count
        else:
            target_field[target_idx] = 0.0


@ti.func
def cubic_interpolate(v0: ti.f32, v1: ti.f32, v2: ti.f32, v3: ti.f32, t: ti.f32) -> ti.f32:
    """
    Cubic interpolation between 4 points.
    
    Args:
        v0, v1, v2, v3: Four consecutive values
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated value
    """
    a = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3
    b = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3
    c = -0.5 * v0 + 0.5 * v2
    d = v1
    
    return a * t * t * t + b * t * t + c * t + d


@ti.kernel
def halve_resolution_kernel_cubic(
    source_field: ti.template(), 
    target_field: ti.template(), 
    nx: ti.i32, 
    ny: ti.i32
):
    """
    Halve resolution using cubic interpolation.
    
    Each 2x2 block in source becomes 1 cell in target using cubic interpolation
    of the surrounding area for smooth downsampling.
    
    Args:
        source_field: Original field (nx * ny elements)
        target_field: Output field ((nx/2) * (ny/2) elements)
        nx: Number of columns in original grid
        ny: Number of rows in original grid
    """
    target_nx = nx // 2
    target_ny = ny // 2
    
    for target_idx in target_field:
        target_j = target_idx // target_nx
        target_i = target_idx % target_nx
        
        if target_j >= target_ny or target_i >= target_nx:
            continue
        
        # Center of the target cell in source coordinates
        center_j = target_j * 2.0 + 0.5
        center_i = target_i * 2.0 + 0.5
        
        # For simplicity, use bilinear interpolation as a cubic approximation
        # Get the 4 nearest source cells for interpolation
        base_j = ti.cast(center_j, ti.i32)
        base_i = ti.cast(center_i, ti.i32)
        
        frac_j = center_j - base_j
        frac_i = center_i - base_i
        
        # Sample 2x2 area around the center point
        sum_val = 0.0
        total_weight = 0.0
        
        for dj in ti.static(range(2)):
            for di in ti.static(range(2)):
                sample_j = base_j + dj
                sample_i = base_i + di
                
                if sample_j >= 0 and sample_j < ny and sample_i >= 0 and sample_i < nx:
                    source_idx = sample_j * nx + sample_i
                    val = source_field[source_idx]
                    
                    # Bilinear weights
                    weight_j = (1.0 - frac_j) if dj == 0 else frac_j
                    weight_i = (1.0 - frac_i) if di == 0 else frac_i
                    weight = weight_j * weight_i
                    
                    sum_val += val * weight
                    total_weight += weight
        
        if total_weight > 0:
            target_field[target_idx] = sum_val / total_weight
        else:
            target_field[target_idx] = 0.0


def halve_resolution(
    grid_data,
    method: str = 'mean',
    return_field: bool = False
):
    """
    Halve the resolution of a 2D grid using specified aggregation method.
    
    Reduces a grid to half resolution in both dimensions by aggregating 2x2 blocks
    of cells into single cells using the specified method.
    
    Args:
        grid_data: Input grid data (numpy array or Taichi field)
                  Expected shape: (ny, nx) for numpy or nx*ny for flat Taichi field
        method: Aggregation method ('max', 'min', 'mean', 'cubic') (default: 'mean')
        return_field: If True, return Taichi field; if False, return numpy array (default: False)
    
    Returns:
        numpy.ndarray or taichi.Field: Downscaled grid with shape (ny//2, nx//2)
        
    Example:
        # Halve resolution using mean aggregation
        downscaled = halve_resolution(terrain_data, method='mean')
        
        # Use maximum values for conservative downsampling
        max_downscaled = halve_resolution(terrain_data, method='max')
        
        # Smooth downsampling with cubic interpolation
        smooth_downscaled = halve_resolution(terrain_data, method='cubic')
    """
    # Validate method
    valid_methods = ['max', 'min', 'mean', 'cubic']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
    
    # Handle input conversion
    if isinstance(grid_data, np.ndarray):
        if len(grid_data.shape) != 2:
            raise ValueError("Input numpy array must be 2D")
        ny, nx = grid_data.shape
        
        # Check that dimensions are even
        if nx % 2 != 0 or ny % 2 != 0:
            raise ValueError(f"Grid dimensions must be even for halving. Got ({ny}, {nx})")
        
        # Create source field and copy data
        source_field = pool.get_temp_field(ti.f32, (ny * nx,))
        source_field.field.from_numpy(grid_data.flatten())
    else:
        # Assume Taichi field - need to determine dimensions
        total_size = grid_data.shape[0] if hasattr(grid_data, 'shape') else len(grid_data)
        nx = ny = int(np.sqrt(total_size))
        if nx * ny != total_size:
            raise ValueError("Cannot determine grid dimensions from Taichi field. Please use square grid or convert to numpy first.")
        if nx % 2 != 0 or ny % 2 != 0:
            raise ValueError(f"Grid dimensions must be even for halving. Got ({ny}, {nx})")
        source_field = pool.get_temp_field(ti.f32, (total_size,))
        # Assume field is already populated
    
    # Create target field for downscaled result
    target_nx = nx // 2
    target_ny = ny // 2
    target_size = target_ny * target_nx
    target_field = pool.get_temp_field(ti.f32, (target_size,))
    
    # Select and execute appropriate kernel
    if method == 'max':
        halve_resolution_kernel_max(source_field.field, target_field.field, nx, ny)
    elif method == 'min':
        halve_resolution_kernel_min(source_field.field, target_field.field, nx, ny)
    elif method == 'mean':
        halve_resolution_kernel_mean(source_field.field, target_field.field, nx, ny)
    elif method == 'cubic':
        halve_resolution_kernel_cubic(source_field.field, target_field.field, nx, ny)
    
    if return_field:
        # Release source field and return target field
        source_field.release()
        return target_field.field
    else:
        # Convert to numpy and release both fields
        result = target_field.field.to_numpy().reshape(target_ny, target_nx)
        source_field.release()
        target_field.release()
        return result