"""
Upscaling operations for PyFastFlow.

Provides GPU-accelerated upscaling functions that double the resolution of 2D grids
while preserving slope characteristics and optionally adding realistic noise variations.

The upscaling algorithm:
1. Analyzes neighboring cells to determine slope direction
2. Performs linear interpolation based on slope direction
3. Optionally adds controlled noise (order of magnitude lower than original slope)

Author: B.G.
"""

import numpy as np
import taichi as ti

from .. import pool
from ..grid import neighbourer_flat_param as nei


@ti.func
def get_slope_direction(
    source_field: ti.template(), idx: ti.i32, nx: ti.i32, ny: ti.i32
) -> ti.Vector:
    """
    Compute slope direction vector for a given cell.

    Args:
        source_field: Original field to analyze
        idx: Flat index of the cell
        nx: Number of columns in original grid
        ny: Number of rows in original grid

    Returns:
        ti.Vector: Normalized slope direction (dx, dy)
    """
    center_val = source_field[idx]

    # Initialize gradient components
    grad_x = 0.0
    grad_y = 0.0
    valid_neighbors = 0

    # Check all 4 cardinal neighbors for gradient computation
    for k in ti.static(range(4)):
        neighbor_idx = nei.neighbour_n_param(idx, k, nx, ny)
        if neighbor_idx != -1:
            neighbor_val = source_field[neighbor_idx]
            diff = center_val - neighbor_val

            # Convert direction to gradient components
            if k == 0:  # top
                grad_y += diff
            elif k == 1:  # left
                grad_x += diff
            elif k == 2:  # right
                grad_x -= diff
            elif k == 3:  # bottom
                grad_y -= diff

            valid_neighbors += 1

    # Normalize gradient
    if valid_neighbors > 0:
        grad_x /= valid_neighbors
        grad_y /= valid_neighbors

    # Return normalized direction vector
    magnitude = ti.sqrt(grad_x * grad_x + grad_y * grad_y)
    result = ti.Vector([0.0, 0.0])
    if magnitude > 1e-8:
        result = ti.Vector([grad_x / magnitude, grad_y / magnitude])
    return result


@ti.kernel
def double_resolution_kernel(
    source_field: ti.template(),
    target_field: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    noise_amplitude: ti.f32,
):
    """
    Double the resolution of a 2D field with slope-preserving interpolation.

    Each cell in the source becomes 4 cells in the target (2x2 block).
    Interpolation preserves the slope direction determined from neighbors.

    Args:
        source_field: Original field (nx * ny elements)
        target_field: Output field (2*nx * 2*ny elements)
        nx: Number of columns in original grid
        ny: Number of rows in original grid
        noise_amplitude: Amplitude of random noise to add
    """
    # Process each cell in the original grid
    for idx in source_field:
        # Convert flat index to 2D coordinates
        j = idx // nx  # row
        i = idx % nx  # column

        # Skip boundary conditions that are out of range
        if j >= ny or i >= nx:
            continue

        # Get slope direction for this cell
        slope_dir = get_slope_direction(source_field, idx, nx, ny)
        center_val = source_field[idx]

        # Calculate base interpolation strength (quarter of the slope magnitude)
        base_interp = ti.abs(slope_dir[0]) + ti.abs(slope_dir[1])
        interp_strength = base_interp * 0.25

        # Generate the 4 new cells (2x2 block) in target grid
        for sub_j in ti.static(range(2)):
            for sub_i in ti.static(range(2)):
                # Target coordinates in upscaled grid
                target_j = j * 2 + sub_j
                target_i = i * 2 + sub_i
                target_idx = target_j * (nx * 2) + target_i

                # Relative position within the 2x2 block ([-0.5, +0.5])
                rel_x = sub_i - 0.5
                rel_y = sub_j - 0.5

                # Apply slope-based interpolation
                slope_contribution = (
                    slope_dir[0] * rel_x + slope_dir[1] * rel_y
                ) * interp_strength
                interpolated_val = center_val + slope_contribution

                # Add controlled noise (order of magnitude lower)
                noise_factor = noise_amplitude * 0.1 * interp_strength
                noise_val = (ti.random(ti.f32) - 0.5) * 2.0 * noise_factor

                # Final value
                final_val = interpolated_val + noise_val
                target_field[target_idx] = final_val


def double_resolution(
    grid_data,
    noise_amplitude: float = 0.0,
    return_field: bool = False,
    nx: int | None = None,
    ny: int | None = None,
):
    """
    Double the resolution of a 2D grid with slope-preserving interpolation.

    Creates a new grid with 2x resolution in both dimensions. Each original cell
    becomes a 2x2 block of cells with values interpolated based on local slope
    direction to preserve terrain characteristics.

    Args:
        grid_data: Input grid data (numpy array or Taichi field)
                  Expected shape: (ny, nx) for numpy arrays or Taichi fields
        noise_amplitude: Amplitude of noise to add for realism (default: 0.0)
                        Noise is scaled down by factor of 10 relative to interpolation
        return_field: If True, return Taichi field; if False, return numpy array (default: False)
        nx: Number of columns when providing a 1D Taichi field
        ny: Number of rows when providing a 1D Taichi field

    Returns:
        numpy.ndarray or taichi.Field: Upscaled grid with shape (2*ny, 2*nx)

    Example:
        # Double resolution of a 100x100 grid with small noise
        upscaled = double_resolution(terrain_data, noise_amplitude=0.1)

        # Return as Taichi field for further GPU processing
        upscaled_field = double_resolution(terrain_data, return_field=True)
    """
    # Handle input conversion
    if isinstance(grid_data, np.ndarray):
        if len(grid_data.shape) != 2:
            raise ValueError("Input numpy array must be 2D")
        ny, nx = grid_data.shape
        data_np = grid_data.flatten()
    elif hasattr(grid_data, "to_numpy"):
        if len(grid_data.shape) == 2:
            ny, nx = grid_data.shape
            data_np = grid_data.to_numpy().reshape(-1)
        elif len(grid_data.shape) == 1:
            total_size = grid_data.shape[0]
            if nx is None or ny is None:
                raise ValueError("nx and ny must be provided for 1D Taichi fields")
            if nx * ny != total_size:
                raise ValueError("nx * ny does not match the size of the Taichi field")
            data_np = grid_data.to_numpy()
        else:
            raise ValueError("Input Taichi field must be 1D or 2D")
    else:
        raise TypeError("grid_data must be a numpy array or Taichi field")

    # Create source field and copy data
    source_field = pool.get_temp_field(ti.f32, (ny * nx,))
    source_field.field.from_numpy(data_np)

    # Create target field for upscaled result
    target_size = (2 * ny) * (2 * nx)
    target_field = pool.get_temp_field(ti.f32, (target_size,))

    # Execute upscaling kernel
    double_resolution_kernel(
        source_field.field,
        target_field.field,
        nx,
        ny,
        noise_amplitude,
    )

    if return_field:
        # Release source field and return target field
        source_field.release()
        return target_field.field
    else:
        # Convert to numpy and release both fields
        result = target_field.field.to_numpy().reshape(2 * ny, 2 * nx)
        source_field.release()
        target_field.release()
        return result
