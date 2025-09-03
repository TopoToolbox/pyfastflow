"""
Downscaling operations for PyFastFlow.

Provides GPU-accelerated downscaling functions that halve the resolution of 2D grids
using various aggregation methods: max, min, mean, and cubic interpolation.

The downscaling algorithm takes each 2x2 block of cells and reduces it to a single cell
using the specified aggregation method.

Author: B.G.
"""

import numpy as np
import taichi as ti

from .. import pool

# Boundary handling modes
_BOUNDARY_CLAMP = 0
_BOUNDARY_WRAP = 1
_BOUNDARY_REFLECT = 2


@ti.func
def _wrap_index(i: ti.i32, n: ti.i32) -> ti.i32:
    return ti.math.mod(i, n)


@ti.func
def _reflect_index(i: ti.i32, n: ti.i32) -> ti.i32:
    period = 2 * (n - 1)
    x = ti.math.mod(i, period)
    if x < 0:
        x += period
    if x >= n:
        x = period - x
    return x


@ti.func
def _resolve_index(i: ti.i32, n: ti.i32, mode: ti.i32) -> ti.i32:
    res = i
    if not (0 <= i < n):
        if mode == _BOUNDARY_CLAMP:
            res = ti.min(ti.max(i, 0), n - 1)
        elif mode == _BOUNDARY_WRAP:
            res = _wrap_index(i, n)
        else:
            res = _reflect_index(i, n)
    return res


@ti.kernel
def halve_resolution_kernel_max(
    source_field: ti.template(),
    target_field: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    boundary_mode: ti.i32,
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
                source_j = _resolve_index(source_base_j + sub_j, ny, boundary_mode)
                source_i = _resolve_index(source_base_i + sub_i, nx, boundary_mode)
                source_idx = source_j * nx + source_i
                val = source_field[source_idx]
                max_val = ti.max(max_val, val)

        target_field[target_idx] = max_val


@ti.kernel
def halve_resolution_kernel_min(
    source_field: ti.template(),
    target_field: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    boundary_mode: ti.i32,
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
                source_j = _resolve_index(source_base_j + sub_j, ny, boundary_mode)
                source_i = _resolve_index(source_base_i + sub_i, nx, boundary_mode)
                source_idx = source_j * nx + source_i
                val = source_field[source_idx]
                min_val = ti.min(min_val, val)

        target_field[target_idx] = min_val


@ti.kernel
def halve_resolution_kernel_mean(
    source_field: ti.template(),
    target_field: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    boundary_mode: ti.i32,
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
        for sub_j in ti.static(range(2)):
            for sub_i in ti.static(range(2)):
                source_j = _resolve_index(source_base_j + sub_j, ny, boundary_mode)
                source_i = _resolve_index(source_base_i + sub_i, nx, boundary_mode)
                source_idx = source_j * nx + source_i
                val = source_field[source_idx]
                sum_val += val

        target_field[target_idx] = sum_val * 0.25


@ti.func
def cubic_interpolate(
    v0: ti.f32, v1: ti.f32, v2: ti.f32, v3: ti.f32, t: ti.f32
) -> ti.f32:
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
    ny: ti.i32,
    boundary_mode: ti.i32,
):
    """
    Halve resolution using true bicubic interpolation.

    Each 2x2 block in source becomes 1 cell in target by sampling a 4x4
    neighborhood with cubic interpolation along both axes.

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

        base_j = ti.floor(center_j, dtype=ti.i32)
        base_i = ti.floor(center_i, dtype=ti.i32)
        frac_j = center_j - base_j
        frac_i = center_i - base_i

        row_vals = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for dj in ti.static(range(4)):
            samples = ti.Vector([0.0, 0.0, 0.0, 0.0])
            jj = _resolve_index(base_j + dj - 1, ny, boundary_mode)
            for di in ti.static(range(4)):
                ii = _resolve_index(base_i + di - 1, nx, boundary_mode)
                source_idx = jj * nx + ii
                samples[di] = source_field[source_idx]
            row_vals[dj] = cubic_interpolate(
                samples[0], samples[1], samples[2], samples[3], frac_i
            )

        target_field[target_idx] = cubic_interpolate(
            row_vals[0], row_vals[1], row_vals[2], row_vals[3], frac_j
        )


def halve_resolution(
    grid_data,
    method: str = "mean",
    return_field: bool = False,
    nx: int | None = None,
    ny: int | None = None,
    kernel=None,
    boundary: str = "clamp",
):
    """
    Halve the resolution of a 2D grid using specified aggregation method.

    Reduces a grid to half resolution in both dimensions by aggregating 2x2 blocks
    of cells into single cells using the specified method.

    Args:
        grid_data: Input grid data (numpy array or Taichi field)
                  Expected shape: (ny, nx) for numpy arrays or Taichi fields
        method: Aggregation method ('max', 'min', 'mean', 'cubic') (default: 'mean').
                Ignored if a custom kernel is provided.
        return_field: If True, return Taichi field; if False, return numpy array (default: False)
        nx: Number of columns when providing a 1D Taichi field
        ny: Number of rows when providing a 1D Taichi field
        kernel: Optional custom Taichi kernel with signature
                (source_field, target_field, nx, ny)
        boundary: Boundary handling ('clamp', 'wrap', 'reflect')

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
    if kernel is None:
        valid_methods = ["max", "min", "mean", "cubic"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")

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

    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError(f"Grid dimensions must be even for halving. Got ({ny}, {nx})")

    boundary_map = {
        "clamp": _BOUNDARY_CLAMP,
        "wrap": _BOUNDARY_WRAP,
        "reflect": _BOUNDARY_REFLECT,
    }
    if boundary not in boundary_map:
        raise ValueError("boundary must be 'clamp', 'wrap', or 'reflect'")
    boundary_mode = boundary_map[boundary]

    # Create source field and copy data
    source_field = pool.get_temp_field(ti.f32, (ny * nx,))
    source_field.field.from_numpy(data_np)

    # Create target field for downscaled result
    target_nx = nx // 2
    target_ny = ny // 2
    target_size = target_ny * target_nx
    target_field = pool.get_temp_field(ti.f32, (target_size,))

    if kernel is not None:
        kernel(source_field.field, target_field.field, nx, ny)
    else:
        if method == "max":
            halve_resolution_kernel_max(
                source_field.field, target_field.field, nx, ny, boundary_mode
            )
        elif method == "min":
            halve_resolution_kernel_min(
                source_field.field, target_field.field, nx, ny, boundary_mode
            )
        elif method == "mean":
            halve_resolution_kernel_mean(
                source_field.field, target_field.field, nx, ny, boundary_mode
            )
        elif method == "cubic":
            halve_resolution_kernel_cubic(
                source_field.field, target_field.field, nx, ny, boundary_mode
            )

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
