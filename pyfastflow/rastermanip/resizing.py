"""General raster resizing utilities for PyFastFlow.

Supports arbitrary scaling factors via bilinear interpolation. Works with both
NumPy arrays and Taichi fields and shares the same memory pool utilities as the
other raster manipulation functions.

Author: OpenAI Assistant
"""

import numpy as np
import taichi as ti

from .. import pool

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
def resize_kernel(
    source_field: ti.template(),
    target_field: ti.template(),
    nx_src: ti.i32,
    ny_src: ti.i32,
    nx_t: ti.i32,
    ny_t: ti.i32,
    boundary_mode: ti.i32,
):
    for idx in target_field:
        j_t = idx // nx_t
        i_t = idx % nx_t
        if j_t >= ny_t or i_t >= nx_t:
            continue

        src_x = (i_t + 0.5) * nx_src / nx_t - 0.5
        src_y = (j_t + 0.5) * ny_src / ny_t - 0.5
        base_x = ti.floor(src_x, dtype=ti.i32)
        base_y = ti.floor(src_y, dtype=ti.i32)
        fx = src_x - base_x
        fy = src_y - base_y

        val = 0.0
        for dy in ti.static(range(2)):
            for dx in ti.static(range(2)):
                ix = _resolve_index(base_x + dx, nx_src, boundary_mode)
                iy = _resolve_index(base_y + dy, ny_src, boundary_mode)
                w = (1 - fx if dx == 0 else fx) * (1 - fy if dy == 0 else fy)
                val += source_field[iy * nx_src + ix] * w

        target_field[idx] = val


def resize_raster(
    grid_data,
    scale_factor: float,
    return_field: bool = False,
    nx: int | None = None,
    ny: int | None = None,
    boundary: str = "clamp",
):
    """Resize a 2D raster by an arbitrary scaling factor.

    Args:
        grid_data: Input grid (NumPy array or Taichi field) of shape (ny, nx)
        scale_factor: Scaling factor (>0). Values >1 upscale, <1 downscale.
        return_field: If True, return Taichi field instead of NumPy array.
        nx: Width when supplying a 1D Taichi field.
        ny: Height when supplying a 1D Taichi field.
        boundary: Boundary handling ('clamp', 'wrap', 'reflect').

    Returns:
        Resized raster as NumPy array or Taichi field.
    """

    if scale_factor <= 0:
        raise ValueError("scale_factor must be > 0")

    # Handle input
    if isinstance(grid_data, np.ndarray):
        if grid_data.ndim != 2:
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

    nx_t = max(1, int(round(nx * scale_factor)))
    ny_t = max(1, int(round(ny * scale_factor)))

    boundary_map = {
        "clamp": _BOUNDARY_CLAMP,
        "wrap": _BOUNDARY_WRAP,
        "reflect": _BOUNDARY_REFLECT,
    }
    if boundary not in boundary_map:
        raise ValueError("boundary must be 'clamp', 'wrap', or 'reflect'")
    boundary_mode = boundary_map[boundary]

    source_field = pool.get_temp_field(ti.f32, (ny * nx,))
    source_field.field.from_numpy(data_np)

    target_field = pool.get_temp_field(ti.f32, (ny_t * nx_t,))

    resize_kernel(
        source_field.field,
        target_field.field,
        nx,
        ny,
        nx_t,
        ny_t,
        boundary_mode,
    )

    if return_field:
        source_field.release()
        return target_field.field
    else:
        result = target_field.field.to_numpy().reshape(ny_t, nx_t)
        source_field.release()
        target_field.release()
        return result


__all__ = ["resize_raster", "resize_kernel"]

