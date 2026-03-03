"""
Standalone hillshading kernel for external 2D arrays.

The higher-level grid-aware helpers now live in ``VisuContext``. This module is
kept as a small standalone kernel for direct ndarray workflows.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte
from ..general_algorithms.math_utils import atan

gridctx = None


@ti.func
def gradient_x_flat(z: ti.template(), i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return x-gradient at node i using the bound grid context.

    Author: B.G (02/2026)
    """
    left_idx = gridctx.tfunc.left_flat(i)
    right_idx = gridctx.tfunc.right_flat(i)
    dx = ti.static(gridctx.dx)

    grad_x = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    if left_idx != -1 and right_idx != -1:
        grad_x = (z[right_idx] - z[left_idx]) / ti.cast(2.0 * dx, cte.FLOAT_TYPE_TI)
    elif left_idx != -1:
        grad_x = (z[i] - z[left_idx]) / ti.cast(dx, cte.FLOAT_TYPE_TI)
    elif right_idx != -1:
        grad_x = (z[right_idx] - z[i]) / ti.cast(dx, cte.FLOAT_TYPE_TI)
    return grad_x


@ti.func
def gradient_y_flat(z: ti.template(), i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return y-gradient at node i using the bound grid context.

    Author: B.G (02/2026)
    """
    top_idx = gridctx.tfunc.top_flat(i)
    bottom_idx = gridctx.tfunc.bottom_flat(i)
    dx = ti.static(gridctx.dx)

    grad_y = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    if top_idx != -1 and bottom_idx != -1:
        grad_y = (z[bottom_idx] - z[top_idx]) / ti.cast(2.0 * dx, cte.FLOAT_TYPE_TI)
    elif top_idx != -1:
        grad_y = (z[i] - z[top_idx]) / ti.cast(dx, cte.FLOAT_TYPE_TI)
    elif bottom_idx != -1:
        grad_y = (z[bottom_idx] - z[i]) / ti.cast(dx, cte.FLOAT_TYPE_TI)
    return grad_y


@ti.func
def hillshade_at_flat(
    z: ti.template(),
    i: ti.i32,
    zenith_rad: cte.FLOAT_TYPE_TI,
    azimuth_rad: cte.FLOAT_TYPE_TI,
    z_factor: cte.FLOAT_TYPE_TI,
) -> cte.FLOAT_TYPE_TI:
    """
    Return hillshade value at node i for one light direction.

    Author: B.G (02/2026)
    """
    dz_dx = gradient_x_flat(z, i) * z_factor
    dz_dy = gradient_y_flat(z, i) * z_factor

    slope_rad = atan(ti.math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))

    aspect_rad = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    if dz_dx != 0.0 or dz_dy != 0.0:
        aspect_rad = ti.cast(ti.math.pi / 2.0, cte.FLOAT_TYPE_TI) - ti.math.atan2(dz_dy, dz_dx)
        if aspect_rad < 0.0:
            aspect_rad += ti.cast(2.0 * ti.math.pi, cte.FLOAT_TYPE_TI)

    hillshade_value = (
        ti.math.cos(zenith_rad) * ti.math.cos(slope_rad)
        + ti.math.sin(zenith_rad)
        * ti.math.sin(slope_rad)
        * ti.math.cos(azimuth_rad - aspect_rad)
    )
    return ti.math.max(
        ti.cast(0.0, cte.FLOAT_TYPE_TI),
        ti.math.min(ti.cast(1.0, cte.FLOAT_TYPE_TI), hillshade_value),
    )


@ti.func
def gradient_x_2d(z: ti.template(), row: ti.i32, col: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return x-gradient at node row, col using simple edge clamping.

    Author: B.G (02/2026)
    """
    dx = ti.static(gridctx.dx)

    grad_x = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    if col > 0 and col < gridctx.nx - 1:
        grad_x = (z[row, col + 1] - z[row, col - 1]) / ti.cast(2.0 * dx, cte.FLOAT_TYPE_TI)
    elif col > 0:
        grad_x = (z[row, col] - z[row, col - 1]) / ti.cast(dx, cte.FLOAT_TYPE_TI)
    elif col < gridctx.nx - 1:
        grad_x = (z[row, col + 1] - z[row, col]) / ti.cast(dx, cte.FLOAT_TYPE_TI)
    return grad_x


@ti.func
def gradient_y_2d(z: ti.template(), row: ti.i32, col: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return y-gradient at node row, col using simple edge clamping.

    Author: B.G (02/2026)
    """
    dx = ti.static(gridctx.dx)

    grad_y = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    if row > 0 and row < gridctx.ny - 1:
        grad_y = (z[row + 1, col] - z[row - 1, col]) / ti.cast(2.0 * dx, cte.FLOAT_TYPE_TI)
    elif row > 0:
        grad_y = (z[row, col] - z[row - 1, col]) / ti.cast(dx, cte.FLOAT_TYPE_TI)
    elif row < gridctx.ny - 1:
        grad_y = (z[row + 1, col] - z[row, col]) / ti.cast(dx, cte.FLOAT_TYPE_TI)
    return grad_y


@ti.func
def hillshade_at_2d(
    z: ti.template(),
    row: ti.i32,
    col: ti.i32,
    zenith_rad: cte.FLOAT_TYPE_TI,
    azimuth_rad: cte.FLOAT_TYPE_TI,
    z_factor: cte.FLOAT_TYPE_TI,
) -> cte.FLOAT_TYPE_TI:
    """
    Return hillshade value at row, col for one light direction.

    Author: B.G (02/2026)
    """
    dz_dx = gradient_x_2d(z, row, col) * z_factor
    dz_dy = gradient_y_2d(z, row, col) * z_factor

    slope_rad = atan(ti.math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))

    aspect_rad = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    if dz_dx != 0.0 or dz_dy != 0.0:
        aspect_rad = ti.cast(ti.math.pi / 2.0, cte.FLOAT_TYPE_TI) - ti.math.atan2(dz_dy, dz_dx)
        if aspect_rad < 0.0:
            aspect_rad += ti.cast(2.0 * ti.math.pi, cte.FLOAT_TYPE_TI)

    hillshade_value = (
        ti.math.cos(zenith_rad) * ti.math.cos(slope_rad)
        + ti.math.sin(zenith_rad)
        * ti.math.sin(slope_rad)
        * ti.math.cos(azimuth_rad - aspect_rad)
    )
    return ti.math.max(
        ti.cast(0.0, cte.FLOAT_TYPE_TI),
        ti.math.min(ti.cast(1.0, cte.FLOAT_TYPE_TI), hillshade_value),
    )


@ti.kernel
def hillshading_flat_kernel(
    z: ti.template(),
    hillshade: ti.template(),
    zenith_rad: cte.FLOAT_TYPE_TI,
    azimuth_rad: cte.FLOAT_TYPE_TI,
    z_factor: cte.FLOAT_TYPE_TI,
):
    """
    Compute single-direction hillshade over the bound grid context.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.nx * gridctx.ny):
        hillshade[i] = hillshade_at_flat(z, i, zenith_rad, azimuth_rad, z_factor)


@ti.kernel
def multishading_flat_kernel(
    z: ti.template(),
    hillshade: ti.template(),
    zenith_rad: cte.FLOAT_TYPE_TI,
    azimuth0_rad: cte.FLOAT_TYPE_TI,
    azimuth1_rad: cte.FLOAT_TYPE_TI,
    azimuth2_rad: cte.FLOAT_TYPE_TI,
    azimuth3_rad: cte.FLOAT_TYPE_TI,
    z_factor: cte.FLOAT_TYPE_TI,
):
    """
    Compute four-direction averaged hillshade over the bound grid context.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.nx * gridctx.ny):
        value = hillshade_at_flat(z, i, zenith_rad, azimuth0_rad, z_factor)
        value += hillshade_at_flat(z, i, zenith_rad, azimuth1_rad, z_factor)
        value += hillshade_at_flat(z, i, zenith_rad, azimuth2_rad, z_factor)
        value += hillshade_at_flat(z, i, zenith_rad, azimuth3_rad, z_factor)
        hillshade[i] = value * ti.cast(0.25, cte.FLOAT_TYPE_TI)


@ti.kernel
def hillshading_2d_kernel(
    z: ti.template(),
    hillshade: ti.template(),
    zenith_rad: cte.FLOAT_TYPE_TI,
    azimuth_rad: cte.FLOAT_TYPE_TI,
    z_factor: cte.FLOAT_TYPE_TI,
):
    """
    Compute single-direction hillshade for a 2D field.

    Author: B.G (02/2026)
    """
    for row, col in ti.ndrange(gridctx.ny, gridctx.nx):
        hillshade[row, col] = hillshade_at_2d(z, row, col, zenith_rad, azimuth_rad, z_factor)


@ti.kernel
def multishading_2d_kernel(
    z: ti.template(),
    hillshade: ti.template(),
    zenith_rad: cte.FLOAT_TYPE_TI,
    azimuth0_rad: cte.FLOAT_TYPE_TI,
    azimuth1_rad: cte.FLOAT_TYPE_TI,
    azimuth2_rad: cte.FLOAT_TYPE_TI,
    azimuth3_rad: cte.FLOAT_TYPE_TI,
    z_factor: cte.FLOAT_TYPE_TI,
):
    """
    Compute four-direction averaged hillshade for a 2D field.

    Author: B.G (02/2026)
    """
    for row, col in ti.ndrange(gridctx.ny, gridctx.nx):
        value = hillshade_at_2d(z, row, col, zenith_rad, azimuth0_rad, z_factor)
        value += hillshade_at_2d(z, row, col, zenith_rad, azimuth1_rad, z_factor)
        value += hillshade_at_2d(z, row, col, zenith_rad, azimuth2_rad, z_factor)
        value += hillshade_at_2d(z, row, col, zenith_rad, azimuth3_rad, z_factor)
        hillshade[row, col] = value * ti.cast(0.25, cte.FLOAT_TYPE_TI)


gradient_x = gradient_x_flat
gradient_y = gradient_y_flat
hillshade_at = hillshade_at_flat
hillshading_kernel = hillshading_flat_kernel
multishading_kernel = multishading_flat_kernel


@ti.kernel
def hillshade_2d(
    z: ti.types.ndarray(dtype=cte.FLOAT_TYPE_TI, ndim=2),
    hillshade: ti.types.ndarray(dtype=cte.FLOAT_TYPE_TI, ndim=2),
    zenith_rad: cte.FLOAT_TYPE_TI,
    azimuth_rad: cte.FLOAT_TYPE_TI,
    z_factor: cte.FLOAT_TYPE_TI,
    dx: cte.FLOAT_TYPE_TI,
):
    """
    Compute hillshade for a 2D ndarray using simple edge clamping.

    Author: B.G (02/2026)
    """
    ny, nx = z.shape

    for row, col in ti.ndrange(ny, nx):
        dz_dx = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        dz_dy = ti.cast(0.0, cte.FLOAT_TYPE_TI)

        if col > 0 and col < nx - 1:
            dz_dx = (z[row, col + 1] - z[row, col - 1]) / (2.0 * dx)
        elif col > 0:
            dz_dx = (z[row, col] - z[row, col - 1]) / dx
        elif col < nx - 1:
            dz_dx = (z[row, col + 1] - z[row, col]) / dx

        if row > 0 and row < ny - 1:
            dz_dy = (z[row + 1, col] - z[row - 1, col]) / (2.0 * dx)
        elif row > 0:
            dz_dy = (z[row, col] - z[row - 1, col]) / dx
        elif row < ny - 1:
            dz_dy = (z[row + 1, col] - z[row, col]) / dx

        dz_dx *= z_factor
        dz_dy *= z_factor

        slope_rad = atan(ti.math.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))

        aspect_rad = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        if dz_dx != 0.0 or dz_dy != 0.0:
            aspect_rad = ti.cast(ti.math.pi / 2.0, cte.FLOAT_TYPE_TI) - ti.math.atan2(dz_dy, dz_dx)
            if aspect_rad < 0.0:
                aspect_rad += ti.cast(2.0 * ti.math.pi, cte.FLOAT_TYPE_TI)

        hillshade_value = (
            ti.math.cos(zenith_rad) * ti.math.cos(slope_rad)
            + ti.math.sin(zenith_rad)
            * ti.math.sin(slope_rad)
            * ti.math.cos(azimuth_rad - aspect_rad)
        )

        hillshade[row, col] = ti.math.max(
            ti.cast(0.0, cte.FLOAT_TYPE_TI),
            ti.math.min(ti.cast(1.0, cte.FLOAT_TYPE_TI), hillshade_value),
        )
