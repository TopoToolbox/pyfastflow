"""
Perlin noise Taichi primitives for PyFastFlow.

This module intentionally only exposes Taichi functions and kernels. Python-side
orchestration lives in ``noisecontext.py``.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None


@ti.func
def fade(t: cte.FLOAT_TYPE_TI) -> cte.FLOAT_TYPE_TI:
    """Return Perlin fade curve value. Author: B.G (02/2026)"""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@ti.func
def lerp(t: cte.FLOAT_TYPE_TI, a: cte.FLOAT_TYPE_TI, b: cte.FLOAT_TYPE_TI) -> cte.FLOAT_TYPE_TI:
    """Return linear interpolation between a and b. Author: B.G (02/2026)"""
    return a + t * (b - a)


@ti.func
def grad(hash_val: ti.i32, dx: cte.FLOAT_TYPE_TI, dy: cte.FLOAT_TYPE_TI) -> cte.FLOAT_TYPE_TI:
    """Return 2D gradient dot product from hashed corner index. Author: B.G (02/2026)"""
    idx = hash_val & 7
    gx = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    gy = ti.cast(0.0, cte.FLOAT_TYPE_TI)

    if idx == 0:
        gx, gy = 1.0, 1.0
    elif idx == 1:
        gx, gy = -1.0, 1.0
    elif idx == 2:
        gx, gy = 1.0, -1.0
    elif idx == 3:
        gx, gy = -1.0, -1.0
    elif idx == 4:
        gx, gy = 1.0, 0.0
    elif idx == 5:
        gx, gy = -1.0, 0.0
    elif idx == 6:
        gx, gy = 0.0, 1.0
    else:
        gx, gy = 0.0, -1.0

    return gx * dx + gy * dy


@ti.func
def perlin_noise_at(
    x: cte.FLOAT_TYPE_TI, y: cte.FLOAT_TYPE_TI, perm: ti.template()
) -> cte.FLOAT_TYPE_TI:
    """Return Perlin noise value at x, y using a permutation field. Author: B.G (02/2026)"""
    x_floor = ti.floor(x)
    y_floor = ti.floor(y)

    X = ti.cast(x_floor, ti.i32) & 255
    Y = ti.cast(y_floor, ti.i32) & 255

    x_local = x - x_floor
    y_local = y - y_floor

    u = fade(x_local)
    v = fade(y_local)

    A = perm[X] + Y
    B = perm[(X + 1) & 255] + Y
    AA = perm[A & 255]
    AB = perm[(A + 1) & 255]
    BA = perm[B & 255]
    BB = perm[(B + 1) & 255]

    return lerp(
        v,
        lerp(u, grad(AA, x_local, y_local), grad(BA, x_local - 1.0, y_local)),
        lerp(
            u,
            grad(AB, x_local, y_local - 1.0),
            grad(BB, x_local - 1.0, y_local - 1.0),
        ),
    )


@ti.kernel
def perlin_noise_2d_kernel(
    noise_field: ti.template(),
    frequency_x: cte.FLOAT_TYPE_TI,
    frequency_y: cte.FLOAT_TYPE_TI,
    octaves: ti.i32,
    persistence: cte.FLOAT_TYPE_TI,
    amplitude: cte.FLOAT_TYPE_TI,
    perm: ti.template(),
):
    """
    Fill a field with multi-octave Perlin noise using the bound grid context.

    Author: B.G (02/2026)
    """
    nx_f = ti.cast(gridctx.nx, cte.FLOAT_TYPE_TI)
    ny_f = ti.cast(gridctx.ny, cte.FLOAT_TYPE_TI)

    for j, i in ti.ndrange(gridctx.ny, gridctx.nx):
        x = ti.cast(i, cte.FLOAT_TYPE_TI) * frequency_x / nx_f
        y = ti.cast(j, cte.FLOAT_TYPE_TI) * frequency_y / ny_f

        total = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        max_value = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        current_amplitude = ti.cast(1.0, cte.FLOAT_TYPE_TI)
        current_frequency = ti.cast(1.0, cte.FLOAT_TYPE_TI)

        for _ in range(octaves):
            total += perlin_noise_at(x * current_frequency, y * current_frequency, perm) * current_amplitude
            max_value += current_amplitude
            current_amplitude *= persistence
            current_frequency *= 2.0

        if max_value > 0.0:
            noise_field[j, i] = (total / max_value) * amplitude
        else:
            noise_field[j, i] = 0.0


@ti.kernel
def perlin_noise_flat_kernel(
    noise_field: ti.template(),
    frequency_x: cte.FLOAT_TYPE_TI,
    frequency_y: cte.FLOAT_TYPE_TI,
    octaves: ti.i32,
    persistence: cte.FLOAT_TYPE_TI,
    amplitude: cte.FLOAT_TYPE_TI,
    perm: ti.template(),
):
    """
    Fill a flat row-major field with multi-octave Perlin noise.

    Author: B.G (02/2026)
    """
    nx_f = ti.cast(gridctx.nx, cte.FLOAT_TYPE_TI)
    ny_f = ti.cast(gridctx.ny, cte.FLOAT_TYPE_TI)

    for j, i in ti.ndrange(gridctx.ny, gridctx.nx):
        x = ti.cast(i, cte.FLOAT_TYPE_TI) * frequency_x / nx_f
        y = ti.cast(j, cte.FLOAT_TYPE_TI) * frequency_y / ny_f

        total = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        max_value = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        current_amplitude = ti.cast(1.0, cte.FLOAT_TYPE_TI)
        current_frequency = ti.cast(1.0, cte.FLOAT_TYPE_TI)

        for _ in range(octaves):
            total += perlin_noise_at(x * current_frequency, y * current_frequency, perm) * current_amplitude
            max_value += current_amplitude
            current_amplitude *= persistence
            current_frequency *= 2.0

        idx = j * gridctx.nx + i
        if max_value > 0.0:
            noise_field[idx] = (total / max_value) * amplitude
        else:
            noise_field[idx] = 0.0


perlin_noise_kernel = perlin_noise_2d_kernel
