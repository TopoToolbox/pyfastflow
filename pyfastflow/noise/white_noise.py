"""
White noise Taichi primitives for PyFastFlow.

This module intentionally only exposes Taichi functions and kernels. Python-side
orchestration lives in ``noisecontext.py``.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None


@ti.func
def _hash_u32(x: ti.u32) -> ti.u32:
    """Small integer hash used for deterministic white noise. Author: B.G (02/2026)"""
    h = x
    h ^= h >> ti.u32(16)
    h *= ti.u32(0x7FEB352D)
    h ^= h >> ti.u32(15)
    h *= ti.u32(0x846CA68B)
    h ^= h >> ti.u32(16)
    return h


@ti.func
def _white_unit(i: ti.i32, j: ti.i32, seed: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return deterministic pseudo-random value in [0, 1). Author: B.G (02/2026)"""
    key = ti.u32(seed)
    key ^= ti.u32(i) * ti.u32(374761393)
    key ^= ti.u32(j) * ti.u32(668265263)
    hashed = _hash_u32(key)
    return ti.cast(hashed, cte.FLOAT_TYPE_TI) / ti.cast(4294967296.0, cte.FLOAT_TYPE_TI)


@ti.kernel
def white_noise_2d_kernel(
    noise_field: ti.template(), amplitude: cte.FLOAT_TYPE_TI, seed: ti.i32
):
    """
    Fill a field with deterministic white noise using the bound grid context.

    Author: B.G (02/2026)
    """
    for j, i in ti.ndrange(gridctx.ny, gridctx.nx):
        noise_field[j, i] = (_white_unit(i, j, seed) - 0.5) * 2.0 * amplitude


@ti.kernel
def white_noise_flat_kernel(
    noise_field: ti.template(), amplitude: cte.FLOAT_TYPE_TI, seed: ti.i32
):
    """
    Fill a flat row-major field with deterministic white noise.

    Author: B.G (02/2026)
    """
    for j, i in ti.ndrange(gridctx.ny, gridctx.nx):
        idx = j * gridctx.nx + i
        noise_field[idx] = (_white_unit(i, j, seed) - 0.5) * 2.0 * amplitude


white_noise_kernel = white_noise_2d_kernel
