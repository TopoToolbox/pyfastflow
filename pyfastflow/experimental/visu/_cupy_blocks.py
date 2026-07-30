"""
cupy (CUDA source) block templates behind make_hillshade.

Mirrors _closure_blocks.py: same gradient rewrite (grid.neighbour(i, k),
falling back to z[i] where there is no neighbour - see that module's
docstring for why), same hillshade formula, written as CUDA text. Every
`__device__`/`__global__` symbol is prefixed with this build's own tag (a
fresh new_uid()) so two make_hillshade() calls in one process never collide
inside a single compiled cupy module - see ../grid/_cupy_blocks.py.

Author: B.G (07/2026)
"""

import functools
import math

from ..core.context.backends import make_helper
from ..core.pool.base import new_uid

_DEG2RAD = math.pi / 180.0
_HALF_PI = math.pi / 2.0
_TWO_PI = 2.0 * math.pi


def build_helpers(HelperCls, KernelCls, *, grid, azimuth_p, altitude_p, z_factor_p):
    """
    `at` (HelperBuilder, shade value at one node) and `hillshade` (unbuilt
    KernelBuilder, writes shade for every node). k-indices for
    left/right/top/bottom are picked from grid.n_neighbours' own const value,
    same as _closure_blocks.build_helpers.

    Author: B.G (07/2026)
    """
    n_neighbours = grid.n_neighbours.get()
    if n_neighbours == 4:
        k_top, k_left, k_right, k_bottom = 0, 1, 2, 3
    elif n_neighbours == 8:
        k_top, k_left, k_right, k_bottom = 1, 3, 4, 6
    else:
        raise ValueError(f"make_hillshade: unsupported grid.n_neighbours {n_neighbours!r}, expected 4 or 8")

    t = f"pf{new_uid()}"
    mk = functools.partial(make_helper, HelperCls)

    grad_x = mk(
        f"""
__device__ float {t}_gradient_x(const float* z, int i) {{
    int zl = $grid.neighbour(i, {k_left})$;
    int zr = $grid.neighbour(i, {k_right})$;
    float left_val = (zl != -1) ? z[zl] : z[i];
    float right_val = (zr != -1) ? z[zr] : z[i];
    return (right_val - left_val) / (2.0f * $grid.dx.get(0)$);
}}
""",
        grid=grid,
    )
    grad_y = mk(
        f"""
__device__ float {t}_gradient_y(const float* z, int i) {{
    int zt = $grid.neighbour(i, {k_top})$;
    int zb = $grid.neighbour(i, {k_bottom})$;
    float top_val = (zt != -1) ? z[zt] : z[i];
    float bottom_val = (zb != -1) ? z[zb] : z[i];
    return (bottom_val - top_val) / (2.0f * $grid.dx.get(0)$);
}}
""",
        grid=grid,
    )
    at = mk(
        f"""
__device__ float {t}_at(const float* z, int i) {{
    float dzdx = $gradient_x(z, i)$ * $ZFACTOR.get(0)$;
    float dzdy = $gradient_y(z, i)$ * $ZFACTOR.get(0)$;

    float slope_rad = atan2f(sqrtf(dzdx * dzdx + dzdy * dzdy), 1.0f);
    float azimuth_rad = $AZIMUTH.get(0)$ * {_DEG2RAD}f;
    float zenith_rad = {_HALF_PI}f - $ALTITUDE.get(0)$ * {_DEG2RAD}f;

    float aspect_rad = 0.0f;
    if (dzdx != 0.0f || dzdy != 0.0f) {{
        aspect_rad = {_HALF_PI}f - atan2f(dzdy, dzdx);
        if (aspect_rad < 0.0f) aspect_rad += {_TWO_PI}f;
    }}

    float hillshade_value = cosf(zenith_rad) * cosf(slope_rad)
        + sinf(zenith_rad) * sinf(slope_rad) * cosf(azimuth_rad - aspect_rad);
    return fmaxf(0.0f, fminf(1.0f, hillshade_value));
}}
""",
        gradient_x=grad_x,
        gradient_y=grad_y,
        AZIMUTH=azimuth_p,
        ALTITUDE=altitude_p,
        ZFACTOR=z_factor_p,
    )

    hillshade_kernel = KernelCls().bind("at", at).ingest(
        f"""
extern "C" __global__ void {t}_hillshade(const float* z, float* out, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = $at(z, i)$;
}}
"""
    )

    return {"at": at, "hillshade": hillshade_kernel}
