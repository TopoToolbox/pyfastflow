"""
Taichi/Quadrants (closure) block templates behind make_hillshade.

Same private/public split as ../grid/_closure_blocks.py: private blocks are
plain python defs, picked - never branched on inside one function body - by
build_helpers() according to the grid's own topology (which fixes which
neighbour index `k` means "left", "right", "top", "south" - see
build_helpers). `_BK` is the bound backend module (ti or qd); `azimuth`,
`altitude` and `z_factor` are mode-overridable Parameters, read uniformly
through `.get(0)` (see ../core/context/parameter.py, "Reading a Parameter in
device code is uniform across modes").

The gradient here is deliberately not the legacy fixed-stencil,
clamped-index version (pyfastflow/visu/hillshading.py's gradient_x_flat /
gradient_y_flat): those are wrong under a periodic grid and wrong at a
nodata cell, since clamping an index at a nodata boundary silently reads
the wrong neighbour instead of recognising there is none. Going through
`grid.neighbour(i, k)` instead gets both cases right for free - see the
grid's own contract (../grid/_closure_blocks.py, lines 184-230): a -1 return
means "no neighbour" (off a bounded edge, off a nodata source, or into a
nodata target), and a periodic axis wraps automatically. A missing sample is
substituted with z[i] itself, i.e. that side of the central difference
degenerates to a one-sided difference against the cell's own elevation
rather than an out-of-range read.

Author: B.G (07/2026)
"""

import functools
import math

from ..core.context.backends import make_helper
from ..core.context.bag import Bag

_DEG2RAD = math.pi / 180.0
_HALF_PI = math.pi / 2.0
_TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------


def _gradient_x_tmpl(z, i):
    zl = _GRID.neighbour(i, _KLEFT)
    zr = _GRID.neighbour(i, _KRIGHT)
    left_val = z[i]
    if zl != -1:
        left_val = z[zl]
    right_val = z[i]
    if zr != -1:
        right_val = z[zr]
    return (right_val - left_val) / (2.0 * _GRID.dx.get(0))


def _gradient_y_tmpl(z, i):
    zt = _GRID.neighbour(i, _KTOP)
    zb = _GRID.neighbour(i, _KBOTTOM)
    top_val = z[i]
    if zt != -1:
        top_val = z[zt]
    bottom_val = z[i]
    if zb != -1:
        bottom_val = z[zb]
    return (bottom_val - top_val) / (2.0 * _GRID.dx.get(0))


# ---------------------------------------------------------------------------
# hillshade at one node
# ---------------------------------------------------------------------------


def _at_tmpl(z, i):
    dzdx = _GRADX(z, i) * _ZFACTOR.get(0)
    dzdy = _GRADY(z, i) * _ZFACTOR.get(0)

    slope_rad = _BK.atan2(_BK.sqrt(dzdx * dzdx + dzdy * dzdy), 1.0)
    azimuth_rad = _AZIMUTH.get(0) * _DEG2RAD
    zenith_rad = _HALF_PI - _ALTITUDE.get(0) * _DEG2RAD

    aspect_rad = 0.0
    if dzdx != 0.0 or dzdy != 0.0:
        aspect_rad = _HALF_PI - _BK.atan2(dzdy, dzdx)
        if aspect_rad < 0.0:
            aspect_rad += _TWO_PI

    hillshade_value = _BK.cos(zenith_rad) * _BK.cos(slope_rad) + _BK.sin(zenith_rad) * _BK.sin(
        slope_rad
    ) * _BK.cos(azimuth_rad - aspect_rad)
    return _BK.max(0.0, _BK.min(1.0, hillshade_value))


# ---------------------------------------------------------------------------
# kernel: write shade for every node
# ---------------------------------------------------------------------------


def _make_hillshade_template(T):
    def hillshade_template(z: T, out: T):
        n = _GRID.nx.get(0) * _GRID.ny.get(0)
        for i in range(n):
            out[i] = _AT(z, i)

    return hillshade_template


def _tensor_annotation(backend_mod, backend: str):
    return backend_mod.template() if backend == "taichi" else backend_mod.Tensor


def build_helpers(HelperCls, KernelCls, *, backend: str, backend_mod, grid: Bag, azimuth_p, altitude_p, z_factor_p):
    """
    `at` (HelperBuilder, shade value at one node) and `hillshade` (unbuilt
    KernelBuilder, writes shade for every node) - the k-indices meaning
    "left"/"right"/"top"/"bottom" are picked once here from
    `grid.n_neighbours`'s own const value (4 -> D4's own delta table, 8 ->
    D8's - see ../grid/_closure_blocks.py's _delta_d4_tmpl/_delta_d8_tmpl),
    since the grid Bag exposes neighbour(i, k) generically but not a
    "which k is west" query.

    Author: B.G (07/2026)
    """
    n_neighbours = grid.n_neighbours.get()
    if n_neighbours == 4:
        k_top, k_left, k_right, k_bottom = 0, 1, 2, 3
    elif n_neighbours == 8:
        k_top, k_left, k_right, k_bottom = 1, 3, 4, 6
    else:
        raise ValueError(f"make_hillshade: unsupported grid.n_neighbours {n_neighbours!r}, expected 4 or 8")

    mk = functools.partial(make_helper, HelperCls)

    grad_x = mk(_gradient_x_tmpl, _GRID=grid, _KLEFT=k_left, _KRIGHT=k_right)
    grad_y = mk(_gradient_y_tmpl, _GRID=grid, _KTOP=k_top, _KBOTTOM=k_bottom)
    at = mk(
        _at_tmpl,
        _BK=backend_mod,
        _GRADX=grad_x,
        _GRADY=grad_y,
        _AZIMUTH=azimuth_p,
        _ALTITUDE=altitude_p,
        _ZFACTOR=z_factor_p,
        _DEG2RAD=_DEG2RAD,
        _HALF_PI=_HALF_PI,
        _TWO_PI=_TWO_PI,
    )

    T = _tensor_annotation(backend_mod, backend)
    hillshade_kernel = KernelCls().bind("_GRID", grid).bind("_AT", at).ingest(_make_hillshade_template(T))

    return {"at": at, "hillshade": hillshade_kernel}
