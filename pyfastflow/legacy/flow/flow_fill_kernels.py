"""
Generic topographic filling kernels for ``FlowContext``.

The filling logic is driven by the minimum-slope getter exposed by the context.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None
flowctx = None
get_min_slope = None
nextafter = None


@ti.kernel
def fill_topography_step_kernel(
    z_ref: ti.template(),
    z_work: ti.template(),
    receivers: ti.template(),
    receivers_next: ti.template(),
    iteration: ti.i32,
):
    """
    Execute one pointer-jumping fill step with a minimum slope constraint.

    Author: B.G (02/2026)
    """
    for i in z_ref:
        receivers_next[i] = receivers[i]
        receivers_next[i] = receivers[receivers[i]]

        if i == receivers[i]:
            continue

        if z_work[i] > z_ref[receivers[i]] and receivers[receivers[i]] == receivers[i]:
            continue

        increment = ti.math.pow(2.0, ti.cast(iteration - 1, cte.FLOAT_TYPE_TI))
        increment *= get_min_slope(i) * ti.cast(gridctx.dx, cte.FLOAT_TYPE_TI)
        z_work[i] = ti.max(z_work[i], z_work[receivers[i]] + increment)

    for i in receivers:
        receivers[i] = receivers_next[i]


@ti.kernel
def fill_h_epsilon_kernel(
    z: ti.template(),
    h: ti.template(),
    receivers: ti.template(),
    receivers_next: ti.template(),
    iteration: ti.i32,
):
    """
    Same as ``fill_topography_step_kernel`` but working value is z+h,
    stored back into h. No separate surface buffer.

    Author: B.G (07/2026)
    """
    for i in z:
        receivers_next[i] = receivers[i]
        receivers_next[i] = receivers[receivers[i]]

        if i == receivers[i]:
            continue

        r = receivers[i]
        dz_ir = z[i] - z[r]

        if (h[i] + dz_ir) > 0.0 and receivers[receivers[i]] == receivers[i]:
            continue

        increment = ti.math.pow(2.0, ti.cast(iteration - 1, cte.FLOAT_TYPE_TI))
        increment *= get_min_slope(i) * ti.cast(gridctx.dx, cte.FLOAT_TYPE_TI)
        h[i] = ti.max(h[i], h[r] - dz_ir + increment)

    for i in receivers:
        receivers[i] = receivers_next[i]


@ti.kernel
def solve_lm_z_kernel(z: ti.template()):
    """
    Cheap single-pass partial local-minima solver on ``z``.

    For every non-outlet node with no strictly-downhill neighbour, raise it
    to just above its lowest neighbour (one ULP via ``nextafter``) plus a
    small random jitter to break ties across simultaneously-raised pits.

    Not exact/complete: resolves single-cell pits and flats only, does
    nothing for multi-cell basins in one pass. Cheap enough to call every
    hydro substep instead of a full depression reroute.

    Author: B.G (07/2026)
    """
    n_neighbours = ti.static(gridctx.n_neighbours)
    for i in z:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            continue

        has_down = False
        lowest = ti.cast(1e18, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                if z[j] < z[i]:
                    has_down = True
                lowest = ti.min(lowest, z[j])

        if not has_down:
            jitter = ti.random(dtype=cte.FLOAT_TYPE_TI) * ti.cast(1e-3, cte.FLOAT_TYPE_TI)
            z[i] = nextafter(lowest, lowest + 1.0) + jitter


@ti.kernel
def solve_lm_zh_kernel(z: ti.template(), h: ti.template()):
    """
    Same as ``solve_lm_z_kernel`` but for the ``z+h`` hydraulic surface,
    raising ``h`` instead of ``z``. Neighbour comparisons use the decomposed
    (z[j]-z[i])+(h[j]-h[i]) form to avoid cancellation between the large z
    and small h magnitudes.

    ``has_down`` must use strict ``< 0.0`` (matching ``monitor_lm_zh``'s
    strict comparison) -- using ``<= 0.0`` marks exact ties as "resolved"
    without ever raising them, which desyncs from the monitor and looks
    like a permanent stall (monitor keeps counting them, solve never
    touches them again).

    Exact ties (``rel == 0.0``) are broken deterministically by comparing
    flat indices (``i`` vs the tied neighbour) instead of random jitter --
    only the higher-index side of a tie ever rises, so the tie is resolved
    in one pass instead of a random walk where both sides climb together
    and never separate. Exception: if the tied neighbour is an outlet, it
    is never touched by this kernel (skipped outright, see below), so an
    index-based "wait for the other side" never resolves -- an interior
    cell tied with a frozen outlet must always raise regardless of index,
    or it deadlocks permanently against a reference that can never move.

    The raise is computed entirely in h-space (``h[i] += lowest_rel +
    increment``), never recombining z and h into an absolute surface value.
    Reconstructing ``z[i] + h[i] + lowest_rel`` (magnitude ~DEM elevation)
    and taking ``nextafter`` of that rounds away anything finer than z's own
    ULP (~1e-4 at typical DEM elevations) *before* nextafter even runs --
    exactly the cancellation the decomposed ``rel`` form exists to avoid,
    just relocated to the raise step. In genuinely low-gradient terrain,
    real inter-cell gaps are sub-ULP, so that reconstruction collapses whole
    clusters onto the same coarse quantized value, re-tying cells that
    aren't each other's tracked ``lowest_j`` and never resolving. Using a
    fixed physical increment (``get_min_slope(i) * dx``, same pattern as
    ``fill_h_epsilon_kernel`` above) keeps the raise in h's own precision
    domain and needs no nextafter/ULP trick at all.

    Author: B.G (07/2026)
    """
    n_neighbours = ti.static(gridctx.n_neighbours)
    for i in h:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            continue

        has_down = False
        lowest_rel = ti.cast(1e18, cte.FLOAT_TYPE_TI)
        lowest_j = -1
        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                rel = (z[j] - z[i]) + (h[j] - h[i])
                if rel < 0.0:
                    has_down = True
                if rel < lowest_rel:
                    lowest_rel = rel
                    lowest_j = j

        if not has_down:
            ref_is_outlet = lowest_j != -1 and gridctx.tfunc.can_out_flat(lowest_j) == 1
            raise_it = lowest_rel > 0.0 or i > lowest_j or ref_is_outlet
            if raise_it:
                increment = get_min_slope(i) * ti.cast(gridctx.dx, cte.FLOAT_TYPE_TI)
                h[i] = h[i] + lowest_rel + increment


@ti.kernel
def apply_fill_delta_kernel(
    z: ti.template(), surplus: ti.template(), z_filled: ti.template()
):
    """
    Apply the filled topography and accumulate the surplus into ``surplus``.

    Author: B.G (02/2026)
    """
    for i in z:
        dh = z_filled[i] - z[i]
        surplus[i] += dh
        z[i] = z_filled[i]
