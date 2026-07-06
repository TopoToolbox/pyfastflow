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
    and small h magnitudes, but the actual raise reconstructs the absolute
    surface (z[i]+lowest_rel) before applying nextafter/jitter -- computing
    the ULP step on the small relative h-offset instead of the absolute
    surface makes the nudge too small to survive being added back to z[i].

    Author: B.G (07/2026)
    """
    n_neighbours = ti.static(gridctx.n_neighbours)
    for i in h:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            continue

        has_down = False
        lowest_rel = ti.cast(1e18, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                rel = (z[j] - z[i]) + h[j]
                if ((z[j] - z[i]) + (h[j] - h[i])) < 0.0:
                    has_down = True
                lowest_rel = ti.min(lowest_rel, rel)

        if not has_down:
            target_abs = z[i] + lowest_rel
            jitter = ti.random(dtype=cte.FLOAT_TYPE_TI) * ti.cast(1e-3, cte.FLOAT_TYPE_TI)
            new_abs = nextafter(target_abs, target_abs + 1.0) + jitter
            h[i] = ti.max(ti.cast(0.0, cte.FLOAT_TYPE_TI), new_abs - z[i])


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
