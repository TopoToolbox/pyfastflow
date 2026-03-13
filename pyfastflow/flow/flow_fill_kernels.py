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
