"""
Generic flow analysis kernels for ``FlowContext``.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None
flowctx = None


@ti.kernel
def sum_at_can_out_kernel(field: ti.template(), out_sum: ti.template()):
    """
    Sum ``field`` values over nodes where ``can_out_flat`` is true.

    Nodata cells are ignored.

    Author: B.G (02/2026)
    """
    out_sum[None] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
    for i in field:
        if gridctx.tfunc.can_out_flat(i) == 1 and gridctx.tfunc.nodata_flat(i) == 0:
            ti.atomic_add(out_sum[None], ti.cast(field[i], cte.FLOAT_TYPE_TI))


@ti.kernel
def monitor_lm_z_kernel(z: ti.template()) -> ti.i32:
    """
    Count local minima on ``z`` (no neighbour with lower z). Skips nodata and outlets.

    Author: B.G (07/2026)
    """
    n_neighbours = ti.static(gridctx.n_neighbours)
    count = 0
    for i in z:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            continue
        has_down = False
        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0 and z[j] < z[i]:
                has_down = True
        if not has_down:
            ti.atomic_add(count, 1)
    return count


@ti.kernel
def monitor_lm_zh_kernel(z: ti.template(), h: ti.template()) -> ti.i32:
    """
    Count local minima on ``z+h`` (no neighbour with lower surface). Skips nodata and outlets.

    Author: B.G (07/2026)
    """
    n_neighbours = ti.static(gridctx.n_neighbours)
    count = 0
    for i in z:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            continue
        has_down = False
        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0 and ((z[j] - z[i]) + (h[j] - h[i])) < 0.0:
                has_down = True
        if not has_down:
            ti.atomic_add(count, 1)
    return count
