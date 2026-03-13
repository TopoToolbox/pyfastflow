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
