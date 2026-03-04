"""
Generic parameter getter helpers for ``FlowContext``.

These Taichi helpers expose one unified access path for flow-related numerical
parameters while allowing compile-time specialization based on the configured
storage mode.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


flowctx = None


@ti.func
def get_weight(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the local accumulation weight for node ``i``.

    Author: B.G (02/2026)
    """
    if ti.static(flowctx.weight_mode == "const"):
        return ti.static(flowctx.weight_const)
    if ti.static(flowctx.weight_mode == "scalar"):
        return flowctx.weight_scalar[None]
    return flowctx.weight_field[i]


@ti.func
def get_min_slope(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the local minimum slope for node ``i``.

    Author: B.G (02/2026)
    """
    if ti.static(flowctx.min_slope_mode == "const"):
        return ti.static(flowctx.min_slope_const)
    if ti.static(flowctx.min_slope_mode == "scalar"):
        return flowctx.min_slope_scalar[None]
    return flowctx.min_slope_field[i]
