"""
Generic parameter and geometry helpers for ``FlowContext``.

These Taichi helpers expose one unified access path for flow-related numerical
parameters and flow-specific geometry corrections while allowing compile-time
specialization based on the configured mode flags.

Author: B.G (02/2026)
"""

import math

import taichi as ti

from .. import constants as cte


gridctx = None
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


@ti.func
def dist_from_k_corrected(k: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return neighbour distance with optional diagonal correction baked in.

    When the diagonal partition correction is enabled on a D8 grid, diagonal
    links are treated as having an effective distance of ``dx`` instead of
    ``sqrt(2) * dx``.

    Author: B.G (02/2026)
    """
    dist = gridctx.tfunc.dist_from_k_flat(k)
    if ti.static(flowctx.diagonal_partition_correction and gridctx.n_neighbours == 8):
        if ti.static(k in (0, 2, 5, 7)):
            dist /= ti.cast(ti.static(math.sqrt(2.0)), cte.FLOAT_TYPE_TI)
    return dist


@ti.func
def dist_between_nodes_corrected(i: ti.i32, j: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return node-to-node distance with the same optional diagonal correction.

    Author: B.G (02/2026)
    """
    dist = gridctx.tfunc.dist_between_nodes_flat(i, j)
    if ti.static(flowctx.diagonal_partition_correction and gridctx.n_neighbours == 8):
        dx = ti.cast(ti.static(gridctx.dx), cte.FLOAT_TYPE_TI)
        if dist > dx * ti.cast(1.1, cte.FLOAT_TYPE_TI):
            dist /= ti.cast(ti.static(math.sqrt(2.0)), cte.FLOAT_TYPE_TI)
    return dist


@ti.func
def slope_from_values_k(
    zi: cte.FLOAT_TYPE_TI, zj: cte.FLOAT_TYPE_TI, k: ti.i32
) -> cte.FLOAT_TYPE_TI:
    """
    Return corrected slope between a node value and its ``k``th neighbour.

    Author: B.G (02/2026)
    """
    return (zi - zj) / dist_from_k_corrected(k)


@ti.func
def slope_between_nodes(
    vi: cte.FLOAT_TYPE_TI, vj: cte.FLOAT_TYPE_TI, i: ti.i32, j: ti.i32
) -> cte.FLOAT_TYPE_TI:
    """
    Return corrected slope between two neighbouring nodes.

    Author: B.G (02/2026)
    """
    return (vi - vj) / dist_between_nodes_corrected(i, j)
