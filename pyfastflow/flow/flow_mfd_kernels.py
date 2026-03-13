"""
Generic multiple-flow-direction kernels for ``FlowContext``.

This first pass keeps the Jacobi-style power-iteration approach while using the
new parameter getters for the source term and specializing the stencil size from
the bound grid topology.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None
flowctx = None
get_weight = None


@ti.kernel
def init_mfd_source_kernel(source: ti.template()):
    """
    Initialize the MFD source field from the configured weight getter.

    Author: B.G (02/2026)
    """
    for i in source:
        source[i] = get_weight(i)


@ti.kernel
def compute_mfd_routing_weights_kernel(
    z: ti.template(), routing_weights: ti.template(), routing_sum: ti.template()
):
    """
    Compute normalized MFD routing weights to the active neighbour stencil.

    Author: B.G (02/2026)
    """
    for i in z:
        n_neighbours = ti.static(gridctx.n_neighbours)
        if gridctx.tfunc.nodata_flat(i):
            routing_sum[i] = 0.0
            for k in ti.static(range(n_neighbours)):
                routing_weights[i, k] = 0.0
            continue

        zi = z[i]
        sum_s = ti.cast(0.0, cte.FLOAT_TYPE_TI)

        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and not gridctx.tfunc.nodata_flat(j):
                slope = flowctx.tfunc.slope_from_values_k(zi, z[j], k)
                if slope > 0.0:
                    routing_weights[i, k] = slope
                    sum_s += slope
                else:
                    routing_weights[i, k] = 0.0
            else:
                routing_weights[i, k] = 0.0

        routing_sum[i] = sum_s
        if sum_s > 0.0:
            for k in ti.static(range(n_neighbours)):
                routing_weights[i, k] /= sum_s


@ti.kernel
def mfd_power_iteration_step_kernel(
    source: ti.template(),
    q_current: ti.template(),
    routing_weights: ti.template(),
    q_next: ti.template(),
):
    """
    Perform one Jacobi MFD accumulation step.

    Author: B.G (02/2026)
    """
    for i in source:
        n_neighbours = ti.static(gridctx.n_neighbours)
        if gridctx.tfunc.nodata_flat(i):
            q_next[i] = 0.0
            continue

        acc = source[i]
        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and not gridctx.tfunc.nodata_flat(j):
                opp_k = ti.cast(0, ti.i32)
                if ti.static(gridctx.n_neighbours == 4):
                    if k == 0:
                        opp_k = 3
                    elif k == 1:
                        opp_k = 2
                    elif k == 2:
                        opp_k = 1
                    else:
                        opp_k = 0
                else:
                    opp_k = 7 - k
                wj = routing_weights[j, opp_k]
                if wj > 0.0:
                    acc += wj * q_current[j]
        q_next[i] = acc


@ti.kernel
def check_mfd_convergence_kernel(
    q_a: ti.template(), q_b: ti.template(), eps: ti.template()
):
    """
    Compute the maximum absolute difference between two MFD fields.

    Author: B.G (02/2026)
    """
    eps[None] = 0.0
    for i in q_a:
        diff = ti.abs(q_a[i] - q_b[i])
        ti.atomic_max(eps[None], diff)
