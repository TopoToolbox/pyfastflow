"""
Generic D4 multiple-flow-direction kernels for ``FlowContext``.

This first pass keeps the Jacobi-style power-iteration approach while using the
new parameter getters for the source term.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None
flowctx = None
get_weight = None


@ti.func
def _opposite_d4(k: ti.i32) -> ti.i32:
    """
    Return the opposite D4 direction index.

    Author: B.G (02/2026)
    """
    out = ti.cast(0, ti.i32)
    if k == 0:
        out = 3
    elif k == 1:
        out = 2
    elif k == 2:
        out = 1
    else:
        out = 0
    return out


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
    Compute normalized MFD routing weights to D4 neighbours.

    Author: B.G (02/2026)
    """
    for i in z:
        if gridctx.tfunc.nodata_flat(i):
            routing_sum[i] = 0.0
            for k in ti.static(range(4)):
                routing_weights[i, k] = 0.0
            continue

        zi = z[i]
        sum_s = ti.cast(0.0, cte.FLOAT_TYPE_TI)

        for k in ti.static(range(4)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and not gridctx.tfunc.nodata_flat(j):
                slope = (zi - z[j]) / gridctx.tfunc.dist_from_k_flat(k)
                if slope > 0.0:
                    routing_weights[i, k] = slope
                    sum_s += slope
                else:
                    routing_weights[i, k] = 0.0
            else:
                routing_weights[i, k] = 0.0

        routing_sum[i] = sum_s
        if sum_s > 0.0:
            for k in ti.static(range(4)):
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
        if gridctx.tfunc.nodata_flat(i):
            q_next[i] = 0.0
            continue

        acc = source[i]
        for k in ti.static(range(4)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and not gridctx.tfunc.nodata_flat(j):
                wj = routing_weights[j, _opposite_d4(k)]
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
