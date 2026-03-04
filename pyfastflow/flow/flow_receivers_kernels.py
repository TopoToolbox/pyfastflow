"""
Generic SFD receiver kernels for ``FlowContext``.

The kernels use the unified ``gridctx`` and ``flowctx`` globals rebound by the
context classes, with the neighbour count specialized statically from the bound
grid topology.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None
flowctx = None


@ti.kernel
def compute_sfd_receivers_kernel(z: ti.template(), receivers: ti.template()):
    """
    Compute deterministic steepest-descent receivers.

    Author: B.G (02/2026)
    """
    for i in receivers:
        if gridctx.tfunc.can_out_flat(i):
            receivers[i] = -1
            continue

        r = i
        sr = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        n_neighbours = ti.static(gridctx.n_neighbours)

        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            valid = j != -1
            tsr = ti.cast(-1.0, cte.FLOAT_TYPE_TI)
            if valid:
                tsr = (z[i] - z[j]) / gridctx.tfunc.dist_from_k_flat(k)

            better = valid and tsr > sr
            sr = tsr if better else sr
            r = j if better else r

        receivers[i] = r


@ti.kernel
def compute_sfd_receivers_stochastic_kernel(z: ti.template(), receivers: ti.template()):
    """
    Compute stochastic steepest-descent receivers.

    Author: B.G (02/2026)
    """
    for i in receivers:
        if gridctx.tfunc.can_out_flat(i):
            receivers[i] = -1
            continue

        r = i
        sr = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        n_neighbours = ti.static(gridctx.n_neighbours)

        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            valid = j != -1
            tsr = ti.cast(-1.0, cte.FLOAT_TYPE_TI)
            if valid:
                tsr = (z[i] - z[j]) / gridctx.tfunc.dist_from_k_flat(k)
                if tsr > 0.0:
                    tsr = ti.random(dtype=cte.FLOAT_TYPE_TI) * ti.math.sqrt(tsr)

            better = valid and tsr > sr
            sr = tsr if better else sr
            r = j if better else r

        receivers[i] = r
