"""
GraphFlood-oriented kernels for ``FloodContext``.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None
flowctx = None
floodctx = None


@ti.kernel
def add_source_to_Q_kernel(Q: ti.template()):
    """Add source term to discharge field. Author: B.G (02/2026)"""
    for i in Q:
        if gridctx.tfunc.nodata_flat(i) == 0:
            Q[i] += floodctx.tfunc.source_to_Q(i)


@ti.kernel
def add_source_to_h_kernel(h: ti.template()):
    """Add source term to depth field. Author: B.G (02/2026)"""
    for i in h:
        if gridctx.tfunc.nodata_flat(i) == 0:
            h[i] += floodctx.tfunc.source_to_h(i)


@ti.kernel
def make_surface_kernel(z: ti.template(), h: ti.template(), surface: ti.template()):
    """Build z+h surface field. Author: B.G (02/2026)"""
    for i in surface:
        surface[i] = z[i] + h[i]


@ti.kernel
def distribute_flow_local_kernel(
    z: ti.template(),
    h: ti.template(),
    Q_in: ti.template(),
    Q_next: ti.template(),
):
    """
    Distribute incoming discharge to downslope neighbours and add source.

    Author: B.G (02/2026)
    """
    n_neigh = ti.static(gridctx.n_neighbours)
    for i in Q_next:
        Q_next[i] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        if gridctx.tfunc.nodata_flat(i) == 0:
            Q_next[i] += floodctx.tfunc.source_to_Q(i)

    for i in Q_in:
        if gridctx.tfunc.nodata_flat(i) == 1:
            continue
        if gridctx.tfunc.can_out_flat(i) == 1:
            continue

        qi = Q_in[i]
        if qi <= 0.0:
            continue

        slopes = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        sum_s = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neigh)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
                s = ti.max(s, ti.cast(0.0, cte.FLOAT_TYPE_TI))
                slopes[k] = s
                sum_s += s

        if sum_s <= 0.0:
            ti.atomic_add(Q_next[i], qi)
            ti.atomic_add(h[i], floodctx.tfunc.get_gf_min_increment(i))
        else:
            # sumslope = 0.
            for k in ti.static(range(n_neigh)):
                j = gridctx.tfunc.neighbour_flat(i, k)
                if j != -1 and slopes[k] > 0.0:
                    # sumslope+=slopes[k] / sum_s
                    ti.atomic_add(Q_next[j], qi * slopes[k] / sum_s)
            # if(abs(sumslope - 1) > 1e-3):
            #     print(sumslope)

@ti.kernel
def graphflood_core_kernel(
    z: ti.template(),
    h: ti.template(),
    receivers: ti.template(),
    Q_in: ti.template(),
    h_next: ti.template(),
):
    """
    Apply friction-law core update and compute next depth.

    Author: B.G (02/2026)
    """
    dx = ti.cast(ti.static(gridctx.dx), cte.FLOAT_TYPE_TI)
    area = dx * dx
    n_neigh = ti.static(gridctx.n_neighbours)
    for i in h:
        if gridctx.tfunc.nodata_flat(i) == 1:
            h_next[i] = h[i]
            continue

        if gridctx.tfunc.can_out_flat(i) == 1:
            h_next[i] = floodctx.tfunc.get_boundary_h(i)
            continue

        # Recompute local steepest slope directly from z+h each pass.
        # `receivers` is intentionally ignored here (kept in signature for API compatibility).
        best_s = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neigh)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
                if s > best_s:
                    best_s = s
        slope = ti.max(best_s, ti.cast(1e-9, cte.FLOAT_TYPE_TI))

        Qo = floodctx.tfunc.compute_qo_from_h_slope(h[i], slope, i)
        dth = floodctx.tfunc.get_dth(i)
        dh = (Q_in[i] - Qo) / area * dth

        min_inc = floodctx.tfunc.get_gf_min_increment(i)
        if Q_in[i] > Qo and dh < min_inc:
            dh = min_inc
        elif Qo > Q_in[i] and dh > -min_inc:
            dh = -min_inc

        h_next[i] = ti.max(ti.cast(0.0, cte.FLOAT_TYPE_TI), h[i] + dh)


@ti.kernel
def localpass_kernel(
    z: ti.template(),
    h: ti.template(),
    receivers: ti.template(),
    Q_in: ti.template(),
    Q_next: ti.template(),
    h_next: ti.template(),
):
    """
    Execute one flat GraphFlood distribute/core pass.

    ``Q_next`` and ``h_next`` receive the next state. ``h`` is left untouched.

    Author: B.G (03/2026)
    """
    n_neigh = ti.static(gridctx.n_neighbours)
    dx = ti.cast(ti.static(gridctx.dx), cte.FLOAT_TYPE_TI)
    area = dx * dx

    for i in Q_next:
        Q_next[i] = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        if gridctx.tfunc.nodata_flat(i) == 0:
            Q_next[i] += floodctx.tfunc.source_to_Q(i)

    for i in Q_in:
        if gridctx.tfunc.nodata_flat(i) == 1:
            continue
        if gridctx.tfunc.can_out_flat(i) == 1:
            continue

        qi = Q_in[i]
        if qi <= 0.0:
            continue

        slopes = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        sum_s = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neigh)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
                s = ti.max(s, ti.cast(0.0, cte.FLOAT_TYPE_TI))
                slopes[k] = s
                sum_s += s

        if sum_s <= 0.0:
            ti.atomic_add(Q_next[i], qi)
        else:
            for k in ti.static(range(n_neigh)):
                j = gridctx.tfunc.neighbour_flat(i, k)
                if j != -1 and slopes[k] > 0.0:
                    ti.atomic_add(Q_next[j], qi * slopes[k] / sum_s)

    for i in h:
        if gridctx.tfunc.nodata_flat(i) == 1:
            h_next[i] = h[i]
            continue
        if gridctx.tfunc.can_out_flat(i) == 1:
            h_next[i] = floodctx.tfunc.get_boundary_h(i)
            continue

        best_s = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neigh)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
                if s > best_s:
                    best_s = s
        slope = ti.max(best_s, ti.cast(1e-9, cte.FLOAT_TYPE_TI))

        qo = floodctx.tfunc.compute_qo_from_h_slope(h[i], slope, i)
        dth = floodctx.tfunc.get_dth(i)
        dh = (Q_next[i] - qo) / area * dth

        min_inc = floodctx.tfunc.get_gf_min_increment(i)
        if Q_next[i] > qo and dh < min_inc:
            dh = min_inc
        elif qo > Q_next[i] and dh > -min_inc:
            dh = -min_inc

        h_next[i] = ti.max(ti.cast(0.0, cte.FLOAT_TYPE_TI), h[i] + dh)


@ti.kernel
def compute_Qo_kernel(
    z: ti.template(), h: ti.template(), Qo: ti.template()
):
    """Compute local outflow capacity field using steepest slope. Author: B.G (02/2026)"""
    n_neigh = ti.static(gridctx.n_neighbours)
    for i in Qo:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            Qo[i] = 0.0
            continue
        slope = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neigh)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
                if s > slope:
                    slope = s
        Qo[i] = floodctx.tfunc.compute_qo_from_h_slope(h[i], ti.max(slope, ti.cast(1e-9, cte.FLOAT_TYPE_TI)), i)


@ti.kernel
def compute_u_kernel(
    z: ti.template(), h: ti.template(), u: ti.template()
):
    """Compute local velocity field from friction law using steepest slope. Author: B.G (02/2026)"""
    n_neigh = ti.static(gridctx.n_neighbours)
    for i in u:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            u[i] = 0.0
            continue
        slope = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neigh)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
                if s > slope:
                    slope = s
        u[i] = floodctx.tfunc.compute_u_from_h_slope(h[i], ti.max(slope, ti.cast(1e-9, cte.FLOAT_TYPE_TI)), i)


@ti.kernel
def compute_tau_kernel(z: ti.template(), h: ti.template(), tau: ti.template()):
    """Compute basal shear-stress proxy field using steepest slope. Author: B.G (02/2026)"""
    for i in tau:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            tau[i] = 0.0
            continue
        slope = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        n_neigh = ti.static(gridctx.n_neighbours)
        for k in ti.static(range(n_neigh)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
                if s > slope:
                    slope = s
        tau[i] = slope * h[i] * floodctx.tfunc.get_rho_w(i) * floodctx.tfunc.get_gravity(i)


@ti.kernel
def compute_Sw_kernel(z: ti.template(), h: ti.template(), Sw: ti.template()):
    """Compute steepest hydraulic slope (z+h). Author: B.G (04/2026)"""
    n_neigh = ti.static(gridctx.n_neighbours)
    for i in Sw:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            Sw[i] = 0.0
            continue
        slope = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neigh)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
                if s > slope:
                    slope = s
        Sw[i] = slope


@ti.kernel
def compute_q_kernel(z: ti.template(), h: ti.template(), q: ti.template()):
    """Compute steepest unit discharge. Author: B.G (04/2026)"""
    n_neigh = ti.static(gridctx.n_neighbours)
    for i in q:
        if gridctx.tfunc.nodata_flat(i) == 1 or gridctx.tfunc.can_out_flat(i) == 1:
            q[i] = 0.0
            continue
        slope = ti.cast(0.0, cte.FLOAT_TYPE_TI)
        for k in ti.static(range(n_neigh)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
                s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
                if s > slope:
                    slope = s
        q[i] = floodctx.tfunc.compute_q_from_h_slope(h[i], ti.max(slope, ti.cast(1e-9, cte.FLOAT_TYPE_TI)), i)


@ti.kernel
def compute_Sw_direction_kernel(z: ti.template(), h: ti.template(), k: ti.template(), Sw_dir: ti.template()):
    """Compute hydraulic slope towards direction k. Author: B.G (04/2026)"""
    for i in Sw_dir:
        if gridctx.tfunc.nodata_flat(i) == 1:
            Sw_dir[i] = 0.0
            continue
        j = gridctx.tfunc.neighbour_flat(i, k)
        if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
            Sw_dir[i] = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
        else:
            Sw_dir[i] = 0.0


@ti.kernel
def compute_u_direction_kernel(z: ti.template(), h: ti.template(), k: ti.template(), u_dir: ti.template()):
    """Compute velocity towards direction k (signed). Author: B.G (04/2026)"""
    for i in u_dir:
        if gridctx.tfunc.nodata_flat(i) == 1:
            u_dir[i] = 0.0
            continue
        j = gridctx.tfunc.neighbour_flat(i, k)
        if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
            s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
            # We use absolute slope for Manning but keep sign of velocity
            u_val = floodctx.tfunc.compute_u_from_h_slope(h[i], ti.abs(s), i)
            u_dir[i] = u_val if s >= 0.0 else -u_val
        else:
            u_dir[i] = 0.0


@ti.kernel
def compute_tau_direction_kernel(z: ti.template(), h: ti.template(), k: ti.template(), tau_dir: ti.template()):
    """Compute shear stress towards direction k (signed). Author: B.G (04/2026)"""
    for i in tau_dir:
        if gridctx.tfunc.nodata_flat(i) == 1:
            tau_dir[i] = 0.0
            continue
        j = gridctx.tfunc.neighbour_flat(i, k)
        if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
            s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
            tau_dir[i] = s * h[i] * floodctx.tfunc.get_rho_w(i) * floodctx.tfunc.get_gravity(i)
        else:
            tau_dir[i] = 0.0


@ti.kernel
def compute_q_direction_kernel(z: ti.template(), h: ti.template(), k: ti.template(), q_dir: ti.template()):
    """Compute unit discharge towards direction k (signed). Author: B.G (04/2026)"""
    for i in q_dir:
        if gridctx.tfunc.nodata_flat(i) == 1:
            q_dir[i] = 0.0
            continue
        j = gridctx.tfunc.neighbour_flat(i, k)
        if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
            s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
            q_val = floodctx.tfunc.compute_q_from_h_slope(h[i], ti.abs(s), i)
            q_dir[i] = q_val if s >= 0.0 else -q_val
        else:
            q_dir[i] = 0.0


@ti.kernel
def compute_Q_direction_kernel(z: ti.template(), h: ti.template(), k: ti.template(), Q_dir: ti.template()):
    """Compute total discharge towards direction k (signed). Author: B.G (04/2026)"""
    for i in Q_dir:
        if gridctx.tfunc.nodata_flat(i) == 1:
            Q_dir[i] = 0.0
            continue
        j = gridctx.tfunc.neighbour_flat(i, k)
        if j != -1 and gridctx.tfunc.nodata_flat(j) == 0:
            s = flowctx.tfunc.slope_from_values_k(z[i] + h[i], z[j] + h[j], k)
            qo_val = floodctx.tfunc.compute_qo_from_h_slope(h[i], ti.abs(s), i)
            Q_dir[i] = qo_val if s >= 0.0 else -qo_val
        else:
            Q_dir[i] = 0.0
