"""
LisFlood (LS) kernels bound by ``FloodContext``.

This cleaned version uses context helpers for physical parameters and source
terms while keeping the algorithm close to the original implementation.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None
floodctx = None


@ti.kernel
def ls_add_source_to_h_kernel(h: ti.template()):
    """Add effective source contribution to water depth. Author: B.G (02/2026)"""
    for i in h:
        if gridctx.tfunc.nodata_flat(i) == 0:
            h[i] += floodctx.tfunc.source_to_h(i)


@ti.kernel
def ls_flow_route_kernel(
    h: ti.template(), z: ti.template(), qx: ti.template(), qy: ti.template()
):
    """
    Route discharge in x and y directions using the local inertial update.

    Author: B.G (02/2026)
    """
    for i in z:
        if gridctx.tfunc.nodata_flat(i) == 1:
            continue

        top = gridctx.tfunc.neighbour_flat(i, 0)
        left = gridctx.tfunc.neighbour_flat(i, 1)
        dt_loc = floodctx.tfunc.dth(i)
        grav = floodctx.tfunc.gravity(i)
        man = ti.max(floodctx.tfunc.friction_coeff(i), ti.cast(1e-9, cte.FLOAT_TYPE_TI))
        flow_exp = floodctx.tfunc.friction_exponent(i)
        denom_pow = ti.cast(2.0, cte.FLOAT_TYPE_TI) + ti.cast(2.0, cte.FLOAT_TYPE_TI) * flow_exp

        if top != -1 and (h[i] > 0.0 or h[top] > 0.0):
            hflow = ti.max(z[i] + h[i], z[top] + h[top]) - ti.max(z[i], z[top])
            if hflow > cte.HFLOW_THRESHOLD:
                tempslope = ((z[top] + h[top]) - (z[i] + h[i])) / ti.cast(
                    gridctx.dx, cte.FLOAT_TYPE_TI
                )
                qy[i] = (qy[i] - grav * hflow * dt_loc * tempslope) / (
                    1.0
                    + grav
                    * hflow
                    * dt_loc
                    * (man * man)
                    * ti.abs(qy[i])
                    / ti.pow(ti.max(hflow, ti.cast(1e-9, cte.FLOAT_TYPE_TI)), denom_pow)
                )

                froude_vel = ti.sqrt(grav * hflow) * cte.FROUDE_LIMIT
                if qy[i] > 0 and (qy[i] / ti.max(hflow, 1e-9)) / ti.sqrt(grav * hflow) > cte.FROUDE_LIMIT:
                    qy[i] = hflow * froude_vel
                if qy[i] < 0 and ti.abs(qy[i] / ti.max(hflow, 1e-9)) / ti.sqrt(grav * hflow) > cte.FROUDE_LIMIT:
                    qy[i] = -hflow * froude_vel
            else:
                qy[i] = 0.0
        elif top == -1:
            qy[i] = 0.0

        if left != -1 and (h[i] > 0.0 or h[left] > 0.0):
            hflow = ti.max(z[i] + h[i], z[left] + h[left]) - ti.max(z[i], z[left])
            if hflow > cte.HFLOW_THRESHOLD:
                tempslope = ((z[left] + h[left]) - (z[i] + h[i])) / ti.cast(
                    gridctx.dx, cte.FLOAT_TYPE_TI
                )
                qx[i] = (qx[i] - grav * hflow * dt_loc * tempslope) / (
                    1.0
                    + grav
                    * hflow
                    * dt_loc
                    * (man * man)
                    * ti.abs(qx[i])
                    / ti.pow(ti.max(hflow, ti.cast(1e-9, cte.FLOAT_TYPE_TI)), denom_pow)
                )

                froude_vel = ti.sqrt(grav * hflow) * cte.FROUDE_LIMIT
                if qx[i] > 0 and (qx[i] / ti.max(hflow, 1e-9)) / ti.sqrt(grav * hflow) > cte.FROUDE_LIMIT:
                    qx[i] = hflow * froude_vel
                if qx[i] < 0 and ti.abs(qx[i] / ti.max(hflow, 1e-9)) / ti.sqrt(grav * hflow) > cte.FROUDE_LIMIT:
                    qx[i] = -hflow * froude_vel
            else:
                qx[i] = 0.0
        elif left == -1:
            qx[i] = 0.0


@ti.kernel
def ls_depth_update_kernel(
    h: ti.template(), z: ti.template(), qx: ti.template(), qy: ti.template()
):
    """
    Update depth from discharge divergence.

    Author: B.G (02/2026)
    """
    dx = ti.cast(ti.static(gridctx.dx), cte.FLOAT_TYPE_TI)
    for i in z:
        if gridctx.tfunc.nodata_flat(i) == 1:
            continue

        right = gridctx.tfunc.neighbour_flat(i, 2)
        bottom = gridctx.tfunc.neighbour_flat(i, 3)

        tqy = -qy[i]
        if bottom != -1:
            tqy += qy[bottom]

        tqx = -qx[i]
        if right != -1:
            tqx += qx[right]

        dh = floodctx.tfunc.dth(i) * (tqy + tqx) / dx
        h[i] = ti.max(ti.cast(0.0, cte.FLOAT_TYPE_TI), h[i] + dh)
        if gridctx.tfunc.can_out_flat(i) == 1:
            h[i] = floodctx.tfunc.boundary_h(i)
