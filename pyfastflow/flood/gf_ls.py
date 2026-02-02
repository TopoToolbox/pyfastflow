"""
Flood LisFlood (LS) shallow water flow computation kernels.

Method implementation adapted from HAIL-CAESAR (https://github.com/dvalters/HAIL-CAESAR) to GPU
Method described in Bates et al. 2010: A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling

This module implements the LisFlood method for 2D shallow water flow simulation
using GPU-accelerated Taichi kernels. The LS method uses explicit time integration
with local time stepping for stability and efficiency.

Key Features:
- LisFlood time integration for shallow water equations
- Manning's friction implementation with stability checks
- Froude number limiting for numerical stability
- Discharge magnitude checks to prevent excessive flow
- 2D flow routing in both x and y directions

The implementation follows the LisFlood method principles:
1. Flow routing: Compute discharge in x and y directions
2. Depth update: Update water depths based on discharge divergence

Implementation Notes:
- Uses separate bed elevation (z) and water depth (h) fields instead of combined surface elevation
- Surface elevation is computed on-the-fly as z + h when needed for slope calculations
- This approach avoids floating-point precision issues with field synchronization

Constants used from constants module:
- HFLOW_THRESHOLD: Minimum flow depth threshold
- EDGESW: Boundary slope for edge conditions
- MANNING: Manning's roughness coefficient
- NY: Number of grid rows
- NX: Number of grid columns
- DT_HYDRO_LS: Time step for LisFlood hydro
- GRAVITY: Gravitational acceleration
- FROUDE_LIMIT: Maximum Froude number

Author: B.G.
"""

import taichi as ti

import pyfastflow as pf
import pyfastflow as pff
import pyfastflow.flow as flow

from .. import constants as cte


@ti.kernel
def init_LS_on_hw_from_constant_effective_prec(hw: ti.template(), z: ti.template()):
    """
    Initialize water depth from constant precipitation rate.

    Adds precipitation as effective water depth increase for LisFlood simulation.
    Only modifies water depth field - bed elevation remains unchanged.

    Args:
            hw (ti.template): Water depth field to be updated
            z (ti.template): Bed elevation field (unchanged)

    Author: B.G.
    """
    for i in hw:
        if pff.grid.neighbourer_flat.nodata(i):
            continue
        dh = cte.PREC * cte.DT_HYDRO_LS
        hw[i] += dh


@ti.kernel
def init_LS_on_hw_from_variable_effective_prec(
    hw: ti.template(), z: ti.template(), rate: ti.template()
):
    """
    Initialize water depth from variable precipitation rates.

    Adds spatially variable precipitation as effective water depth increase
    for LisFlood simulation. Only modifies water depth field.

    Args:
            hw (ti.template): Water depth field to be updated
            z (ti.template): Bed elevation field (unchanged)
            rate (ti.template): Spatially variable precipitation rates (m/s)

    Author: B.G.
    """
    for i in hw:
        if pff.grid.neighbourer_flat.nodata(i):
            continue
        dh = rate[i] * cte.DT_HYDRO_LS
        hw[i] += dh


# Placeholder for future discharge initialization kernel
# @ti.kernel
# def init_LS_q(qx, qy):
#     """Initialize discharge fields for LisFlood simulation."""


@ti.kernel
def flow_route(
    hw: ti.template(), z: ti.template(), qx: ti.template(), qy: ti.template()
):
    """
    Compute flow routing in x and y directions using LisFlood method.

    Updates discharge fields (qx, qy) based on water surface gradients,
    Manning's friction, and stability constraints. The method applies:
    1. Momentum equation with pressure and friction terms
    2. Froude number limiting for numerical stability
    3. Discharge magnitude checks to prevent excessive flow

    The routing is performed separately for x and y directions using
    staggered grid approach where discharges are computed at cell faces.

    Args:
            hw (ti.template): Water depth field
            z (ti.template): Bed elevation field
            qx (ti.template): Discharge in x-direction (m²/s)
            qy (ti.template): Discharge in y-direction (m²/s)

    Author: B.G.
    """

    # Process each grid cell for flow routing
    for i in z:
        if pff.grid.neighbourer_flat.nodata(i):
            continue
        # Get neighboring cell indices
        top = pf.grid.neighbourer_flat.neighbour(i, 0)  # North neighbor
        left = pf.grid.neighbourer_flat.neighbour(i, 1)  # West neighbor

        # Get row and column indices for boundary condition checks
        row, col = flow.neighbourer_flat.rc_from_i(i)

        if row == 0 or col == 0:
            continue

        # =======================================
        # FLOW ROUTING IN Y-DIRECTION (NORTH-SOUTH)
        # =======================================
        if top > -1:  # Check if top neighbor exists
            # Only compute flow if there's water in current or neighboring cell
            if hw[i] > 0 or hw[top] > 0:
                # Calculate effective flow depth (depth over the higher bed elevation)
                # Surface elevation = bed + water, effective depth = max_surface - max_bed
                hflow = ti.math.max(z[i] + hw[i], z[top] + hw[top]) - ti.math.max(
                    z[top], z[i]
                )

                # Only compute flow if effective depth exceeds threshold
                if hflow > cte.HFLOW_THRESHOLD:
                    # Calculate water surface slope between cells
                    # Surface slope = (neighbor_surface - current_surface) / distance
                    tempslope = ((z[top] + hw[top]) - (z[i] + hw[i])) / cte.DX

                    # Apply boundary conditions for slope
                    if row == cte.NY - 1:  # Top boundary
                        tempslope = cte.EDGESW
                    elif row <= 2:  # Bottom boundary
                        tempslope = 0 - cte.EDGESW
                    # elif pf.flow.neighbourer_flat.can_leave_domain(i):  # Domain edges
                    #     tempslope = cte.EDGESW

                    # Update y-discharge using implicit friction approach
                    # Based on momentum equation: dq/dt = -g*h*dh/dx - friction
                    qy[i] = (
                        qy[i] - (cte.GRAVITY * hflow * cte.DT_HYDRO_LS * tempslope)
                    ) / (
                        1
                        + cte.GRAVITY
                        * hflow
                        * cte.DT_HYDRO_LS
                        * (cte.MANNING * cte.MANNING)
                        * ti.abs(qy[i])
                        / ti.pow(hflow, (10.0 / 3.0))
                    )

                    # STABILITY CONSTRAINT 1: Froude number limiting
                    # Prevent supercritical flow (Fr > FROUDE_LIMIT)
                    froude_vel = ti.sqrt(cte.GRAVITY * hflow) * cte.FROUDE_LIMIT
                    if (
                        qy[i] > 0
                        and (qy[i] / hflow) / ti.sqrt(cte.GRAVITY * hflow)
                        > cte.FROUDE_LIMIT
                    ):
                        qy[i] = hflow * froude_vel

                    if (
                        qy[i] < 0
                        and ti.abs(qy[i] / hflow) / ti.sqrt(cte.GRAVITY * hflow)
                        > cte.FROUDE_LIMIT
                    ):
                        qy[i] = -hflow * froude_vel

                    # STABILITY CONSTRAINT 2: Discharge magnitude/timestep checks
                    # Prevent excessive water transfer in single timestep
                    max_transfer = (hw[i] * cte.DX) / 5  # Max 20% of water per timestep
                    if qy[i] > 0 and (qy[i] * cte.DT_HYDRO_LS / cte.DX) > (hw[i] / 4):
                        qy[i] = max_transfer / cte.DT_HYDRO_LS

                    max_transfer_top = (hw[top] * cte.DX) / 5
                    if qy[i] < 0 and ti.abs(qy[i] * cte.DT_HYDRO_LS / cte.DX) > (
                        hw[top] / 4
                    ):
                        qy[i] = -max_transfer_top / cte.DT_HYDRO_LS
                else:
                    qy[i] = 0  # No flow if effective depth too small
            else:
                qy[i] = 0  # No flow if effective depth too small

        # =======================================
        # FLOW ROUTING IN X-DIRECTION (WEST-EAST)
        # =======================================
        if left > -1:  # Check if left neighbor exists
            if hw[i] > 0 or hw[left] > 0:
                # Calculate effective flow depth (same approach as y-direction)
                # Surface elevation = bed + water, effective depth = max_surface - max_bed
                hflow = ti.math.max(z[i] + hw[i], z[left] + hw[left]) - ti.math.max(
                    z[i], z[left]
                )

                # Only compute flow if effective depth exceeds threshold
                if hflow > cte.HFLOW_THRESHOLD:
                    # Calculate water surface slope between cells
                    # Surface slope = (neighbor_surface - current_surface) / distance
                    tempslope = ((z[left] + hw[left]) - (z[i] + hw[i])) / cte.DX

                    # Apply boundary conditions for slope
                    if col == cte.NX - 1:  # Right boundary
                        tempslope = cte.EDGESW
                    elif col <= 2:  # Left boundary
                        tempslope = 0 - cte.EDGESW
                    # elif pf.flow.neighbourer_flat.can_leave_domain(i):  # Domain edges
                    #     tempslope = cte.EDGESW

                    # Update x-discharge using implicit friction approach
                    # Same momentum equation as y-direction
                    qx[i] = (
                        qx[i] - (cte.GRAVITY * hflow * cte.DT_HYDRO_LS * tempslope)
                    ) / (
                        1
                        + cte.GRAVITY
                        * hflow
                        * cte.DT_HYDRO_LS
                        * (cte.MANNING * cte.MANNING)
                        * ti.abs(qx[i])
                        / ti.pow(hflow, (10.0 / 3.0))
                    )

                    # STABILITY CONSTRAINT 1: Froude number limiting
                    froude_vel = ti.sqrt(cte.GRAVITY * hflow) * cte.FROUDE_LIMIT
                    if (
                        qx[i] > 0
                        and (qx[i] / hflow) / ti.sqrt(cte.GRAVITY * hflow)
                        > cte.FROUDE_LIMIT
                    ):
                        qx[i] = hflow * froude_vel

                    if (
                        qx[i] < 0
                        and ti.abs(qx[i] / hflow) / ti.sqrt(cte.GRAVITY * hflow)
                        > cte.FROUDE_LIMIT
                    ):
                        qx[i] = -hflow * froude_vel

                    # STABILITY CONSTRAINT 2: Discharge magnitude/timestep checks
                    max_transfer = (hw[i] * cte.DX) / 5
                    if qx[i] > 0 and (qx[i] * cte.DT_HYDRO_LS / cte.DX) > (hw[i] / 4):
                        qx[i] = max_transfer / cte.DT_HYDRO_LS

                    max_transfer_left = (hw[left] * cte.DX) / 5
                    if qx[i] < 0 and ti.abs(qx[i] * cte.DT_HYDRO_LS / cte.DX) > (
                        hw[left] / 4
                    ):
                        qx[i] = -max_transfer_left / cte.DT_HYDRO_LS
                else:
                    qx[i] = 0  # No flow if effective depth too small
            else:
                qx[i] = 0  # No flow if effective depth too small


@ti.kernel
def depth_update(
    hw: ti.template(), z: ti.template(), qx: ti.template(), qy: ti.template()
):
    """
    Update water depths based on discharge divergence using continuity equation.

    Applies the continuity equation to update water depths based on the
    divergence of discharge in x and y directions. This implements the
    conservation of mass for shallow water flow.

    The continuity equation: dh/dt + d(qx)/dx + d(qy)/dy = 0
    Discretized as: h_new = h_old - dt/dx * (qx_right - qx_left + qy_top - qy_bottom)

    Args:
            hw (ti.template): Water depth field to be updated
            z (ti.template): Bed elevation field (unchanged)
            qx (ti.template): Discharge in x-direction (m²/s)
            qy (ti.template): Discharge in y-direction (m²/s)

    Author: B.G.
    """

    # Process each grid cell for depth update
    for i in z:
        if pff.grid.neighbourer_flat.nodata(i):
            continue
        # Get row and column indices (for potential boundary checks)
        row, col = flow.neighbourer_flat.rc_from_i(i)

        # Get neighboring cell indices for discharge divergence calculation
        right = pf.grid.neighbourer_flat.neighbour(i, 2)  # East neighbor
        bottom = pf.grid.neighbourer_flat.neighbour(i, 3)  # South neighbor

        # Calculate discharge divergence in y-direction
        # Net inflow = (inflow from south) - (outflow to north)
        tqy = -qy[i]
        if bottom > -1:  # Check if south neighbor exists
            tqy += qy[bottom]  # qy[bottom] flows into i, qy[i] flows out of i

        # Calculate discharge divergence in x-direction
        # Net inflow = (inflow from west) - (outflow to east)
        tqx = -qx[i]
        if right > -1:  # Check if east neighbor exists
            tqx += qx[right]  # qx[right] flows into i, qx[i] flows out of i

        # Update water depths using continuity equation
        # dh/dt = -(dqx/dx + dqy/dy) -> h_new = h_old + dt * net_inflow / area
        # Only water depth is updated - bed elevation remains constant
        dh = cte.DT_HYDRO_LS * (tqy + tqx) / cte.DX

        hw[i] += dh
