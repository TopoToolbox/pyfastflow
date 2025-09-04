"""
Hydrodynamic computation kernels for Flood shallow water flow.

This module implements the core hydrodynamic algorithms for 2D shallow water
flow simulation using GPU-accelerated Taichi kernels. It provides methods for
discharge diffusion and Manning's equation-based flow depth updates.

Key algorithms:
- Discharge diffusion for multiple flow path simulation
- Manning's equation for flow resistance and depth updates
- Integration with FastFlow's flow routing system

Based on methods from Gailleton et al. 2024 for efficient
shallow water flow approximation, adapted to GPU (Gailleton et al., in prep).

Author: B.G.
"""

import taichi as ti

import pyfastflow.flow as flow

from .. import constants as cte


@ti.kernel
def diffuse_Q_constant_prec(z: ti.template(), Q: ti.template(), temp: ti.template()):
    """
    Diffuse discharge field to simulate multiple flow paths.

    Redistributes discharge from each cell to its neighbors based on slope
    gradients, creating a more realistic multiple flow direction pattern
    from the original single flow direction (SFD) routing.

    The method:
    1. Initializes precipitation input for each cell
    2. Computes slope-weighted diffusion to neighbors
    3. Redistributes discharge proportionally to slope gradients

    Args:
            z (ti.template): Combined surface elevation field (topography + water depth)
            Q (ti.template): Discharge field to diffuse
            temp (ti.template): Temporary field for intermediate calculations

    Author: B.G.
    """

    # Initialize precipitation input and handle boundary conditions
    for i in Q:
        temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input

    # Diffuse discharge based on slope gradients
    for i in z:
        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i):
            continue

        # Calculate total slope gradient sum for normalization
        sums = 0.0
        for k in range(4):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            sums += ti.max(0.0, (((z[i]) - (z[j])) / cte.DX) if j != -1 else 0.0)

        # Skip cells with no downslope neighbors
        if sums == 0.0:
            continue

        # Distribute discharge proportionally to slope gradients
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            tS = ti.max(0.0, (((z[i]) - (z[j])) / cte.DX) if j != -1 else 0.0)
            ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

    # Update discharge field with diffused values
    for i in Q:
        Q[i] = temp[i]


@ti.kernel
def graphflood_core_cte_mannings(
    h: ti.template(),
    z: ti.template(),
    dh: ti.template(),
    rec: ti.template(),
    Q: ti.template(),
):
    """
    Core shallow water flow computation using Manning's equation.

    Implements the main hydrodynamic computation for 2D shallow water flow
    using Manning's equation for flow resistance. Updates water depth based
    on discharge input and outflow capacity.

    The method:
    1. Computes local slope from flow receivers
    2. Calculates outflow capacity using Manning's equation
    3. Updates water depth based on discharge balance
    4. Ensures non-negative depth values
    5. Maintains separate water depth field (bed elevation unchanged)

    Based on core methods from Gailleton et al. 2024.

    Args:
            h (ti.template): Flow depth field
            z (ti.template): Combined surface elevation field (topography + water depth)
            dh (ti.template): Depth change field for intermediate calculations
            rec (ti.template): Flow receiver field from flow routing
            Q (ti.template): Discharge field

    Author: B.G.
    """

    # Compute depth changes using Manning's equation
    for i in h:
        # Determine local slope
        tS = cte.EDGESW  # Use edge slope for boundary/sink cells
        if rec[i] != i:  # Interior cells with valid receivers
            tS = ti.max(
                ((z[i] + h[i]) - (z[rec[i]] + h[rec[i]])) / cte.DX, 1e-4
            )  # Slope to receiver (minimum 1e-4)

        # Calculate outflow capacity using Manning's equation
        # Q = (1/n) * A * R^(2/3) * S^(1/2), where R ≈ h for wide channels
        Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(tS)

        # Update depth based on discharge balance (inflow - outflow)
        tdh = (Q[i] - Qo) / (cte.DX**2) * cte.DT_HYDRO  # Volume change per unit area

        # Apply depth change and ensure non-negative depths
        # h[i] += tdh
        if h[i] + tdh < 0:  # Prevent negative depths
            tdh = -h[i]  # Adjust change to reach zero depth
        dh[i] = tdh

    # Apply final water depth changes
    for i in h:
        h[i] += dh[i]  # Apply final depth change


@ti.kernel
def graphflood_diffuse_cte_P_cte_man(z: ti.template(), h:ti.template(), Q: ti.template(), temp: ti.template(), dh: ti.template(), srecs: ti.template(), LM: ti.template(), temporal_filtering:ti.f32):
    """
    NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

    Author: B.G.
    """

    # Initialize precipitation input and handle boundary conditions
    for i in Q:
        temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input
        dh[i] = 0.

    # Diffuse discharge based on slope gradients
    for i in z:
        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i):
            continue

        # Calculate total slope gradient sum for normalization
        sums = 0.0
        msx  = 0.0
        msy  = 0.0
        mz   = (z[i] + h[i])
        for k in ti.static(range(4)):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
            sums += ts
            msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
            msy = ti.max(ts, msy) if k == 0 or k == 3 else msy
            mz = ti.max(mz, z[j]+h[j])


        # Skip cells with no downslope neighbors
        if sums == 0.0:
            # h[i] = mz - z[i] + 2e-3
            continue

        # Distribute discharge proportionally to slope gradients
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            tS = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
            ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

        norms = ti.math.sqrt(msx**2 + msy**2)
        Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
        dh[i] -= cte.DT_HYDRO * Qo/cte.DX**2

    # Update discharge field with diffused values
    for i in Q:
        Q[i] = temp[i] * (1 - temporal_filtering ) + Q[i] * temporal_filtering
        dh[i] += cte.DT_HYDRO * Q[i]/cte.DX**2
        h[i] += dh[i]