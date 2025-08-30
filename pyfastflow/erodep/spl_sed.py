"""
Stream Power Law (SPL) erosion model implementation for landscape evolution.

This module implements the Stream Power Law erosion model using GPU-accelerated
Taichi kernels. The SPL model describes bedrock erosion rate as a function of
drainage area (discharge proxy) and local slope, following the equation:

E = K * A^m * S^n

where:
- E: erosion rate
- K: erodibility coefficient
- A: drainage area (or discharge)
- S: local slope
- m, n: empirical exponents

The implementation uses an implicit finite difference scheme for numerical
stability, allowing larger time steps while maintaining accuracy.

Key Features:
- Implicit SPL erosion with stability guarantees
- Block uplift implementation
- Integration with FastFlow routing system
- GPU-accelerated computation using Taichi
- Supports landscape evolution modeling

Author: B.G.
"""

import math

import taichi as ti

import pyfastflow as pf


@ti.kernel
def init_erode_SPL(
    z: ti.template(),
    z_: ti.template(),
    z__: ti.template(),
    alpha_: ti.template(),
    alpha__: ti.template(),
    Q: ti.template(),
):
    """
    Initialize fields for implicit Stream Power Law erosion computation.

    Sets up the implicit finite difference scheme for SPL erosion by computing
    initial values for the erosion coefficient (alpha) and adjusted elevations.
    The implicit scheme allows stable computation with large time steps.

    The erosion follows: dz/dt = -K * Q^m * |∇z|^n
    where K = Kr * Q^m * dt/dx (dimensionless erosion coefficient)

    Args:
            z (ti.template): Current topographic elevation field
            z_ (ti.template): Working elevation field for iterations
            z__ (ti.template): Secondary working elevation field
            alpha_ (ti.template): Primary erosion coefficient field
            alpha__ (ti.template): Secondary erosion coefficient field
            Q (ti.template): Discharge field (proxy for drainage area)

    Note:
            The implicit formulation: z_new = (z_old + K*z_downstream)/(1+K)
            results in: alpha = K/(1+K) and z_adjusted = z/(1+K)

    Author: B.G.
    """

    for i in z:
        if pf.flow.neighbourer_flat.can_leave_domain(i):
            continue
        # Compute dimensionless erosion coefficient
        # K = erodibility * discharge^m * timestep / grid_spacing
        K = (
            pf.constants.KR
            * Q[i] ** pf.constants.MEXP
            * pf.constants.DT_SPL
            / pf.constants.DX
        )

        # Implicit scheme coefficients
        alpha_[i] = K / (1 + K)  # Erosion weight factor
        z_[i] = z[i] / (1 + K)  # Adjusted elevation (working field)
        z__[i] = z[i] / (1 + K)  # Adjusted elevation (secondary field)


@ti.kernel
def iteration_erode_SPL(
    z_: ti.template(),
    z__: ti.template(),
    rec_: ti.template(),
    rec__: ti.template(),
    alpha_: ti.template(),
    alpha__: ti.template(),
):
    """
    Perform one iteration of implicit SPL erosion using receiver chains.

    This kernel implements the core iteration step of the implicit SPL solver.
    It follows drainage paths (receiver chains) to propagate erosion effects
    upstream, ensuring mass conservation and numerical stability.

    The algorithm performs parallel graph traversal along flow paths,
    updating elevations based on downstream erosion potential. Multiple
    iterations are needed to fully propagate effects through the drainage network.

    Args:
            z_ (ti.template): Working elevation field being updated
            z__ (ti.template): Secondary elevation field for stable updates
            rec_ (ti.template): Primary receiver field (flow routing)
            rec__ (ti.template): Secondary receiver field for stable updates
            alpha_ (ti.template): Primary erosion coefficient field
            alpha__ (ti.template): Secondary erosion coefficient field

    Note:
            This implements the implicit update: z_new[i] += alpha[i] * z[receiver[i]]
            where alpha represents the erosion coupling strength between connected nodes.

    Author: B.G.
    """

    # First pass: Update elevations and propagate coefficients downstream
    for i in rec_:
        # Add contribution from downstream node (receiver)
        z_[i] += alpha_[i] * z__[rec_[i]]

        # Propagate erosion coefficient through receiver chain
        # This compounds erosion effects along flow paths
        alpha__[i] = alpha_[i] * alpha_[rec_[i]]

        # Advance receiver chain by one step (follow flow path)
        rec__[i] = rec_[rec_[i]]

    # Second pass: Update working fields for next iteration
    for i in rec_:
        if pf.flow.neighbourer_flat.can_leave_domain(i):
            continue
        z__[i] = z_[i]  # Copy updated elevations
        rec_[i] = rec__[i]  # Advance receiver chains
        alpha_[i] = alpha__[i]  # Update erosion coefficients


def SPL(router, alpha_, alpha__):
    """
    Execute complete Stream Power Law erosion for one time step.

    This function orchestrates the full SPL erosion computation using an implicit
    finite difference scheme. It initializes erosion coefficients, then performs
    iterative updates to propagate erosion effects throughout the drainage network.

    The number of iterations is set to log2(N) where N is the total number of nodes,
    which ensures that erosion effects can propagate across the entire domain through
    the longest possible drainage paths.

    Args:
            router: FastFlow router object containing topography and flow routing
            alpha_ (ti.field): Primary erosion coefficient field (ti.f32, shape=(N,))
            alpha__ (ti.field): Secondary erosion coefficient field (ti.f32, shape=(N,))

    Note:
            The router object is modified in-place, with final eroded topography
            stored in router.grid.z. The function requires pre-computed discharge (router.Q)
            and flow routing (router.receivers).

    Example:
            # Setup erosion coefficient fields
            alpha_ = ti.field(ti.f32, shape=(nx*ny,))
            alpha__ = ti.field(ti.f32, shape=(nx*ny,))

            # Run one erosion time step
            SPL(router, alpha_, alpha__)

    Author: B.G.
    """

    # Get temporary fields from pool for SPL computation
    z_ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(router.nx * router.ny))
    z__ = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(router.nx * router.ny))
    receivers_ = pf.pool.taipool.get_tpfield(
        dtype=ti.i32, shape=(router.nx * router.ny)
    )
    receivers__ = pf.pool.taipool.get_tpfield(
        dtype=ti.i32, shape=(router.nx * router.ny)
    )

    # Calculate number of iterations needed for domain-wide propagation
    # log2(N) ensures erosion can propagate through longest drainage paths
    log2 = math.ceil(math.log2(router.nx * router.ny))

    # Initialize erosion coefficients and adjusted elevations
    init_erode_SPL(
        router.grid.z.field, z_.field, z__.field, alpha_, alpha__, router.Q.field
    )

    # Setup receiver chains for efficient parallel traversal
    receivers_.field.copy_from(router.receivers.field)
    receivers__.field.copy_from(router.receivers.field)

    # Perform iterative erosion computation
    # Each iteration propagates erosion effects one step further upstream
    for _ in range(log2):
        iteration_erode_SPL(
            z_.field, z__.field, receivers_.field, receivers__.field, alpha_, alpha__
        )

    # Copy final eroded topography back to main elevation field
    router.grid.z.field.copy_from(z_.field)

    # Release temporary fields back to pool
    z_.release()
    z__.release()
    receivers_.release()
    receivers__.release()
