"""
Steepest descent receiver computation for flow routing.

Implements the steepest descent algorithm to compute flow receivers for each
grid node. Supports both deterministic and stochastic receiver selection
for hydrological flow routing applications.

Author: B.G.
"""

import taichi as ti

import pyfastflow.grid.neighbourer_flat as nei

from .. import constants as cte


@ti.kernel
def compute_receivers(z: ti.template(), receivers: ti.template()):
    """
    Compute steepest descent receivers for each node in the grid.

    Args:
        z: Elevation field
        receivers: Output array for receiver indices
        gradient: Output array for steepest gradients

    Note:
        Uses the neighbourer_flat system for boundary-aware neighbor access.
        Supports stochastic receiver selection when RAND_RCV=1.

    Author: B.G.
    """
    for i in z:
        # Initialize with self-receiver (pit condition)
        r: ti.i32 = i  # Current node receives to itself by default
        sr: ti.f32 = 0.0  # Steepest descent gradient found so far

        # Check all 4 cardinal neighbors
        for k in ti.static(range(4)):
            # Get neighbor using boundary-aware neighbor system
            tr: ti.i32 = nei.neighbour(i, k)
            valid: ti.u1 = tr != -1  # Check if neighbor is valid

            # Compute gradient to this neighbor
            tsr = (z[i] - z[tr]) / cte.DX if valid else -1.0

            # Update steepest receiver if this gradient is steeper
            valid = valid and tsr > sr
            sr = tsr if valid else sr
            r = tr if valid else r

        # Store results
        receivers[i] = r  # Receiver node index


@ti.kernel
def compute_receivers_stochastic(z: ti.template(), receivers: ti.template()):
    """
    Compute steepest descent receivers for each node in the grid.

    Args:
        z: Elevation field
        receivers: Output array for receiver indices
        gradient: Output array for steepest gradients

    Note:
        Uses the neighbourer_flat system for boundary-aware neighbor access.
        Supports stochastic receiver selection when RAND_RCV=1.

    Author: B.G.
    """
    for i in z:
        # Initialize with self-receiver (pit condition)
        r: ti.i32 = i  # Current node receives to itself by default
        sr: ti.f32 = 0.0  # Steepest descent gradient found so far

        # Check all 4 cardinal neighbors
        for k in ti.static(range(4)):
            # Get neighbor using boundary-aware neighbor system
            tr: ti.i32 = nei.neighbour(i, k)
            valid: ti.u1 = tr != -1  # Check if neighbor is valid

            # Compute gradient to this neighbor
            tsr = (z[i] - z[tr]) / cte.DX if valid else -1.0

            # Stochastic slope
            tsr = ti.random() * tsr**0.5

            # Update steepest receiver if this gradient is steeper
            valid = valid and tsr > sr
            sr = tsr if valid else sr
            r = tr if valid else r

        # Store results
        receivers[i] = r  # Receiver node index
