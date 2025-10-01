"""
Upstream affine propagation via pointer jumping.

Propagates values upstream along the receiver tree using affine relations:
    x[i] = a[i] * x[rcv[i]] + b[i]

Supports constant a and spatially varying a. Uses pointer jumping to compose
affine transforms in O(log N) iterations on GPU.

Author: B.G. - G.C.
"""

import taichi as ti


@ti.kernel
def init_upstream_AB_const(
    rcv: ti.template(), A: ti.template(), B: ti.template(), P: ti.template(), a: ti.f32, b: ti.template()
):
    """
    Initialize affine parameters and parent pointers for constant a.

    For non-root nodes (rcv[i] != i):
        A[i] = a, B[i] = b[i], P[i] = rcv[i]
    For roots (rcv[i] == i):
        A[i] = 1, B[i] = 0, P[i] = i  (identity), so the boundary value supplied via
        ``x_root`` (or the in-place snapshot) is used unchanged during finalization.
    """
    for i in rcv:
        if rcv[i] == i:
            A[i] = 1.0
            B[i] = 0.0
            P[i] = i
        else:
            A[i] = a
            B[i] = b[i]
            P[i] = rcv[i]


@ti.kernel
def init_upstream_AB_var(
    rcv: ti.template(), A: ti.template(), B: ti.template(), P: ti.template(), a: ti.template(), b: ti.template()
):
    """
    Initialize affine parameters and parent pointers for spatially varying a.

    For non-root nodes: A[i] = a[i], B[i] = b[i], P[i] = rcv[i]
    For roots (self-receivers): A[i] = 1, B[i] = 0, P[i] = i so that the boundary value
    captured outside the kernel is injected verbatim in the last step.
    """
    for i in rcv:
        if rcv[i] == i:
            A[i] = 1.0
            B[i] = 0.0
            P[i] = i
        else:
            A[i] = a[i]
            B[i] = b[i]
            P[i] = rcv[i]


@ti.kernel
def pointer_jump_affine(
    P_in: ti.template(), A_in: ti.template(), B_in: ti.template(),
    P_out: ti.template(), A_out: ti.template(), B_out: ti.template(),
):
    """
    One pointer-jumping step composing transforms by skipping one ancestor level:
        Let j = P_in[i]
        A_out[i] = A_in[i] * A_in[j]
        B_out[i] = B_in[i] + A_in[i] * B_in[j]
        P_out[i] = P_in[j]

    Roots (j==i) remain stable since A=1, B=0 at roots.
    """
    for i in P_in:
        j = P_in[i]
        if j == i:
            A_out[i] = A_in[i]
            B_out[i] = B_in[i]
            P_out[i] = j
        else:
            A_out[i] = A_in[i] * A_in[j]
            B_out[i] = B_in[i] + A_in[i] * B_in[j]
            P_out[i] = P_in[j]


@ti.kernel
def finalize_upstream_x(
    P: ti.template(), A: ti.template(), B: ti.template(),
    x_root: ti.template(), x_out: ti.template()
):
    """Compute final ``x`` using the composed affine transforms.

    Args:
        P: Parent pointers (each entry points to the root after pointer jumping).
        A: Cumulative coefficients along the collapsed path.
        B: Cumulative offsets along the collapsed path.
        x_root: Field supplying boundary values at roots (read-only).
        x_out: Output field receiving the solved ``x`` values.
    """
    for i in P:
        x_out[i] = A[i] * x_root[P[i]] + B[i]
