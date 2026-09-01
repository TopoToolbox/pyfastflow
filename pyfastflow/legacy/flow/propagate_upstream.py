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
            continue
            # A_out[i] = A_in[i]
            # B_out[i] = B_in[i]
            # P_out[i] = j
        else:
            A_out[i] = A_in[i] * A_in[j]
            B_out[i] = B_in[i] + A_in[i] * B_in[j]
            P_out[i] = P_in[j]
