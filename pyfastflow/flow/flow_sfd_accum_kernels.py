"""
Generic single-flow-direction accumulation kernels for ``FlowContext``.

These kernels implement the rake-and-compress downstream accumulation path with
the donor stencil size specialized from the bound grid topology.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None
flowctx = None
get_weight = None
get_src = None
update_src = None


@ti.kernel
def init_weighted_source_kernel(q: ti.template()):
    """
    Initialize the source field from the configured weight getter.

    Author: B.G (02/2026)
    """
    for i in q:
        q[i] = get_weight(i)


@ti.kernel
def receivers_to_donors_kernel(
    receivers: ti.template(), donors: ti.template(), ndonors: ti.template()
):
    """
    Build the donor adjacency from the receiver map.

    Author: B.G (02/2026)
    """
    for tid in receivers:
        rcv = receivers[tid]
        if rcv != tid:
            old_val = ti.atomic_add(ndonors[rcv], 1)
            donors[rcv * gridctx.n_neighbours + old_val] = tid


@ti.kernel
def rake_compress_accum_kernel(
    donors: ti.template(),
    ndonors: ti.template(),
    q: ti.template(),
    src: ti.template(),
    donors_alt: ti.template(),
    ndonors_alt: ti.template(),
    q_alt: ti.template(),
    iteration: ti.i32,
):
    """
    Execute one rake-and-compress sweep on the donor forest.

    Author: B.G (02/2026)
    """
    for tid in q:
        flip = get_src(src, tid, iteration)
        n_neighbours = ti.static(gridctx.n_neighbours)

        worked = False
        todo = ndonors[tid] if not flip else ndonors_alt[tid]
        base = tid * gridctx.n_neighbours
        donors_local = ti.Vector([-1, -1, -1, -1, -1, -1, -1, -1])
        q_added = ti.cast(0.0, cte.FLOAT_TYPE_TI)

        i = 0
        while i < todo and i < n_neighbours:
            if donors_local[i] == -1:
                donors_local[i] = donors[base + i] if not flip else donors_alt[base + i]
            did = donors_local[i]

            flip_donor = get_src(src, did, iteration)
            ndnr_val = ndonors[did] if not flip_donor else ndonors_alt[did]

            if ndnr_val <= 1:
                if not worked:
                    q_added = q[tid] if not flip else q_alt[tid]
                worked = True

                q_val = q[did] if not flip_donor else q_alt[did]
                q_added += q_val

                if ndnr_val == 0:
                    todo -= 1
                    if todo > i:
                        donors_local[i] = donors[base + todo] if not flip else donors_alt[base + todo]
                    i -= 1
                else:
                    donor_base = did * gridctx.n_neighbours
                    donors_local[i] = donors[donor_base] if not flip_donor else donors_alt[donor_base]
            i += 1

        if worked:
            if flip:
                ndonors[tid] = todo
                q[tid] = q_added
                for j in ti.static(range(n_neighbours)):
                    if j < todo:
                        donors[base + j] = donors_local[j]
            else:
                ndonors_alt[tid] = todo
                q_alt[tid] = q_added
                for j in ti.static(range(n_neighbours)):
                    if j < todo:
                        donors_alt[base + j] = donors_local[j]
            update_src(src, tid, iteration, flip)


@ti.kernel
def fuse_accum_buffers_kernel(
    q: ti.template(), src: ti.template(), q_alt: ti.template(), iteration: ti.i32
):
    """
    Consolidate ping-pong accumulation buffers back into ``q``.

    Author: B.G (02/2026)
    """
    for tid in q:
        if get_src(src, tid, iteration):
            q[tid] = q_alt[tid]
