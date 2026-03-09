"""
Generic lake rerouting kernels for ``FlowContext``.

This module ports the operational core of ``lakeflow.py`` to the new context
style while keeping the logic close to the reference implementation and
specializing the neighbour stencil from the bound grid topology.

Author: B.G (02/2026)
"""

import taichi as ti

from .f32_i32_struct import pack_float_index, unpack_float_index


gridctx = None
flowctx = None


@ti.func
def _invalid_pack() -> ti.i64:
    """
    Return the sentinel packed value used for invalid basin links.

    Author: B.G (02/2026)
    """
    return pack_float_index(1e8, 42)


@ti.kernel
def depression_counter_kernel(rec: ti.template()) -> ti.i32:
    """
    Count non-draining pit cells in the current receiver map.

    Author: B.G (02/2026)
    """
    ndep = 0
    for i in rec:
        if rec[i] == i and not gridctx.tfunc.can_out_flat(i):
            ti.atomic_add(ndep, 1)
    return ndep


@ti.kernel
def basin_id_init_kernel(bid: ti.template()):
    """
    Initialize basin IDs before pointer-jumping propagation.

    Author: B.G (02/2026)
    """
    for i in bid:
        bid[i] = 0 if gridctx.tfunc.can_out_flat(i) else (i + 1)


@ti.kernel
def propagate_basin_iter_kernel(rec_work: ti.template()):
    """
    Pointer-jump the temporary receiver forest by one step.

    Author: B.G (02/2026)
    """
    for i in rec_work:
        if rec_work[i] != rec_work[rec_work[i]]:
            rec_work[i] = rec_work[rec_work[i]]


@ti.kernel
def propagate_basin_final_kernel(bid: ti.template(), rec_work: ti.template()):
    """
    Finalize basin IDs from the compressed receiver forest.

    Author: B.G (02/2026)
    """
    for i in bid:
        bid[i] = bid[rec_work[i]]


@ti.kernel
def saddlesort_kernel(
    bid: ti.template(),
    is_border: ti.template(),
    z_prime: ti.template(),
    basin_saddle: ti.template(),
    basin_saddlenode: ti.template(),
    outlet: ti.template(),
    z: ti.template(),
):
    """
    Identify basin borders, saddles, and outlets.

    Author: B.G (02/2026)
    """
    invalid = _invalid_pack()
    n_neighbours = ti.static(gridctx.n_neighbours)

    for i in z:
        if gridctx.tfunc.can_out_flat(i):
            z_prime[i] = z[i]
            continue

        is_border[i] = False
        z_prime[i] = 1e9
        zn = 1e9

        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and bid[j] != bid[i]:
                is_border[i] = True
                zn = ti.min(zn, z[j])

        if is_border[i]:
            z_prime[i] = ti.max(z[i], zn)

    for i in bid:
        basin_saddle[i] = invalid
        outlet[i] = invalid
        basin_saddlenode[i] = -1

    for i in bid:
        if not is_border[i]:
            continue

        tbid = bid[i]
        res = invalid

        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and bid[j] != tbid:
                candidate = pack_float_index(z_prime[i], bid[j])
                res = ti.min(res, candidate)

        if res != invalid:
            ti.atomic_min(basin_saddle[tbid], res)

    for i in bid:
        if not is_border[i] or bid[i] == 0:
            continue

        target_z, target_b = unpack_float_index(basin_saddle[bid[i]])
        is_here = False

        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(i, k)
            if j != -1 and bid[j] == target_b and z_prime[i] == target_z:
                is_here = True

        if is_here:
            basin_saddlenode[bid[i]] = i

    for i in bid:
        if i == 0 or basin_saddle[i] == invalid:
            continue

        tbid = i
        node = basin_saddlenode[tbid]
        tz = 1e9
        rec_out = -1

        for k in ti.static(range(n_neighbours)):
            j = gridctx.tfunc.neighbour_flat(node, k)
            if j != -1 and bid[j] != tbid and tz > z[j]:
                tz = z[j]
                rec_out = j

        if rec_out > -1:
            candidate = pack_float_index(tz, rec_out)
            ti.atomic_min(outlet[tbid], candidate)

    for i in bid:
        bid_d = i
        if bid_d == 0 or outlet[bid_d] == invalid:
            continue

        _, rec_out = unpack_float_index(outlet[bid_d])
        bid_d_prime = bid[rec_out]

        if bid_d_prime == 0:
            continue

        _, rec_out_prime = unpack_float_index(outlet[bid_d_prime])
        bid_d_prime_prime = bid[rec_out_prime]

        if bid_d_prime_prime == bid_d:
            if bid_d_prime < bid_d:
                outlet[bid_d] = invalid
                basin_saddle[bid_d] = invalid
                basin_saddlenode[bid_d] = -1


@ti.kernel
def reroute_jump_kernel(
    rec_work: ti.template(), outlet: ti.template(), rerouted: ti.template()
):
    """
    Reroute directly from each basin pit to its outlet.

    Author: B.G (02/2026)
    """
    invalid = _invalid_pack()

    for i in rerouted:
        rerouted[i] = False

    for i in rec_work:
        if outlet[i] != invalid:
            _, rrec = unpack_float_index(outlet[i])
            rec_work[i - 1] = rrec
            rerouted[i - 1] = True


@ti.kernel
def init_reroute_carve_kernel(
    tag: ti.template(), tag_alt: ti.template(), saddlenode: ti.template()
):
    """
    Initialize the tag arrays used by the carving path.

    Author: B.G (02/2026)
    """
    for i in tag:
        tag[i] = False

    for i in tag:
        if saddlenode[i] != -1:
            tag[saddlenode[i]] = True

    for i in tag:
        tag_alt[i] = tag[i]


@ti.kernel
def iteration_reroute_carve_kernel(
    tag: ti.template(),
    tag_alt: ti.template(),
    rec: ti.template(),
    rec_work: ti.template(),
    bid: ti.template(),
):
    """
    Execute one pointer-jumping carve iteration.

    Author: B.G (02/2026)
    """
    for i in tag:
        if bid[i] == 0:
            continue
        if tag[i] and rec[i] != i:
            tag_alt[rec[i]] = True
        rec_work[i] = rec[i]

    for i in tag:
        if bid[i] == 0:
            continue
        if rec_work[i] != i:
            rec[i] = rec_work[rec_work[i]]
        tag[i] = tag_alt[i]


@ti.kernel
def finalise_reroute_carve_kernel(
    rec: ti.template(),
    rec_work: ti.template(),
    tag: ti.template(),
    saddlenode: ti.template(),
    outlet: ti.template(),
    rerouted: ti.template(),
):
    """
    Finalize the carving reroute by creating reverse links and outlet jumps.

    Author: B.G (02/2026)
    """
    invalid = _invalid_pack()

    for i in rec:
        rec[i] = rec_work[i]

    for i in rec:
        if tag[rec_work[i]] and tag[i] and i != rec_work[i]:
            rec[rec_work[i]] = i
            rerouted[rec_work[i]] = True

    for i in rec:
        if outlet[i] != invalid:
            _, node = unpack_float_index(outlet[i])
            rec[saddlenode[i]] = node
            rerouted[saddlenode[i]] = True
