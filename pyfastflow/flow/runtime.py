"""
Runtime services for flow workflows built from ``FlowContext`` callables.

Author: B.G (03/2026)
"""

from math import ceil, log2

import taichi as ti

from .. import constants as cte
from .. import pool as ppool
from ..context import require_flat_field, unwrap_field


def sum_at_can_out(flowctx, field) -> float:
    """
    Sum one flat field over outlet nodes and return the host scalar.

    Author: B.G (03/2026)
    """
    out_sum = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=())
    try:
        flowctx.sum_at_can_out(require_flat_field(field, "field"), out_sum.field)
        return float(out_sum.field[None])
    finally:
        out_sum.release()


def accumulate_sfd_with_temps(
    flowctx,
    receivers,
    q,
    donors,
    ndonors,
    donors_alt,
    ndonors_alt,
    q_alt,
    src,
):
    """
    Execute D4/D8 SFD accumulation with caller-provided temporary buffers.

    Author: B.G (03/2026)
    """
    rec_field = require_flat_field(receivers, "receivers")
    q_field = require_flat_field(q, "q")
    donors_field = require_flat_field(donors, "donors")
    ndonors_field = require_flat_field(ndonors, "ndonors")
    donors_alt_field = require_flat_field(donors_alt, "donors_alt")
    ndonors_alt_field = require_flat_field(ndonors_alt, "ndonors_alt")
    q_alt_field = require_flat_field(q_alt, "q_alt")
    src_field = require_flat_field(src, "src")

    ndonors_field.fill(0)
    ndonors_alt_field.fill(0)
    src_field.fill(0)

    flowctx.init_weighted_source(q_field)
    flowctx.receivers_to_donors(rec_field, donors_field, ndonors_field)

    for iteration in range(flowctx.logn + 1):
        flowctx.rake_compress_accum(
            donors_field,
            ndonors_field,
            q_field,
            src_field,
            donors_alt_field,
            ndonors_alt_field,
            q_alt_field,
            iteration,
        )

    flowctx.fuse_accum_buffers(q_field, src_field, q_alt_field, flowctx.logn)


def accumulate_sfd(flowctx, receivers, q):
    """
    Execute SFD accumulation with pooled temporary buffers.

    Author: B.G (03/2026)
    """
    donors = ppool.taipool.get_tpfield(
        dtype=ti.i32, shape=(flowctx.n_flat * flowctx.gridctx.n_neighbours)
    )
    ndonors = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))
    donors_alt = ppool.taipool.get_tpfield(
        dtype=ti.i32, shape=(flowctx.n_flat * flowctx.gridctx.n_neighbours)
    )
    ndonors_alt = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))
    q_alt = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(flowctx.n_flat))
    src = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))

    try:
        accumulate_sfd_with_temps(
            flowctx,
            receivers,
            q,
            donors,
            ndonors,
            donors_alt,
            ndonors_alt,
            q_alt,
            src,
        )
    finally:
        donors.release()
        ndonors.release()
        donors_alt.release()
        ndonors_alt.release()
        q_alt.release()
        src.release()


def accumulate_mfd_with_temps(
    flowctx,
    z,
    q_out,
    routing_weights,
    routing_sum,
    source,
    q_tmp,
    eps,
    *,
    max_iterations=2000,
    tol=1e-6,
    check_interval=20,
):
    """
    Execute MFD power-iteration accumulation with caller buffers.

    Author: B.G (03/2026)
    """
    z_field = require_flat_field(z, "z")
    q_out_field = require_flat_field(q_out, "q_out")
    routing_weights_field = unwrap_field(routing_weights)
    routing_sum_field = require_flat_field(routing_sum, "routing_sum")
    source_field = require_flat_field(source, "source")
    q_tmp_field = require_flat_field(q_tmp, "q_tmp")
    eps_field = unwrap_field(eps)

    flowctx.init_mfd_source(source_field)
    q_out_field.copy_from(source_field)
    q_tmp_field.copy_from(source_field)
    flowctx.compute_mfd_routing_weights(z_field, routing_weights_field, routing_sum_field)

    for iteration in range(int(max_iterations)):
        flowctx.mfd_power_iteration_step(
            source_field,
            q_tmp_field,
            routing_weights_field,
            q_out_field,
        )
        if iteration > 0 and iteration % int(check_interval) == 0:
            flowctx.check_mfd_convergence(q_out_field, q_tmp_field, eps_field)
            if eps_field[None] < tol:
                break
        q_tmp_field.copy_from(q_out_field)


def accumulate_mfd(
    flowctx,
    z,
    q_out,
    *,
    max_iterations=2000,
    tol=1e-6,
    check_interval=20,
):
    """
    Execute MFD accumulation with pooled temporary buffers.

    Author: B.G (03/2026)
    """
    routing_weights = ppool.taipool.get_tpfield(
        dtype=cte.FLOAT_TYPE_TI, shape=(flowctx.n_flat, flowctx.gridctx.n_neighbours)
    )
    routing_sum = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(flowctx.n_flat))
    source = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(flowctx.n_flat))
    q_tmp = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(flowctx.n_flat))
    eps = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=())

    try:
        accumulate_mfd_with_temps(
            flowctx,
            z,
            q_out,
            routing_weights,
            routing_sum,
            source,
            q_tmp,
            eps,
            max_iterations=max_iterations,
            tol=tol,
            check_interval=check_interval,
        )
    finally:
        routing_weights.release()
        routing_sum.release()
        source.release()
        q_tmp.release()
        eps.release()


def fill_topography_inplace_with_temps(flowctx, z, receivers, z_work, receivers_work, receivers_next):
    """
    Fill one flat topography in place with caller-provided buffers.

    Author: B.G (03/2026)
    """
    z_field = require_flat_field(z, "z")
    rec_field = require_flat_field(receivers, "receivers")
    z_work_field = require_flat_field(z_work, "z_work")
    receivers_work_field = require_flat_field(receivers_work, "receivers_work")
    receivers_next_field = require_flat_field(receivers_next, "receivers_next")

    z_work_field.copy_from(z_field)
    receivers_work_field.copy_from(rec_field)
    receivers_next_field.copy_from(rec_field)

    for iteration in range(flowctx.logn):
        flowctx.fill_topography_step(
            z_field,
            z_work_field,
            receivers_work_field,
            receivers_next_field,
            iteration + 1,
        )

    z_field.copy_from(z_work_field)


def fill_topography_inplace(flowctx, z, receivers):
    """
    Fill one flat topography in place with pooled buffers.

    Author: B.G (03/2026)
    """
    z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(flowctx.n_flat))
    receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))
    receivers_next = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))

    try:
        fill_topography_inplace_with_temps(
            flowctx,
            z,
            receivers,
            z_work,
            receivers_work,
            receivers_next,
        )
    finally:
        z_work.release()
        receivers_work.release()
        receivers_next.release()


def fill_h_epsilon_inplace_with_temps(flowctx, z, h, receivers, receivers_work, receivers_next):
    """
    Fill h (surface z+h) in place with caller-provided receivers buffers.

    Author: B.G (07/2026)
    """
    z_field = require_flat_field(z, "z")
    h_field = require_flat_field(h, "h")
    rec_field = require_flat_field(receivers, "receivers")
    receivers_work_field = require_flat_field(receivers_work, "receivers_work")
    receivers_next_field = require_flat_field(receivers_next, "receivers_next")

    receivers_work_field.copy_from(rec_field)
    receivers_next_field.copy_from(rec_field)

    for iteration in range(flowctx.logn):
        flowctx.fill_h_epsilon(
            z_field,
            h_field,
            receivers_work_field,
            receivers_next_field,
            iteration + 1,
        )


def fill_h_epsilon_inplace(flowctx, z, h, receivers):
    """
    Fill h (surface z+h) in place with pooled receivers buffers.

    Author: B.G (07/2026)
    """
    receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))
    receivers_next = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))

    try:
        fill_h_epsilon_inplace_with_temps(
            flowctx,
            z,
            h,
            receivers,
            receivers_work,
            receivers_next,
        )
    finally:
        receivers_work.release()
        receivers_next.release()


def fill_topography_delta_with_temps(
    flowctx,
    z,
    surplus,
    receivers,
    z_work,
    receivers_work,
    receivers_next,
):
    """
    Fill one topography and accumulate the fill surplus with caller buffers.

    Author: B.G (03/2026)
    """
    z_field = require_flat_field(z, "z")
    rec_field = require_flat_field(receivers, "receivers")
    surplus_field = require_flat_field(surplus, "surplus")
    z_work_field = require_flat_field(z_work, "z_work")
    receivers_work_field = require_flat_field(receivers_work, "receivers_work")
    receivers_next_field = require_flat_field(receivers_next, "receivers_next")

    z_work_field.copy_from(z_field)
    receivers_work_field.copy_from(rec_field)
    receivers_next_field.copy_from(rec_field)

    for iteration in range(flowctx.logn):
        flowctx.fill_topography_step(
            z_field,
            z_work_field,
            receivers_work_field,
            receivers_next_field,
            iteration + 1,
        )

    flowctx.apply_fill_delta(z_field, surplus_field, z_work_field)


def fill_topography_delta(flowctx, z, surplus, receivers):
    """
    Fill one topography and accumulate surplus with pooled buffers.

    Author: B.G (03/2026)
    """
    z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(flowctx.n_flat))
    receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))
    receivers_next = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))

    try:
        fill_topography_delta_with_temps(
            flowctx,
            z,
            surplus,
            receivers,
            z_work,
            receivers_work,
            receivers_next,
        )
    finally:
        z_work.release()
        receivers_work.release()
        receivers_next.release()


def reroute_flow_with_temps(
    flowctx,
    z,
    receivers,
    bid,
    receivers_work,
    receivers_jump,
    z_prime,
    is_border,
    outlet,
    basin_saddle,
    basin_saddlenode,
    tag,
    tag_alt,
    rerouted,
    *,
    carve=True,
):
    """
    Execute lake rerouting with caller-provided buffers.

    Author: B.G (03/2026)
    """
    z_field = require_flat_field(z, "z")
    rec_field = require_flat_field(receivers, "receivers")
    bid_field = require_flat_field(bid, "bid")
    rec_work_field = require_flat_field(receivers_work, "receivers_work")
    rec_jump_field = require_flat_field(receivers_jump, "receivers_jump")
    z_prime_field = require_flat_field(z_prime, "z_prime")
    is_border_field = require_flat_field(is_border, "is_border")
    outlet_field = require_flat_field(outlet, "outlet")
    basin_saddle_field = require_flat_field(basin_saddle, "basin_saddle")
    basin_saddlenode_field = require_flat_field(basin_saddlenode, "basin_saddlenode")
    tag_field = require_flat_field(tag, "tag")
    tag_alt_field = require_flat_field(tag_alt, "tag_alt")
    rerouted_field = require_flat_field(rerouted, "rerouted")

    rec_work_field.copy_from(rec_field)
    rerouted_field.fill(False)

    ndep = flowctx.depression_counter(rec_field)
    if ndep == 0:
        return

    ndep_iters = ceil(log2(max(1, int(ndep)))) + 1

    for _ in range(ndep_iters):
        ndep_bis = flowctx.depression_counter(rec_work_field)

        flowctx.basin_id_init(bid_field)
        rec_jump_field.copy_from(rec_work_field)

        for _ in range(flowctx.logn + 1):
            flowctx.propagate_basin_iter(rec_jump_field)
        flowctx.propagate_basin_final(bid_field, rec_jump_field)

        if ndep_bis == 0:
            break

        flowctx.saddlesort(
            bid_field,
            is_border_field,
            z_prime_field,
            basin_saddle_field,
            basin_saddlenode_field,
            outlet_field,
            z_field,
        )

        if carve:
            flowctx.init_reroute_carve(tag_field, tag_alt_field, basin_saddlenode_field)
            rec_field.copy_from(rec_work_field)
            rec_jump_field.copy_from(rec_work_field)

            for _ in range(flowctx.logn + 1):
                flowctx.iteration_reroute_carve(
                    tag_field,
                    tag_alt_field,
                    rec_field,
                    rec_work_field,
                    bid_field,
                )

            flowctx.finalise_reroute_carve(
                rec_field,
                rec_jump_field,
                tag_field,
                basin_saddlenode_field,
                outlet_field,
                rerouted_field,
            )
            rec_work_field.copy_from(rec_field)
        else:
            flowctx.reroute_jump(rec_work_field, outlet_field, rerouted_field)

    rec_field.copy_from(rec_work_field)


def reroute_flow(flowctx, z, receivers, *, carve=True):
    """
    Execute lake rerouting with pooled temporary buffers.

    Author: B.G (03/2026)
    """
    bid = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))
    receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))
    receivers_jump = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))
    z_prime = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(flowctx.n_flat))
    is_border = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(flowctx.n_flat))
    outlet = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(flowctx.n_flat))
    basin_saddle = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(flowctx.n_flat))
    basin_saddlenode = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(flowctx.n_flat))
    tag = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(flowctx.n_flat))
    tag_alt = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(flowctx.n_flat))
    rerouted = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(flowctx.n_flat))

    try:
        reroute_flow_with_temps(
            flowctx,
            z,
            receivers,
            bid,
            receivers_work,
            receivers_jump,
            z_prime,
            is_border,
            outlet,
            basin_saddle,
            basin_saddlenode,
            tag,
            tag_alt,
            rerouted,
            carve=carve,
        )
    finally:
        bid.release()
        receivers_work.release()
        receivers_jump.release()
        z_prime.release()
        is_border.release()
        outlet.release()
        basin_saddle.release()
        basin_saddlenode.release()
        tag.release()
        tag_alt.release()

    return rerouted


def reroute_carve_optimized(
    flowctx,
    z,
    receivers,
    receivers_jump,
    bid,
    is_border,
    z_prime,
    basin_saddle,
    basin_saddlenode,
    outlet,
):
    """
    Optimized full depression handling: single-launch basin labeling and
    single-launch serial carve per pass, saddlesort unchanged. Modifies
    ``receivers`` in place. All arguments after flowctx are raw ti fields,
    caller-allocated. Returns the number of unresolved depressions.

    Author: B.G (07/2026)
    """
    ndep = flowctx.depression_counter(receivers)
    if ndep == 0:
        return 0
    for _ in range(ceil(log2(max(2, int(ndep)))) + 2):
        flowctx.label_basins_walk(receivers, receivers_jump, bid)
        flowctx.saddlesort(
            bid, is_border, z_prime, basin_saddle, basin_saddlenode, outlet, z
        )
        flowctx.carve_basins_serial(receivers, basin_saddlenode, outlet)
        ndep = flowctx.depression_counter(receivers)
        if ndep == 0:
            break
    return int(ndep)


def accumulate_sfd_atomic(flowctx, receivers, q):
    """
    SFD accumulation by direct atomic descent, result in ``q``. Raw ti
    fields, no temporaries. Receiver graph must be acyclic.

    Author: B.G (07/2026)
    """
    flowctx.accum_downstream_atomic(receivers, q)


def accumulate_sfd_pointer_jump_push(
    flowctx, receivers, receivers_work, receivers_work2, q, q_work
):
    """
    Push pointer-jumping accumulation with retirement, logn+1 full
    ping-pong rounds (rec and q buffers), result in ``q``. Exact. Raw ti
    fields, caller-allocated; ``receivers`` itself is never written.

    Author: B.G (07/2026)
    """
    flowctx.init_weighted_source(q)
    rec_a, rec_b = receivers, receivers_work
    q_a, q_b = q, q_work
    for _ in range(flowctx.logn + 1):
        flowctx.accum_pointer_jump_push_step(rec_a, rec_b, q_a, q_b)
        rec_a, rec_b = rec_b, (receivers_work2 if rec_b is receivers_work else receivers_work)
        q_a, q_b = q_b, q_a
    if q_a is not q:
        q.copy_from(q_a)
