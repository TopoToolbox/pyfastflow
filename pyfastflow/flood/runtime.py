"""
Runtime services for flood workflows built from ``FloodContext``.

Author: B.G (03/2026)
"""

import numpy as np

from .. import constants as cte
from .. import pool as ppool
from ..context import require_flat_field
from ..flow import runtime as flow_runtime


def graphflood_localpass_n(flowctx, z, h, receivers, q, q_next, h_next, *, n_iterations):
    """
    Execute repeated compiled GraphFlood local passes with caller buffers.

    Author: B.G (03/2026)
    """
    q_field = require_flat_field(q, "q")
    q_next_field = require_flat_field(q_next, "q_next")
    h_field = require_flat_field(h, "h")
    h_next_field = require_flat_field(h_next, "h_next")
    z_field = require_flat_field(z, "z")
    rec_field = require_flat_field(receivers, "receivers")

    for _ in range(int(n_iterations)):
        flowctx.localpass(z_field, h_field, rec_field, q_field, q_next_field, h_next_field)
        q_field.copy_from(q_next_field)
        h_field.copy_from(h_next_field)


def graphflood_localpass_n_pooled(flowctx, z, h, receivers, q, *, n_iterations):
    """
    Execute repeated GraphFlood local passes with pooled buffers.

    Author: B.G (03/2026)
    """
    q_next = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (flowctx.n_flat))
    h_next = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (flowctx.n_flat))
    try:
        graphflood_localpass_n(
            flowctx,
            z,
            h,
            receivers,
            q,
            q_next,
            h_next,
            n_iterations=n_iterations,
        )
    finally:
        q_next.release()
        h_next.release()


def graphflood_propagate_with_fields(
    floodctx,
    z,
    h,
    receivers,
    q_out,
    source_q,
    *,
    mode="sfd",
    reroute=False,
    fill=False,
    carve=True,
    sfd_buffers=None,
    mfd_buffers=None,
    reroute_buffers=None,
    fill_buffers=None,
):
    """
    Propagate source-driven flow using the compiled flood and flow APIs.

    Author: B.G (03/2026)
    """
    z_field = require_flat_field(z, "z")
    h_field = require_flat_field(h, "h")
    rec_field = require_flat_field(receivers, "receivers")
    q_out_field = require_flat_field(q_out, "q_out")
    source_q_field = require_flat_field(source_q, "source_q")

    mode_l = str(mode).lower()
    if mode_l not in {"sfd", "mfd"}:
        raise ValueError("mode must be 'sfd' or 'mfd'")

    source_q_field.fill(0.0)
    floodctx.add_source_to_Q(source_q_field)
    floodctx._accum_flowctx.set_weight(source_q_field)

    if mode_l == "mfd":
        fill = True

    surface_owned = False
    if mfd_buffers is not None and "surface" in mfd_buffers:
        surface_field = require_flat_field(mfd_buffers["surface"], "surface")
    else:
        surface = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (floodctx.n_flat))
        surface_field = surface.field
        surface_owned = True

    try:
        floodctx.make_surface(z_field, h_field, surface_field)
        floodctx.flowctx.compute_receivers(surface_field, rec_field)

        if fill:
            if fill_buffers is None:
                flow_runtime.fill_topography_delta(floodctx.flowctx, surface_field, h_field, rec_field)
            else:
                require_flat_field(fill_buffers["z_ref"], "z_ref").copy_from(surface_field)
                flow_runtime.fill_topography_delta_with_temps(
                    floodctx.flowctx,
                    fill_buffers["z_ref"],
                    h_field,
                    rec_field,
                    fill_buffers["z_work"],
                    fill_buffers["receivers_work"],
                    fill_buffers["receivers_next"],
                )
            floodctx.make_surface(z_field, h_field, surface_field)
            floodctx.flowctx.compute_receivers(surface_field, rec_field)

        if reroute:
            if reroute_buffers is None:
                rerouted = flow_runtime.reroute_flow(
                    floodctx.flowctx,
                    surface_field,
                    rec_field,
                    carve=carve,
                )
                rerouted.release()
            else:
                flow_runtime.reroute_flow_with_temps(
                    floodctx.flowctx,
                    surface_field,
                    rec_field,
                    reroute_buffers["bid"],
                    reroute_buffers["receivers_work"],
                    reroute_buffers["receivers_jump"],
                    reroute_buffers["z_prime"],
                    reroute_buffers["is_border"],
                    reroute_buffers["outlet"],
                    reroute_buffers["basin_saddle"],
                    reroute_buffers["basin_saddlenode"],
                    reroute_buffers["tag"],
                    reroute_buffers["tag_alt"],
                    reroute_buffers["change"],
                    reroute_buffers["rerouted"],
                    carve=carve,
                )

        if mode_l == "sfd":
            if sfd_buffers is None:
                flow_runtime.accumulate_sfd(floodctx._accum_flowctx, rec_field, q_out_field)
            else:
                flow_runtime.accumulate_sfd_with_temps(
                    floodctx._accum_flowctx,
                    rec_field,
                    q_out_field,
                    sfd_buffers["donors"],
                    sfd_buffers["ndonors"],
                    sfd_buffers["donors_alt"],
                    sfd_buffers["ndonors_alt"],
                    sfd_buffers["q_alt"],
                    sfd_buffers["src"],
                )
        else:
            if mfd_buffers is None:
                flow_runtime.accumulate_mfd(floodctx._accum_flowctx, surface_field, q_out_field)
            else:
                flow_runtime.accumulate_mfd_with_temps(
                    floodctx._accum_flowctx,
                    surface_field,
                    q_out_field,
                    mfd_buffers["routing_weights"],
                    mfd_buffers["routing_sum"],
                    mfd_buffers["source"],
                    mfd_buffers["q_tmp"],
                    mfd_buffers["eps"],
                )
    finally:
        if surface_owned:
            surface.release()


def graphflood_propagate(
    floodctx,
    z,
    h,
    receivers,
    *,
    mode="sfd",
    reroute=False,
    fill=False,
    carve=True,
):
    """
    Propagate source-driven flow with pooled output and temporary buffers.

    Author: B.G (03/2026)
    """
    q_out = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (floodctx.n_flat))
    source_q = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (floodctx.n_flat))

    try:
        graphflood_propagate_with_fields(
            floodctx,
            z,
            h,
            receivers,
            q_out,
            source_q,
            mode=mode,
            reroute=reroute,
            fill=fill,
            carve=carve,
        )
        out = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (floodctx.n_flat))
        out.field.copy_from(q_out.field)
        return out
    finally:
        q_out.release()
        source_q.release()


def field_to_numpy(field_like):
    """
    Convert one flat field-like object to a numpy array copy.

    Author: B.G (03/2026)
    """
    field = require_flat_field(field_like, "field")
    return np.asarray(field.to_numpy(), dtype=np.float32).reshape(-1)
