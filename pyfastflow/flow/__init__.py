"""
Flow routing tools for PyFastFlow.

The cleaned context-driven surface is exported unconditionally. The older
FlowRouter-centric surface is kept as a best-effort legacy import block so the
package root remains usable while the migration is in progress.

Author: B.G (02/2026)
"""

from .flowcontext import FlowContext

__all__ = ["FlowContext"]

# LEGACY:
try:
    from .. import constants
    from ..general_algorithms import util_taichi
    from ..grid.legacy import neighbourer_flat
    from ..grid.legacy.gridfields import Grid as GridField
    from ..grid.legacy.neighbourer_flat import (
        bottom,
        bottom_custom,
        bottom_n,
        bottom_pew,
        bottom_pns,
        can_leave_domain,
        can_leave_domain_custom,
        can_leave_domain_n,
        can_leave_domain_pew,
        can_leave_domain_pns,
        compile_neighbourer,
        fill_edges,
        flow_out_nodes,
        i_from_rc,
        is_on_edge,
        left,
        left_custom,
        left_n,
        left_pew,
        left_pns,
        neighbour,
        neighbour_custom,
        neighbour_n,
        neighbour_pew,
        neighbour_pns,
        rc_from_i,
        right,
        right_custom,
        right_n,
        right_pew,
        right_pns,
        top,
        top_custom,
        top_n,
        top_pew,
        top_pns,
        validate_link,
        validate_link_custom,
        validate_link_n,
        validate_link_pew,
        validate_link_pns,
        which_edge,
    )
    from . import (
        downstream_propag,
        f32_i32_struct,
        fill_topo,
        flowfields,
        lakeflow,
        level_acc,
        level_based,
        locarve,
        receivers,
        sweeper,
    )
    from .fill_topo import fill_z_add_delta, topofill
    from .flowfields import FlowRouter

    __all__ += [
        "FlowRouter",
        "GridField",
        "fill_z_add_delta",
        "topofill",
        "neighbour",
        "can_leave_domain",
        "is_on_edge",
        "which_edge",
        "validate_link",
        "top",
        "bottom",
        "left",
        "right",
        "neighbour_custom",
        "can_leave_domain_custom",
        "validate_link_custom",
        "top_custom",
        "bottom_custom",
        "left_custom",
        "right_custom",
        "neighbour_n",
        "can_leave_domain_n",
        "validate_link_n",
        "top_n",
        "bottom_n",
        "left_n",
        "right_n",
        "neighbour_pew",
        "can_leave_domain_pew",
        "validate_link_pew",
        "top_pew",
        "bottom_pew",
        "left_pew",
        "right_pew",
        "neighbour_pns",
        "can_leave_domain_pns",
        "validate_link_pns",
        "top_pns",
        "bottom_pns",
        "left_pns",
        "right_pns",
        "compile_neighbourer",
        "fill_edges",
        "flow_out_nodes",
        "i_from_rc",
        "rc_from_i",
        "neighbourer_flat",
        "receivers",
        "downstream_propag",
        "lakeflow",
        "level_acc",
        "level_based",
        "f32_i32_struct",
        "util_taichi",
        "flowfields",
        "fill_topo",
        "sweeper",
        "locarve",
        "constants",
    ]
except ImportError:
    pass
