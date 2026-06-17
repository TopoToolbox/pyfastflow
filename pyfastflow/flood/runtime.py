"""
Runtime services for flood workflows built from ``FloodContext``.

Author: B.G (03/2026)
"""

import numpy as np

from .. import constants as cte
from .. import pool as ppool


def graphflood_core(floodctx, z, h, Q_in):
    """
    Run the safe (two-pass) graphflood core update on h.

    Allocates and releases a temporary dh field from the pool.

    Author: B.G (03/2026)
    """
    dh = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, floodctx.n_flat)
    try:
        floodctx.graphflood_core(z, h, Q_in, dh.field)
    finally:
        dh.release()


def compute_metrics(
    floodctx,
    z,
    h,
    *,
    u=False,
    u_direction=False,
    tau=False,
    tau_direction=False,
    Sw=False,
    Sw_direction=False,
    q=False,
    q_direction=False,
    Q=False,
    Q_direction=False,
):
    """
    Compute and return a dictionary of hydraulic metrics.

    Scalar metrics (e.g., 'u', 'Sw') are returned as 2D numpy arrays matching
    the grid shape (ny, nx).

    Directional metrics (e.g., 'u_direction') are returned as sub-dictionaries
    mapping direction names (e.g., 'top', 'topleft') to 2D numpy arrays.

    Author: B.G (04/2026)
    """
    n_flat = floodctx.n_flat
    nx, ny = floodctx.gridctx.nx, floodctx.gridctx.ny
    rshp = (ny, nx)
    n_neigh = floodctx.gridctx.n_neighbours
    results = {}

    if n_neigh == 8:
        dir_names = [
            "topleft", "top", "topright",
            "left", "right",
            "bottomleft", "bottom", "bottomright",
        ]
    else:
        dir_names = ["top", "left", "right", "bottom"]

    tmp = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, n_flat)
    try:
        def to_2d(field):
            return np.asarray(field.to_numpy(), dtype=np.float32).reshape(rshp).copy()

        if Sw:
            floodctx.compute_Sw(z, h, tmp.field)
            results["Sw"] = to_2d(tmp.field)
        if u:
            floodctx.compute_u(z, h, tmp.field)
            results["u"] = to_2d(tmp.field)
        if tau:
            floodctx.compute_tau(z, h, tmp.field)
            results["tau"] = to_2d(tmp.field)
        if q:
            floodctx.compute_q(z, h, tmp.field)
            results["q"] = to_2d(tmp.field)
        if Q:
            floodctx.compute_Qo(z, h, tmp.field)
            results["Q"] = to_2d(tmp.field)

        directional_configs = [
            (Sw_direction, "Sw_direction", floodctx.compute_Sw_direction),
            (u_direction, "u_direction", floodctx.compute_u_direction),
            (tau_direction, "tau_direction", floodctx.compute_tau_direction),
            (q_direction, "q_direction", floodctx.compute_q_direction),
            (Q_direction, "Q_direction", floodctx.compute_Q_direction),
        ]
        for flag, name, kernel in directional_configs:
            if flag:
                dir_results = {}
                for k in range(n_neigh):
                    kernel(z, h, k, tmp.field)
                    dir_results[dir_names[k]] = to_2d(tmp.field)
                results[name] = dir_results

    finally:
        tmp.release()

    return results
