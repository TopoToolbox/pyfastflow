"""
make_grid: the GridContext-equivalent Bag factory, built on the
backend-agnostic core (see ..core.context: parameter.py for Parameter, compile.py for HelperBuilder, bag.py for Bag).

There is no stateful Context class here - by design, the core has none (see
core/context/parameter.py's module docstring). make_grid just builds a Bag once: a
uniform public surface (grid.nx, grid.neighbour(i, k), ...) whatever the
backend and whatever the grid's own topology/boundary/nodata/outlet config.

Two kinds of knobs:
  - value params (nx, ny, dx) - mode-overridable (const/scalar, dx also
    field), always read in device code through `.get(...)`. Default mode is
    "const" for all three.
  - structural selectors (topology, boundary, nodata, outlet) - each one
    picks which variant of a private block gets bound into the public
    composite helpers; see _closure_blocks.py / _cupy_blocks.py.

Masks are independent, optional bag members: nodata_mask (u8, 1 == inactive)
when nodata=True, outlet_mask (u8, 1 == outlet) when outlet=="mask". Neither
exists in the bag when its feature is off, so a caller that never asked for
nodata/mask-outlet never sees them.

Author: B.G (07/2026)
"""

import numpy as np

from ..core.context.backends import backend_classes
from ..core.context.bag import Bag

_TOPOLOGIES = {"D4": 4, "D8": 8}
_BOUNDARIES = frozenset({"normal", "periodic_EW", "periodic_NS"})
_OUTLETS = frozenset({"edge", "mask"})


def _blocks_for(backend: str):
    """
    The private block module implementing make_grid's device code for one
    backend name: the closure blocks (shared by Taichi and Quadrants) or the
    cupy blocks.

    Author: B.G (07/2026)
    """
    if backend in ("taichi", "quadrants"):
        from . import _closure_blocks as blocks
    elif backend == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"make_grid: unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")
    return blocks


def make_grid(
    backend: str,
    pool,
    nx: int,
    ny: int,
    dx: float,
    *,
    topology: str = "D8",
    boundary: str = "normal",
    nodata: bool = False,
    outlet: str = "edge",
    nx_mode: str = "const",
    ny_mode: str = "const",
    dx_mode: str = "const",
) -> Bag:
    """
    Build one grid's Bag: nx/ny/dx/n_neighbours params, the optional
    nodata_mask/outlet_mask fields, and the neighbour/distance/edge helper
    surface - all uniform by name regardless of backend or config.

    `topology` "D4"|"D8", `boundary` "normal"|"periodic_EW"|"periodic_NS",
    `outlet` "edge"|"mask" pick block variants at build time (see
    _closure_blocks.py / _cupy_blocks.py). `nodata` allocates and folds in
    nodata_mask (u8, 1 == inactive) wherever a block needs it.

    `nx_mode`/`ny_mode` default "const", may be overridden to "scalar".
    `dx_mode` defaults "const", may be overridden to "scalar" or "field" - a
    field-mode dx is allocated (one cell per node, caller fills it) but the
    public helpers that read dx (dist_from_k, dist_between_nodes) only ever
    read index 0: neither's signature carries a node to key a per-node value
    off, so a genuinely spatially-varying dx is not wired through those two
    helpers as things stand - only reachable by reading grid.dx.get(i)
    directly in a caller's own template.

    Author: B.G (07/2026)
    """
    if topology not in _TOPOLOGIES:
        raise ValueError(f"make_grid: topology must be one of {sorted(_TOPOLOGIES)}, got {topology!r}")
    if boundary not in _BOUNDARIES:
        raise ValueError(f"make_grid: boundary must be one of {sorted(_BOUNDARIES)}, got {boundary!r}")
    if outlet not in _OUTLETS:
        raise ValueError(f"make_grid: outlet must be one of {sorted(_OUTLETS)}, got {outlet!r}")
    if nx_mode not in ("const", "scalar"):
        raise ValueError(f"make_grid: nx_mode must be 'const' or 'scalar', got {nx_mode!r}")
    if ny_mode not in ("const", "scalar"):
        raise ValueError(f"make_grid: ny_mode must be 'const' or 'scalar', got {ny_mode!r}")
    if dx_mode not in ("const", "scalar", "field"):
        raise ValueError(f"make_grid: dx_mode must be 'const', 'scalar' or 'field', got {dx_mode!r}")

    backend_mod, ParamCls, HelperCls, dtypes = backend_classes(backend)
    blocks = _blocks_for(backend)
    n_flat = int(nx) * int(ny)

    nx_p = ParamCls("GRID_NX", dtype=dtypes["i32"], mode=nx_mode, value=int(nx), pool=pool)
    ny_p = ParamCls("GRID_NY", dtype=dtypes["i32"], mode=ny_mode, value=int(ny), pool=pool)

    if dx_mode == "field":
        dx_p = ParamCls(
            "GRID_DX",
            dtype=dtypes["f32"],
            mode="field",
            value=np.full(n_flat, dx, dtype=np.float32),
            pool=pool,
            n_flat=n_flat,
        )
    else:
        dx_p = ParamCls("GRID_DX", dtype=dtypes["f32"], mode=dx_mode, value=float(dx), pool=pool)

    n_neighbours_p = ParamCls(
        "GRID_NNEIGHBOURS", dtype=dtypes["i32"], mode="const", value=_TOPOLOGIES[topology], pool=pool
    )

    nodata_mask_p = None
    if nodata:
        nodata_mask_p = ParamCls(
            "GRID_NODATA_MASK",
            dtype=dtypes["u8"],
            mode="field",
            value=np.zeros(n_flat, dtype=np.uint8),
            pool=pool,
            n_flat=n_flat,
        )

    outlet_mask_p = None
    if outlet == "mask":
        outlet_mask_p = ParamCls(
            "GRID_OUTLET_MASK",
            dtype=dtypes["u8"],
            mode="field",
            value=np.zeros(n_flat, dtype=np.uint8),
            pool=pool,
            n_flat=n_flat,
        )

    helpers = blocks.build_helpers(
        HelperCls,
        nx_p=nx_p,
        ny_p=ny_p,
        dx_p=dx_p,
        nodata_mask_p=nodata_mask_p,
        outlet_mask_p=outlet_mask_p,
        topology=topology,
        boundary=boundary,
        nodata=nodata,
        outlet=outlet,
        backend_mod=backend_mod,
    )

    items = {"nx": nx_p, "ny": ny_p, "dx": dx_p, "n_neighbours": n_neighbours_p}
    if nodata_mask_p is not None:
        items["nodata_mask"] = nodata_mask_p
    if outlet_mask_p is not None:
        items["outlet_mask"] = outlet_mask_p
    items.update(helpers)
    return Bag(items)
