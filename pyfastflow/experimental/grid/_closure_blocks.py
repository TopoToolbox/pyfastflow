"""
Taichi/Quadrants (closure) block templates behind make_grid.

Every private block below is one plain python def, PICKED - never branched on
inside a single function body - by build_helpers() according to the grid's
config: topology (D4/D8), boundary (normal/periodic_EW/periodic_NS), nodata
(on/off), outlet (edge/mask). Composability is per axis: a periodic boundary
swaps in the "periodic" variant of a row or column block, never both, and the
untouched axis keeps its "identity"/"bounded" variant - there is no
ti.static/#if choosing between them inside one function. The one runtime
if-ladder is _delta(k): k is genuine per-call device data, not a structural
choice, so it cannot be resolved by picking a python function ahead of time.
A dynamically-indexed local array for that ladder would spill to local memory
on GPU, hence the explicit if/elif chain instead.

Every public helper below is a HelperBuilder that binds the private blocks it
needs BY NAME - helper binds helper - so a block reached from two composites
(e.g. _row reached from both neighbour_raw and dist_between_nodes) is
specialized once per compile and shared at both call sites (see compile.py,
_SpecializeCtx).

nx/ny/dx are read exclusively through `.get(...)`, uniformly across whatever
mode they end up in (const, scalar, field) - see parameter.py, "Reading a
Parameter in device code is uniform across modes." This is what lets any of
them be overridden to a runtime-modifiable mode without touching a single
block template.

Author: B.G (07/2026)
"""

import functools

from ..core.context.backends import helper_need, make_helper, param_need

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _row_tmpl(i):
    return i // NX.get(0)


def _col_tmpl(i):
    return i % NX.get(0)


def _index_tmpl(row, col):
    return row * NX.get(0) + col


def _in_bounds_tmpl(row, col):
    ok = 0
    if row >= 0 and row < NY.get(0) and col >= 0 and col < NX.get(0):
        ok = 1
    return ok


# ---------------------------------------------------------------------------
# topology: _delta(k) - runtime if-ladder, k is per-call device data
# ---------------------------------------------------------------------------


def _delta_d4_tmpl(k):
    dr = 0
    dc = 0
    if k == 0:
        dr, dc = -1, 0
    elif k == 1:
        dr, dc = 0, -1
    elif k == 2:
        dr, dc = 0, 1
    else:
        dr, dc = 1, 0
    return dr, dc


def _delta_d8_tmpl(k):
    dr = 0
    dc = 0
    if k == 0:
        dr, dc = -1, -1
    elif k == 1:
        dr, dc = -1, 0
    elif k == 2:
        dr, dc = -1, 1
    elif k == 3:
        dr, dc = 0, -1
    elif k == 4:
        dr, dc = 0, 1
    elif k == 5:
        dr, dc = 1, -1
    elif k == 6:
        dr, dc = 1, 0
    else:
        dr, dc = 1, 1
    return dr, dc


# ---------------------------------------------------------------------------
# per-axis wrap (boundary): identity or periodic, chosen per axis
# ---------------------------------------------------------------------------


def _row_wrap_identity_tmpl(row):
    return row


def _row_wrap_periodic_tmpl(row):
    r = row
    if r < 0:
        r += NY.get(0)
    elif r >= NY.get(0):
        r -= NY.get(0)
    return r


def _col_wrap_identity_tmpl(col):
    return col


def _col_wrap_periodic_tmpl(col):
    c = col
    if c < 0:
        c += NX.get(0)
    elif c >= NX.get(0):
        c -= NX.get(0)
    return c


def _wrap_tmpl(row, col):
    return _ROWWRAP(row), _COLWRAP(col)


# ---------------------------------------------------------------------------
# per-axis edge gate (boundary): is a candidate coordinate still in range on
# that axis - a periodic axis never blocks, since it wraps instead.
# ---------------------------------------------------------------------------


def _row_edge_ok_bounded_tmpl(row):
    ok = 0
    if row >= 0 and row < NY.get(0):
        ok = 1
    return ok


def _row_edge_ok_periodic_tmpl(row):
    return 1


def _col_edge_ok_bounded_tmpl(col):
    ok = 0
    if col >= 0 and col < NX.get(0):
        ok = 1
    return ok


def _col_edge_ok_periodic_tmpl(col):
    return 1


# ---------------------------------------------------------------------------
# source-cell nodata gate: on/off
# ---------------------------------------------------------------------------


def _source_ok_always_tmpl(i):
    return 1


def _source_ok_nodata_tmpl(i):
    ok = 1
    if NODATA_MASK.get(i) == 1:
        ok = 0
    return ok


# ---------------------------------------------------------------------------
# _move_allowed(i, k): per-axis edge gate x source-cell nodata gate
# ---------------------------------------------------------------------------


def _move_allowed_tmpl(i, k):
    row = _ROW(i)
    col = _COL(i)
    dr, dc = _DELTA(k)
    return _ROWEDGEOK(row + dr) * _COLEDGEOK(col + dc) * _SOURCEOK(i)


# ---------------------------------------------------------------------------
# _valid(j): in range, and (nodata ? active : true)
# ---------------------------------------------------------------------------


def _valid_no_nodata_tmpl(j):
    ok = 0
    if j >= 0 and j < NX.get(0) * NY.get(0):
        ok = 1
    return ok


def _valid_nodata_tmpl(j):
    ok = 0
    if j >= 0 and j < NX.get(0) * NY.get(0):
        ok = 1
        if NODATA_MASK.get(j) == 1:
            ok = 0
    return ok


# ---------------------------------------------------------------------------
# public: neighbour / neighbour_raw
# ---------------------------------------------------------------------------


def _neighbour_raw_tmpl(i, k):
    row = _ROW(i)
    col = _COL(i)
    dr, dc = _DELTA(k)
    wrow, wcol = _WRAP(row + dr, col + dc)
    return _INDEX(wrow, wcol)


def _neighbour_tmpl(i, k):
    j = -1
    if _MOVEALLOWED(i, k) == 1:
        cand = _NEIGHBOURRAW(i, k)
        if _VALID(cand) == 1:
            j = cand
    return j


# ---------------------------------------------------------------------------
# public: is_active / nodata
# ---------------------------------------------------------------------------


def _is_active_always_tmpl(i):
    return 1


def _is_active_mask_tmpl(i):
    ok = 1
    if NODATA_MASK.get(i) == 1:
        ok = 0
    return ok


def _nodata_tmpl(i):
    return 1 - _ISACTIVE(i)


# ---------------------------------------------------------------------------
# public: is_on_edge / which_edge (per-axis, mirrors _wrap/_edge_ok)
# ---------------------------------------------------------------------------


def _row_is_edge_active_tmpl(row):
    e = 0
    if row == 0 or row == NY.get(0) - 1:
        e = 1
    return e


def _row_is_edge_periodic_tmpl(row):
    return 0


def _col_is_edge_active_tmpl(col):
    e = 0
    if col == 0 or col == NX.get(0) - 1:
        e = 1
    return e


def _col_is_edge_periodic_tmpl(col):
    return 0


def _is_on_edge_tmpl(i):
    row = _ROW(i)
    col = _COL(i)
    e = 0
    if _ROWISEDGE(row) == 1 or _COLISEDGE(col) == 1:
        e = 1
    return e


def _row_edge_code_active_tmpl(row):
    code = -1
    if row == 0:
        code = 0
    elif row == NY.get(0) - 1:
        code = 3
    return code


def _row_edge_code_periodic_tmpl(row):
    return -1


def _col_edge_code_active_tmpl(col):
    code = -1
    if col == 0:
        code = 1
    elif col == NX.get(0) - 1:
        code = 2
    return code


def _col_edge_code_periodic_tmpl(col):
    return -1


def _which_edge_tmpl(i):
    row = _ROW(i)
    col = _COL(i)
    code = _ROWEDGECODE(row)
    if code == -1:
        code = _COLEDGECODE(col)
    return code


# ---------------------------------------------------------------------------
# public: can_out
# ---------------------------------------------------------------------------


def _can_out_mask_tmpl(i):
    out = 0
    if OUTLET_MASK.get(i) == 1:
        out = 1
    return out


def _can_out_edge_tmpl(i):
    return _ISONEDGE(i)


# ---------------------------------------------------------------------------
# public: dist_from_k / dist_between_nodes
# ---------------------------------------------------------------------------


def _dist_from_k_d4_tmpl(k):
    return DX.get(0)


def _dist_from_k_d8_tmpl(k):
    d = DX.get(0)
    if k == 0 or k == 2 or k == 5 or k == 7:
        d = DX.get(0) * SQRT2
    return d


def _row_dist_normal_tmpl(raw):
    return _BK.abs(raw)


def _row_dist_periodic_tmpl(raw):
    d = _BK.abs(raw)
    return _BK.min(d, NY.get(0) - d)


def _col_dist_normal_tmpl(raw):
    return _BK.abs(raw)


def _col_dist_periodic_tmpl(raw):
    d = _BK.abs(raw)
    return _BK.min(d, NX.get(0) - d)


def _dist_between_d4_tmpl(i, j):
    out = -1.0
    if j >= 0:
        dr = _ROWDIST(_ROW(j) - _ROW(i))
        dc = _COLDIST(_COL(j) - _COL(i))
        if dr == 0 and dc == 1:
            out = DX.get(0)
        elif dr == 1 and dc == 0:
            out = DX.get(0)
    return out


def _dist_between_d8_tmpl(i, j):
    out = -1.0
    if j >= 0:
        dr = _ROWDIST(_ROW(j) - _ROW(i))
        dc = _COLDIST(_COL(j) - _COL(i))
        if dr == 0 and dc == 1:
            out = DX.get(0)
        elif dr == 1 and dc == 0:
            out = DX.get(0)
        elif dr == 1 and dc == 1:
            out = DX.get(0) * SQRT2
    return out


# ---------------------------------------------------------------------------
# public: neighbour_and_distance
# ---------------------------------------------------------------------------


def _neighbour_and_distance_tmpl(i, k):
    j = _NEIGHBOUR(i, k)
    d = -1.0
    if j != -1:
        d = _DISTFROMK(k)
    return j, d


def build_helpers(
    HelperCls,
    *,
    nx_p,
    ny_p,
    dx_p,
    nodata_mask_p,
    outlet_mask_p,
    topology,
    boundary,
    nodata,
    outlet,
):
    """
    Wire one grid's private blocks and public composites for a closure
    backend (Taichi or Quadrants), picking each block's variant from
    `topology`/`boundary`/`nodata`/`outlet` and binding private blocks into
    public ones by name.

    Every bind below goes through a Need (param_need/helper_need, see
    backends.py) and every HelperBuilder is constructed with
    `strict_needs=True` (via `mk`) - the reference conversion the
    Need-restructuring plan calls for; see need.py/compile.py's module
    docstrings. `SQRT2` stays a plain, un-Needed bind - a python float has no
    dtype/mode/identity for a Need to check (see compile.py's `_bind_raw`,
    "a plain value ... is bound directly even under strict_needs=True").
    `backend_mod` no longer appears here at all: `_BK`, the one thing this
    file used it for (row_dist/col_dist's `_BK.abs`/`_BK.min`), is now
    auto-injected into every closure template - see _closure_backend.py's
    module docstring.

    Returns {public_name: HelperBuilder}, meant to be merged straight into
    the Bag make_grid() returns.

    Author: B.G (07/2026)
    """
    import math

    sqrt2 = math.sqrt(2.0)
    d8 = topology == "D8"

    mk = functools.partial(make_helper, HelperCls, strict_needs=True)

    row = mk(_row_tmpl, NX=param_need("NX", nx_p))
    col = mk(_col_tmpl, NX=param_need("NX", nx_p))
    index = mk(_index_tmpl, NX=param_need("NX", nx_p))

    delta = mk(_delta_d8_tmpl if d8 else _delta_d4_tmpl)

    row_wrap = mk(
        _row_wrap_periodic_tmpl if boundary == "periodic_NS" else _row_wrap_identity_tmpl,
        NY=param_need("NY", ny_p),
    )
    col_wrap = mk(
        _col_wrap_periodic_tmpl if boundary == "periodic_EW" else _col_wrap_identity_tmpl,
        NX=param_need("NX", nx_p),
    )
    wrap = mk(_wrap_tmpl, _ROWWRAP=helper_need("_ROWWRAP", row_wrap), _COLWRAP=helper_need("_COLWRAP", col_wrap))

    row_edge_ok = mk(
        _row_edge_ok_periodic_tmpl if boundary == "periodic_NS" else _row_edge_ok_bounded_tmpl,
        NY=param_need("NY", ny_p),
    )
    col_edge_ok = mk(
        _col_edge_ok_periodic_tmpl if boundary == "periodic_EW" else _col_edge_ok_bounded_tmpl,
        NX=param_need("NX", nx_p),
    )

    source_ok = (
        mk(_source_ok_nodata_tmpl, NODATA_MASK=param_need("NODATA_MASK", nodata_mask_p))
        if nodata
        else mk(_source_ok_always_tmpl)
    )

    move_allowed = mk(
        _move_allowed_tmpl,
        _ROW=helper_need("_ROW", row),
        _COL=helper_need("_COL", col),
        _DELTA=helper_need("_DELTA", delta),
        _ROWEDGEOK=helper_need("_ROWEDGEOK", row_edge_ok),
        _COLEDGEOK=helper_need("_COLEDGEOK", col_edge_ok),
        _SOURCEOK=helper_need("_SOURCEOK", source_ok),
    )

    valid_binds = {"NX": param_need("NX", nx_p), "NY": param_need("NY", ny_p)}
    if nodata:
        valid_binds["NODATA_MASK"] = param_need("NODATA_MASK", nodata_mask_p)
    valid = mk(_valid_nodata_tmpl if nodata else _valid_no_nodata_tmpl, **valid_binds)

    neighbour_raw = mk(
        _neighbour_raw_tmpl,
        _ROW=helper_need("_ROW", row),
        _COL=helper_need("_COL", col),
        _DELTA=helper_need("_DELTA", delta),
        _WRAP=helper_need("_WRAP", wrap),
        _INDEX=helper_need("_INDEX", index),
    )
    neighbour = mk(
        _neighbour_tmpl,
        _MOVEALLOWED=helper_need("_MOVEALLOWED", move_allowed),
        _NEIGHBOURRAW=helper_need("_NEIGHBOURRAW", neighbour_raw),
        _VALID=helper_need("_VALID", valid),
    )

    is_active = (
        mk(_is_active_mask_tmpl, NODATA_MASK=param_need("NODATA_MASK", nodata_mask_p))
        if nodata
        else mk(_is_active_always_tmpl)
    )
    nodata_fn = mk(_nodata_tmpl, _ISACTIVE=helper_need("_ISACTIVE", is_active))

    row_is_edge = mk(
        _row_is_edge_periodic_tmpl if boundary == "periodic_NS" else _row_is_edge_active_tmpl,
        NY=param_need("NY", ny_p),
    )
    col_is_edge = mk(
        _col_is_edge_periodic_tmpl if boundary == "periodic_EW" else _col_is_edge_active_tmpl,
        NX=param_need("NX", nx_p),
    )
    is_on_edge = mk(
        _is_on_edge_tmpl,
        _ROW=helper_need("_ROW", row),
        _COL=helper_need("_COL", col),
        _ROWISEDGE=helper_need("_ROWISEDGE", row_is_edge),
        _COLISEDGE=helper_need("_COLISEDGE", col_is_edge),
    )

    row_edge_code = mk(
        _row_edge_code_periodic_tmpl if boundary == "periodic_NS" else _row_edge_code_active_tmpl,
        NY=param_need("NY", ny_p),
    )
    col_edge_code = mk(
        _col_edge_code_periodic_tmpl if boundary == "periodic_EW" else _col_edge_code_active_tmpl,
        NX=param_need("NX", nx_p),
    )
    which_edge = mk(
        _which_edge_tmpl,
        _ROW=helper_need("_ROW", row),
        _COL=helper_need("_COL", col),
        _ROWEDGECODE=helper_need("_ROWEDGECODE", row_edge_code),
        _COLEDGECODE=helper_need("_COLEDGECODE", col_edge_code),
    )

    if outlet == "mask":
        can_out = mk(_can_out_mask_tmpl, OUTLET_MASK=param_need("OUTLET_MASK", outlet_mask_p))
    else:
        can_out = mk(_can_out_edge_tmpl, _ISONEDGE=helper_need("_ISONEDGE", is_on_edge))

    dist_from_k = mk(_dist_from_k_d8_tmpl if d8 else _dist_from_k_d4_tmpl, DX=param_need("DX", dx_p), SQRT2=sqrt2)

    row_dist = mk(
        _row_dist_periodic_tmpl if boundary == "periodic_NS" else _row_dist_normal_tmpl,
        NY=param_need("NY", ny_p),
    )
    col_dist = mk(
        _col_dist_periodic_tmpl if boundary == "periodic_EW" else _col_dist_normal_tmpl,
        NX=param_need("NX", nx_p),
    )
    dist_between = mk(
        _dist_between_d8_tmpl if d8 else _dist_between_d4_tmpl,
        _ROW=helper_need("_ROW", row),
        _COL=helper_need("_COL", col),
        _ROWDIST=helper_need("_ROWDIST", row_dist),
        _COLDIST=helper_need("_COLDIST", col_dist),
        DX=param_need("DX", dx_p),
        SQRT2=sqrt2,
    )

    neighbour_and_distance = mk(
        _neighbour_and_distance_tmpl,
        _NEIGHBOUR=helper_need("_NEIGHBOUR", neighbour),
        _DISTFROMK=helper_need("_DISTFROMK", dist_from_k),
    )

    return {
        "neighbour": neighbour,
        "neighbour_raw": neighbour_raw,
        "nodata": nodata_fn,
        "is_active": is_active,
        "can_out": can_out,
        "dist_from_k": dist_from_k,
        "dist_between_nodes": dist_between,
        "is_on_edge": is_on_edge,
        "which_edge": which_edge,
        "neighbour_and_distance": neighbour_and_distance,
    }
