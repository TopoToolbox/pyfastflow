"""
cupy (CUDA source) block templates behind make_grid.

Mirrors _closure_blocks.py block for block - same private/public split, same
per-axis composability (a periodic boundary swaps in the "periodic" __device__
variant of a row or column block, the untouched axis keeps its
"identity"/"bounded" variant) - written as CUDA text instead of python defs,
since that is what CupyHelperBuilder/CupyKernelBuilder compile (see
cupy_backend.py's module docstring for the `$...$` span mechanism).

_delta(k) is the one runtime if-ladder equivalent: k is per-call device data,
not a structural choice, so here it is a `__constant__` int table indexed by
k at runtime instead - a dynamically-indexed local array would live in local
memory on GPU, `__constant__` does not. Callers that loop over k statically
should add `#pragma unroll` at the call site; that is a caller concern, not
something a block enforces.

Every device function name is prefixed with this grid's own tag (a fresh
new_uid()), so two make_grid() calls in one process never collide inside a
single compiled cupy module even if both are bound into the same kernel.

Author: B.G (07/2026)
"""

import math

from ..core.pool.base import new_uid


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
    backend_mod=None,
):
    """
    Wire one grid's private blocks and public composites for the cupy
    backend, picking each block's variant from
    `topology`/`boundary`/`nodata`/`outlet` and binding private blocks into
    public ones by name (a bound name resolves to the real emitted C symbol
    at span-expansion time - see cupy_backend.py's _SpanParser).

    Returns {public_name: HelperBuilder}, meant to be merged straight into
    the Bag make_grid() returns. `backend_mod` is accepted for signature
    parity with the closure backend's build_helpers and unused here - cupy
    templates call plain C (abs, ternaries) rather than a bound backend
    module.

    Author: B.G (07/2026)
    """
    t = f"pf{new_uid()}"
    d8 = topology == "D8"
    sqrt2 = math.sqrt(2.0)

    def mk(template, **binds):
        b = HelperCls().ingest(template)
        for name, obj in binds.items():
            b.bind(name, obj)
        return b

    row = mk(
        f"__device__ int {t}_row(int i) {{ return i / $NX.get(0)$; }}",
        NX=nx_p,
    )
    col = mk(
        f"__device__ int {t}_col(int i) {{ return i % $NX.get(0)$; }}",
        NX=nx_p,
    )
    index = mk(
        f"__device__ int {t}_index(int row, int col) {{ return row * $NX.get(0)$ + col; }}",
        NX=nx_p,
    )

    if d8:
        delta = mk(
            f"""
__constant__ int {t}_DELTA_DR[8] = {{-1, -1, -1, 0, 0, 1, 1, 1}};
__constant__ int {t}_DELTA_DC[8] = {{-1, 0, 1, -1, 1, -1, 0, 1}};
__device__ void {t}_delta(int k, int* dr, int* dc) {{
    *dr = {t}_DELTA_DR[k];
    *dc = {t}_DELTA_DC[k];
}}
"""
        )
    else:
        delta = mk(
            f"""
__constant__ int {t}_DELTA_DR[4] = {{-1, 0, 0, 1}};
__constant__ int {t}_DELTA_DC[4] = {{0, -1, 1, 0}};
__device__ void {t}_delta(int k, int* dr, int* dc) {{
    *dr = {t}_DELTA_DR[k];
    *dc = {t}_DELTA_DC[k];
}}
"""
        )

    row_wrap = (
        mk(f"__device__ int {t}_row_wrap(int row) {{ return row; }}")
        if boundary != "periodic_NS"
        else mk(
            f"""
__device__ int {t}_row_wrap(int row) {{
    int r = row;
    if (r < 0) r += $NY.get(0)$;
    else if (r >= $NY.get(0)$) r -= $NY.get(0)$;
    return r;
}}
""",
            NY=ny_p,
        )
    )
    col_wrap = (
        mk(f"__device__ int {t}_col_wrap(int col) {{ return col; }}")
        if boundary != "periodic_EW"
        else mk(
            f"""
__device__ int {t}_col_wrap(int col) {{
    int c = col;
    if (c < 0) c += $NX.get(0)$;
    else if (c >= $NX.get(0)$) c -= $NX.get(0)$;
    return c;
}}
""",
            NX=nx_p,
        )
    )
    wrap = mk(
        f"""
__device__ void {t}_wrap(int row, int col, int* wrow, int* wcol) {{
    *wrow = $row_wrap(row)$;
    *wcol = $col_wrap(col)$;
}}
""",
        row_wrap=row_wrap,
        col_wrap=col_wrap,
    )

    row_edge_ok = (
        mk(f"__device__ int {t}_row_edge_ok(int row) {{ return 1; }}")
        if boundary == "periodic_NS"
        else mk(
            f"__device__ int {t}_row_edge_ok(int row) {{ return (row >= 0 && row < $NY.get(0)$) ? 1 : 0; }}",
            NY=ny_p,
        )
    )
    col_edge_ok = (
        mk(f"__device__ int {t}_col_edge_ok(int col) {{ return 1; }}")
        if boundary == "periodic_EW"
        else mk(
            f"__device__ int {t}_col_edge_ok(int col) {{ return (col >= 0 && col < $NX.get(0)$) ? 1 : 0; }}",
            NX=nx_p,
        )
    )

    source_ok = (
        mk(
            f"__device__ int {t}_source_ok(int i) {{ return ($NODATA_MASK.get(i)$ == 1) ? 0 : 1; }}",
            NODATA_MASK=nodata_mask_p,
        )
        if nodata
        else mk(f"__device__ int {t}_source_ok(int i) {{ return 1; }}")
    )

    move_allowed = mk(
        f"""
__device__ int {t}_move_allowed(int i, int k) {{
    int row = $row(i)$;
    int col = $col(i)$;
    int dr, dc;
    $delta(k, &dr, &dc)$;
    int a = $row_edge_ok(row + dr)$;
    int b = $col_edge_ok(col + dc)$;
    int c = $source_ok(i)$;
    return a * b * c;
}}
""",
        row=row,
        col=col,
        delta=delta,
        row_edge_ok=row_edge_ok,
        col_edge_ok=col_edge_ok,
        source_ok=source_ok,
    )

    valid_binds = {"NX": nx_p, "NY": ny_p}
    if nodata:
        valid_binds["NODATA_MASK"] = nodata_mask_p
        valid = mk(
            f"""
__device__ int {t}_valid(int j) {{
    if (j < 0 || j >= $NX.get(0)$ * $NY.get(0)$) return 0;
    return ($NODATA_MASK.get(j)$ == 1) ? 0 : 1;
}}
""",
            **valid_binds,
        )
    else:
        valid = mk(
            f"__device__ int {t}_valid(int j) {{ return (j >= 0 && j < $NX.get(0)$ * $NY.get(0)$) ? 1 : 0; }}",
            **valid_binds,
        )

    neighbour_raw = mk(
        f"""
__device__ int {t}_neighbour_raw(int i, int k) {{
    int row = $row(i)$;
    int col = $col(i)$;
    int dr, dc;
    $delta(k, &dr, &dc)$;
    int wrow, wcol;
    $wrap(row + dr, col + dc, &wrow, &wcol)$;
    return $index(wrow, wcol)$;
}}
""",
        row=row,
        col=col,
        delta=delta,
        wrap=wrap,
        index=index,
    )

    neighbour = mk(
        f"""
__device__ int {t}_neighbour(int i, int k) {{
    int j = -1;
    if ($move_allowed(i, k)$ == 1) {{
        int cand = $neighbour_raw(i, k)$;
        if ($valid(cand)$ == 1) j = cand;
    }}
    return j;
}}
""",
        move_allowed=move_allowed,
        neighbour_raw=neighbour_raw,
        valid=valid,
    )

    is_active = (
        mk(
            f"__device__ int {t}_is_active(int i) {{ return ($NODATA_MASK.get(i)$ == 1) ? 0 : 1; }}",
            NODATA_MASK=nodata_mask_p,
        )
        if nodata
        else mk(f"__device__ int {t}_is_active(int i) {{ return 1; }}")
    )
    nodata_fn = mk(
        f"__device__ int {t}_nodata(int i) {{ return 1 - $is_active(i)$; }}",
        is_active=is_active,
    )

    row_is_edge = (
        mk(f"__device__ int {t}_row_is_edge(int row) {{ return 0; }}")
        if boundary == "periodic_NS"
        else mk(
            f"__device__ int {t}_row_is_edge(int row) {{ return (row == 0 || row == $NY.get(0)$ - 1) ? 1 : 0; }}",
            NY=ny_p,
        )
    )
    col_is_edge = (
        mk(f"__device__ int {t}_col_is_edge(int col) {{ return 0; }}")
        if boundary == "periodic_EW"
        else mk(
            f"__device__ int {t}_col_is_edge(int col) {{ return (col == 0 || col == $NX.get(0)$ - 1) ? 1 : 0; }}",
            NX=nx_p,
        )
    )
    is_on_edge = mk(
        f"""
__device__ int {t}_is_on_edge(int i) {{
    int row = $row(i)$;
    int col = $col(i)$;
    return ($row_is_edge(row)$ == 1 || $col_is_edge(col)$ == 1) ? 1 : 0;
}}
""",
        row=row,
        col=col,
        row_is_edge=row_is_edge,
        col_is_edge=col_is_edge,
    )

    row_edge_code = (
        mk(f"__device__ int {t}_row_edge_code(int row) {{ return -1; }}")
        if boundary == "periodic_NS"
        else mk(
            f"""
__device__ int {t}_row_edge_code(int row) {{
    if (row == 0) return 0;
    if (row == $NY.get(0)$ - 1) return 3;
    return -1;
}}
""",
            NY=ny_p,
        )
    )
    col_edge_code = (
        mk(f"__device__ int {t}_col_edge_code(int col) {{ return -1; }}")
        if boundary == "periodic_EW"
        else mk(
            f"""
__device__ int {t}_col_edge_code(int col) {{
    if (col == 0) return 1;
    if (col == $NX.get(0)$ - 1) return 2;
    return -1;
}}
""",
            NX=nx_p,
        )
    )
    which_edge = mk(
        f"""
__device__ int {t}_which_edge(int i) {{
    int row = $row(i)$;
    int col = $col(i)$;
    int code = $row_edge_code(row)$;
    if (code == -1) code = $col_edge_code(col)$;
    return code;
}}
""",
        row=row,
        col=col,
        row_edge_code=row_edge_code,
        col_edge_code=col_edge_code,
    )

    if outlet == "mask":
        can_out = mk(
            f"__device__ int {t}_can_out(int i) {{ return ($OUTLET_MASK.get(i)$ == 1) ? 1 : 0; }}",
            OUTLET_MASK=outlet_mask_p,
        )
    else:
        can_out = mk(
            f"__device__ int {t}_can_out(int i) {{ return $is_on_edge(i)$; }}",
            is_on_edge=is_on_edge,
        )

    if d8:
        dist_from_k = mk(
            f"""
__device__ float {t}_dist_from_k(int k) {{
    float d = $DX.get(0)$;
    if (k == 0 || k == 2 || k == 5 || k == 7) d = $DX.get(0)$ * {sqrt2}f;
    return d;
}}
""",
            DX=dx_p,
        )
    else:
        dist_from_k = mk(
            f"__device__ float {t}_dist_from_k(int k) {{ return $DX.get(0)$; }}",
            DX=dx_p,
        )

    row_dist = (
        mk(f"__device__ int {t}_row_dist(int raw) {{ return abs(raw); }}")
        if boundary != "periodic_NS"
        else mk(
            f"""
__device__ int {t}_row_dist(int raw) {{
    int d = abs(raw);
    int n = $NY.get(0)$;
    return d < (n - d) ? d : (n - d);
}}
""",
            NY=ny_p,
        )
    )
    col_dist = (
        mk(f"__device__ int {t}_col_dist(int raw) {{ return abs(raw); }}")
        if boundary != "periodic_EW"
        else mk(
            f"""
__device__ int {t}_col_dist(int raw) {{
    int d = abs(raw);
    int n = $NX.get(0)$;
    return d < (n - d) ? d : (n - d);
}}
""",
            NX=nx_p,
        )
    )

    diag_line = f"        else if (dr == 1 && dc == 1) out = $DX.get(0)$ * {sqrt2}f;\n" if d8 else ""
    dist_between = mk(
        f"""
__device__ float {t}_dist_between(int i, int j) {{
    float out = -1.0f;
    if (j >= 0) {{
        int ri = $row(i)$;
        int ci = $col(i)$;
        int rj = $row(j)$;
        int cj = $col(j)$;
        int dr = $row_dist(rj - ri)$;
        int dc = $col_dist(cj - ci)$;
        if (dr == 0 && dc == 1) out = $DX.get(0)$;
        else if (dr == 1 && dc == 0) out = $DX.get(0)$;
{diag_line}    }}
    return out;
}}
""",
        row=row,
        col=col,
        row_dist=row_dist,
        col_dist=col_dist,
        DX=dx_p,
    )

    neighbour_and_distance = mk(
        f"""
__device__ void {t}_neighbour_and_distance(int i, int k, int* j_out, float* d_out) {{
    int j = $neighbour(i, k)$;
    float d = -1.0f;
    if (j != -1) d = $dist_from_k(k)$;
    *j_out = j;
    *d_out = d;
}}
""",
        neighbour=neighbour,
        dist_from_k=dist_from_k,
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
