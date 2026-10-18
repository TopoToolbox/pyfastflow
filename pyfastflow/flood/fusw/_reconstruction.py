"""States on the faces of a cell: neighbour-or-ghost state and the
reconstruction of a cell along one axis (FullSWOF).

D4 directions k: 0 north (row - 1), 1 west, 2 east, 3 south (row + 1);
axis 0 (x, columns) holds k = 1, 2 and axis 1 (y, rows) k = 0, 3. Along an
axis, the normal velocity is u on x and v on y, the tangential one the other
component; "left" is the west or north side.

``make_side(grid, ghost, real)`` returns a helper

    void fusw_side(const real* h, const real* u, const real* v,
                   const real* z, const unsigned char* code,
                          const unsigned char* fmask, int i, int k,
                   real dx, real factor,
                   real* hs, real* un, real* ut, real* zs)

giving the state beyond face k of cell i: its D4 neighbour j when the grid
has one (periodic edges wrap), else the ghost state of the boundary code of
cell i when bit k of its face mask ``fmask`` is set, of a wall otherwise
(``_boundary.py``; ``factor``: inflow time factor; the face count passed to
the ghost counts the boundary faces selected by the mask).

``make_axis(order, grid, side, limiter, real)`` returns a helper

    void fusw_axis(const real* h, const real* u, const real* v,
                   const real* z, const unsigned char* code,
                          const unsigned char* fmask, int i, int axis,
                   real dx, real factor,
                   real* hl, real* hr, real* unl, real* unr,
                   real* utl, real* utr, real* zl, real* zr)

giving the reconstructed depth, velocities and bed of cell i on its left
(l) and right (r) faces along ``axis``:

  - ``order=1``: the cell values on both faces;
  - ``order=2`` (MUSCL with the hydrostatic bed reconstruction): with the
    one-sided differences d1 (cell - left state) and d2 (right state - cell)
    of h, z, u_n, u_t and the ``limiter`` L,
      dh = L(dh1, dh2), dhz = L(dh1 + dz1, dh2 + dz2),
      h_r,l = h +- dh / 2, z_r = z + (dhz - dh) / 2, z_l = z + (dh - dhz) / 2,
      u_r = u + h_l du / (2 h), u_l = u - h_r du / (2 h)
    (u +- du / 2 on a dry cell), du = L(du1, du2), the same for u_t.

Author: B.G (10/2026)
"""

from pyfastflow.core import FrozenHelper, HelperBuilder

from ._common import literal


def make_side(grid, ghost, real: str) -> FrozenHelper:
    return (HelperBuilder(f'''
__device__ void fusw_side(const {real}* h, const {real}* u, const {real}* v,
                          const {real}* z, const unsigned char* code,
                          const unsigned char* fmask, int i, int k,
                          {real} dx, {real} factor,
                          {real}* hs, {real}* un, {real}* ut, {real}* zs) {{
    bool x_axis = (k == 1 || k == 2);
    int j = $ctx.grid.neighbour(i, k)$;
    if (j >= 0) {{
        *hs = h[j];
        *un = x_axis ? u[j] : v[j];
        *ut = x_axis ? v[j] : u[j];
        *zs = z[j];
        return;
    }}
    int opp = $ctx.grid.neighbour(i, 3 - k)$;
    int faces = 0;
    for (int kk = 0; kk < 4; ++kk)
        faces += ($ctx.grid.neighbour(i, kk)$ < 0 && ((fmask[i] >> kk) & 1)) ? 1 : 0;
    $ctx.ghost(
        ((fmask[i] >> k) & 1) ? (int)code[i] : 0, i, (k == 2 || k == 3) ? 1 : -1, h[i],
        x_axis ? u[i] : v[i], x_axis ? v[i] : u[i], z[i],
        opp >= 0 ? z[opp] : z[i], dx, factor, faces, hs, un, ut, zs)$;
}}
''').compose("grid", grid).compose("ghost", ghost).freeze())


def make_axis(order: int, grid, side, limiter, real: str) -> FrozenHelper:
    c = literal(real)
    signature = f'''
__device__ void fusw_axis(const {real}* h, const {real}* u, const {real}* v,
                          const {real}* z, const unsigned char* code,
                          const unsigned char* fmask, int i, int axis,
                          {real} dx, {real} factor,
                          {real}* hl, {real}* hr, {real}* unl, {real}* unr,
                          {real}* utl, {real}* utr, {real}* zl, {real}* zr) {{'''
    if int(order) == 1:
        return HelperBuilder(signature + f'''
    (void)code; (void)fmask; (void)dx; (void)factor;
    *hl = *hr = h[i];
    *unl = *unr = axis == 0 ? u[i] : v[i];
    *utl = *utr = axis == 0 ? v[i] : u[i];
    *zl = *zr = z[i];
}}
''').freeze()
    return (HelperBuilder(signature + f'''
    {real} h0, n0, t0, z0, h2, n2, t2, z2;
    $ctx.side(h, u, v, z, code, fmask, i, axis == 0 ? 1 : 0, dx, factor, &h0, &n0, &t0, &z0)$;
    $ctx.side(h, u, v, z, code, fmask, i, axis == 0 ? 2 : 3, dx, factor, &h2, &n2, &t2, &z2)$;
    {real} h1 = h[i];
    {real} n1 = axis == 0 ? u[i] : v[i];
    {real} t1 = axis == 0 ? v[i] : u[i];
    {real} z1 = z[i];
    {real} dh = $ctx.limiter(h1 - h0, h2 - h1)$;
    {real} dhz = $ctx.limiter(h1 - h0 + z1 - z0, h2 - h1 + z2 - z1)$;
    {real} dn = $ctx.limiter(n1 - n0, n2 - n1)$;
    {real} dt = $ctx.limiter(t1 - t0, t2 - t1)$;
    *hr = h1 + dh * {c(0.5)};
    *hl = h1 - dh * {c(0.5)};
    *zr = z1 + {c(0.5)} * (dhz - dh);
    *zl = z1 + {c(0.5)} * (dh - dhz);
    if (h1 > {c(0)}) {{
        *unr = n1 + *hl * dn * {c(0.5)} / h1;
        *unl = n1 - *hr * dn * {c(0.5)} / h1;
        *utr = t1 + *hl * dt * {c(0.5)} / h1;
        *utl = t1 - *hr * dt * {c(0.5)} / h1;
    }} else {{
        *unr = n1 + dn * {c(0.5)};
        *unl = n1 - dn * {c(0.5)};
        *utr = t1 + dt * {c(0.5)};
        *utl = t1 - dt * {c(0.5)};
    }}
}}
''').compose("side", side).compose("limiter", limiter).freeze())
