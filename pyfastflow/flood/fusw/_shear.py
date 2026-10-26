"""Bed shear stress of the current state, on request (analysis kernels).

Two estimates, with rho = RHO, g = GRAVITY, dry below HE_CA (0):

  - ``friction``: the stress of the friction law of the solver,
    tau = rho g h S_f u / |u|. With D(h, |u|, dt) the friction divisor
    (``_friction.py``), g S_f / |u| = D(h, |u|, 1) - 1, hence
    tau = rho h (D(h, |u|, 1) - 1) (u, v);
  - ``surface``: the driving stress of the water-surface slope,
    tau = -rho g h grad(eta), eta = z + h, with grad(eta) by
    ``surface_slope``: ``gradient`` (central differences, below) or ``d8``
    (the steepest descent among the 8 neighbours: S = max (eta - eta_j) / d
    over the neighbours j at distance d = dx or dx sqrt(2), grad(eta) = -S
    times the unit vector towards that neighbour; 0 without a lower
    neighbour).

Neighbour states come from ``fusw_side`` (``_reconstruction.py``): the D4
neighbour, or the ghost state of the boundary face, so edges follow the
boundary codes of the solver (a wall gives a zero normal slope). Beyond a
dry neighbour state (depth <= HE_CA) the surface is min(bed, eta of the wet
side): a dry bed above the water stops the slope, one below it gives the
drop.

Cells (``tau_cx``, ``tau_cy``, (ny, nx)): ``friction`` from the cell state;
``surface`` with the central difference (eta_r - eta_l) / (2 dx) per axis,
or the D8 slope (the 4 sides as above, the diagonals from the cells two D4
steps away, skipped beyond an edge or a closed cell).

Faces (x faces ``tau_fxx``, ``tau_fxy`` (ny, nx + 1), y faces ``tau_fyx``,
``tau_fyy`` (ny + 1, nx), the x and y components, face layout of
``_kernels.py``), from the states L, R on both sides (cells, or the ghost
beyond a boundary face):

  - ``friction``: h_f = (h_L + h_R) / 2, discharges averaged, velocity
    q_f / h_f, D averaged over the friction coefficients of the two cells
    (the active one beyond a boundary face);
  - ``surface``: ``gradient``, normal slope (eta_R - eta_L) / dx,
    tangential slope the mean of the cell slopes of the wet active cells;
    ``d8``, the mean of the D8 slope vectors of the wet active cells; depth
    h_f.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder

from ._common import C_INFLOW, literal

SHEAR_METHODS = ("friction", "surface")
SURFACE_SLOPES = ("gradient", "d8")
CELL_OUT = ("tau_cx", "tau_cy")
FACE_OUT = ("tau_fxx", "tau_fxy", "tau_fyx", "tau_fyy")


def _slope_text(real, c):
    """Cell slope of eta along ``axis`` (0: +x, 1: +row)."""
    return f'''
__device__ {real} fusw_slope(const {real}* h, const {real}* u, const {real}* v,
                         const {real}* z, const unsigned char* code,
                         const unsigned char* fmask, int i, int axis,
                         {real} dx, {real} factor, {real} he) {{
    {real} eta = z[i] + h[i];
    {real} side[2];
    for (int s = 0; s < 2; ++s) {{
        int k = axis == 0 ? (s == 0 ? 1 : 2) : (s == 0 ? 0 : 3);
        {real} hs, un, ut, zs;
        $ctx.side(h, u, v, z, code, fmask, i, k, dx, factor, &hs, &un, &ut, &zs)$;
        side[s] = hs > he ? zs + hs : fmin(zs, eta);
    }}
    return (side[1] - side[0]) / ({c(2)} * dx);
}}
'''


def _d8_text(real, c):
    """D8 steepest-descent slope of eta as a gradient (gx, gy)."""
    return f'''
__device__ void fusw_slope_d8(const {real}* h, const {real}* u, const {real}* v,
                              const {real}* z, const unsigned char* code,
                              const unsigned char* fmask, int i,
                              {real} dx, {real} factor, {real} he,
                              {real}* gx, {real}* gy) {{
    {real} eta = z[i] + h[i];
    {real} best = {c(0)}, ox = {c(0)}, oy = {c(0)};
    const int kx[4] = {{0, -1, 1, 0}};
    const int ky[4] = {{-1, 0, 0, 1}};
    for (int k = 0; k < 4; ++k) {{
        {real} hs, un, ut, zs;
        $ctx.side(h, u, v, z, code, fmask, i, k, dx, factor, &hs, &un, &ut, &zs)$;
        {real} e = hs > he ? zs + hs : fmin(zs, eta);
        {real} s = (eta - e) / dx;
        if (s > best) {{ best = s; ox = ({real})kx[k]; oy = ({real})ky[k]; }}
    }}
    for (int a = 0; a < 4; a += 3) {{
        int m = $ctx.grid.neighbour(i, a)$;
        if (m < 0) continue;
        for (int b = 1; b < 3; ++b) {{
            int j = $ctx.grid.neighbour(m, b)$;
            if (j < 0) continue;
            {real} e = h[j] > he ? z[j] + h[j] : fmin(z[j], eta);
            {real} s = (eta - e) / (dx * {c(2 ** 0.5)});
            if (s > best) {{
                best = s; ox = ({real})kx[b] * {c(0.5 ** 0.5)}; oy = ({real})ky[a] * {c(0.5 ** 0.5)};
            }}
        }}
    }}
    *gx = -best * ox;
    *gy = -best * oy;
}}
'''


def face_state_text(real):
    """States L, R on both sides of the face between cells a and b (-1:
    the ghost beyond the boundary face of the other cell)."""
    return f'''
__device__ void fusw_face_state(const {real}* h, const {real}* u, const {real}* v,
                                const {real}* z, const unsigned char* code,
                                const unsigned char* fmask, int a, int b, int axis,
                                {real} dx, {real} factor,
                                {real}* hL, {real}* nL, {real}* tL, {real}* zL,
                                {real}* hR, {real}* nR, {real}* tR, {real}* zR) {{
    if (a >= 0) {{
        *hL = h[a]; *nL = axis == 0 ? u[a] : v[a]; *tL = axis == 0 ? v[a] : u[a]; *zL = z[a];
    }} else $ctx.side(h, u, v, z, code, fmask, b, axis == 0 ? 1 : 0, dx, factor, hL, nL, tL, zL)$;
    if (b >= 0) {{
        *hR = h[b]; *nR = axis == 0 ? u[b] : v[b]; *tR = axis == 0 ? v[b] : u[b]; *zR = z[b];
    }} else $ctx.side(h, u, v, z, code, fmask, a, axis == 0 ? 2 : 3, dx, factor, hR, nR, tR, zR)$;
}}
'''


def slope_texts(real, c, d8):
    """Cell (``fusw_cell_slope``: gx, gy) and face (``fusw_face_slope``:
    normal, tangential; false without a wet side) water-surface gradients
    by ``surface_slope`` (see the module docstring)."""
    if d8:
        cell = f'''
__device__ void fusw_cell_slope(const {real}* h, const {real}* u, const {real}* v,
                                const {real}* z, const unsigned char* code,
                                const unsigned char* fmask, int i,
                                {real} dx, {real} factor, {real} he, {real}* gx, {real}* gy) {{
    fusw_slope_d8(h, u, v, z, code, fmask, i, dx, factor, he, gx, gy);
}}
'''
        face = f'''
__device__ bool fusw_face_slope(const {real}* h, const {real}* u, const {real}* v,
                                const {real}* z, const unsigned char* code,
                                const unsigned char* fmask, int a, int b, int axis,
                                {real} hL, {real} zL, {real} hR, {real} zR,
                                {real} dx, {real} factor, {real} he, {real}* sn, {real}* st) {{
    (void)hL; (void)zL; (void)hR; (void)zR;
    *sn = {c(0)}; *st = {c(0)};
    int count = 0;
    for (int s = 0; s < 2; ++s) {{
        int j = s == 0 ? a : b;
        if (j < 0 || !(h[j] > he)) continue;
        {real} gx, gy;
        fusw_slope_d8(h, u, v, z, code, fmask, j, dx, factor, he, &gx, &gy);
        *sn += axis == 0 ? gx : gy;
        *st += axis == 0 ? gy : gx;
        ++count;
    }}
    if (count == 0) return false;
    *sn /= ({real})count;
    *st /= ({real})count;
    return true;
}}
'''
        return _d8_text(real, c) + cell + face
    cell = f'''
__device__ void fusw_cell_slope(const {real}* h, const {real}* u, const {real}* v,
                                const {real}* z, const unsigned char* code,
                                const unsigned char* fmask, int i,
                                {real} dx, {real} factor, {real} he, {real}* gx, {real}* gy) {{
    *gx = fusw_slope(h, u, v, z, code, fmask, i, 0, dx, factor, he);
    *gy = fusw_slope(h, u, v, z, code, fmask, i, 1, dx, factor, he);
}}
'''
    face = f'''
__device__ bool fusw_face_slope(const {real}* h, const {real}* u, const {real}* v,
                                const {real}* z, const unsigned char* code,
                                const unsigned char* fmask, int a, int b, int axis,
                                {real} hL, {real} zL, {real} hR, {real} zR,
                                {real} dx, {real} factor, {real} he, {real}* sn, {real}* st) {{
    *sn = {c(0)}; *st = {c(0)};
    bool wl = hL > he, wr = hR > he;
    if (!wl && !wr) return false;
    {real} el = zL + hL, er = zR + hR;
    if (!wl) el = fmin(zL, er);
    if (!wr) er = fmin(zR, el);
    int count = 0;
    if (a >= 0 && h[a] > he) {{
        *st += fusw_slope(h, u, v, z, code, fmask, a, 1 - axis, dx, factor, he); ++count; }}
    if (b >= 0 && h[b] > he) {{
        *st += fusw_slope(h, u, v, z, code, fmask, b, 1 - axis, dx, factor, he); ++count; }}
    if (count > 0) *st /= ({real})count;
    *sn = (er - el) / dx;
    return true;
}}
'''
    return _slope_text(real, c) + cell + face


def face_loop(config, c, call, x_out, y_out):
    """Kernel body visiting every face once (one thread per cell: its west
    and north faces, plus the east / south face of the last column / row of
    a non-periodic axis). ``call(a, b, axis)`` is the statement filling the
    locals O0, O1, ... written to ``x_out`` / ``y_out`` (faces with no active
    side: 0)."""
    nx, ny = int(config["nx"]), int(config["ny"])
    n = nx * ny
    px = config["boundary"] == "periodic_EW"
    py = config["boundary"] == "periodic_NS"

    def face(a, b, axis, index):
        names = x_out if axis == 0 else y_out
        stores = " ".join(f"{name}[{index}] = O{k};" for k, name in enumerate(names))
        return f"{call(a, b, axis)}\n            {stores}"

    def zero(axis, index):
        return " ".join(f"{name}[{index}] = {c(0)};" for name in (x_out if axis == 0 else y_out))

    west_raw = f"x > 0 ? i - 1 : {f'i + {nx - 1}' if px else '-1'}"
    north_raw = f"y > 0 ? i - {nx} : {f'i + {n - nx}' if py else '-1'}"
    east = (zero(0, "fxw + 1") if px else
            f"if (act) {{ {face('i', '-1', 0, 'fxw + 1')} }} else {{ {zero(0, 'fxw + 1')} }}")
    south = (zero(1, f"fyn + {nx}") if py else
             f"if (act) {{ {face('i', '-1', 1, f'fyn + {nx}')} }} else {{ {zero(1, f'fyn + {nx}')} }}")
    return f'''
    int x = i % {nx};
    int y = i / {nx};
    int fxw = y * {nx + 1} + x;
    int fyn = y * {nx} + x;
    bool act = $ctx.grid.is_active(i)$;
    int a = -1, b = -1;
    if (act) {{ b = i; a = $ctx.grid.neighbour(i, 1)$; }}
    else {{ int w = {west_raw}; if (w >= 0 && $ctx.grid.is_active(w)$) a = w; }}
    if (a >= 0 || b >= 0) {{ {face('a', 'b', 0, 'fxw')} }} else {{ {zero(0, 'fxw')} }}
    if (x == {nx - 1}) {{ {east} }}
    a = -1; b = -1;
    if (act) {{ b = i; a = $ctx.grid.neighbour(i, 0)$; }}
    else {{ int w = {north_raw}; if (w >= 0 && $ctx.grid.is_active(w)$) a = w; }}
    if (a >= 0 || b >= 0) {{ {face('a', 'b', 1, 'fyn')} }} else {{ {zero(1, 'fyn')} }}
    if (y == {ny - 1}) {{ {south} }}'''


def common_head(real, n):
    return f'''
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    {real} dx = ({real})$ctx.grid.DX.get(0)$;
    {real} he = ({real})$ctx.HE_CA.get(0)$;
    {real} rho = ({real})$ctx.RHO.get(0)$;
    {real} g = ({real})$ctx.GRAVITY.get(0)$;
    {real} factor = ({real})clock[{C_INFLOW}];
    (void)g; (void)factor; (void)dx;'''


ARGS = '''const {r}* h, const {r}* u, const {r}* v, const {r}* z,
        const unsigned char* boundary_type, const unsigned char* boundary_faces,
        const double* clock'''


def cell_kernel(bundles, config, helpers, real, method):
    """Per-cell shear stress kernel (see the module docstring)."""
    n = int(config["nx"]) * int(config["ny"])
    c = literal(real)
    if method == "friction":
        body = f'''
    {real} s = sqrt(u[i] * u[i] + v[i] * v[i]);
    {real} k = rho * hi * ($ctx.friction(hi, s, {c(1)}, i)$ - {c(1)});
    tau_cx[i] = k * u[i];
    tau_cy[i] = k * v[i];'''
    else:
        body = f'''
    {real} gx, gy;
    fusw_cell_slope(h, u, v, z, boundary_type, boundary_faces, i, dx, factor, he, &gx, &gy);
    tau_cx[i] = -rho * g * hi * gx;
    tau_cy[i] = -rho * g * hi * gy;'''
    text = (slope_texts(real, c, config["surface_slope"] == "d8") if method == "surface"
            else "") + f'''
extern "C" __global__ void fusw_tau_cell(
        {ARGS.format(r=real)}, {real}* tau_cx, {real}* tau_cy) {{{common_head(real, n)}
    {real} hi = h[i];
    if (!$ctx.grid.is_active(i)$ || hi <= he) {{ tau_cx[i] = {c(0)}; tau_cy[i] = {c(0)}; return; }}{body}
}}
'''
    kb = KernelBuilder(text, domain=n).compose("grid", bundles["grid"])
    kb = kb.compose("friction" if method == "friction" else "side",
                    helpers["friction" if method == "friction" else "side"])
    return kb.freeze()


def face_kernel(bundles, config, helpers, real, method):
    """Per-face shear stress kernel (see the module docstring)."""
    n = int(config["nx"]) * int(config["ny"])
    c = literal(real)
    if method == "friction":
        law = f'''
    {real} hf = {c(0.5)} * (hL + hR);
    if (hf <= he) return;
    {real} un = {c(0.5)} * (hL * nL + hR * nR) / hf;
    {real} ut = {c(0.5)} * (hL * tL + hR * tR) / hf;
    {real} s = sqrt(un * un + ut * ut);
    int ia = a >= 0 ? a : b, ib = b >= 0 ? b : a;
    {real} d = {c(0.5)} * ($ctx.friction(hf, s, {c(1)}, ia)$
                     + $ctx.friction(hf, s, {c(1)}, ib)$) - {c(1)};
    *tn = rho * hf * d * un;
    *tt = rho * hf * d * ut;'''
        slopes = ""
    else:
        law = f'''
    {real} sn, st;
    if (!fusw_face_slope(h, u, v, z, code, fmask, a, b, axis, hL, zL, hR, zR,
                         dx, factor, he, &sn, &st)) return;
    {real} k = -rho * g * {c(0.5)} * (hL + hR);
    *tn = k * sn;
    *tt = k * st;'''
        slopes = slope_texts(real, c, config["surface_slope"] == "d8")

    def call_xy(a, b, axis):
        # x faces store (normal, tangential) = (tau_x, tau_y); y faces (tau_y, tau_x).
        first, second = ("O0", "O1") if axis == 0 else ("O1", "O0")
        return (f"{real} O0, O1; fusw_tau_face(h, u, v, z, boundary_type, boundary_faces, "
                f"{a}, {b}, {axis}, dx, factor, he, rho, g, &{first}, &{second});")

    text = face_state_text(real) + slopes + f'''
__device__ void fusw_tau_face(const {real}* h, const {real}* u, const {real}* v,
                              const {real}* z, const unsigned char* code,
                              const unsigned char* fmask, int a, int b, int axis,
                              {real} dx, {real} factor, {real} he, {real} rho, {real} g,
                              {real}* tn, {real}* tt) {{
    (void)g; (void)rho;
    *tn = {c(0)}; *tt = {c(0)};
    {real} hL, nL, tL, zL, hR, nR, tR, zR;
    fusw_face_state(h, u, v, z, code, fmask, a, b, axis, dx, factor,
                    &hL, &nL, &tL, &zL, &hR, &nR, &tR, &zR);{law}
}}

extern "C" __global__ void fusw_tau_face_kernel(
        {ARGS.format(r=real)}, {real}* tau_fxx, {real}* tau_fxy,
        {real}* tau_fyx, {real}* tau_fyy) {{{common_head(real, n)}{face_loop(config, c, call_xy, ("tau_fxx", "tau_fxy"), ("tau_fyx", "tau_fyy"))}
}}
'''
    kb = KernelBuilder(text, domain=n).compose("grid", bundles["grid"]).compose("side", helpers["side"])
    if method == "friction":
        kb = kb.compose("friction", helpers["friction"])
    return kb.freeze()
