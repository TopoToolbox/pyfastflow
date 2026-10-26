"""Flow-regime diagnostics of the current state, on request (analysis
kernels): Froude number and acceleration ratio, per cell and per face.

With g = GRAVITY, dry below HE_CA:

  - Froude number Fr = |u| / sqrt(g h) (0 when dry). Cells: the cell state;
    faces (``fr_fx`` (ny, nx + 1), ``fr_fy`` (ny + 1, nx)): h_f = (h_L +
    h_R) / 2 and the averaged discharges of the states on both sides
    (``fusw_face_state``, ``_shear.py``: cells, or the ghost beyond a
    boundary face).
  - Acceleration ratio R = |du/dt + (u . grad) u| / (g |S_w|), S_w the
    water-surface gradient by ``surface_slope`` (``fusw_cell_slope`` /
    ``fusw_face_slope``, ``_shear.py``). du/dt = (u - u0) / dt from the
    velocities u0 a time dt earlier (``acc_u0``, ``acc_v0``, ``acc_dt``);
    the advective term by central differences over the D4 neighbour or
    ghost states. Cells: ``r_c``; faces: the mean acceleration of the wet
    active cells on both sides over the face slope. NaN when dry, with no
    wet side or with S_w = 0 (R is ill-conditioned as S_w -> 0); 0 on the
    faces with no active cell.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder

from ._common import literal
from ._shear import ARGS, common_head, face_loop, face_state_text, slope_texts

FROUDE_CELL = ("fr_c",)
FROUDE_FACE = ("fr_fx", "fr_fy")
ACC_IN = ("acc_u0", "acc_v0", "acc_dt")
ACC_CELL = ("r_c",)
ACC_FACE = ("r_fx", "r_fy")


def _nan(real):
    return ("__int_as_float(0x7fc00000)" if real == "float"
            else "__longlong_as_double(0x7ff8000000000000LL)")


def froude_cell_kernel(bundles, config, helpers, real):
    n = int(config["nx"]) * int(config["ny"])
    c = literal(real)
    text = f'''
extern "C" __global__ void fusw_froude_cell({ARGS.format(r=real)}, {real}* fr_c) {{{common_head(real, n)}
    (void)rho;
    {real} hi = h[i];
    fr_c[i] = (!$ctx.grid.is_active(i)$ || hi <= he) ? {c(0)}
        : sqrt(u[i] * u[i] + v[i] * v[i]) / sqrt(g * hi);
}}
'''
    return KernelBuilder(text, domain=n).compose("grid", bundles["grid"]).freeze()


def froude_face_kernel(bundles, config, helpers, real):
    n = int(config["nx"]) * int(config["ny"])
    c = literal(real)

    def call(a, b, axis):
        return (f"{real} O0 = fusw_froude_face(h, u, v, z, boundary_type, boundary_faces, "
                f"{a}, {b}, {axis}, dx, factor, he, g);")

    text = face_state_text(real) + f'''
__device__ {real} fusw_froude_face(const {real}* h, const {real}* u, const {real}* v,
                                  const {real}* z, const unsigned char* code,
                                  const unsigned char* fmask, int a, int b, int axis,
                                  {real} dx, {real} factor, {real} he, {real} g) {{
    {real} hL, nL, tL, zL, hR, nR, tR, zR;
    fusw_face_state(h, u, v, z, code, fmask, a, b, axis, dx, factor,
                    &hL, &nL, &tL, &zL, &hR, &nR, &tR, &zR);
    {real} hf = {c(0.5)} * (hL + hR);
    if (hf <= he) return {c(0)};
    {real} qn = {c(0.5)} * (hL * nL + hR * nR);
    {real} qt = {c(0.5)} * (hL * tL + hR * tR);
    return sqrt(qn * qn + qt * qt) / hf / sqrt(g * hf);
}}

extern "C" __global__ void fusw_froude_face_kernel({ARGS.format(r=real)}, {real}* fr_fx, {real}* fr_fy) {{{common_head(real, n)}
    (void)rho;{face_loop(config, c, call, ("fr_fx",), ("fr_fy",))}
}}
'''
    return (KernelBuilder(text, domain=n).compose("grid", bundles["grid"])
            .compose("side", helpers["side"]).freeze())


def _acceleration_text(real, c):
    """Cell acceleration du/dt + (u . grad) u (ax, ay)."""
    return f'''
__device__ void fusw_cell_acc(const {real}* h, const {real}* u, const {real}* v,
                              const {real}* z, const unsigned char* code,
                              const unsigned char* fmask, const {real}* u0,
                              const {real}* v0, {real} dtw, int i, {real} dx,
                              {real} factor, {real}* ax, {real}* ay) {{
    {real} us[4], vs[4];
    for (int k = 0; k < 4; ++k) {{
        {real} hs, un, ut, zs;
        $ctx.side(h, u, v, z, code, fmask, i, k, dx, factor, &hs, &un, &ut, &zs)$;
        bool x_axis = (k == 1 || k == 2);
        us[k] = x_axis ? un : ut;
        vs[k] = x_axis ? ut : un;
    }}
    {real} d = {c(0.5)} / dx;
    {real} dudx = (us[2] - us[1]) * d, dvdx = (vs[2] - vs[1]) * d;
    {real} dudy = (us[3] - us[0]) * d, dvdy = (vs[3] - vs[0]) * d;
    *ax = (u[i] - u0[i]) / dtw + u[i] * dudx + v[i] * dudy;
    *ay = (v[i] - v0[i]) / dtw + u[i] * dvdx + v[i] * dvdy;
}}
'''


_ACC_ARGS = "const {r}* acc_u0, const {r}* acc_v0, const {r}* acc_dt"


def acceleration_cell_kernel(bundles, config, helpers, real):
    n = int(config["nx"]) * int(config["ny"])
    c = literal(real)
    text = slope_texts(real, c, config["surface_slope"] == "d8") + _acceleration_text(real, c) + f'''
extern "C" __global__ void fusw_acc_cell({ARGS.format(r=real)}, {_ACC_ARGS.format(r=real)},
                                         {real}* r_c) {{{common_head(real, n)}
    (void)rho;
    if (!$ctx.grid.is_active(i)$ || h[i] <= he) {{ r_c[i] = {_nan(real)}; return; }}
    {real} ax, ay, gx, gy;
    fusw_cell_acc(h, u, v, z, boundary_type, boundary_faces, acc_u0, acc_v0, acc_dt[0],
                  i, dx, factor, &ax, &ay);
    fusw_cell_slope(h, u, v, z, boundary_type, boundary_faces, i, dx, factor, he, &gx, &gy);
    {real} s = sqrt(gx * gx + gy * gy);
    r_c[i] = s > {c(0)} ? sqrt(ax * ax + ay * ay) / (g * s) : {_nan(real)};
}}
'''
    return (KernelBuilder(text, domain=n).compose("grid", bundles["grid"])
            .compose("side", helpers["side"]).freeze())


def acceleration_face_kernel(bundles, config, helpers, real):
    n = int(config["nx"]) * int(config["ny"])
    c = literal(real)

    def call(a, b, axis):
        return (f"{real} O0 = fusw_acc_face(h, u, v, z, boundary_type, boundary_faces, "
                f"acc_u0, acc_v0, acc_dt[0], {a}, {b}, {axis}, dx, factor, he, g);")

    text = (face_state_text(real) + slope_texts(real, c, config["surface_slope"] == "d8")
            + _acceleration_text(real, c) + f'''
__device__ {real} fusw_acc_face(const {real}* h, const {real}* u, const {real}* v,
                               const {real}* z, const unsigned char* code,
                               const unsigned char* fmask, const {real}* u0,
                               const {real}* v0, {real} dtw, int a, int b, int axis,
                               {real} dx, {real} factor, {real} he, {real} g) {{
    {real} hL, nL, tL, zL, hR, nR, tR, zR;
    fusw_face_state(h, u, v, z, code, fmask, a, b, axis, dx, factor,
                    &hL, &nL, &tL, &zL, &hR, &nR, &tR, &zR);
    {real} sn, st;
    if (!fusw_face_slope(h, u, v, z, code, fmask, a, b, axis, hL, zL, hR, zR,
                         dx, factor, he, &sn, &st)) return {_nan(real)};
    {real} ax = {c(0)}, ay = {c(0)};
    int count = 0;
    for (int s = 0; s < 2; ++s) {{
        int j = s == 0 ? a : b;
        if (j < 0 || !(h[j] > he)) continue;
        {real} jx, jy;
        fusw_cell_acc(h, u, v, z, code, fmask, u0, v0, dtw, j, dx, factor, &jx, &jy);
        ax += jx; ay += jy; ++count;
    }}
    {real} s = sqrt(sn * sn + st * st);
    if (count == 0 || !(s > {c(0)})) return {_nan(real)};
    return sqrt(ax * ax + ay * ay) / ({real})count / (g * s);
}}

extern "C" __global__ void fusw_acc_face_kernel({ARGS.format(r=real)}, {_ACC_ARGS.format(r=real)},
                                         {real}* r_fx, {real}* r_fy) {{{common_head(real, n)}
    (void)rho;{face_loop(config, c, call, ("r_fx",), ("r_fy",))}
}}
''')
    return (KernelBuilder(text, domain=n).compose("grid", bundles["grid"])
            .compose("side", helpers["side"]).freeze())
