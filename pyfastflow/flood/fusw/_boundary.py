"""Ghost states of the boundary faces (FullSWOF boundary conditions plus
three outlet laws).

``make_ghost(real, inflow_unit)`` returns a helper

    void fusw_ghost(int code, int i, int sgn, real hin, real un, real ut,
                    real zc, real zopp, real dx, real factor, int faces,
                    real* hb, real* unb, real* utb, real* zb)

giving the ghost state (depth hb, normal and tangential velocities unb and
utb, bed zb) beyond a boundary face of the active cell i, from the cell
state (hin, un, ut), its bed zc and the bed zopp of its neighbour opposite
the face (zc without one). sgn is the outward direction of the face along
its axis (+1 east or south, -1 west or north), ``faces`` the number of
boundary faces of the cell and ``factor`` the inflow time factor.

The bed beyond the face is zc for walls and sinks and the linear
extrapolation 2 zc - zopp otherwise. The imposed depth is
hfix = max(STAGE(i) - zb, 0) and the imposed unit discharge, positive into
the domain, is q = factor INFLOW(i) with ``inflow_unit="unit"`` (m2/s per
face) or factor INFLOW(i) / (faces dx) with ``inflow_unit="discharge"``
(m3/s per cell, split over its boundary faces); qfix = -sgn q is its
signed value along the axis. With g = GRAVITY, codes (_common.py):

  - wall, sink: (hin, -un, ut);
  - neumann: (hin, un, ut);
  - stage (FullSWOF imposed height): subcritical cell, hb = hfix and
    unb = un + 2 sgn sqrt(g) (sqrt(hin) - sqrt(hfix)); if the ghost is then
    supercritical, inflow takes unb = qfix / hfix and outflow the cell
    state. Supercritical cell: inflow (hfix, qfix / hfix, 0), outflow the
    cell state;
  - discharge (FullSWOF imposed discharge): subcritical cell, hb solves
    2 sqrt(g h) h - sgn (un + 2 sgn sqrt(g hin)) h = |qfix| by Newton from
    2 (|qfix| / sqrt(g))^(2/3) + 1 (tolerance 0.1 HE_CA, at most
    NEWTON_ITERATIONS iterations), unb = qfix / hb (dry below HE_CA).
    Supercritical cell: (hfix, qfix / hfix, 0);
  - critical (free outfall): hb = 2 E / 3 with E = hin + un^2 / (2 g),
    unb = sgn sqrt(g hb), utb = ut;
  - normal_depth (Manning uniform flow): hb = hin,
    unb = sgn hin^(2/3) sqrt(S) / OUTLET_MANNING(i), S = OUTLET_SLOPE(i)
    when positive, else the bed slope max(zopp - zc, 0) / dx;
  - rating: hb = hin, unb = sgn q / hin with
    q = RATING_A(i) (zc + hin - max(RATING_ETA0(i), zc))^RATING_B(i) when
    positive, else 0 (m2/s): the stage counts from the sill RATING_ETA0 or
    the cell bed, whichever is higher, so q vanishes with the water held.

  - inflow: (hin, un, ut) on the bed zc; the face flux itself is imposed
    by ``fusw_inflow_face`` and this ghost only enters the reconstruction.

Imposed velocities vanish below HE_CA.

``make_inflow_face(real)`` returns a helper

    void fusw_inflow_face(real hin, real un, int sgn, real q, real hn,
                          real hc, real uc, real* f1, real* fn, real* hb,
                          real* c)

giving the fluxes through an inflow face (code inflow) of a cell of depth
hin and normal velocity un, sgn its outward direction, q >= 0 the unit
discharge entering through it (m2/s) and hn the normal depth of its line
there (``_kernels.py``): mass flux f1 = -sgn q exactly, normal momentum
flux fn = f1 ub + g hb^2 / 2 with ub = f1 / hb (0 when hb <= HE_CA), the
face depth hb and the wave speed c, the larger of |ub| + sqrt(g hb) and
|uc| + sqrt(g hc) for the cell-side face state (hc, uc). hb solves the
imposed-discharge relation of the discharge code (Newton as above) when the
cell is subcritical, else hb = hn, at least the critical depth
(q^2 / g)^(1/3); with q = 0 and a still cell hb = hin, so a closed inflow
face holds still water.

Author: B.G (10/2026)
"""

from pyfastflow.core import FrozenHelper, HelperBuilder

from ._common import BOUNDARY_CODES as B
from ._common import literal

INFLOW_UNITS = ("discharge", "unit")


def make_ghost(real: str, inflow_unit: str) -> FrozenHelper:
    if inflow_unit not in INFLOW_UNITS:
        raise ValueError(f"inflow_unit must be one of {INFLOW_UNITS}, got {inflow_unit!r}")
    c = literal(real)
    q_in = (f"factor * ({real})$ctx.INFLOW.get(i)$" if inflow_unit == "unit"
            else f"factor * ({real})$ctx.INFLOW.get(i)$ / (({real})faces * dx)")
    return HelperBuilder(f'''
__device__ void fusw_ghost(int code, int i, int sgn, {real} hin, {real} un, {real} ut,
                           {real} zc, {real} zopp, {real} dx, {real} factor, int faces,
                           {real}* hb, {real}* unb, {real}* utb, {real}* zb) {{
    {real} g = ({real})$ctx.GRAVITY.get(0)$;
    {real} he = ({real})$ctx.HE_CA.get(0)$;
    {real} s = ({real})sgn;
    *zb = (code == {B["wall"]} || code == {B["sink"]} || code == {B["inflow"]})
        ? zc : {c(2)} * zc - zopp;
    *hb = hin; *unb = un; *utb = ut;
    if (code == {B["wall"]} || code == {B["sink"]}) {{
        *unb = -un;
    }} else if (code == {B["stage"]} || code == {B["discharge"]}) {{
        {real} hfix = fmax(({real})$ctx.STAGE.get(i)$ - *zb, {c(0)});
        {real} qfix = -s * ({q_in});
        {real} ufix = hfix < he ? {c(0)} : qfix / hfix;
        if (hfix < he) hfix = {c(0)};
        bool sub = fabs(un) - sqrt(g * hin) <= {c(0)};
        bool inflow = s * un <= {c(0)};
        if (code == {B["stage"]}) {{
            if (sub) {{
                *hb = hfix;
                *unb = un + s * {c(2)} * sqrt(g) * (sqrt(hin) - sqrt(hfix));
                *utb = {c(0)};
                if (fabs(*unb) - sqrt(g * hfix) > {c(0)}) {{
                    if (inflow) {{ *unb = ufix; *utb = {c(0)}; }}
                    else {{ *hb = hin; *unb = un; *utb = ut; }}
                }}
            }} else if (inflow) {{
                *hb = hfix; *unb = ufix; *utb = {c(0)};
            }}
        }} else if (sub) {{
            {real} a = s * (un + s * {c(2)} * sqrt(g * hin));
            {real} q = fabs(qfix);
            {real} h = {c(2)} * pow(q / sqrt(g), {c(2.0 / 3.0)}) + {c(1)};
            {real} f = {c(2)} * sqrt(g * h) * h - a * h - q;
            {real} tol = {c(0.1)} * he;
            int iterations = $ctx.NEWTON_ITERATIONS.get(0)$;
            for (int it = 0; it < iterations && fabs(f) > tol; ++it) {{
                h -= f / ({c(3)} * sqrt(g * h) - a);
                f = {c(2)} * sqrt(g * h) * h - a * h - q;
            }}
            if (h <= he) {{ *hb = {c(0)}; *unb = {c(0)}; *utb = {c(0)}; }}
            else {{ *hb = h; *unb = qfix / h; *utb = {c(0)}; }}
        }} else {{
            *hb = hfix; *unb = ufix; *utb = {c(0)};
        }}
    }} else if (code == {B["critical"]}) {{
        if (hin <= he) {{ *hb = {c(0)}; *unb = {c(0)}; *utb = {c(0)}; }}
        else {{
            *hb = {c(2.0 / 3.0)} * (hin + un * un / ({c(2)} * g));
            *unb = s * sqrt(g * *hb);
        }}
    }} else if (code == {B["normal_depth"]}) {{
        {real} slope = ({real})$ctx.OUTLET_SLOPE.get(i)$;
        if (!(slope > {c(0)})) slope = fmax(zopp - zc, {c(0)}) / dx;
        *unb = hin <= he ? {c(0)}
            : s * pow(hin, {c(2.0 / 3.0)}) * sqrt(slope) / ({real})$ctx.OUTLET_MANNING.get(i)$;
    }} else if (code == {B["rating"]}) {{
        {real} rise = zc + hin - fmax(({real})$ctx.RATING_ETA0.get(i)$, zc);
        {real} q = rise > {c(0)}
            ? ({real})$ctx.RATING_A.get(i)$ * pow(rise, ({real})$ctx.RATING_B.get(i)$) : {c(0)};
        *unb = hin <= he ? {c(0)} : s * q / hin;
    }}
}}
''').freeze()


def make_inflow_face(real: str) -> FrozenHelper:
    c = literal(real)
    return HelperBuilder(f'''
__device__ void fusw_inflow_face({real} hin, {real} un, int sgn, {real} q, {real} hn,
                                 {real} hc, {real} uc,
                                 {real}* f1, {real}* fn, {real}* hb, {real}* cw) {{
    {real} g = ({real})$ctx.GRAVITY.get(0)$;
    {real} he = ({real})$ctx.HE_CA.get(0)$;
    {real} s = ({real})sgn;
    {real} h;
    if (fabs(un) - sqrt(g * hin) <= {c(0)}) {{
        {real} a = s * (un + s * {c(2)} * sqrt(g * hin));
        h = {c(2)} * pow(q / sqrt(g), {c(2.0 / 3.0)}) + {c(1)};
        {real} f = {c(2)} * sqrt(g * h) * h - a * h - q;
        {real} tol = {c(0.1)} * he;
        int iterations = $ctx.NEWTON_ITERATIONS.get(0)$;
        for (int it = 0; it < iterations && fabs(f) > tol; ++it) {{
            h -= f / ({c(3)} * sqrt(g * h) - a);
            f = {c(2)} * sqrt(g * h) * h - a * h - q;
        }}
    }} else {{
        h = hn;
    }}
    h = fmax(h, pow(q * q / g, {c(1.0 / 3.0)}));
    if (!(h > he)) h = {c(0)};
    {real} flux = -s * q;
    {real} ub = h > he ? flux / h : {c(0)};
    *f1 = flux;
    *fn = flux * ub + {c(0.5)} * g * h * h;
    *hb = h;
    *cw = fmax(fabs(ub) + sqrt(g * h), fabs(uc) + sqrt(g * fmax(hc, {c(0)})));
}}
''').freeze()
