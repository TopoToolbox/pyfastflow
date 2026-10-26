"""Numerical fluxes of the shallow-water equations (FullSWOF).

``make_flux(name, real)`` returns a helper

    void fusw_flux(real hL, real uL, real vL, real hR, real uR, real vR,
                   real* f1, real* f2, real* f3, real* cfl)

giving, along the face normal, the mass flux f1, the normal momentum flux
f2 and the tangential momentum flux f3 between the left state (hL, uL, vL)
and the right state (hR, uR, vR) (u normal, v tangential velocity), and the
wave speed ``cfl`` setting the time step. Both sides dry (h < HE_CA) give
zero fluxes. With c1 = min(uL - sqrt(g hL), uR - sqrt(g hR)) and
c2 = max(uL + sqrt(g hL), uR + sqrt(g hR)):

  - ``rusanov``: centred flux with the dissipation
    c = max(|uL| + sqrt(g hL), |uR| + sqrt(g hR)) (Bouchut 2004);
  - ``hll``: Harten-Lax-van Leer, upwind outside c1 < 0 < c2;
  - ``hll2``: the same HLL in the reduced t1 / t2 / t3 form (Batten 1997);
  - ``hllc``: HLL with the tangential flux f1 vL or f1 vR from the sign of
    the contact speed c*, and dry-side speeds uR - 2 sqrt(g hR) /
    uL + 2 sqrt(g hL) (Toro 2001);
  - ``hllc2``: ``hllc`` in the reduced form.

EPSILON is the speed below which c1 and c2 count as zero. g = GRAVITY.

Author: B.G (10/2026)
"""

from pyfastflow.core import FrozenHelper, HelperBuilder

from ._common import literal

FLUXES = ("rusanov", "hll", "hll2", "hllc", "hllc2")


def _speeds(c, real: str, dry_corrected: bool) -> str:
    if not dry_corrected:
        return f'''
    {real} c1 = fmin(uL - sqrt(g * hL), uR - sqrt(g * hR));
    {real} c2 = fmax(uL + sqrt(g * hL), uR + sqrt(g * hR));'''
    return f'''
    {real} c1 = hL < he ? uR - {c(2)} * sqrt(g * hR) : fmin(uL - sqrt(g * hL), uR - sqrt(g * hR));
    {real} c2 = hR < he ? uL + {c(2)} * sqrt(g * hL) : fmax(uL + sqrt(g * hL), uR + sqrt(g * hR));'''


def _branched(c, real: str, contact: bool) -> str:
    """HLL with the dry / left-supersonic / right-supersonic branches."""
    tangential = (
        f'''{real} c_star = (c1 * hR * (uR - c2) - c2 * hL * (uL - c1))
            / (hR * (uR - c2) - hL * (uL - c1));
        *f3 = c_star > eps ? *f1 * vL : *f1 * vR;'''
        if contact else
        f"*f3 = (c2 * (qL * vL) - c1 * (qR * vR)) * t + c1 * c2 * (hR * vR - hL * vL) * t;"
    )
    return f'''
    if (fabs(c1) < eps && fabs(c2) < eps) {{
        *f1 = {c(0)}; *f2 = {c(0)}; *f3 = {c(0)}; *cfl = {c(0)};
    }} else if (c1 >= eps) {{
        *f1 = qL; *f2 = qL * uL + g * hL * hL * {c(0.5)}; *f3 = qL * vL; *cfl = c2;
    }} else if (c2 <= -eps) {{
        *f1 = qR; *f2 = qR * uR + g * hR * hR * {c(0.5)}; *f3 = qR * vR; *cfl = fabs(c1);
    }} else {{
        {real} t = {c(1)} / (c2 - c1);
        *f1 = (c2 * qL - c1 * qR) * t + c1 * c2 * (hR - hL) * t;
        *f2 = (c2 * (qL * uL + g * hL * hL * {c(0.5)}) - c1 * (qR * uR + g * hR * hR * {c(0.5)})) * t
            + c1 * c2 * (qR - qL) * t;
        {tangential}
        *cfl = fmax(fabs(c1), fabs(c2));
    }}'''


def _reduced(c, real: str, contact: bool) -> str:
    """HLL in the t1 / t2 / t3 form."""
    tangential = (
        f'''{real} c_star = (c1 * hR * (uR - c2) - c2 * hL * (uL - c1))
        / (hR * (uR - c2) - hL * (uL - c1));
    *f3 = c_star > eps ? *f1 * vL : *f1 * vR;'''
        if contact else
        "*f3 = t1 * qR * vR + t2 * qL * vL - t3 * (hR * vR - hL * vL);"
    )
    return f'''
    {real} t = {c(1)} / (c2 - c1);
    {real} t1 = (fmin(c2, {c(0)}) - fmin(c1, {c(0)})) * t;
    {real} t2 = {c(1)} - t1;
    {real} t3 = (c2 * fabs(c1) - c1 * fabs(c2)) * {c(0.5)} * t;
    *f1 = t1 * qR + t2 * qL - t3 * (hR - hL);
    *f2 = t1 * (qR * uR + g * hR * hR * {c(0.5)}) + t2 * (qL * uL + g * hL * hL * {c(0.5)})
        - t3 * (qR - qL);
    {tangential}
    *cfl = fmax(fabs(c1), fabs(c2));'''


def make_flux(name: str, real: str) -> FrozenHelper:
    if name not in FLUXES:
        raise ValueError(f"flux must be one of {FLUXES}, got {name!r}")
    c = literal(real)
    if name == "rusanov":
        body = f'''
    {real} s = fmax(fabs(uL) + sqrt(g * hL), fabs(uR) + sqrt(g * hR));
    {real} sd = s * {c(0.5)};
    *f1 = (qL + qR) * {c(0.5)} - sd * (hR - hL);
    *f2 = (uL * qL + {c(0.5)} * g * hL * hL + uR * qR + {c(0.5)} * g * hR * hR) * {c(0.5)}
        - sd * (qR - qL);
    *f3 = (qL * vL + qR * vR) * {c(0.5)} - sd * (hR * vR - hL * vL);
    *cfl = s;'''
    elif name == "hll":
        body = _speeds(c, real, False) + _branched(c, real, False)
    elif name == "hll2":
        body = _speeds(c, real, False) + _reduced(c, real, False)
    elif name == "hllc":
        body = _speeds(c, real, True) + _branched(c, real, True)
    else:
        body = _speeds(c, real, True) + _reduced(c, real, True)
    return HelperBuilder(f'''
__device__ void fusw_flux({real} hL, {real} uL, {real} vL, {real} hR, {real} uR, {real} vR,
                          {real}* f1, {real}* f2, {real}* f3, {real}* cfl) {{
    {real} g = ({real})$ctx.GRAVITY.get(0)$;
    {real} he = ({real})$ctx.HE_CA.get(0)$;
    {real} eps = ({real})$ctx.EPSILON.get(0)$;
    if (hL < he && hR < he) {{
        *f1 = {c(0)}; *f2 = {c(0)}; *f3 = {c(0)}; *cfl = {c(0)};
        return;
    }}
    {real} qL = uL * hL;
    {real} qR = uR * hR;
    (void)eps;{body}
}}
''').freeze()
