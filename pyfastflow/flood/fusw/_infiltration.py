"""Infiltration laws of the shallow-water update (FullSWOF).

``make_infiltration(law, real)`` returns a helper

    real fusw_infiltration(real h, real infiltrated, real dt, int i)

giving the depth infiltrated over dt from a cell of depth h whose
cumulative infiltrated depth is ``infiltrated`` (at most h):

  - ``none``: 0;
  - ``green_ampt``: min(h, min(I_c, IMAX) dt) with the Green-Ampt capacity
    of the cumulative depth V = max(infiltrated, IE_CA),
      I_c = K + K (PSI + h) DTHETA / V,
    K = KS without crust (ZCRUST < EPSILON), K = KC while V <= ZCRUST
    DTHETA + IE_CA, else the series conductivity
    1 / ((1 - ZCRUST DTHETA / V) / KS + (ZCRUST DTHETA / V) / KC).

Author: B.G (10/2026)
"""

from pyfastflow.core import FrozenHelper, HelperBuilder

from ._common import literal

INFILTRATIONS = ("none", "green_ampt")


def make_infiltration(law: str, real: str) -> FrozenHelper:
    if law not in INFILTRATIONS:
        raise ValueError(f"infiltration must be one of {INFILTRATIONS}, got {law!r}")
    c = literal(real)
    if law == "none":
        body = f"(void)h; (void)infiltrated; (void)dt; (void)i; return {c(0)};"
    else:
        body = f'''{real} ks = ({real})$ctx.KS.get(i)$;
    {real} kc = ({real})$ctx.KC.get(i)$;
    {real} dtheta = ({real})$ctx.DTHETA.get(i)$;
    {real} psi = ({real})$ctx.PSI.get(i)$;
    {real} zcrust = ({real})$ctx.ZCRUST.get(i)$;
    {real} v = fmax(infiltrated, ({real})$ctx.IE_CA.get(0)$);
    {real} k;
    if (zcrust < ({real})$ctx.EPSILON.get(0)$) {{
        k = ks;
    }} else if (v <= zcrust * dtheta + ({real})$ctx.IE_CA.get(0)$) {{
        k = kc;
    }} else {{
        {real} f = zcrust * dtheta / v;
        k = {c(1)} / (({c(1)} - f) / ks + f / kc);
    }}
    {real} capacity = fmin(k + k * (psi + h) * dtheta / v, ({real})$ctx.IMAX.get(i)$);
    return fmin(h, capacity * dt);'''
    return HelperBuilder(
        f"__device__ {real} fusw_infiltration({real} h, {real} infiltrated, {real} dt, int i) {{\n    {body}\n}}\n"
    ).freeze()
