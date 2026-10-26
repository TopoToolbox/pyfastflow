"""Semi-implicit friction laws of the shallow-water update (FullSWOF).

``make_friction(law, real)`` returns a helper

    real fusw_friction(real h, real speed, real dt, int i)

giving the divisor D of the updated discharges, q <- q / D, from the new
depth h, the velocity norm ``speed`` of the stage input and the step dt,
with c = FRICTION(i) and g = GRAVITY:

  - ``none``: D = 1;
  - ``manning``: D = 1 + c^2 g speed dt / h^(4/3) (c: Manning n);
  - ``darcy_weisbach``: D = 1 + c speed dt / (8 h) (c: friction factor f);
  - ``laminar``: D = 1 + c dt / h^2.

D = 1 on a dry cell (h <= 0).

``make_uniform(law, real)`` returns a helper

    real fusw_uniform_q(real h, real slope, int i)

giving the unit discharge (m2/s) of uniform flow of depth h on the energy
slope ``slope`` under the same law (0 for h <= 0):

  - ``none``: critical flow, sqrt(g) h^(3/2) (slope unused);
  - ``manning``: h^(5/3) sqrt(slope) / c;
  - ``darcy_weisbach``: sqrt(8 g slope / c) h^(3/2);
  - ``laminar``: g slope h^3 / c.

Author: B.G (10/2026)
"""

from pyfastflow.core import FrozenHelper, HelperBuilder

from ._common import literal

FRICTIONS = ("none", "manning", "darcy_weisbach", "laminar")


def make_friction(law: str, real: str) -> FrozenHelper:
    if law not in FRICTIONS:
        raise ValueError(f"friction must be one of {FRICTIONS}, got {law!r}")
    c = literal(real)
    term = {
        "none": None,
        "manning": f"cf * cf * ({real})$ctx.GRAVITY.get(0)$ * speed * dt / pow(h, {c(4.0 / 3.0)})",
        "darcy_weisbach": f"cf * speed * dt / ({c(8)} * h)",
        "laminar": "cf * dt / (h * h)",
    }[law]
    if term is None:
        body = f"(void)h; (void)speed; (void)dt; (void)i; return {c(1)};"
    else:
        body = f'''if (!(h > {c(0)})) return {c(1)};
    {real} cf = ({real})$ctx.FRICTION.get(i)$;
    return {c(1)} + {term};'''
    return HelperBuilder(
        f"__device__ {real} fusw_friction({real} h, {real} speed, {real} dt, int i) {{\n    {body}\n}}\n"
    ).freeze()


def make_uniform(law: str, real: str) -> FrozenHelper:
    if law not in FRICTIONS:
        raise ValueError(f"friction must be one of {FRICTIONS}, got {law!r}")
    c = literal(real)
    g = f"({real})$ctx.GRAVITY.get(0)$"
    q = {
        "none": f"(void)slope; (void)i; return sqrt({g}) * h * sqrt(h);",
        "manning": f"return pow(h, {c(5.0 / 3.0)}) * sqrt(slope) / ({real})$ctx.FRICTION.get(i)$;",
        "darcy_weisbach": f"return sqrt({c(8)} * {g} * slope / ({real})$ctx.FRICTION.get(i)$) * h * sqrt(h);",
        "laminar": f"return {g} * slope * h * h * h / ({real})$ctx.FRICTION.get(i)$;",
    }[law]
    return HelperBuilder(
        f"__device__ {real} fusw_uniform_q({real} h, {real} slope, int i) {{\n"
        f"    if (!(h > {c(0)})) return {c(0)};\n    {q}\n}}\n"
    ).freeze()
