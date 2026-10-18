"""Slope limiters of the MUSCL reconstruction (FullSWOF).

``make_limiter(name, real)`` returns a helper
``real fusw_limiter(real a, real b)`` limiting the two one-sided differences
``a`` and ``b``:

  - ``minmod``: min(a, b) if both >= 0, max(a, b) if both <= 0, else 0;
  - ``vanalbada``: 0 if a b < 0, else
    (a (b^2 + eps) + b (a^2 + eps)) / (a^2 + b^2 + 2 eps), eps = EPSILON,
    written with explicit fma so that its rounding does not depend on the
    kernel it is inlined in (``flux`` and ``stage`` reconstruct the same
    cells and must agree to the bit);
  - ``vanleer``: 2 a b / (a + b) if a and b have the same strict sign,
    else 0.

Author: B.G (10/2026)
"""

from pyfastflow.core import FrozenHelper, HelperBuilder

from ._common import literal

LIMITERS = ("minmod", "vanalbada", "vanleer")


def make_limiter(name: str, real: str) -> FrozenHelper:
    if name not in LIMITERS:
        raise ValueError(f"limiter must be one of {LIMITERS}, got {name!r}")
    c = literal(real)
    if name == "minmod":
        body = f'''
    if (a >= {c(0)} && b >= {c(0)}) return fmin(a, b);
    if (a <= {c(0)} && b <= {c(0)}) return fmax(a, b);
    return {c(0)};'''
    elif name == "vanalbada":
        body = f'''
    if (a * b < {c(0)}) return {c(0)};
    {real} eps = ({real})$ctx.EPSILON.get(0)$;
    return fma(a, fma(b, b, eps), b * fma(a, a, eps)) / fma(a, a, fma(b, b, {c(2)} * eps));'''
    else:
        body = f'''
    if ((a > {c(0)} && b > {c(0)}) || (a < {c(0)} && b < {c(0)})) return {c(2)} * a * b / (a + b);
    return {c(0)};'''
    return HelperBuilder(
        f"__device__ {real} fusw_limiter({real} a, {real} b) {{{body}\n}}\n"
    ).freeze()
