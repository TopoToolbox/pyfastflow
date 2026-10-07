"""Shared erosion-speed expression of the Salève link-slope kernels.

Every solver assigns each receiver link a slope ``uplift / speed``. The
speed is the term ``a`` of Tzathas et al. (2024), Eqn. 26: the fluvial part
``K * A**m`` plus the hillslope part ``D / C * A**-h``, the diffusivity over
a Hack-law hillslope length ``C * A**h`` (the steady state of linear
hillslope diffusion near the ridge, Section 6.1). ``D = 0`` leaves the plain
stream power law.

Author: B.G (09/2026)
"""


def speed_source(config) -> str:
    """Return CUDA statements defining ``float speed`` for link ``i``.

    Requires ``i`` and ``area`` in scope and the ERODIBILITY and HILLSLOPE
    parameter accessors.
    """
    m = float(config["m"])
    c = float(config["hack_constant"])
    h = float(config["hack_exponent"])
    return "\n    ".join([
        f"float speed = fmaxf($ctx.ERODIBILITY.get(i)$, 1.0e-20f)"
        f" * powf(area, {m:.9e}f)",
        f"    + $ctx.HILLSLOPE.get(i)$ / {c:.9e}f * powf(area, -{h:.9e}f);",
        "speed = fmaxf(speed, 1.0e-20f);",
    ])
