"""Shared erosion-speed expression of the Salève link-slope kernels.

Every solver assigns each receiver link a slope ``uplift / speed``. The
speed has a fluvial part ``K * A**m`` and an opt-in hillslope part selected
by ``hillslope_model``:

- ``hack``: ``D / (c * A**h)`` — the diffusivity over a Hack-law hillslope
  length, so the slope is set by drainage area alone.
- ``divide_linear``: ``D / x`` with ``x`` the longest upstream flow path
  (distance from the divide). Steady linear diffusion, ``S = U x / D``,
  giving parabolic hilltops.
- ``divide_roering``: steady nonlinear diffusion ``U x = D S / (1 - (S/Sc)^2)``
  solved for ``S`` in closed form, with ``Sc`` the critical slope. Linear
  near the divide, saturating at ``Sc`` downslope.

``channel_area`` (``A_c``) selects how the two parts combine: ``0`` adds the
speeds everywhere; a positive value uses the hillslope speed alone below
``A_c`` and the fluvial speed alone at or above it.

Author: B.G (09/2026)
"""


def speed_source(config) -> str:
    """Return CUDA statements defining ``float speed`` for link ``i``.

    Requires ``i``, ``area``, ``dx``, ``uplift`` and ``divide_distance`` in
    scope and the ERODIBILITY, HILLSLOPE and CRITICAL parameter accessors.
    """
    m = float(config["m"])
    c = float(config["hack_constant"])
    h = float(config["hack_exponent"])
    model = config["hillslope_model"]
    channel_area = float(config["channel_area"])
    lines = [
        f"float speed_fluvial = fmaxf($ctx.ERODIBILITY.get(i)$, 1.0e-20f)"
        f" * powf(area, {m:.9e}f);",
    ]
    if model == "hack":
        lines.append(
            f"float speed_hill = $ctx.HILLSLOPE.get(i)$ / {c:.9e}f"
            f" * powf(area, -{h:.9e}f);")
    elif model == "divide_linear":
        lines.append(
            "float speed_hill = $ctx.HILLSLOPE.get(i)$"
            " / fmaxf(divide_distance[i], 0.5f * dx);")
    elif model == "divide_roering":
        lines += [
            "float speed_hill;",
            "{",
            "    float sc = $ctx.CRITICAL.get(0)$;",
            "    float x = fmaxf(divide_distance[i], 0.5f * dx);",
            "    float a = uplift * x / (fmaxf($ctx.HILLSLOPE.get(i)$, 1.0e-20f) * sc);",
            "    float s = 2.0f * a / (1.0f + sqrtf(1.0f + 4.0f * a * a));",
            "    speed_hill = uplift / fmaxf(sc * s, 1.0e-20f);",
            "}",
        ]
    else:
        raise ValueError(f"unknown hillslope_model {model!r}")
    if channel_area > 0.0:
        lines.append(
            f"float speed = fmaxf(area < {channel_area:.9e}f"
            " ? speed_hill : speed_fluvial, 1.0e-20f);")
    else:
        lines.append("float speed = fmaxf(speed_fluvial + speed_hill, 1.0e-20f);")
    return "\n    ".join(lines)


HILLSLOPE_MODELS = ("hack", "divide_linear", "divide_roering")
