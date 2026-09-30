"""Hydraulic friction helpers for CuPy GraphFlood solvers."""

from ..core import FrozenHelper, HelperBuilder, new_uid

_MIN_SLOPE = 1.0e-5
_MIN_MANNING = 1.0e-9


def _velocity_manning(t: str) -> FrozenHelper:
    return HelperBuilder(
        f"""
__device__ float {t}_velocity_manning(float h, float slope, int node) {{
    float hh = h > 0.0f ? h : 0.0f;
    float ss = slope > {_MIN_SLOPE}f ? slope : {_MIN_SLOPE}f;
    float coeff = $ctx.MANNING.get(node)$;
    coeff = coeff > {_MIN_MANNING}f ? coeff : {_MIN_MANNING}f;
    return powf(hh, $ctx.EXPO.get(node)$) / coeff * sqrtf(ss);
}}
"""
    ).freeze()


_VELOCITY_LAWS = {"manning": _velocity_manning}


def build_friction_velocity(law: str) -> FrozenHelper:
    """Return a hydraulic velocity helper, leaving flow width to its caller."""
    if law not in _VELOCITY_LAWS:
        raise ValueError(
            f"build_friction_velocity: law must be one of "
            f"{sorted(_VELOCITY_LAWS)}, got {law!r}"
        )
    return _VELOCITY_LAWS[law](f"gfv{new_uid()}")

