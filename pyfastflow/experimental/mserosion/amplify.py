"""Host driver of the multi-scale erosion port: preset sequence and x2 upsampling.

``PRESET`` is ``PredefinedErosion`` of the release's ``main.cpp``: four stages
(256, 512, 1024, 2048 cells per side there), each ``(erosion, thermal,
deposition)`` step counts, with a x2 upsampling before every stage but the
first. Every stage starts its erosion stream and its deposition stream and
sediment at zero, as the release's ``Init`` calls do.

``upsample_x2`` is ``ScalarField2::SetResolution(2 nx, 2 ny)``: the same box
sampled on twice the vertices by bilinear interpolation (corners kept, so
the vertex spacing becomes ``dx * (n - 1) / (2 n - 1)``).

Author: B.G (10/2026)
"""

import cupy as cp

from .program import MultiScaleErosionProgram

#: (erosion, thermal, deposition) steps per stage, from the release's preset.
PRESET = ((3000, 600, 2000), (1500, 1000, 700), (700, 2000, 200), (400, 6000, 150))

_PARAMS = ("flow_p", "k", "p_sa", "p_sl", "dt", "max_spe", "eps", "tan_angle",
           "noisified", "noise_min", "noise_max", "noise_wavelength",
           "deposition_strength")


def _resample_axis(a, n_new, axis):
    """Linear resampling of ``a`` along ``axis`` from n to ``n_new`` vertices
    spanning the same extent."""
    n = a.shape[axis]
    t = cp.arange(n_new, dtype=cp.float64) * ((n - 1) / (n_new - 1))
    i0 = cp.clip(cp.floor(t).astype(cp.int64), 0, n - 2)
    w = t - i0
    lo = cp.take(a, i0, axis=axis)
    hi = cp.take(a, i0 + 1, axis=axis)
    shape = [1, 1]
    shape[axis] = n_new
    w = w.reshape(shape)
    return lo * (1.0 - w) + hi * w


def upsample_x2(z):
    """``z`` (ny, nx) bilinearly resampled on (2 ny, 2 nx) vertices of the same
    box, as float32 (computed in float64, like the release's doubles)."""
    z = cp.asarray(z, dtype=cp.float64)
    ny, nx = z.shape
    z = _resample_axis(z, 2 * nx, 1)
    z = _resample_axis(z, 2 * ny, 0)
    return z.astype(cp.float32)


def amplify(backend, z, dx, *, stages=1, schedule=PRESET, **params):
    """Run the multi-scale erosion preset on heightfield ``z`` (ny, nx) of
    vertex spacing ``dx``.

    ``stages`` (1 to ``len(schedule)``) runs that many stages of ``schedule``;
    each stage after the first doubles the resolution. ``params`` overrides
    the shader uniforms (names of ``MultiScaleErosionProgram``'s params).
    Returns the eroded heightfield (CuPy float32) and its vertex spacing.
    """
    unknown = set(params) - set(_PARAMS)
    if unknown:
        raise ValueError(f"unknown multi-scale erosion params {sorted(unknown)}")
    stages = int(stages)
    if not 1 <= stages <= len(schedule):
        raise ValueError(f"stages must be in [1, {len(schedule)}]")
    z = cp.asarray(z, dtype=cp.float32)
    dx = float(dx)
    for stage in range(stages):
        if stage:
            n = z.shape[1]
            z = upsample_x2(z)
            dx = dx * (n - 1) / (2 * n - 1)
        ny, nx = z.shape
        with MultiScaleErosionProgram(backend, nx=nx, ny=ny) as prog:
            prog.dx.set(dx)
            for name, value in params.items():
                getattr(prog, name).set(value)
            # Program data is flat: viewed as (ny, nx).
            buffers = {name: tuple(getattr(prog, f"{name}_{p}").array.reshape(ny, nx)
                                   for p in ("a", "b"))
                       for name in ("z", "stream", "sed")}
            buffers["z"][0][...] = z
            current = 0
            erosion, thermal, deposition = schedule[stage]
            for kind, steps in (("erosion", erosion), ("thermal", thermal),
                                ("deposition", deposition)):
                # Each step's Init: its stream (and sediment) start at zero.
                for name in ("stream", "sed"):
                    for buf in buffers[name]:
                        buf.fill(0.0)
                for _ in range(int(steps)):
                    src = "a" if current == 0 else "b"
                    if kind != "thermal":
                        getattr(prog, f"weights_{kind}_{src}")()
                    getattr(prog, f"{kind}_{src}_to_{'b' if current == 0 else 'a'}")()
                    current = 1 - current
            z = buffers["z"][current].copy()
    return z, dx
