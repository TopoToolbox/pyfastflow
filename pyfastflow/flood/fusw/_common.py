"""Shared constants and CUDA text utilities of FuswProgram.

Precision: every state, flux and reconstruction value is ``real``, the CUDA
type picked by the ``precision`` config (``f32`` -> float, ``f64`` ->
double). Parameters stay f32 and are cast on read; time, budget and
time-factor tables are always double.

Boundary codes (u8 field ``boundary_type``, read on every boundary face of
an active cell, i.e. a D4 face whose neighbour lies outside a non-periodic
edge or is a closed cell):

  0 wall, 1 neumann, 2 stage, 3 critical, 4 normal_depth, 5 rating,
  6 discharge, 7 sink (the cell absorbs its water; its faces act as walls),
  8 inflow (face of an inflow line: imposed mass flux, ``_boundary.py``).

Device clock (f64): time, dt of the current step, end time (<= 0: none),
the rain, inflow and source time factors at the current time, the rain
and source rates of the active cells at factor 1 (sum of RAIN dx^2 and of
SOURCE, m3/s, evaluated by ``_prepare`` at every ``run``) and the wave
speed growth predicted for the first stage (order 2 with a CFL time step:
the ratio of the intermediate to the step-start wave speed of the last
step; below 1 read as 1).

Device control (u32): the largest wave speed of the step-start and
intermediate fluxes (float bits), the redo flag, redos of the current step,
the CFL error flag (fixed time step), the step counter, the recording base,
the total number of redos and the recording overflow flag.

Budget (f64, volumes in m3, momentum in m4/s, all cumulative):

  rain, source, infiltration, sink, threshold (volume removed by the HE_CA
  cut; negative when a negative depth is cut to 0), then for each boundary
  code c at 5 + 4c: volume out, outward normal momentum out, tangential
  momentum out, bed-pressure momentum out (outward normal momentum lost
  through the face is the sum of the last-but-two and the last entries).

Author: B.G (10/2026)
"""

from pyfastflow.core import Backend

PRECISIONS = {"f32": "float", "f64": "double"}

BOUNDARY_CODES = {
    "wall": 0, "neumann": 1, "stage": 2, "critical": 3,
    "normal_depth": 4, "rating": 5, "discharge": 6, "sink": 7, "inflow": 8,
}
N_CODES = len(BOUNDARY_CODES)
OUTLET_TYPES = ("neumann", "stage", "critical", "normal_depth", "rating", "sink")

# clock (f64)
(C_TIME, C_DT, C_T_END, C_RAIN, C_INFLOW, C_SOURCE, C_RAIN_RATE, C_SOURCE_RATE,
 C_GROWTH) = range(9)
N_CLOCK = 9

# control (u32)
(K_CMAX0, K_CMAX1, K_REDO, K_RESTARTS, K_CFL_ERROR, K_STEP, K_REC_BASE,
 K_TOTAL_RESTARTS, K_REC_OVERFLOW) = range(9)
N_CONTROL = 9

# budget (f64)
B_RAIN, B_SOURCE, B_INFILTRATION, B_SINK, B_THRESHOLD = range(5)
B_FACE = 5
N_FACE_TERMS = 4
N_BUDGET = B_FACE + N_FACE_TERMS * N_CODES
FACE_TERMS = ("volume", "normal_momentum", "tangential_momentum", "pressure")

SERIES = ("rain", "inflow", "source")


def cupy_only(be: Backend) -> None:
    if be.name != "cupy":
        raise ValueError(f"FuswProgram is cupy-only, got {be.name!r}")


def literal(real: str):
    """Return the function writing a number as a ``real`` literal."""
    if real == "float":
        return lambda x: f"{float(x)!r}f"
    return lambda x: f"{float(x)!r}"


def warp_reduce(real: str) -> str:
    """Warp sum of doubles and warp max of ``real``, full-warp only."""
    return f'''
__device__ __forceinline__ double fusw_warp_sum(double v) {{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}}
__device__ __forceinline__ {real} fusw_warp_max({real} v) {{
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}}
'''
