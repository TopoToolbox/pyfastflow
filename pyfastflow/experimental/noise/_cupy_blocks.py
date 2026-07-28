"""
cupy (CUDA source) block templates behind make_noise.

Mirrors _closure_blocks.py block for block - same private/public split, same
`kind` selector deciding which chain `at(i)` is wired to - written as CUDA
text instead of python defs, since that is what CupyHelperBuilder compiles
(see cupy_backend.py's module docstring for the `$...$` span mechanism).

The arithmetic is the same port of pyfastflow/noise/white_noise.py and
perlin_noise.py the closure blocks carry, so the two backends agree bit for
bit on the white-noise hash and octave for octave on Perlin.

Every device function name is prefixed with this noise bag's own tag (a fresh
new_uid()), so two make_noise() calls in one process never collide inside a
single compiled cupy module even if both are bound into the same kernel.

Perlin's permutation lookups go through a named local (`int ia = A & 255;`)
before the span rather than inlining the expression into `$PERM.get(...)$` -
spans are resolved as text, and keeping their argument a bare identifier
keeps that resolution unambiguous.

Author: B.G (07/2026)
"""

import functools

from ..core.context.backends import make_helper
from ..core.pool.base import new_uid


def build_helpers(
    HelperCls,
    *,
    grid,
    kind,
    amplitude_p,
    seed_p,
    perm_p,
    frequency_x_p,
    frequency_y_p,
    octaves_p,
    persistence_p,
    backend_mod=None,
):
    """
    Wire one noise bag's private blocks and its public `at` for the cupy
    backend, picking the white or Perlin chain from `kind` and binding
    private blocks into public ones by name (a bound name resolves to the
    real emitted C symbol at span-expansion time - see cupy_backend.py's
    _SpanParser).

    Returns {public_name: HelperBuilder}, meant to be merged straight into
    the Bag make_noise() returns. `backend_mod` is accepted for signature
    parity with the closure backend's build_helpers and unused here - cupy
    templates call plain C (floorf, casts) rather than a bound backend
    module.

    Author: B.G (07/2026)
    """
    t = f"pn{new_uid()}"

    mk = functools.partial(make_helper, HelperCls)

    row = mk(f"__device__ int {t}_row(int i) {{ return i / $GRID.nx.get(0)$; }}", GRID=grid)
    col = mk(f"__device__ int {t}_col(int i) {{ return i % $GRID.nx.get(0)$; }}", GRID=grid)

    if kind == "white":
        hash_u32 = mk(
            f"""
__device__ unsigned int {t}_hash_u32(unsigned int x) {{
    unsigned int h = x;
    h ^= h >> 16u;
    h *= 0x7FEB352Du;
    h ^= h >> 15u;
    h *= 0x846CA68Bu;
    h ^= h >> 16u;
    return h;
}}
"""
        )
        white_unit = mk(
            f"""
__device__ float {t}_white_unit(int i) {{
    // column first, row second - the argument order white_noise.py hashes in
    int c = $col(i)$;
    int r = $row(i)$;
    unsigned int key = (unsigned int)$SEED.get(0)$;
    key ^= (unsigned int)c * 374761393u;
    key ^= (unsigned int)r * 668265263u;
    unsigned int hashed = $hash_u32(key)$;
    return (float)hashed / 4294967296.0f;
}}
""",
            row=row,
            col=col,
            hash_u32=hash_u32,
            SEED=seed_p,
        )
        at = mk(
            f"""
__device__ float {t}_at(int i) {{
    return ($white_unit(i)$ - 0.5f) * 2.0f * $AMP.get(0)$;
}}
""",
            white_unit=white_unit,
            AMP=amplitude_p,
        )
        return {"at": at, "white_unit": white_unit}

    fade = mk(f"__device__ float {t}_fade(float t) {{ return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }}")
    lerp = mk(f"__device__ float {t}_lerp(float t, float a, float b) {{ return a + t * (b - a); }}")
    grad = mk(
        f"""
__device__ float {t}_grad(int hash_val, float dx, float dy) {{
    int idx = hash_val & 7;
    float gx = 0.0f;
    float gy = 0.0f;
    if (idx == 0)      {{ gx =  1.0f; gy =  1.0f; }}
    else if (idx == 1) {{ gx = -1.0f; gy =  1.0f; }}
    else if (idx == 2) {{ gx =  1.0f; gy = -1.0f; }}
    else if (idx == 3) {{ gx = -1.0f; gy = -1.0f; }}
    else if (idx == 4) {{ gx =  1.0f; gy =  0.0f; }}
    else if (idx == 5) {{ gx = -1.0f; gy =  0.0f; }}
    else if (idx == 6) {{ gx =  0.0f; gy =  1.0f; }}
    else               {{ gx =  0.0f; gy = -1.0f; }}
    return gx * dx + gy * dy;
}}
"""
    )
    perlin_at = mk(
        f"""
__device__ float {t}_perlin_at(float x, float y) {{
    float x_floor = floorf(x);
    float y_floor = floorf(y);

    int X = ((int)x_floor) & 255;
    int Y = ((int)y_floor) & 255;

    float x_local = x - x_floor;
    float y_local = y - y_floor;

    float u = $fade(x_local)$;
    float v = $fade(y_local)$;

    int iX1 = (X + 1) & 255;
    int A = $PERM.get(X)$ + Y;
    int B = $PERM.get(iX1)$ + Y;

    int iAA = A & 255;
    int iAB = (A + 1) & 255;
    int iBA = B & 255;
    int iBB = (B + 1) & 255;

    int AA = $PERM.get(iAA)$;
    int AB = $PERM.get(iAB)$;
    int BA = $PERM.get(iBA)$;
    int BB = $PERM.get(iBB)$;

    float gaa = $grad(AA, x_local, y_local)$;
    float gba = $grad(BA, x_local - 1.0f, y_local)$;
    float gab = $grad(AB, x_local, y_local - 1.0f)$;
    float gbb = $grad(BB, x_local - 1.0f, y_local - 1.0f)$;

    float lo = $lerp(u, gaa, gba)$;
    float hi = $lerp(u, gab, gbb)$;
    return $lerp(v, lo, hi)$;
}}
""",
        fade=fade,
        lerp=lerp,
        grad=grad,
        PERM=perm_p,
    )
    at = mk(
        f"""
__device__ float {t}_at(int i) {{
    float nx_f = (float)$GRID.nx.get(0)$;
    float ny_f = (float)$GRID.ny.get(0)$;

    float x = (float)$col(i)$ * $FX.get(0)$ / nx_f;
    float y = (float)$row(i)$ * $FY.get(0)$ / ny_f;

    float total = 0.0f;
    float max_value = 0.0f;
    float current_amplitude = 1.0f;
    float current_frequency = 1.0f;

    int octaves = $OCTAVES.get(0)$;
    for (int o = 0; o < octaves; ++o) {{
        total += $perlin_at(x * current_frequency, y * current_frequency)$ * current_amplitude;
        max_value += current_amplitude;
        current_amplitude *= $PERSISTENCE.get(0)$;
        current_frequency *= 2.0f;
    }}

    if (max_value > 0.0f) return (total / max_value) * $AMP.get(0)$;
    return 0.0f;
}}
""",
        row=row,
        col=col,
        perlin_at=perlin_at,
        GRID=grid,
        FX=frequency_x_p,
        FY=frequency_y_p,
        OCTAVES=octaves_p,
        PERSISTENCE=persistence_p,
        AMP=amplitude_p,
    )
    return {"at": at, "perlin_at": perlin_at}
