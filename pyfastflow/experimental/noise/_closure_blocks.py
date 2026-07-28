"""
Taichi/Quadrants (closure) block templates behind make_noise.

Same shape as the grid's _closure_blocks.py: every block is a plain python
def, PICKED by build_helpers() from the noise config rather than branched on
inside one body. Here the only structural selector is `kind` - it decides
whether the public `at(i)` is wired to the white-noise chain or the Perlin
chain, and nothing downstream of `at` is shared between the two.

The arithmetic is a port of pyfastflow/noise/white_noise.py and
perlin_noise.py, kept value-for-value identical: same integer hash constants,
same fade/lerp/grad, same octave accumulation, and the same argument order
(column first, row second) into the hash, so a bag built here reproduces what
NoiseContext produced for the same seed.

Row/column come from the bound grid Bag (GRID.nx/GRID.ny read through
`.get(0)` like any other value param), so a noise bag inherits whatever mode
the grid's geometry is in and never carries its own copy of nx/ny.

`_BK` is the backend module (taichi or quadrants) - both expose the same
cast/u32/i32/f32/floor surface, which is all these blocks need.

Author: B.G (07/2026)
"""

import functools

from ..core.context.backends import make_helper

# ---------------------------------------------------------------------------
# shared: flat index -> row / column, off the bound grid bag
# ---------------------------------------------------------------------------


def _row_tmpl(i):
    return i // GRID.nx.get(0)


def _col_tmpl(i):
    return i % GRID.nx.get(0)


# ---------------------------------------------------------------------------
# white noise
# ---------------------------------------------------------------------------


def _hash_u32_tmpl(x):
    h = x
    h ^= h >> _BK.u32(16)
    h *= _BK.u32(0x7FEB352D)
    h ^= h >> _BK.u32(15)
    h *= _BK.u32(0x846CA68B)
    h ^= h >> _BK.u32(16)
    return h


def _white_unit_tmpl(i):
    # column first, row second - the argument order white_noise.py hashes in
    col = _COL(i)
    row = _ROW(i)
    key = _BK.u32(SEED.get(0))
    key ^= _BK.u32(col) * _BK.u32(374761393)
    key ^= _BK.u32(row) * _BK.u32(668265263)
    hashed = _HASH(key)
    return _BK.cast(hashed, _BK.f32) / 4294967296.0


def _at_white_tmpl(i):
    return (_WHITEUNIT(i) - 0.5) * 2.0 * AMP.get(0)


# ---------------------------------------------------------------------------
# perlin noise
# ---------------------------------------------------------------------------


def _fade_tmpl(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp_tmpl(t, a, b):
    return a + t * (b - a)


def _grad_tmpl(hash_val, dx, dy):
    idx = hash_val & 7
    gx = 0.0
    gy = 0.0

    if idx == 0:
        gx, gy = 1.0, 1.0
    elif idx == 1:
        gx, gy = -1.0, 1.0
    elif idx == 2:
        gx, gy = 1.0, -1.0
    elif idx == 3:
        gx, gy = -1.0, -1.0
    elif idx == 4:
        gx, gy = 1.0, 0.0
    elif idx == 5:
        gx, gy = -1.0, 0.0
    elif idx == 6:
        gx, gy = 0.0, 1.0
    else:
        gx, gy = 0.0, -1.0

    return gx * dx + gy * dy


def _perlin_at_tmpl(x, y):
    x_floor = _BK.floor(x)
    y_floor = _BK.floor(y)

    X = _BK.cast(x_floor, _BK.i32) & 255
    Y = _BK.cast(y_floor, _BK.i32) & 255

    x_local = x - x_floor
    y_local = y - y_floor

    u = _FADE(x_local)
    v = _FADE(y_local)

    A = PERM.get(X) + Y
    B = PERM.get((X + 1) & 255) + Y
    AA = PERM.get(A & 255)
    AB = PERM.get((A + 1) & 255)
    BA = PERM.get(B & 255)
    BB = PERM.get((B + 1) & 255)

    return _LERP(
        v,
        _LERP(u, _GRAD(AA, x_local, y_local), _GRAD(BA, x_local - 1.0, y_local)),
        _LERP(u, _GRAD(AB, x_local, y_local - 1.0), _GRAD(BB, x_local - 1.0, y_local - 1.0)),
    )


def _at_perlin_tmpl(i):
    nx_f = _BK.cast(GRID.nx.get(0), _BK.f32)
    ny_f = _BK.cast(GRID.ny.get(0), _BK.f32)

    x = _BK.cast(_COL(i), _BK.f32) * FX.get(0) / nx_f
    y = _BK.cast(_ROW(i), _BK.f32) * FY.get(0) / ny_f

    total = 0.0
    max_value = 0.0
    current_amplitude = 1.0
    current_frequency = 1.0

    for _ in range(OCTAVES.get(0)):
        total += _PERLINAT(x * current_frequency, y * current_frequency) * current_amplitude
        max_value += current_amplitude
        current_amplitude *= PERSISTENCE.get(0)
        current_frequency *= 2.0

    out = 0.0
    if max_value > 0.0:
        out = (total / max_value) * AMP.get(0)
    return out


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
    backend_mod,
):
    """
    Wire one noise bag's private blocks and its public `at` for a closure
    backend (Taichi or Quadrants), picking the white or Perlin chain from
    `kind` and binding private blocks into public ones by name.

    Returns {public_name: HelperBuilder}, meant to be merged straight into
    the Bag make_noise() returns. The parameters not used by the chosen
    `kind` arrive as None and are simply never bound.

    Author: B.G (07/2026)
    """

    mk = functools.partial(make_helper, HelperCls)

    row = mk(_row_tmpl, GRID=grid)
    col = mk(_col_tmpl, GRID=grid)

    if kind == "white":
        hash_u32 = mk(_hash_u32_tmpl, _BK=backend_mod)
        white_unit = mk(_white_unit_tmpl, _ROW=row, _COL=col, _HASH=hash_u32, SEED=seed_p, _BK=backend_mod)
        at = mk(_at_white_tmpl, _WHITEUNIT=white_unit, AMP=amplitude_p)
        return {"at": at, "white_unit": white_unit}

    fade = mk(_fade_tmpl)
    lerp = mk(_lerp_tmpl)
    grad = mk(_grad_tmpl)
    perlin_at = mk(_perlin_at_tmpl, _FADE=fade, _LERP=lerp, _GRAD=grad, PERM=perm_p, _BK=backend_mod)
    at = mk(
        _at_perlin_tmpl,
        _ROW=row,
        _COL=col,
        _PERLINAT=perlin_at,
        GRID=grid,
        FX=frequency_x_p,
        FY=frequency_y_p,
        OCTAVES=octaves_p,
        PERSISTENCE=persistence_p,
        AMP=amplitude_p,
        _BK=backend_mod,
    )
    return {"at": at, "perlin_at": perlin_at}
