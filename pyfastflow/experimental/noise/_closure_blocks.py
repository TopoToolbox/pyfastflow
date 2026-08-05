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
cast/u32/i32/f32/floor surface, which is all these blocks need. It is never
bound explicitly here - every closure HelperBuilder/KernelBuilder auto-
injects its own `_BK` (see _closure_backend.py's module docstring).

Author: B.G (07/2026)
"""

import functools

from ..core.context.backends import bag_need, helper_need, make_helper, param_need
from ..core.context.need import Kind, Need

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


def _grid_nx_need(name: str, grid) -> Need:
    """
    A `Need(name, kind=Kind.BAG)` bound to `grid`, requiring only the `nx`
    member row/col actually read (`GRID.nx.get(0)`).

    Author: B.G (08/2026)
    """
    return bag_need(name, grid, contains=[Need("nx", kind=Kind.PARAM, dtype=grid.nx.dtype, modes={grid.nx.mode})])


def _grid_nx_ny_need(name: str, grid) -> Need:
    """
    A `Need(name, kind=Kind.BAG)` bound to `grid`, requiring the `nx`/`ny`
    members `_at_perlin_tmpl` actually reads.

    Author: B.G (08/2026)
    """
    return bag_need(
        name,
        grid,
        contains=[
            Need("nx", kind=Kind.PARAM, dtype=grid.nx.dtype, modes={grid.nx.mode}),
            Need("ny", kind=Kind.PARAM, dtype=grid.ny.dtype, modes={grid.ny.mode}),
        ],
    )


def build_hash_u32(HelperCls):
    """
    The standalone hash_u32(x) HelperBuilder - no Parameters, so it can be
    built without a grid, a pool, or any of make_noise's other config. Used
    both by build_helpers below and by make_noise's own make_hash_u32(),
    which callers outside make_noise (e.g. flow's rand_unit) reach when they
    want the same integer hash without building a whole noise Bag.

    `strict_needs=True`: the reference conversion the Need-restructuring plan
    calls for (see grid/_closure_blocks.py's build_helpers). `_BK` needs no
    bind at all here - see the module docstring's note on auto-injection.

    Author: B.G (07/2026)
    """
    mk = functools.partial(make_helper, HelperCls, strict_needs=True)
    return mk(_hash_u32_tmpl)


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
):
    """
    Wire one noise bag's private blocks and its public `at` for a closure
    backend (Taichi or Quadrants), picking the white or Perlin chain from
    `kind` and binding private blocks into public ones by name.

    Every bind below goes through a Need (param_need/helper_need/bag_need,
    see backends.py) and every HelperBuilder is constructed with
    `strict_needs=True` (via `mk`) - mirrors grid/_closure_blocks.py's own
    conversion. `GRID=grid` - a whole Bag bound under one name, read by
    dotted path (`GRID.nx.get(0)`) - is the first real use of `Kind.BAG`
    (need.py): see _grid_nx_need/_grid_nx_ny_need, one contract per bind site
    since row/col only ever read `nx` while Perlin's `at` also reads `ny`.
    `backend_mod`/`_BK` no longer appears here at all - see build_hash_u32's
    docstring and _closure_backend.py's module docstring on auto-injection.

    Returns {public_name: HelperBuilder}, meant to be merged straight into
    the Bag make_noise() returns. The parameters not used by the chosen
    `kind` arrive as None and are simply never bound.

    Author: B.G (07/2026)
    """

    mk = functools.partial(make_helper, HelperCls, strict_needs=True)

    row = mk(_row_tmpl, GRID=_grid_nx_need("GRID", grid))
    col = mk(_col_tmpl, GRID=_grid_nx_need("GRID", grid))
    # Built unconditionally (not just for kind="white") - hash_u32 is also a
    # public bag member, reused by callers outside make_noise (e.g. flow's
    # rand_unit) that want the same integer hash without pulling in the rest
    # of the white-noise chain.
    hash_u32 = build_hash_u32(HelperCls)

    if kind == "white":
        white_unit = mk(
            _white_unit_tmpl,
            _ROW=helper_need("_ROW", row),
            _COL=helper_need("_COL", col),
            _HASH=helper_need("_HASH", hash_u32),
            SEED=param_need("SEED", seed_p),
        )
        at = mk(_at_white_tmpl, _WHITEUNIT=helper_need("_WHITEUNIT", white_unit), AMP=param_need("AMP", amplitude_p))
        return {"at": at, "white_unit": white_unit, "hash_u32": hash_u32}

    fade = mk(_fade_tmpl)
    lerp = mk(_lerp_tmpl)
    grad = mk(_grad_tmpl)
    perlin_at = mk(
        _perlin_at_tmpl,
        _FADE=helper_need("_FADE", fade),
        _LERP=helper_need("_LERP", lerp),
        _GRAD=helper_need("_GRAD", grad),
        PERM=param_need("PERM", perm_p),
    )
    at = mk(
        _at_perlin_tmpl,
        _ROW=helper_need("_ROW", row),
        _COL=helper_need("_COL", col),
        _PERLINAT=helper_need("_PERLINAT", perlin_at),
        GRID=_grid_nx_ny_need("GRID", grid),
        FX=param_need("FX", frequency_x_p),
        FY=param_need("FY", frequency_y_p),
        OCTAVES=param_need("OCTAVES", octaves_p),
        PERSISTENCE=param_need("PERSISTENCE", persistence_p),
        AMP=param_need("AMP", amplitude_p),
    )
    return {"at": at, "perlin_at": perlin_at, "hash_u32": hash_u32}
