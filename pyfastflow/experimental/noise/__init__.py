"""
make_noise: the NoiseContext-equivalent Bag factory, built on the
backend-agnostic core (see ..core.context: parameter.py for Parameter, compile.py for HelperBuilder, bag.py for Bag)
and on a grid Bag from ..grid.

Like make_grid there is no stateful context class - make_noise builds a Bag
once and hands it back. The bag is device helpers plus their parameters,
nothing else: no allocation, no fill kernel, no host-side generate_*. A caller
binds the bag into its own kernel and reads `noise.at(i)` inline, so noise
composes with whatever that kernel is already doing rather than forcing a
separate pass over a temporary field.

    noise = make_noise("taichi", pool, grid, kind="perlin", octaves=4)

    def init_template(z: ti.template()):
        for i in z:
            z[i] = noise.at(i)

    TaichiKernelBuilder().bind("noise", noise).ingest(init_template).compile()

Two kinds of knobs, same split as make_grid:
  - value params (amplitude, seed, frequency_x/y, octaves, persistence) -
    mode-overridable, always read in device code through `.get(0)`.
  - one structural selector, `kind` ("white" or "perlin") - it picks which
    chain of private blocks the public `at` is wired to. The public surface
    is `at(i)` whatever the kind, so swapping generators is a build-time
    config change and the calling template never moves.

Mode defaults are "const" - the same "flexible at build time, dense at
runtime" stance make_grid takes - with one exception: `seed` (white noise
only) defaults to "scalar", so reseeding is a host write rather than a
rebuild. That keeps it symmetric with Perlin, whose seed is not a device
parameter at all: it drives the permutation table, which is built on the host
and uploaded. Reseed Perlin by refilling that field:

    noise.perm.set(permutation_table(7))

Bag members are `at` and `hash_u32` plus whatever the kind actually uses:
white carries `amplitude`, `seed`, `white_unit`; perlin carries `amplitude`,
`perm`, `frequency_x`, `frequency_y`, `octaves`, `persistence`, `perlin_at`. A
member that a kind does not use is absent from the bag rather than present
and inert. `hash_u32` is always present regardless of `kind` - it is the
integer hash white noise builds on, exposed so other bags (e.g. flow's
rand_unit) can reuse the exact same hash rather than reimplementing it.

Values match pyfastflow/noise/ for the same seed and settings - this is a port
of that arithmetic, not a reimplementation. White noise lands in
[-amplitude, amplitude]; Perlin is the octave-averaged lattice noise scaled by
amplitude.

_closure_blocks.py/_cupy_blocks.py's own internal wiring goes through a Need
(need.py) now, every HelperBuilder built `strict_needs=True` (compile.py) -
the second factory converted, after grid/, per the Need-restructuring plan.
`GRID=grid` (a whole Bag bound under one name) is the first real use of
`Kind.BAG` there. Internal only - make_noise's/make_hash_u32's own signatures
and the returned Bag's member names/types are unchanged.

Author: B.G (07/2026)
"""

import numpy as np

from ..core.context.backends import backend_classes
from ..core.context.bag import Bag

_KINDS = frozenset({"white", "perlin"})
_MODES = ("const", "scalar")


def permutation_table(seed: int) -> np.ndarray:
    """
    The 512-entry Perlin permutation table for one seed: a Fisher-Yates
    shuffle of 0..255 concatenated with itself, so a lattice lookup can index
    past 255 without wrapping by hand.

    Same construction (and same numpy Generator) as
    pyfastflow/noise/noisecontext.py, so a given seed yields the same table
    and therefore the same noise.

    Author: B.G (07/2026)
    """
    rng = np.random.default_rng(seed)
    perm = np.arange(256, dtype=np.int32)
    for i in range(255, 0, -1):
        j = rng.integers(0, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return np.concatenate([perm, perm])


def _blocks_for(backend: str):
    """
    The private block module implementing make_noise's device code for one
    backend name: the closure blocks (shared by Taichi and Quadrants) or the
    cupy blocks.

    Author: B.G (07/2026)
    """
    if backend in ("taichi", "quadrants"):
        from . import _closure_blocks as blocks
    elif backend == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"make_noise: unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")
    return blocks


def make_hash_u32(backend: str):
    """
    The standalone hash_u32(x) HelperBuilder make_noise's white-noise chain
    is built on - no Parameters, no grid, no pool. A caller that only wants
    the same integer hash (e.g. flow's rand_unit - see ../flow/__init__.py)
    reaches it here rather than building a whole noise Bag just to pull
    hash_u32 back out of it.

    Author: B.G (07/2026)
    """
    _, _, HelperCls, _ = backend_classes(backend)
    blocks = _blocks_for(backend)
    return blocks.build_hash_u32(HelperCls)


def make_noise(
    backend: str,
    pool,
    grid: Bag,
    *,
    kind: str = "perlin",
    amplitude: float = 1.0,
    seed: int = 42,
    frequency: float = 8.0,
    frequency_x: float | None = None,
    frequency_y: float | None = None,
    octaves: int = 4,
    persistence: float = 0.5,
    amplitude_mode: str = "const",
    seed_mode: str = "scalar",
    frequency_mode: str = "const",
    octaves_mode: str = "const",
    persistence_mode: str = "const",
) -> Bag:
    """
    Build one noise bag: the public `at(i)` device helper and the parameters
    the chosen `kind` needs, all reading row/column off the `grid` bag rather
    than carrying their own geometry.

    `kind` "white"|"perlin" picks the block chain (see _closure_blocks.py /
    _cupy_blocks.py). `frequency` sets both axes; `frequency_x`/`frequency_y`
    override one axis each. `octaves`/`persistence`/`frequency_*` are Perlin
    only and are not allocated for white noise.

    The `*_mode` arguments are "const" or "scalar" and decide whether a value
    is folded in at compile time or lives in a one-cell device field the host
    can retune. Defaults are "const" except `seed_mode`, which is "scalar" -
    see the module docstring.

    Author: B.G (07/2026)
    """
    if kind not in _KINDS:
        raise ValueError(f"make_noise: kind must be one of {sorted(_KINDS)}, got {kind!r}")
    for label, mode in (
        ("amplitude_mode", amplitude_mode),
        ("seed_mode", seed_mode),
        ("frequency_mode", frequency_mode),
        ("octaves_mode", octaves_mode),
        ("persistence_mode", persistence_mode),
    ):
        if mode not in _MODES:
            raise ValueError(f"make_noise: {label} must be 'const' or 'scalar', got {mode!r}")

    backend_mod, ParamCls, HelperCls, dtypes = backend_classes(backend)
    blocks = _blocks_for(backend)

    amplitude_p = ParamCls(
        "NOISE_AMPLITUDE", dtype=dtypes["f32"], mode=amplitude_mode, value=float(amplitude), pool=pool
    )

    seed_p = None
    perm_p = None
    frequency_x_p = None
    frequency_y_p = None
    octaves_p = None
    persistence_p = None

    if kind == "white":
        seed_p = ParamCls("NOISE_SEED", dtype=dtypes["u32"], mode=seed_mode, value=int(seed), pool=pool)
        members = {"amplitude": amplitude_p, "seed": seed_p}
    else:
        perm_p = ParamCls(
            "NOISE_PERM",
            dtype=dtypes["i32"],
            mode="field",
            value=permutation_table(seed),
            pool=pool,
            n_flat=512,
        )
        fx = float(frequency_x if frequency_x is not None else frequency)
        fy = float(frequency_y if frequency_y is not None else frequency)
        frequency_x_p = ParamCls("NOISE_FX", dtype=dtypes["f32"], mode=frequency_mode, value=fx, pool=pool)
        frequency_y_p = ParamCls("NOISE_FY", dtype=dtypes["f32"], mode=frequency_mode, value=fy, pool=pool)
        octaves_p = ParamCls("NOISE_OCTAVES", dtype=dtypes["i32"], mode=octaves_mode, value=int(octaves), pool=pool)
        persistence_p = ParamCls(
            "NOISE_PERSISTENCE", dtype=dtypes["f32"], mode=persistence_mode, value=float(persistence), pool=pool
        )
        members = {
            "amplitude": amplitude_p,
            "perm": perm_p,
            "frequency_x": frequency_x_p,
            "frequency_y": frequency_y_p,
            "octaves": octaves_p,
            "persistence": persistence_p,
        }

    helpers = blocks.build_helpers(
        HelperCls,
        grid=grid,
        kind=kind,
        amplitude_p=amplitude_p,
        seed_p=seed_p,
        perm_p=perm_p,
        frequency_x_p=frequency_x_p,
        frequency_y_p=frequency_y_p,
        octaves_p=octaves_p,
        persistence_p=persistence_p,
    )

    members.update(helpers)
    return Bag(members)
