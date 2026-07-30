"""
make_hillshade: the hillshading Bag factory, built on the backend-agnostic
core (see ../core/context: parameter.py for Parameter, compile.py for
HelperBuilder, bag.py for Bag) and on a grid Bag from ../grid.

Like make_grid/make_noise there is no stateful context class - make_hillshade
builds a Bag once: `at(z, i)` (a device helper, shade value at one node) and
`hillshade` (an unbuilt KernelBuilder that writes shade for every node - call
`.compile()` on it before launching). A caller wanting hillshade inline in
its own kernel binds the bag and reads `visu.at(z, i)`; a caller wanting a
standalone pass compiles `visu.hillshade` and launches it with (z, out) -
and, on cupy, the node count as a third argument, since a RawModule kernel
has no auto-ranging (see ../core/context/cupy_backend.py's module docstring).

azimuth/altitude/z_factor are mode-overridable Parameters (default "const"),
read in device code through `.get(0)` uniformly across modes - the same
"flexible at build time, dense at runtime" stance make_grid/make_noise take.
azimuth and altitude are both in degrees, matching the hillshading
convention (315/45 is the classic NW-lit default); z_factor scales the
elevation gradient before it enters the slope/aspect computation, letting a
caller exaggerate relief without rescaling z itself.

The gradient the `at` helper is built on deliberately does not clamp indices
against a fixed 3x3 stencil the way the legacy standalone kernel
(pyfastflow/visu/hillshading.py) does - see _closure_blocks.py's module
docstring for why that is wrong under a periodic grid or at a nodata cell,
and how going through grid.neighbour(i, k) fixes both for free.

Author: B.G (07/2026)
"""

from ..core.context.backends import backend_classes
from ..core.context.bag import Bag


def _blocks_for(backend: str):
    """
    The private block module implementing make_hillshade's device code for
    one backend name: the closure blocks (shared by Taichi and Quadrants) or
    the cupy blocks.

    Author: B.G (07/2026)
    """
    if backend in ("taichi", "quadrants"):
        from . import _closure_blocks as blocks
    elif backend == "cupy":
        from . import _cupy_blocks as blocks
    else:
        raise ValueError(f"make_hillshade: unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")
    return blocks


def _kernel_cls(backend: str):
    """
    The KernelBuilder class for `backend` - backend_classes() only returns
    HelperBuilder, so the `hillshade` kernel member looks this up directly.

    Author: B.G (07/2026)
    """
    if backend == "taichi":
        from ..core.context.taichi_backend import TaichiKernelBuilder

        return TaichiKernelBuilder
    if backend == "quadrants":
        from ..core.context.quadrants_backend import QuadrantsKernelBuilder

        return QuadrantsKernelBuilder
    if backend == "cupy":
        from ..core.context.cupy_backend import CupyKernelBuilder

        return CupyKernelBuilder
    raise ValueError(f"unknown backend {backend!r}")


def make_hillshade(
    backend: str,
    pool,
    grid: Bag,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    azimuth_mode: str = "const",
    altitude_mode: str = "const",
    z_factor_mode: str = "const",
) -> Bag:
    """
    Build one hillshade Bag: `at(z, i)` and the unbuilt `hillshade` kernel
    builder, both reading row/column/dx/neighbours off `grid` rather than
    carrying their own geometry.

    `azimuth`/`altitude` are light-source angles in degrees (315/45 is the
    classic NW-lit default); `z_factor` scales the gradient before it enters
    the slope/aspect computation. The `*_mode` arguments are "const" or
    "scalar", deciding whether a value is folded in at compile time or lives
    in a one-cell device field the host can retune - same convention as
    make_grid/make_noise.

    Author: B.G (07/2026)
    """
    for label, mode in (
        ("azimuth_mode", azimuth_mode),
        ("altitude_mode", altitude_mode),
        ("z_factor_mode", z_factor_mode),
    ):
        if mode not in ("const", "scalar"):
            raise ValueError(f"make_hillshade: {label} must be 'const' or 'scalar', got {mode!r}")

    backend_mod, ParamCls, HelperCls, dtypes = backend_classes(backend)
    KernelCls = _kernel_cls(backend)
    blocks = _blocks_for(backend)

    azimuth_p = ParamCls("HS_AZIMUTH", dtype=dtypes["f32"], mode=azimuth_mode, value=float(azimuth), pool=pool)
    altitude_p = ParamCls("HS_ALTITUDE", dtype=dtypes["f32"], mode=altitude_mode, value=float(altitude), pool=pool)
    z_factor_p = ParamCls("HS_ZFACTOR", dtype=dtypes["f32"], mode=z_factor_mode, value=float(z_factor), pool=pool)

    if backend == "cupy":
        members = blocks.build_helpers(
            HelperCls, KernelCls, grid=grid, azimuth_p=azimuth_p, altitude_p=altitude_p, z_factor_p=z_factor_p
        )
    else:
        members = blocks.build_helpers(
            HelperCls,
            KernelCls,
            backend=backend,
            backend_mod=backend_mod,
            grid=grid,
            azimuth_p=azimuth_p,
            altitude_p=altitude_p,
            z_factor_p=z_factor_p,
        )

    return Bag({"azimuth": azimuth_p, "altitude": altitude_p, "z_factor": z_factor_p, **members})
