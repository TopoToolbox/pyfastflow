"""ShearParcelSedProgram: shear-stress entrainment and stochastic sediment
parcels driven by the time-averaged flow of a hydrodynamic solver.

CuPy only. The program reads a depth field ``h`` and the link fluxes ``qx``,
``qy`` (and ``qxy``, ``qyx`` in D8) in the InertialFloodProgram layout
(typically adopted from one, with ``z`` shared the same way) and changes only
``z``; ``h`` is held fixed, so the water surface moves with the bed. All
times are in seconds. The erosion-deposition model is the transport-length
one of Davy and Lague (2009): the bed loses the entrainment rate and gains
q_s / xi, here as parcels travelling an exponentially distributed distance
of mean xi.

``sample()`` adds the current flow to a window; the caller runs the flow
solver between samples. For every wet cell (links not deeper than
``min_depth`` are dry; link depth max(z + h) - max(z) as the flow solver
sees it; u the width-weighted mean of the link velocities, d the mean wet
link depth) it averages

  - the entrainment rate (m/s) ``entrainment_factor`` / (``sediment_density``
    w_s) * max(tau - tau_r, 0) (Le Minor et al. 2025, ejection velocity =
    w_s), with tau = ``water_density`` g ``manning``^2 u^2 / d^(1/3)
    and tau_r = ``critical_shields`` (``sediment_density`` -
    ``water_density``) g ``grain_size``;
  - the transport length xi = h_s v_s / w_s (Le Minor et al. 2025): the
    saltation height h_s = min(z0 + 0.6 D + 0.025 D T*, h) with z0 =
    ``bed_roughness`` and the transport stage T* = max(tau / tau_r - 1, 0),
    the log-profile mean velocity v_s = min(u* / ``von_karman`` (ln(h_s /
    z0) - 1 + z0 / h_s), u) with u* = sqrt(tau / ``water_density``), and the
    Ferguson and Church (2004) settling velocity w_s = R g D^2 /
    (``settling_c1`` ``viscosity`` + sqrt(0.75 ``settling_c2`` R g D^3));

as well as the link fluxes. Inactive cells and outlets do not erode.

``run_sediment(window)`` closes a window of ``window`` hydraulic seconds
with one sediment step of morphological duration
dt = ``window`` * ``morpho_factor``, reduced so that the net bed change rate
of the previous step (the leaving rate below, on the first step) changes no
cell at least ``cap_depth`` deep by more than ``max_bed_change`` times its
depth:

  1. erosion: of the material entrained from uniformly distributed points
     of a cell, the fraction (xi / dx) (1 - exp(-dx / xi)) crosses its edge
     and the rest settles back in it; the bed lowers by that leaving
     fraction of rate * dt;
  2. deposition: each cell's eroded volume is split into
     ceil(volume / ``parcel_volume``) equal parcels, walked by a persistent
     kernel. Each hop goes to a neighbour drawn with probability
     proportional to max(outflow, 0) + ``bed_slope_weight`` * max(bed
     slope, 0), outflow the mean discharge of the link (cardinal links
     weighted 1, diagonal 1/2), and costs dist / xi of the cell left, except
     the first (the parcel has already left its source cell). ``deposition`` config: ``stochastic``
     drops the whole parcel once an optical depth tau = -ln(u) is consumed
     by the hop costs; ``gradual`` leaves the fraction 1 - exp(-cost) at
     each hop, down to ``min_volume`` (m3). On a cell without candidate a
     parcel leaves just the volume that raises the bed ``fill_epsilon`` (m)
     above its lowest neighbour and moves there (everything, if it carries
     less); parcels reaching an outlet leave the domain.

``run_sediment`` returns dt, the parcel count and the eroded (leaving),
deposited and exported volumes (m3).

Author: B.G (10/2026)
"""

from pyfastflow.core import ProgramError
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.grid._mask import GridMaskAccessor

from ._shear_parcels import (
    apply_factory, erosion_factory, sample_factory, walk_factory,
)

_MAX_PARCELS = 2**31 - 1


def _diagonal_shape(config):
    """Diagonal link fluxes exist only in D8."""
    if config["topology"] == "D8":
        return (config["ny"], config["nx"])
    return (0, 0)


def build_shear_parcel_sed_program() -> type:
    """Build the ShearParcelSedProgram class (see the module docstring)."""
    b = ProgramBuilder("ShearParcelSedProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("topology", choices=("D4", "D8"), default="D4")
    b.config(
        "boundary", choices=("normal", "periodic_EW", "periodic_NS"),
        default="normal",
    )
    b.config("outlet", choices=("edge", "mask"), default="edge")
    b.config("nodata", choices=(False, True), default=False)
    b.config("deposition", choices=("stochastic", "gradual"),
             default="stochastic")
    b.config("threads", default=65536)

    shape = (Dim("ny"), Dim("nx"))
    flat = Dim("ny") * Dim("nx")
    b.param("entrainment_factor", "auto", "f32", value=2.51, shape=shape)
    b.param("critical_shields", "auto", "f32", value=0.047, shape=shape)
    b.param("grain_size", "auto", "f32", value=1.0e-2, shape=shape)
    b.param("bed_roughness", "auto", "f32", value=1.0e-3, shape=shape)
    b.param("manning", "auto", "f32", value=0.033, shape=shape)
    b.param("sediment_density", "auto", "f32", value=2650.0)
    b.param("water_density", "auto", "f32", value=1000.0)
    b.param("gravity", "auto", "f32", value=9.81)
    b.param("von_karman", "auto", "f32", value=0.41)
    b.param("viscosity", "auto", "f32", value=1.0e-6)
    b.param("settling_c1", "auto", "f32", value=18.0)
    b.param("settling_c2", "auto", "f32", value=1.0)
    b.param("min_depth", "auto", "f32", value=1.0e-3)
    b.param("morpho_factor", "auto", "f32", value=1.0)
    b.param("max_bed_change", "auto", "f32", value=0.05)
    b.param("cap_depth", "auto", "f32", value=1.0e-2)
    b.param("morpho_dt", "scalar", "f32", value=0.0)
    b.param("bed_slope_weight", "auto", "f32", value=0.0)
    b.param("parcel_volume", "auto", "f32", value=1.0)
    b.param("min_volume", "auto", "f32", value=1.0e-2)
    b.param("fill_epsilon", "auto", "f32", value=1.0e-3)
    b.param("cell_count", "const", "i32",
            value=lambda dims: dims["nx"] * dims["ny"])
    b.param("seed", "scalar", "i32", value=1)
    b.param("clock", "scalar", "i32", value=0)

    xshape = (Dim("ny"), Dim("nx") + 1)
    yshape = (Dim("ny") + 1, Dim("nx"))
    b.data("z", "f32", shape, role="state", shape_source=True)
    b.data("h", "f32", shape, role="input")
    b.data("qx", "f32", xshape, role="input")
    b.data("qy", "f32", yshape, role="input")
    b.data("qxy", "f32", _diagonal_shape, role="input")
    b.data("qyx", "f32", _diagonal_shape, role="input")
    b.data("rate", "f32", shape, role="output")
    b.data("length", "f32", shape, role="output")
    b.data("qx_mean", "f32", xshape, role="internal")
    b.data("qy_mean", "f32", yshape, role="internal")
    b.data("qxy_mean", "f32", _diagonal_shape, role="internal")
    b.data("qyx_mean", "f32", _diagonal_shape, role="internal")
    b.data("erosion", "f32", shape, role="output")
    b.data("parcel_count", "f64", (flat,), role="internal")
    b.data("parcel_cdf", "f64", (flat,), role="internal")
    b.data("parcel_claim", "i32", (1,), role="internal")
    b.data("parcel_stats", "f64", (4,), role="internal")

    def grid_structure(be, *, topology, boundary, outlet, nodata, **_):
        _cupy_only(be)
        return make_grid_group(
            be, topology=topology, boundary=boundary, outlet=outlet,
            nodata=nodata,
        )

    def grid_params(be, pool, *, nx, ny, dx, topology, boundary, outlet, nodata):
        del boundary
        return make_grid_parameters(
            be, pool, nx, ny, dx, topology=topology, outlet=outlet,
            nodata=nodata,
        )

    b.bundle("grid", grid_structure, grid_params, dims=("nx", "ny"),
             config=("dx", "topology", "boundary", "outlet", "nodata"))

    b.add("sample_flow", sample_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "h": "h", "qx": "qx", "qy": "qy", "qxy": "qxy",
              "qyx": "qyx", "rate": "rate", "length": "length",
              "CELL_COUNT": "cell_count", "MIN_DEPTH": "min_depth",
              "MANNING": "manning", "GRAVITY": "gravity",
              "RHO_W": "water_density", "RHO_S": "sediment_density",
              "GRAIN_SIZE": "grain_size", "THETA_R": "critical_shields",
              "ALPHA": "entrainment_factor", "Z0": "bed_roughness",
              "KAPPA": "von_karman", "NU": "viscosity", "C1": "settling_c1",
              "C2": "settling_c2",
          }))
    b.add("compute_erosion", erosion_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "rate": "rate", "length": "length", "erosion": "erosion",
              "count": "parcel_count",
              "CELL_COUNT": "cell_count", "MORPHO_DT": "morpho_dt",
              "PARCEL_VOLUME": "parcel_volume",
          }))
    b.add("apply_erosion", apply_factory,
          bind={"z": "z", "erosion": "erosion", "CELL_COUNT": "cell_count"})
    b.add("walk_parcels", walk_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "cdf": "parcel_cdf", "erosion": "erosion", "length": "length",
              "qx": "qx_mean", "qy": "qy_mean", "qxy": "qxy_mean",
              "qyx": "qyx_mean", "z": "z", "claim": "parcel_claim",
              "stats": "parcel_stats", "CELL_COUNT": "cell_count",
              "BED_SLOPE_WEIGHT": "bed_slope_weight", "SEED": "seed",
              "CLOCK": "clock", "MIN_VOLUME": "min_volume",
              "FILL_EPS": "fill_epsilon",
          }))

    b.pipeline("_sample", ("sample_flow",))
    b.pipeline("_erode", ("compute_erosion", "apply_erosion"))
    b.pipeline("_deposit", ("walk_parcels",))

    program = b.freeze()

    def _accumulators(self):
        names = ("rate", "length", "qx_mean", "qy_mean")
        if self._config["topology"] == "D8":
            names += ("qxy_mean", "qyx_mean")
        return [self._handle(name).array for name in names]

    def sample(self):
        """Add the current flow (z, h and the link fluxes) to the window."""
        samples = getattr(self, "_samples", 0)
        if samples == 0:
            for array in self._accumulators():
                array.fill(0)
        self._sample()
        self._handle("qx_mean").array[...] += self._handle("qx").array
        self._handle("qy_mean").array[...] += self._handle("qy").array
        if self._config["topology"] == "D8":
            self._handle("qxy_mean").array[...] += self._handle("qxy").array
            self._handle("qyx_mean").array[...] += self._handle("qyx").array
        object.__setattr__(self, "_samples", samples + 1)

    def run_sediment(self, window):
        """One sediment step closing the sampled window of ``window`` hydraulic
        seconds; return ``{"dt", "parcels", "eroded", "deposited",
        "exported"}`` (dt the morphological duration in s, volumes in m3)."""
        import cupy as cp

        samples = getattr(self, "_samples", 0)
        if samples == 0:
            raise ProgramError("run_sediment needs at least one sample()")
        for array in self._accumulators():
            array *= 1.0 / samples
        h = self._handle("h").array
        deep = h >= float(self.cap_depth.read())
        cap = float(self.max_bed_change.read())
        dt = float(window) * float(self.morpho_factor.read())
        peak = getattr(self, "_net_rate", None)
        if peak is None:
            # first step: the leaving rate bounds the net lowering
            dx = float(self.dx)
            length = self._handle("length").array
            leave = cp.where(length > 0,
                             -(length / dx) * cp.expm1(-dx / cp.maximum(length, 1e-30)),
                             0.0)
            peak = float(cp.max(cp.where(
                deep, self._handle("rate").array * leave / h, 0.0)).item())
        if peak * dt > cap:
            dt = cap / peak
        self.morpho_dt.set(dt)
        z = self._handle("z").array
        z_before = z.copy()

        count = self._handle("parcel_count").array.ravel()
        cdf = self._handle("parcel_cdf").array.ravel()
        self._handle("parcel_stats").array.fill(0)
        self._erode()
        cp.cumsum(count, out=cdf)
        if float(cdf[-1].item()) > _MAX_PARCELS:
            raise ProgramError(
                f"{float(cdf[-1].item()):.3g} parcels in one step exceed "
                f"{_MAX_PARCELS}: raise parcel_volume")
        self._handle("parcel_claim").array.fill(0)
        clock = getattr(self, "_clock", 0)
        self.clock.set(clock)
        self._deposit()
        object.__setattr__(self, "_clock", clock + 1)
        net = cp.where(deep, cp.abs(z - z_before) / h, 0.0)
        object.__setattr__(self, "_net_rate", float(cp.max(net).item()) / dt)
        object.__setattr__(self, "_samples", 0)
        raw = self._handle("parcel_stats").to_numpy()
        return {"dt": dt, "parcels": int(raw[0]), "eroded": float(raw[1]),
                "deposited": float(raw[2]), "exported": float(raw[3])}

    program._accumulators = _accumulators
    program.sample = sample
    program.run_sediment = run_sediment
    program.outlet_mask = property(
        lambda self: GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"),
    )
    program.nodata_mask = property(
        lambda self: GridMaskAccessor(self, "NODATA_MASK", "nodata=True"),
    )
    return program


ShearParcelSedProgram = build_shear_parcel_sed_program()

__all__ = ["ShearParcelSedProgram", "build_shear_parcel_sed_program"]
