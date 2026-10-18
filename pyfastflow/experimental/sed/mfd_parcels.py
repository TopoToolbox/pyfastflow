"""MFDParcelSedProgram: MFD stream-power erosion with stochastic sediment parcels.

CuPy only. Outlets are held at z = 0 (base level). One step of ``dt`` (yr):

  1. uplift by ``uplift`` * ``dt`` (outlets set to 0);
  2. MFD graph of z with GraphFlood's routing (``h`` stays 0):
     ``mfd_local_minima`` (default carve_cordonnier) resolves the local
     minima, giving the SFD receivers ``rec`` and the MFD receivers and
     fractions; the persistent MFD kernel accumulates dx^2 per cell over it
     into the drainage area ``Qi`` (m2);
  3. implicit stream-power incision on ``rec`` with the drainage area
     ``Qi``, solved by Newton pointer jumping (GOLEM fluvial): lowers z and
     writes ``erosion_rate``; its volume ``erosion_rate`` * ``dt`` * dx^2 is
     the entrained volume of the cell, split into parcels of at most
     ``parcel_volume``;
  4. each parcel walks downstream, its receivers set by ``parcel_path``:
       - ``graph``: a receiver of the MFD graph of step 2 drawn in
         proportion to its fraction (paths fixed during the walk);
       - ``surface``: a lower neighbour of the current z (deposits of the
         walk included) drawn in proportion to its slope, a cell without
         lower neighbour taking a receiver of the MFD graph of step 2;
     and leaves on the cell it departs from the volume of the
     ``deposition`` rule for the hop's stretch [x, x + d] of travelled
     distance:
       - ``gaussian``: v0 times the share of that stretch in a gaussian of
         the travelled distance centred on ``transport_distance`` (m) with
         spread ``transport_spread`` (m), truncated to x >= 0;
       - ``exponential``: the fraction 1 - exp(-d / ``transport_distance``)
         of what it carries;
     (both of the source cell); a remainder that would fall below
     ``min_volume`` (m3) is left whole. Parcels reaching an outlet export
     what they carry; deposits are applied to z as they happen.

``initialize()`` sets the outlets to 0 and h to 0. ``run_n_step(n)`` returns
the parcel count and the entrained, deposited and exported volumes (m3)
summed over the n steps.

Author: B.G (10/2026)
"""

from pyfastflow.experimental.golem._fluvial import fluvial_factory, fluvial_plan
from pyfastflow.flow._program_cupy import _grid_leaf_plan
from pyfastflow.graphflood._base import FLAT, SHAPE, finish_program, graphflood_builder
from pyfastflow.graphflood._routing import _accumulation_factory

from ._mfd_parcels import base_level_factory, entrain_factory, uplift_factory, walk_factory

_MAX_PARCELS = 2**31 - 1


def build_mfd_parcel_sed_program() -> type:
    """Build the MFDParcelSedProgram class (see the module docstring)."""
    b = graphflood_builder("MFDParcelSedProgram")
    b.config("newton_iterations", default=8)
    b.config("threads", default=65536)
    b.config("deposition", choices=("gaussian", "exponential"),
             default="gaussian")
    b.config("parcel_path", choices=("graph", "surface"), default="graph")

    b.param("uplift", "auto", "f32", value=0.0, shape=SHAPE)
    b.param("erodibility", "auto", "f32", value=0.0, shape=SHAPE)
    b.param("dt", "auto", "f32", value=1.0, shape=SHAPE)
    b.param("m", "auto", "f32", value=0.45, shape=SHAPE)
    b.param("n", "auto", "f32", value=1.0, shape=SHAPE)
    b.param("slope_floor", "auto", "f32", value=1.0e-6, shape=SHAPE)
    b.param("transport_distance", "auto", "f32", value=1.0e3, shape=SHAPE)
    b.param("transport_spread", "auto", "f32", value=5.0e2, shape=SHAPE)
    b.param("parcel_volume", "auto", "f32", value=1.0e4)
    b.param("min_volume", "auto", "f32", value=1.0e2)
    b.param("unit_source", "const", "f32", value=1.0)
    b.param("cell_count", "const", "i32",
            value=lambda dims: dims["nx"] * dims["ny"])
    b.param("seed", "scalar", "i32", value=1)
    b.param("clock", "scalar", "i32", value=0)

    b.data("erosion_rate", "f32", SHAPE, role="output")
    b.data("entrained", "f32", SHAPE, role="output")
    b.data("parcel_count", "f64", (FLAT,), role="internal")
    b.data("parcel_cdf", "f64", (FLAT,), role="internal")
    b.data("parcel_claim", "i32", (1,), role="internal")
    b.data("parcel_stats", "f64", (3,), role="internal")
    for name in ("fluvial_rec_a", "fluvial_rec_b"):
        b.data(name, "i32", (FLAT,), lifetime="temp")
    for name in ("fluvial_z0", "fluvial_guess", "fluvial_z_a", "fluvial_z_b",
                 "fluvial_alpha_a", "fluvial_alpha_b"):
        b.data(name, "f32", (FLAT,), lifetime="temp")

    b.add("accumulate_area", _accumulation_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "PRECIPITATION": "unit_source", "Qi": "Qi",
              "frontier0": "mfd_frontier0", "frontier1": "mfd_frontier1",
              "count": "mfd_count", "barrier": "mfd_barrier",
              "dirs": "directions", "mfd_w": "weights", "accum": "Qi",
              "indegree": "indegree",
          }))
    b.add("hold_base_level", base_level_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "CELL_COUNT": "cell_count",
          }))
    b.add("apply_uplift", uplift_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "CELL_COUNT": "cell_count", "UPLIFT": "uplift",
              "DT": "dt",
          }))

    def fluvial_area_plan(frozen, be):
        return {key: "Qi" if value == "drainage_area" else value
                for key, value in fluvial_plan(frozen, be).items()}

    b.add("fluvial", fluvial_factory, bind=fluvial_area_plan)
    b.add("entrain", entrain_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "erosion_rate": "erosion_rate", "entrained": "entrained",
              "count": "parcel_count", "CELL_COUNT": "cell_count",
              "DT": "dt", "PARCEL_VOLUME": "parcel_volume",
          }))
    b.add("walk_parcels", walk_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "cdf": "parcel_cdf", "entrained": "entrained",
              "dirs": "directions", "weights": "weights", "z": "z",
              "claim": "parcel_claim", "stats": "parcel_stats",
              "CELL_COUNT": "cell_count", "MIN_VOLUME": "min_volume",
              "SEED": "seed", "CLOCK": "clock",
              "TRANSPORT_DISTANCE": "transport_distance",
              "TRANSPORT_SPREAD": "transport_spread",
          }))

    b.pipeline("_initialize", ("reset_h", "hold_base_level"))
    b.pipeline("_erode", (
        "apply_uplift", "make_surface", "route_local_minima",
        "snapshot_local_minima", "resolve_minima", "prepare_mfd_surface",
        "refresh_hydraulic_surface", "build_topology", "prepare_frontier",
        "accumulate_area", "fluvial", "hold_base_level", "entrain",
    ))
    b.pipeline("_walk", ("walk_parcels", "hold_base_level"))

    program = b.freeze()

    def initialize(self):
        """Outlets to 0, h to 0."""
        self._initialize()

    def run_n_step(self, n=1):
        """Run ``n`` steps; return ``{"parcels", "entrained", "deposited",
        "exported"}`` summed over them (volumes in m3)."""
        import cupy as cp

        totals = {"parcels": 0, "entrained": 0.0, "deposited": 0.0,
                  "exported": 0.0}
        clock = getattr(self, "_clock", 0)
        cdf = self._handle("parcel_cdf").array.ravel()
        for _ in range(int(n)):
            self._erode()
            entrained = self._handle("entrained").array
            totals["entrained"] += float(cp.sum(entrained, dtype=cp.float64).item())
            cp.cumsum(self._handle("parcel_count").array.ravel(), out=cdf)
            if float(cdf[-1].item()) > _MAX_PARCELS:
                raise ValueError("too many parcels in one step: raise parcel_volume")
            self._handle("parcel_claim").array.fill(0)
            self._handle("parcel_stats").array.fill(0)
            self.clock.set(clock)
            self._walk()
            raw = self._handle("parcel_stats").to_numpy()
            totals["parcels"] += int(raw[0])
            totals["deposited"] += float(raw[1])
            totals["exported"] += float(raw[2])
            clock += 1
        object.__setattr__(self, "_clock", clock)
        return totals

    program.initialize = initialize
    program.run_n_step = run_n_step
    return finish_program(program)


MFDParcelSedProgram = build_mfd_parcel_sed_program()

__all__ = ["MFDParcelSedProgram", "build_mfd_parcel_sed_program"]
