"""PrecipitonSedProgram: erosion/deposition carried by precipitons.

CuPy only. Precipitons are small water volumes carrying sediment over the
current surface (after Davy et al. 2026, Earth Surf. Dynam. 14). Outlets
are held at z = 0. A launch covers ``launch_time`` (yr):

  1. uplift by ``uplift`` * ``launch_time`` (outlets set to 0);
  2. local minima routed on z (GraphFlood carve_cordonnier, h = 0): the
     receivers ``rec`` and spill level ``z_prime`` used by the receiver
     rule of ``_sed_law.py``;
  3. ``n_precipitons`` precipitons land on uniformly drawn cells, each with
     the water volume V = ``precipitation`` ``launch_time`` dx^2 cells /
     ``n_precipitons``, and walk at most ``walk_steps`` cells to a receiver
     of the current surface (lower neighbour drawn in proportion to its
     slope, else ``rec``). On each cell a precipiton adds V to the water of
     the launch and applies the ``law`` (davy_lague | cover | lift_impact,
     ``_sed_law.py``) over the time it represents, dt = V / Q, with its
     load as the incoming volume: z and the sediment layer ``sediment``
     change at once (atomic adds), and it carries the leaving volume.
     Outlets export the load; a precipiton stopped before one leaves it on
     its cell;
  4. the discharge ``Qi`` (m3/yr) relaxes towards the water of the launch
     / ``launch_time`` (floored at the cell's own rain) with
     ``discharge_relaxation`` in [0, 1].

``initialize()`` sets the outlets and h to 0, h_s to 0, and Qi to
``precipitation`` dx^2 accumulated over the MFD graph of the initial
surface. ``run(n)`` performs n launches and returns, summed over them, the
precipitons, the eroded, deposited and exported volumes (m3) and the
precipitons stopped before an outlet.

Author: B.G (10/2026)
"""

from pyfastflow.flow._program_cupy import _grid_leaf_plan
from pyfastflow.graphflood._base import FLAT, SHAPE, TOPOLOGY, finish_program, graphflood_builder

from ._mfd_parcels import base_level_factory, uplift_factory
from ._precipiton import discharge_factory, walk_factory
from ._sed_law import LAW_BIND, LAWS, law_params

_ROUTE = ("make_surface", "route_local_minima", "snapshot_local_minima",
          "resolve_minima", "prepare_mfd_surface")
_STATS = ("precipitons", "eroded", "deposited", "exported", "stopped")


def build_precipiton_sed_program() -> type:
    """Build the PrecipitonSedProgram class (see the module docstring)."""
    b = graphflood_builder("PrecipitonSedProgram")
    b.config("law", choices=LAWS, default="davy_lague")
    b.config("threads", default=65536)

    law_params(b, SHAPE)
    b.param("uplift", "auto", "f32", value=0.0, shape=SHAPE)
    b.param("launch_time", "auto", "f32", value=1.0)
    b.param("n_precipitons", "auto", "i32", value=65536)
    b.param("walk_steps", "auto", "i32", value=10000)
    b.param("discharge_relaxation", "auto", "f32", value=0.5)
    b.param("cell_count", "const", "i32",
            value=lambda dims: dims["nx"] * dims["ny"])
    b.param("seed", "scalar", "i32", value=1)
    b.param("clock", "scalar", "i32", value=0)

    b.data("sediment", "f32", SHAPE, role="output")
    b.data("water", "f32", (FLAT,), role="internal")
    b.data("claim", "i32", (1,), role="internal")
    b.data("walk_stats", "f64", (len(_STATS),), role="internal")

    b.add("hold_base_level", base_level_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "CELL_COUNT": "cell_count",
          }))
    b.add("apply_uplift", uplift_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "CELL_COUNT": "cell_count", "UPLIFT": "uplift",
              "DT": "launch_time",
          }))
    b.add("walk", walk_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "rec": "rec", "spill": "z_prime", "Q": "Qi", "z": "z",
              "hs": "sediment", "water": "water", "claim": "claim",
              "stats": "walk_stats", "CELL_COUNT": "cell_count",
              "N_PRECIPITONS": "n_precipitons", "WALK_STEPS": "walk_steps",
              "LAUNCH_TIME": "launch_time", "SEED": "seed",
              "CLOCK": "clock", "PRECIPITATION": "precipitation",
              **LAW_BIND,
          }))
    b.add("update_discharge", discharge_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "Q": "Qi", "water": "water", "hs": "sediment",
              "CELL_COUNT": "cell_count", "LAUNCH_TIME": "launch_time",
              "PRECIPITATION": "precipitation",
              "DISCHARGE_RELAXATION": "discharge_relaxation",
          }))

    b.pipeline("_initialize", ("reset_h", "hold_base_level") + TOPOLOGY)
    b.pipeline("_launch", ("apply_uplift",) + _ROUTE
               + ("walk", "update_discharge", "hold_base_level"))

    program = b.freeze()

    def initialize(self):
        """Outlets and h to 0, h_s to 0, Qi from the initial MFD graph."""
        for name in ("sediment", "water"):
            self._handle(name).array.fill(0)
        self._initialize()
        object.__setattr__(self, "_clock", 0)

    def run(self, n=1):
        """Perform ``n`` launches; return the statistics summed over them."""
        totals = dict.fromkeys(_STATS, 0.0)
        clock = getattr(self, "_clock", 0)
        for _ in range(int(n)):
            self._handle("claim").array.fill(0)
            self._handle("walk_stats").array.fill(0)
            self.clock.set(clock)
            self._launch()
            raw = self._handle("walk_stats").to_numpy()
            for name, value in zip(_STATS, raw):
                totals[name] += float(value)
            clock += 1
        object.__setattr__(self, "_clock", clock)
        return totals

    program.initialize = initialize
    program.run = run
    return finish_program(program)


PrecipitonSedProgram = build_precipiton_sed_program()

__all__ = ["PrecipitonSedProgram", "build_precipiton_sed_program"]
