"""CellularSedProgram: cellular erosion/deposition with dynamic receivers.

CuPy only. Every quantity is local and advances by explicit steps of
``dt`` (yr), so the routing follows the surface as it changes (after Jain,
Benes and Cordonnier 2024). A step:

  1. receivers of the current z: a lower neighbour drawn with probability
     proportional to its slope among those with a lower spill level, else
     the receiver of the last routing (see ``_sed_law.py``), the upwind
     slope and the drop to the lowest neighbour;
  2. the sediment flux ``Qs`` (m3/yr) advances one hop: a cell takes that
     of the cells draining to it at the previous step; the discharge
     ``Qi`` (m3/yr) follows ``discharge``:
       - ``hop``: one hop as Qs, plus the cell's ``precipitation`` dx^2;
       - ``accumulate``: ``precipitation`` dx^2 accumulated over the
         current receivers (pointer jumping);
  3. the ``law`` (davy_lague | cover | lift_impact, ``_sed_law.py``) with
     the incoming volume Qs_in dt updates z, the sediment layer
     ``sediment`` (h_s) and the leaving flux; with ``deposition_cap`` the
     deposit cannot raise z above the Qs-weighted mean elevation of the
     givers (the excess leaves with Qs); z gains ``uplift`` dt;
  4. outlets stay at z = 0 and export their incoming Qs.

Every ``routing_interval`` steps the local minima are routed again
(GraphFlood carve_cordonnier on z, h = 0): ``rec`` and the spill level
(path maximum) ``z_prime`` that keep the receivers acyclic.
``initialize()`` routes, accumulates ``precipitation`` dx^2 over the MFD
graph into Qi, and sets Qs, h_s and the exports to 0.

``run_n_step(n)`` returns the time (yr) and the volume exported (m3) over
the n steps.

Author: B.G (10/2026)
"""

from pyfastflow.flow._program_cupy import _grid_leaf_plan
from pyfastflow.graphflood._base import FLAT, SHAPE, TOPOLOGY, finish_program, graphflood_builder

from ._cellular import (
    accumulation_factory, commit_factory, receivers_factory, scale_factory,
    update_factory,
)
from ._mfd_parcels import base_level_factory
from ._sed_law import LAW_BIND, LAWS, law_params

_ROUTE = ("make_surface", "route_local_minima", "snapshot_local_minima",
          "resolve_minima", "prepare_mfd_surface")


def build_cellular_sed_program() -> type:
    """Build the CellularSedProgram class (see the module docstring)."""
    b = graphflood_builder("CellularSedProgram")
    b.config("law", choices=LAWS, default="davy_lague")
    b.config("deposition_cap", choices=(True, False), default=True)
    b.config("routing_interval", default=20)
    b.config("discharge", choices=("accumulate", "hop"), default="accumulate")

    law_params(b, SHAPE)
    b.param("uplift", "auto", "f32", value=0.0, shape=SHAPE)
    b.param("dt", "auto", "f32", value=1.0)
    b.param("cell_count", "const", "i32",
            value=lambda dims: dims["nx"] * dims["ny"])
    b.param("seed", "scalar", "i32", value=1)

    b.data("sediment", "f32", SHAPE, role="output")
    b.data("Qs", "f32", SHAPE, role="output")
    b.data("exported", "f32", SHAPE, role="output")
    for name in ("z_next", "sediment_next", "Q_next", "Qs_next", "slope",
                 "drop"):
        b.data(name, "f32", (FLAT,), role="internal")
    b.data("receiver", "i32", (FLAT,), role="internal")
    for name in ("pj_work", "pj_work2"):
        b.data(name, "i32", (FLAT,), lifetime="temp")
    b.data("pj_q_work", "f32", (FLAT,), lifetime="temp")
    b.data("step", "u32", (1,), role="internal")

    b.add("hold_base_level", base_level_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "CELL_COUNT": "cell_count",
          }))
    b.add("receivers", receivers_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "spill": "z_prime", "rec": "rec", "step": "step",
              "recv": "receiver", "slope_out": "slope", "drop_out": "drop",
              "CELL_COUNT": "cell_count", "SEED": "seed",
          }))
    b.add("update", update_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "hs": "sediment", "Q": "Qi", "Qs": "Qs",
              "recv": "receiver", "slope": "slope", "drop": "drop",
              "z_next": "z_next", "hs_next": "sediment_next",
              "Q_next": "Q_next", "Qs_next": "Qs_next",
              "exported": "exported", "CELL_COUNT": "cell_count",
              "DT": "dt", "UPLIFT": "uplift",
              "PRECIPITATION": "precipitation", **LAW_BIND,
          }))
    b.add("accumulate_discharge", accumulation_factory, bind={
        "q_init.SOURCE": "precipitation", "q_init.q": "Q_next",
        "copy_rec_to_work.rec": "receiver", "copy_rec_to_work.work": "pj_work",
        "step_a_copy.q_curr": "Q_next", "step_a_copy.q_next": "pj_q_work",
        "step_a_core.rec_curr": "pj_work", "step_a_core.rec_next": "pj_work2",
        "step_a_core.q_curr": "Q_next", "step_a_core.q_next": "pj_q_work",
        "step_b_copy.q_curr": "pj_q_work", "step_b_copy.q_next": "Q_next",
        "step_b_core.rec_curr": "pj_work2", "step_b_core.rec_next": "pj_work",
        "step_b_core.q_curr": "pj_q_work", "step_b_core.q_next": "Q_next",
    })
    b.add("scale_discharge", scale_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "Q": "Q_next", "CELL_COUNT": "cell_count",
          }))
    b.add("commit", commit_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z_next": "z_next", "hs_next": "sediment_next",
              "Q_next": "Q_next", "Qs_next": "Qs_next", "z": "z",
              "hs": "sediment", "Q": "Qi", "Qs": "Qs", "step": "step",
              "CELL_COUNT": "cell_count",
          }))

    b.pipeline("_initialize", ("reset_h", "hold_base_level") + TOPOLOGY)
    b.pipeline("_route", _ROUTE)
    b.pipeline("_step_hop", ("receivers", "update", "commit"))
    b.pipeline("_step_accumulate", ("receivers", "accumulate_discharge",
                                    "scale_discharge", "update", "commit"))

    program = b.freeze()

    def initialize(self):
        """Route, accumulate the rain into Qi; Qs, h_s, exports to 0."""
        for name in ("sediment", "Qs", "exported", "step"):
            self._handle(name).array.fill(0)
        self._initialize()
        object.__setattr__(self, "_steps", 0)

    def run_n_step(self, n=1):
        """Run ``n`` steps; return ``{"time", "exported"}`` (yr, m3)."""
        import cupy as cp

        interval = max(int(self.routing_interval), 1)
        steps = getattr(self, "_steps", 0)
        exported = self._handle("exported").array
        exported.fill(0)
        step = self._step_hop if self.discharge == "hop" else self._step_accumulate
        for _ in range(int(n)):
            if steps % interval == 0:
                self._route()
            step()
            steps += 1
        object.__setattr__(self, "_steps", steps)
        return {"time": steps * float(self.dt.read()),
                "exported": float(cp.sum(exported, dtype=cp.float64).item())}

    program.initialize = initialize
    program.run_n_step = run_n_step
    return finish_program(program)


CellularSedProgram = build_cellular_sed_program()

__all__ = ["CellularSedProgram", "build_cellular_sed_program"]
