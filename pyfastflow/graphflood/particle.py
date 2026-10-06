"""GraphFlood steady state by live particle processors (CuPy only).

``GraphFloodParticleProgram`` owns z, h, rain (scalar or field), the grid
masks and every buffer of the method.

Pre-step (``prestep``), once:
  1. reconstruction + epsilon fill of z+h (depressions fill with water),
     MFD topology on that surface and one accumulation of the rain into Qi;
  2. cells with Qi >= ``channel_q`` are river cells; the discharge field
     (per-cell discharge and per-link sends) starts from that accumulation;
  3. one hillslope accumulation with the river cells as sinks, then every
     other cell takes the Manning depth of its hillslope discharge at the
     pre-step steepest slope (undamped analytical solve);
  4. the D8 paths downstream of the river cells (steepest receiver of the
     pre-step DAG) are the spawning area; ``reset_h`` sets depth to 0 there
     and ``reset_q`` brings their discharge back to rain plus the inflow
     from outside the area.

Particles (``run_particles``) carry no water: each is a processor that starts on a
cell of the spawning area (uniform, + up to ``spawn_pad`` rows and columns of
noise; with ``focus``, starts are redrawn with a probability falling with
the cell's live |Qo/Q - 1| down to ``focus_floor``) and walks at most
``walk_steps`` cells down the live surface. At each cell it fills a pit to
just above its lowest neighbour, takes as discharge the rain plus what its
currently higher neighbours send, rewrites its own drop-weighted MFD sends
and relaxes h by ``h_relaxation`` towards the depth of that discharge
(``h_update``: "newton", solved against the receiver's head, or
"pointwise", the Manning depth at the steepest slope floored at
``min_slope``). With ``propagate``, cells whose inflow changed by more than
``propagate_tolerance`` are processed by the same thread before its next
particle. Cells only write their own sends, so water is conserved; outlets
pass their discharge out.

``finish`` writes ``discharge`` and ``outflow`` (Manning outflow of the
final depth) and returns the residual sum|Qo - Q| / sum Q.

Typical use::

    gfp = GraphFloodParticleProgram(be, nx=nx, ny=ny, dx=dx,
                                    precipitation=rain)
    gfp.z.from_numpy(z)
    gfp.fill_topography()           # or initialize_h_from_fill()
    gfp.channel_q.set(...)
    gfp.prestep()
    for _ in range(launches):
        gfp.run_particles(particles)
    gfp.finish()

Author: B.G (10/2026)
"""

from pyfastflow.core import ProgramError
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow._program_cupy import (
    _cupy_only, _grid_leaf_plan, _reconstruct_epsilon_factory,
)
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.grid._mask import GridMaskAccessor as _GridMaskAccessor

from ._particle import (
    accumulation_factory, count_indegree_factory, fill_depth_factory,
    outflow_factory, qinit_factory, reach_factory, recount_inflow_factory,
    reset_area_factory, sources_factory, split_factory, topology_factory,
    walk_factory,
)
from ._routing import _frontier_factory
from ._surface import (
    _copy_fill_depth_factory, _fill_surface_plan, _make_surface_factory,
    _refresh_hydraulic_surface_factory, _reset_h_factory,
)


def build_graphflood_particle_program() -> type:
    """Build the GraphFloodParticleProgram class."""
    b = ProgramBuilder("GraphFloodParticleProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("threads", default=32768)
    b.config("h_update", choices=("newton", "pointwise"), default="newton")
    b.config("nodata", choices=(False, True), default=False)
    b.config("outlet", choices=("edge", "mask"), default="edge")
    flat = Dim("ny") * Dim("nx")
    shape = (Dim("ny"), Dim("nx"))

    b.param("cell_count", "const", "i32",
            value=lambda dims: dims["nx"] * dims["ny"])
    b.param("precipitation", "auto", "f32", value=0.0, shape=shape)
    b.param("manning", "scalar", "f32", value=0.033)
    b.param("friction_exponent", "scalar", "f32", value=2.0 / 3.0)
    b.param("channel_q", "scalar", "f32", value=1.0)
    b.param("drop_min", "scalar", "f32", value=1.0e-6)
    b.param("h_relaxation", "scalar", "f32", value=0.01)
    b.param("walk_steps", "scalar", "i32", value=100)
    b.param("spawn_pad", "scalar", "i32", value=2)
    b.param("focus", "scalar", "i32", value=0)
    b.param("focus_floor", "scalar", "f32", value=0.05)
    b.param("min_slope", "scalar", "f32", value=1.0e-5)
    b.param("propagate", "scalar", "i32", value=0)
    b.param("propagate_tolerance", "scalar", "f32", value=0.2)
    b.param("reset_h", "scalar", "i32", value=1)
    b.param("reset_q", "scalar", "i32", value=0)
    b.param("n_particles", "scalar", "i32", value=0)
    b.param("clock_base", "scalar", "i32", value=0)
    b.param("seed", "scalar", "i32", value=1)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    b.data("z", "f32", shape, role="input", shape_source=True)
    b.data("h", "f32", shape, role="state")
    b.data("Qi", "f32", shape, role="internal")
    b.data("discharge", "f32", shape, role="output")
    b.data("outflow", "f32", shape, role="output")
    b.data("owned", "u8", shape, role="output")
    b.data("sources", "f32", shape, role="internal")
    b.data("Qacc", "f64", shape, role="internal")
    b.data("surface", "f32", (flat,), role="internal")
    b.data("hydraulic_surface", "f64", (flat,), role="internal")
    b.data("filled", "f32", (flat,), role="internal")
    b.data("epsilon_distance", "f32", (flat,), role="internal")
    b.data("conditioned", "u8", (flat,), role="internal")
    b.data("directions", "u8", (flat,), role="internal")
    b.data("route_dirs", "u8", (flat,), role="internal")
    b.data("weights", "f32", (8 * flat,), role="internal")
    b.data("indegree", "i32", (flat,), role="internal")
    b.data("steepest_slope", "f32", (flat,), role="internal")
    b.data("flow_width", "f32", (flat,), role="internal")
    b.data("mfd_frontier0", "i32", (flat,), role="internal")
    b.data("mfd_frontier1", "i32", (flat,), role="internal")
    b.data("mfd_count", "i32", (3,), role="internal")
    b.data("mfd_barrier", "u32", (1,), role="internal")
    b.data("pushed", "f32", (8 * flat,), role="internal")
    b.data("visits", "u32", (flat,), role="internal")
    b.data("locks", "i32", (flat,), role="internal")
    b.data("imbalance", "f32", (flat,), role="internal")
    b.data("claim", "i32", (2,), role="internal")
    b.data("stats", "f64", (8,), role="internal")
    b.data("reach", "i32", (flat,), role="internal")
    b.data("reach_frontier0", "i32", (flat,), role="internal")
    b.data("reach_frontier1", "i32", (flat,), role="internal")
    b.data("reach_count", "i32", (3,), role="internal")
    b.data("reach_barrier", "u32", (1,), role="internal")
    b.data("spawn_cdf", "f64", (flat,), role="internal")
    b.data("final_stats", "f32", (4,), role="internal")
    for name in ("rec", "fill_parent", "fill_counters", "fill_queued_gen",
                 "fill_epsilon_ancestor", "fill_epsilon_ancestor_work"):
        b.data(name, "i32", (flat,), lifetime="temp")
    for name in ("fill_epsilon_distance", "fill_epsilon_distance_work"):
        b.data(name, "f32", (flat,), lifetime="temp")
    b.data("fill_frontier", "i32", (2 * flat,), lifetime="temp")

    def grid_structure(be, *, nodata, outlet, **_):
        _cupy_only(be)
        return make_grid_group(be, topology="D8", outlet=outlet, nodata=nodata)

    def grid_params(be, pool, *, nx, ny, dx, nodata, outlet):
        return make_grid_parameters(be, pool, nx, ny, dx, topology="D8",
                                    outlet=outlet, nodata=nodata)

    b.bundle("grid", grid_structure, grid_params, dims=("nx", "ny"),
             config=("dx", "nodata", "outlet"))

    # Initial surface.
    b.add("reset_h_all", _reset_h_factory, bind={"h": "h"})
    b.add("reconstruct_fill_surface", _reconstruct_epsilon_factory,
          bind=_fill_surface_plan("z", "surface"))
    b.add("copy_fill_depth", _copy_fill_depth_factory,
          bind={"z": "z", "filled": "surface", "h": "h"})
    b.pipeline("initialize_h_from_fill",
               ("reconstruct_fill_surface", "copy_fill_depth"))

    # Pre-step topology: reconstruct + epsilon fill of z+h, MFD on it.
    b.add("make_surface", _make_surface_factory, bind={
        "z": "z", "h": "h", "surface": "surface",
        "hydraulic_surface": "hydraulic_surface",
    })
    b.add("reconstruct_surface", _reconstruct_epsilon_factory,
          bind=_fill_surface_plan("surface", "filled",
                                  distance="epsilon_distance"))
    b.add("fill_depth", fill_depth_factory, bind={
        "z": "z", "filled": "filled", "h": "h", "conditioned": "conditioned",
        "CELL_COUNT": "cell_count",
    })
    b.add("refresh_surface", _refresh_hydraulic_surface_factory, bind={
        "z": "z", "h": "h", "hydraulic_surface": "hydraulic_surface",
    })
    b.add("build_topology", topology_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "filled": "filled", "dist": "epsilon_distance",
              "physical_surface": "hydraulic_surface",
              "conditioned": "conditioned", "dirs": "directions",
              "mfd_w": "weights", "steepest_slope": "steepest_slope",
              "flow_width": "flow_width", "indegree": "indegree",
          }))
    b.pipeline("prestep_topology", (
        "make_surface", "reconstruct_surface", "fill_depth",
        "refresh_surface", "build_topology",
    ))
    b.add("prepare_frontier", _frontier_factory, bind={
        "clear.count": "mfd_count", "clear.barrier": "mfd_barrier",
        "compact.indegree": "indegree", "compact.frontier": "mfd_frontier0",
        "compact.count": "mfd_count",
    })
    b.add("accumulate", accumulation_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "PRECIPITATION": "precipitation", "Qi": "Qi",
              "frontier0": "mfd_frontier0", "frontier1": "mfd_frontier1",
              "count": "mfd_count", "barrier": "mfd_barrier",
              "dirs": "directions", "mfd_w": "weights", "accum": "Qi",
              "indegree": "indegree",
          }))
    b.add("count_indegree", count_indegree_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "dirs": "directions", "indegree": "indegree",
          }))

    # Discharge field, river split and analytical hillslope.
    b.add("init_field", qinit_factory, bind={
        "Qi": "Qi", "directions": "directions", "weights": "weights",
        "Qacc": "Qacc", "pushed": "pushed", "CELL_COUNT": "cell_count",
    })
    b.add("split_rivers", split_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "Qi": "Qi", "directions": "directions",
              "indegree": "indegree", "route_dirs": "route_dirs",
              "owned": "owned", "CELL_COUNT": "cell_count",
              "CHANNEL_Q": "channel_q",
          }))
    b.add("solve_hillslope", sources_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "Qi": "Qi", "owned": "owned",
              "steepest_slope": "steepest_slope", "flow_width": "flow_width",
              "h": "h", "sources": "sources", "CELL_COUNT": "cell_count",
              "MANNING": "manning", "FRICTION_EXPONENT": "friction_exponent",
          }))

    # Spawning area.
    b.add("sweep_spawning_area", reach_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "sources": "sources", "reach": "reach", "dirs": "route_dirs",
              "weights": "weights", "frontier0": "reach_frontier0",
              "frontier1": "reach_frontier1", "count": "reach_count",
              "barrier": "reach_barrier", "CELL_COUNT": "cell_count",
          }))
    b.add("reset_area", reset_area_factory, bind={
        "reach": "reach", "h": "h", "pushed": "pushed",
        "CELL_COUNT": "cell_count", "RESET_H": "reset_h",
        "RESET_Q": "reset_q",
    })
    b.add("recount_inflow", recount_inflow_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "reach": "reach", "pushed": "pushed", "Qacc": "Qacc",
              "CELL_COUNT": "cell_count", "PRECIPITATION": "precipitation",
          }))

    # Particles and output.
    b.add("walk", walk_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "cdf": "spawn_cdf", "z": "z", "h": "h", "Qacc": "Qacc",
              "pushed": "pushed", "dirs": "route_dirs",
              "weights": "weights", "visits": "visits", "locks": "locks",
              "imbalance": "imbalance", "claim": "claim", "stats": "stats",
              "CELL_COUNT": "cell_count", "N_PARTICLES": "n_particles",
              "CLOCK_BASE": "clock_base", "PRECIPITATION": "precipitation",
              "MANNING": "manning", "FRICTION_EXPONENT": "friction_exponent",
              "H_RELAXATION": "h_relaxation", "DROP_MIN": "drop_min",
              "WALK_STEPS": "walk_steps", "SPAWN_PAD": "spawn_pad",
              "SEED": "seed", "PROPAGATE": "propagate",
              "PROPAGATE_TOLERANCE": "propagate_tolerance",
              "FOCUS": "focus", "FOCUS_FLOOR": "focus_floor",
              "MIN_SLOPE": "min_slope",
          }))
    b.add("compute_outflow", outflow_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "h": "h", "Qacc": "Qacc", "discharge": "discharge",
              "outflow": "outflow", "stats": "final_stats",
              "CELL_COUNT": "cell_count", "MANNING": "manning",
              "FRICTION_EXPONENT": "friction_exponent",
              "MIN_SLOPE": "min_slope",
          }))
    program = b.freeze()

    def fill_topography(self):
        """Replace z by its reconstructed epsilon fill and set h to 0.

        Depressions become topography (flat lakes with epsilon gradients).
        ``initialize_h_from_fill`` instead keeps z and fills them with water.
        """
        self.reconstruct_fill_surface()
        z = self._handle("z").array
        z[...] = self._handle("surface").array.reshape(z.shape)
        self.reset_h_all()

    def prestep(self, reset_h=True, reset_q=False, h_init=None):
        """One-shot pre-step; returns the number of spawning cells.

        See the module docstring. ``h_init`` (an (ny, nx) array) replaces
        the pre-step depth everywhere before the spawning area is reset.
        Set z, h, the masks, precipitation and ``channel_q`` first.
        """
        import cupy as cp

        self.prestep_topology()
        self.prepare_frontier()
        self.accumulate()
        self.count_indegree()
        self.init_field()
        self.split_rivers()
        self.prepare_frontier()
        self.accumulate()
        self.solve_hillslope()
        if h_init is not None:
            h = self._handle("h").array
            h[...] = cp.asarray(h_init, dtype=cp.float32).reshape(h.shape)
        self._handle("locks").array.fill(0)
        self._handle("visits").array.fill(0)
        self._handle("imbalance").array.fill(1.0)
        self._handle("reach_count").array.fill(0)
        self._handle("reach_barrier").array.fill(0)
        self.sweep_spawning_area()
        cdf = self._handle("spawn_cdf").array
        cp.cumsum(self._handle("reach").array != 0, dtype=cp.float64, out=cdf)
        if reset_h or reset_q:
            self.reset_h.set(int(bool(reset_h)))
            self.reset_q.set(int(bool(reset_q)))
            self.reset_area()
            if reset_q:
                self.recount_inflow()
        object.__setattr__(self, "_clock", 0)
        return int(cdf[-1].item())

    def run_particles(self, particles):
        """Release ``particles`` particles and return their statistics.

        Fractions of the particles: ``launched``, ``exit`` (reached an
        outlet), ``limited`` (stopped at ``walk_steps``), ``stuck``;
        ``skipped``: fraction of visits that found the cell locked;
        ``propagated``: correction processings per particle processing;
        ``rejected``: start draws refused (focus, nodata) per particle.
        """
        if getattr(self, "_clock", None) is None:
            raise ProgramError("call prestep() first")
        particles = int(particles)
        if particles < 1:
            raise ProgramError("particles must be >= 1")
        self._handle("claim").array.fill(0)
        self._handle("stats").array.fill(0)
        self.n_particles.set(particles)
        self.clock_base.set(self._clock)
        self.walk()
        object.__setattr__(self, "_clock", self._clock + particles)
        raw = self._handle("stats").to_numpy()
        visits = float(raw[4] + raw[5])
        return {
            "launched": float(raw[0]) / particles,
            "exit": float(raw[1]) / particles,
            "limited": float(raw[2]) / particles,
            "stuck": float(raw[3]) / particles,
            "skipped": float(raw[4]) / visits if visits else 0.0,
            "propagated": float(raw[6]) / float(raw[5]) if raw[5] else 0.0,
            "rejected": float(raw[7]) / float(raw[0]) if raw[0] else 0.0,
        }

    def finish(self):
        """Write ``discharge`` and ``outflow``; return the residual.

        ``residual`` is sum|Qo - Q| / sum Q over valid non-outlet cells,
        ``cells`` their count.
        """
        self._handle("final_stats").array.fill(0)
        self.compute_outflow()
        stats = self._handle("final_stats").to_numpy()
        return {
            "residual": float(stats[0] / stats[1]) if stats[1] > 0 else 0.0,
            "cells": int(stats[2]),
        }

    program.fill_topography = fill_topography
    program.prestep = prestep
    program.run_particles = run_particles
    program.finish = finish
    program.outlet_mask = property(
        lambda self: _GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"))
    program.nodata_mask = property(
        lambda self: _GridMaskAccessor(self, "NODATA_MASK", "nodata=True"))
    return program


GraphFloodParticleProgram = build_graphflood_particle_program()

__all__ = ["GraphFloodParticleProgram", "build_graphflood_particle_program"]
