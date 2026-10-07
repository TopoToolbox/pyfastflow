"""GraphFloodParticles: GraphFlood steady state by live particle processors.

CuPy only. Shared config, inputs and initial surfaces: see ``_base``.

``warmup(n=5)``, once:
  1. n passes of the capped analytical warm-up (``warmup_relaxation``,
     ``depth_cap``), which fills the hillslopes;
  2. MFD topology on the warmed surface and one accumulation of the rain
     into Qi; the discharge field (per-cell discharge and per-link sends)
     starts from it;
  3. sources: valid cells whose Qi is at or above the
     ``source_percentile``-th percentile of the non-zero Qi;
  4. active area: the sources plus the steepest-descent paths downstream of
     them (steepest receiver of the warm-up DAG). Particles spawn there.
  Returns the number of active cells and resets the convergence checks.

``run(n=1)``: n launches of ``n_particles`` particles. Particles carry no
water: each is a processor that starts on a cell of the active area
(uniform), moved by a random offset of up to ``spawn_pad`` rows and columns
through the grid neighbours (redrawn if it lands on a cell without
discharge), and walks at most ``walk_steps`` cells down the live surface.
At each cell it fills a pit to just above its lowest neighbour, takes as
discharge the rain plus what its currently higher neighbours send, rewrites
its own drop-weighted MFD sends and relaxes h by ``relaxation`` towards the
newton depth of that discharge against its steepest receiver. With
``propagate`` > 0, cells whose inflow changed by more than that fraction are
processed by the same thread before its next particle (0: off). Cells only
write their own sends, so water is conserved; outlets pass their discharge
out. Returns the launch statistics.

Convergence checks (``convergence()``, ``run_until_converged``, with
``run_until_converged``'s run unit a launch): see ``_convergence``; they
measure the discharge field Qacc and add staleness, outlet balance and
coverage of the active area.

``finish()`` writes Qi (the discharge field) and Qo (Manning outflow of the
final depth) and returns the residual sum|Qo - Qi| / sum Qi.

Typical use::

    gf = GraphFloodParticles(be, nx=nx, ny=ny, dx=dx, precipitation=rain)
    gf.z.from_numpy(z)
    gf.fill_topography()
    gf.warmup()
    gf.run(10)
    gf.finish()

Author: B.G (10/2026)
"""

from pyfastflow.core import ProgramError
from pyfastflow.flow._program_cupy import _grid_leaf_plan

from ._base import FLAT, TOPOLOGY, add_warmup, finish_program, graphflood_builder
from ._convergence import add_convergence, attach_convergence
from ._particle import outflow_factory, qinit_factory, reach_factory, walk_factory


def build_graphflood_particles() -> type:
    """Build the GraphFloodParticles class."""
    b = add_warmup(graphflood_builder("GraphFloodParticles"))
    b.config("threads", default=32768)

    b.param("cell_count", "const", "i32",
            value=lambda dims: dims["nx"] * dims["ny"])
    b.param("n_particles", "auto", "i32", value=1_000_000)
    b.param("relaxation", "auto", "f32", value=0.01)
    b.param("propagate", "auto", "f32", value=0.5)
    b.param("spawn_pad", "auto", "i32", value=2)
    b.param("walk_steps", "auto", "i32", value=100)
    b.param("source_percentile", "auto", "f32", value=99.0)
    b.param("source_q", "scalar", "f32", value=0.0)
    b.param("clock_base", "scalar", "i32", value=0)
    b.param("seed", "scalar", "i32", value=1)

    b.data("Qacc", "f64", (FLAT,), role="internal")
    b.data("pushed", "f32", (8 * FLAT,), role="internal")
    b.data("visits", "u32", (FLAT,), role="internal")
    b.data("locks", "i32", (FLAT,), role="internal")
    b.data("claim", "i32", (2,), role="internal")
    b.data("stats", "f64", (8,), role="internal")
    b.data("reach", "i32", (FLAT,), role="internal")
    b.data("reach_frontier0", "i32", (FLAT,), role="internal")
    b.data("reach_frontier1", "i32", (FLAT,), role="internal")
    b.data("reach_count", "i32", (3,), role="internal")
    b.data("reach_barrier", "u32", (1,), role="internal")
    b.data("spawn_cdf", "f64", (FLAT,), role="internal")
    b.data("final_stats", "f32", (4,), role="internal")

    b.pipeline("_topology", TOPOLOGY)
    b.add("init_field", qinit_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "Qi": "Qi", "directions": "directions", "weights": "weights",
              "Qacc": "Qacc", "pushed": "pushed", "CELL_COUNT": "cell_count",
          }))
    b.add("sweep_active_area", reach_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "Qi": "Qi", "reach": "reach", "dirs": "directions",
              "weights": "weights", "frontier0": "reach_frontier0",
              "frontier1": "reach_frontier1", "count": "reach_count",
              "barrier": "reach_barrier", "CELL_COUNT": "cell_count",
              "SOURCE_Q": "source_q",
          }))
    b.add("walk", walk_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "cdf": "spawn_cdf", "z": "z", "h": "h", "Qacc": "Qacc",
              "pushed": "pushed", "dirs": "directions",
              "weights": "weights", "visits": "visits", "locks": "locks",
              "claim": "claim", "stats": "stats",
              "CELL_COUNT": "cell_count", "N_PARTICLES": "n_particles",
              "CLOCK_BASE": "clock_base", "PRECIPITATION": "precipitation",
              "MANNING": "friction_coefficient",
              "FRICTION_EXPONENT": "friction_exponent",
              "RELAXATION": "relaxation", "WALK_STEPS": "walk_steps",
              "SPAWN_PAD": "spawn_pad", "SEED": "seed",
              "PROPAGATE": "propagate",
          }))
    b.add("compute_outflow", outflow_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "h": "h", "Qacc": "Qacc", "Qi": "Qi", "Qo": "Qo",
              "stats": "final_stats", "CELL_COUNT": "cell_count",
              "MANNING": "friction_coefficient",
              "FRICTION_EXPONENT": "friction_exponent",
          }))
    add_convergence(b, particles=True)
    program = b.freeze()

    def warmup(self, n=5):
        """Warm-up, sources and active area; returns the active cell count.

        See the module docstring. Set z, h, the masks and precipitation
        first.
        """
        import cupy as cp

        if n > 0:
            self._warmup_pass(n)
        self._topology()
        qi = self._handle("Qi").array.ravel()
        positive = qi[qi > 0.0]
        if positive.size == 0:
            raise ProgramError("no discharge: set precipitation")
        self.source_q.set(float(cp.percentile(
            positive, float(self.source_percentile.read()))))
        self.init_field()
        self._handle("locks").array.fill(0)
        self._handle("visits").array.fill(0)
        self._handle("reach_count").array.fill(0)
        self._handle("reach_barrier").array.fill(0)
        self.sweep_active_area()
        cdf = self._handle("spawn_cdf").array
        cp.cumsum(self._handle("reach").array != 0, dtype=cp.float64, out=cdf)
        object.__setattr__(self, "_clock", 0)
        self.reset_convergence()
        return int(cdf[-1].item())

    def run(self, n=1):
        """Launch ``n`` times ``n_particles`` particles; return statistics.

        Fractions of the particles: ``launched``, ``exit`` (reached an
        outlet), ``limited`` (stopped at ``walk_steps``), ``stuck``;
        ``skipped``: fraction of visits that found the cell locked;
        ``propagated``: correction processings per particle processing;
        ``rejected``: spawn draws redrawn (no discharge, nodata) per
        particle.
        """
        if getattr(self, "_clock", None) is None:
            raise ProgramError("call warmup() first")
        particles = int(self.n_particles.read())
        if particles < 1:
            raise ProgramError("n_particles must be >= 1")
        self._handle("stats").array.fill(0)
        for _ in range(int(n)):
            self._handle("claim").array.fill(0)
            self.clock_base.set(self._clock)
            self.walk()
            object.__setattr__(self, "_clock", self._clock + particles)
        raw = self._handle("stats").to_numpy()
        launched = float(raw[0])
        visits = float(raw[4] + raw[5])
        return {
            "launched": launched / (particles * max(int(n), 1)),
            "exit": float(raw[1]) / launched if launched else 0.0,
            "limited": float(raw[2]) / launched if launched else 0.0,
            "stuck": float(raw[3]) / launched if launched else 0.0,
            "skipped": float(raw[4]) / visits if visits else 0.0,
            "propagated": float(raw[6]) / float(raw[5]) if raw[5] else 0.0,
            "rejected": float(raw[7]) / launched if launched else 0.0,
        }

    def finish(self):
        """Write Qi and Qo; return ``{"residual", "cells"}``.

        ``residual`` is sum|Qo - Qi| / sum Qi over valid non-outlet cells,
        ``cells`` their count.
        """
        self._handle("final_stats").array.fill(0)
        self.compute_outflow()
        stats = self._handle("final_stats").to_numpy()
        return {
            "residual": float(stats[0] / stats[1]) if stats[1] > 0 else 0.0,
            "cells": int(stats[2]),
        }

    program.warmup = warmup
    program.run = run
    program.finish = finish
    return attach_convergence(finish_program(program), discharge="Qacc",
                              particles=True)


GraphFloodParticles = build_graphflood_particles()

__all__ = ["GraphFloodParticles", "build_graphflood_particles"]
