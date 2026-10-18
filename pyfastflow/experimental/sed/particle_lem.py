"""ParticleLEMProgram: a landscape evolution model run by stochastic particles.

CuPy only. Outlets are held at z = 0 (base level): ``initialize()`` and
every launch set them to 0. ``initialize()`` computes the drainage area by pointer jumping on
the initial surface (SFD receivers with ``local_minima`` handling) and seeds
the MFD state with it: every cell sends its whole area to its steepest
receiver, no sediment flux, every cell last visited at time 0.

``run(n)`` performs n launches. A launch advances the clock by
``launch_time`` (yr), uplifts the grid by ``uplift`` * ``launch_time``
(outlets set to 0 instead), then walks ``n_particles`` particles, each spawned on a
uniformly drawn active cell and moving at most ``walk_steps`` cells to a
lower neighbour drawn with probability proportional to
slope^``mfd_exponent``, stopping at an outlet. Particles carry no water;
they are processors. A visited cell (skipped if another particle holds its
lock):

  - is raised ``fill_epsilon`` (m) above its lowest neighbour if it has no
    lower one, from a carried load first, the rest counted as filled volume
    (a particle finding such a cell locked waits for it and fills it, then
    goes on to its lowest neighbour);
  - takes its drainage area A = dx^2 + the area its higher neighbours send
    it;
  - integrates the time since its last visit dt: stream-power incision
    against its steepest receiver r, implicit for n = 1,
    z <- (z + c z_r) / (1 + c), c = ``erodibility`` A^``m`` dt / dist_r;
  - deposits with the transport length xi = ``transport_coefficient`` A / dx
    (Davy and Lague 2009): of the material entrained from uniformly
    distributed points of the cell, the fraction (xi / dx)(1 - exp(-dx / xi))
    crosses its edge, and a sediment flux entering it passes with the
    fraction exp(-dx / xi). ``sediment`` config:
      - ``pulled``: the sediment flux (m3/yr) is a field like the area: the
        cell takes it from the sends of its higher neighbours, adds its
        entrainment rate, deposits what does not pass over dt and sends the
        rest;
      - ``carried``: the particle carries a volume: it drops on the cell the
        part of its load that does not pass and picks up the leaving part of
        the cell's erosion; what it carries when it stops stays there, what
        reaches an outlet is exported;
  - rewrites its area (and, pulled, sediment) sends in proportion to
    slope^``mfd_exponent`` towards its lower neighbours.

That is ``scheme="downstream"``. With ``scheme="precipiton"`` no drainage
area is transported: launch k covers the years (t - ``launch_time``, t] and
its precipiton p lands at its own time t_p = t - ``launch_time`` + (p + 0.5)
/ ``n_particles`` * ``launch_time`` on a uniformly drawn cell, then walks as
above. tau = ``launch_time`` * cells / ``n_particles`` is the mean time
between two landings on one cell, so a cell of drainage area A is crossed
every dx^2 tau / A on average and the passage rate gives A = dx^2 tau rate.
A passage, holding the cell's lock (``contention``: ``wait`` for it or
``skip`` the cell), takes dt = t_p - its last passage (a passage older
than the last one is late and incises over no time), updates the rate
(``smoothing`` False: 1 / |t_p - last passage|, the gap floored at
``launch_time`` / ``n_particles``; True: exponentially weighted passage
count with time scale ``rate_timescale``, sum of exp(-age / ts) / ts),
incises over dt as above and carries its load with the transport length
(as ``carried``). Pits (``pits``):
  - ``fill``: a pit receives the whole load and is raised ``fill_epsilon``
    above its lowest neighbour if still needed (counted as filled volume);
  - ``graph``: each launch recomputes the receivers ``rec`` of the uplifted
    surface with ``local_minima`` handling; a precipiton reaching a pit
    follows ``rec`` (lake mode: no incision, no fill, deposition by the
    transport length rule only) until it reaches a cell lower than that
    pit, outside the pit's basin, and moves on stochastically from there. ``window`` > 0 defers the
incision: the precipiton records (cell, dt, A) of the passages with dt > 0
and, every ``window`` recorded cells, at an outlet and at the end of its
walk, incises them from the most downstream one up (each against its
current steepest receiver, waiting for the locks), then carries their
eroded volumes down the recorded cells with the transport length rule; what
leaves the last one joins its load. ``initialize()`` seeds the rates
from the pointer-jump area and places each cell's last passage one mean
interval before 0.

``run`` returns the launch statistics summed over the n launches: particles,
processed and locked visits, outlet arrivals, and the eroded, deposited,
exported (carried, precipiton) and filled volumes (m3), and late passages
(precipiton). ``time`` is the clock (yr).

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.experimental.golem.program import _pointer_jump_plan
from pyfastflow.flow import (
    depression_binding_plan,
    make_accumulation,
    make_depression_solver,
    make_depressions,
    make_receivers,
)
from pyfastflow.flow._program_cupy import (
    BLOCK,
    _cupy_only,
    _grid_leaf_plan,
    _noop_factory,
    _reconstruct_epsilon_factory,
    _reconstruct_epsilon_plan,
)
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.grid._mask import GridMaskAccessor

from ._particle_lem import (
    base_level_factory,
    init_factory,
    precipiton_factory,
    seed_rate_factory,
    uplift_factory,
    walk_factory,
)

_STATS = ("particles", "processed", "locked", "outlet", "eroded",
          "deposited", "exported", "filled", "late")


def build_particle_lem_program() -> type:
    """Build the ParticleLEMProgram class (see the module docstring)."""
    b = ProgramBuilder("ParticleLEMProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("topology", choices=("D4", "D8"), default="D8")
    b.config(
        "boundary", choices=("normal", "periodic_EW", "periodic_NS"),
        default="normal",
    )
    b.config("outlet", choices=("edge", "mask"), default="edge")
    b.config("nodata", choices=(False, True), default=False)
    b.config(
        "local_minima",
        choices=("none", "reconstruct_epsilon", "cordonnier_carve", "cordonnier_jump"),
        default="cordonnier_carve",
    )
    b.config("scheme", choices=("downstream", "precipiton"), default="downstream")
    b.config("contention", choices=("wait", "skip"), default="wait")
    b.config("smoothing", choices=(False, True), default=False)
    b.config("window", default=0)
    b.config("pits", choices=("fill", "graph"), default="graph")
    b.config("sediment", choices=("pulled", "carried"), default="pulled")
    b.config("threads", default=65536)

    shape = (Dim("ny"), Dim("nx"))
    flat = Dim("ny") * Dim("nx")
    b.param("uplift", "auto", "f32", value=0.0, shape=shape)
    b.param("erodibility", "auto", "f32", value=0.0, shape=shape)
    b.param("m", "auto", "f32", value=0.45, shape=shape)
    b.param("transport_coefficient", "auto", "f32", value=1.0, shape=shape)
    b.param("launch_time", "auto", "f32", value=1.0)
    b.param("n_particles", "auto", "i32", value=1_000_000)
    b.param("walk_steps", "auto", "i32", value=1000)
    b.param("mfd_exponent", "auto", "f32", value=1.0)
    b.param("rate_timescale", "auto", "f32", value=1.0e4)
    b.param("fill_epsilon", "auto", "f32", value=1.0e-3)
    b.param("source", "auto", "f32", value=1.0, shape=shape)
    b.param("cell_count", "const", "i32",
            value=lambda dims: dims["nx"] * dims["ny"])
    b.param("seed", "scalar", "i32", value=1)
    b.param("clock", "scalar", "i32", value=0)
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    b.data("z", "f32", shape, role="state", shape_source=True)
    b.data("rec", "i32", shape, role="output")
    b.data("drainage_area", "f32", shape, role="output")
    b.data("sediment_flux", "f32", shape, role="output")
    b.data("filled", "f32", shape, role="output")
    b.data("epsilon_distance", "f32", shape, role="output")
    b.data("area_send", "f32", (8 * flat,), role="internal")
    b.data("sediment_send", "f32", (8 * flat,), role="internal")
    b.data("last_visit", "f64", (flat,), role="internal")
    b.data("now", "f64", (1,), role="internal")
    b.data("locks", "i32", (flat,), role="internal")
    b.data("claim", "i32", (1,), role="internal")
    b.data("passage_rate", "f32", (flat,), role="internal")
    b.data("walk_stats", "f64", (9,), role="internal")

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

    for name in (
        "bid", "rec_jump", "basin_saddlenode", "basin_route", "b_rcv",
        "parent", "epsilon_ancestor", "epsilon_ancestor_work",
    ):
        b.data(name, "i32", (flat,), role="internal")
    for name in ("z_prime", "epsilon_distance_work"):
        b.data(name, "f32", (flat,), role="internal")
    for name in ("is_border", "rerouted"):
        b.data(name, "u8", (flat,), role="internal")
    for name in ("basin_saddle", "basin_outlet"):
        b.data(name, "i64", (flat,), role="internal")
    b.data("frontier", "i32", (2 * flat,), role="internal")
    b.data("counters", "i32", (flat,), role="internal")
    b.data("queued_gen", "i32", (flat,), role="internal")
    for name in ("pj_work", "pj_work2"):
        b.data(name, "i32", (flat,), lifetime="temp")
    b.data("pj_q_work", "f32", (flat,), lifetime="temp")

    def route_factory(be, bundles, config):
        _cupy_only(be)
        return make_receivers(
            be, bundles["grid"], topology=config["topology"], mode="steepest",
        )["receivers"]

    b.add("route", route_factory, bind={"grid": "grid", "z": "z", "rec": "rec"})

    def cordonnier_factory(reroute):
        def factory(be, bundles, config):
            _cupy_only(be)
            deps = make_depressions(
                be, bundles["grid"], bundles.param("ndep"),
                method="optimized", reroute=reroute,
                n_flat=config["nx"] * config["ny"],
            )
            return make_depression_solver(
                be, deps, bundles.bundle_params("grid"), method="optimized",
                reroute=reroute, n_flat=config["nx"] * config["ny"],
                block_size=BLOCK,
            )[0]
        return factory

    def depression_plan(frozen, reroute):
        plan = depression_binding_plan(frozen, method="optimized", reroute=reroute)
        return {key: "basin_outlet" if value == "outlet" else value
                for key, value in plan.items()}

    b.add("resolve_carve", cordonnier_factory("carve"),
          bind=lambda f, be: depression_plan(f, "carve"))
    b.add("resolve_jump", cordonnier_factory("jump"),
          bind=lambda f, be: depression_plan(f, "jump"))
    b.add("resolve_reconstruct_epsilon", _reconstruct_epsilon_factory,
          bind=_reconstruct_epsilon_plan)
    b.add("resolve_none", _noop_factory, bind={})
    b.dispatch("resolve_minima", on="local_minima", cases={
        "none": "resolve_none",
        "reconstruct_epsilon": "resolve_reconstruct_epsilon",
        "cordonnier_carve": "resolve_carve",
        "cordonnier_jump": "resolve_jump",
    })

    def accumulation_factory(be, bundles, config):
        _cupy_only(be)
        return make_accumulation(
            be, bundles["grid"], method="pointer_jump_push",
            n_flat=config["nx"] * config["ny"],
        )["sequence"].freeze()

    b.add("accumulate", accumulation_factory, bind=_pointer_jump_plan)

    def scale_factory(be, bundles, config):
        _cupy_only(be)
        n_cells = config["nx"] * config["ny"]
        return KernelBuilder(
            f'''extern "C" __global__ void plem_scale_area(float* area) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= {n_cells}) return;
                if (!$ctx.grid.is_active(i)$) {{ area[i] = 0.0f; return; }}
                float dx = $ctx.grid.DX.get(0)$;
                area[i] *= dx * dx;
            }}''', domain=n_cells,
        ).compose("grid", bundles["grid"]).freeze()

    b.add("scale_drainage_area", scale_factory,
          bind={"grid": "grid", "area": "drainage_area"})
    b.add("seed_state", init_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "rec": "rec", "area": "drainage_area", "a_send": "area_send",
              "qs_send": "sediment_send", "qs": "sediment_flux",
              "t_last": "last_visit", "locks": "locks",
              "CELL_COUNT": "cell_count",
          }))
    b.add("hold_base_level", base_level_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "CELL_COUNT": "cell_count",
          }))
    b.add("apply_uplift", uplift_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "CELL_COUNT": "cell_count", "UPLIFT": "uplift",
              "LAUNCH_TIME": "launch_time",
          }))
    b.add("walk_particles", walk_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "a_send": "area_send", "qs_send": "sediment_send",
              "area": "drainage_area", "qs": "sediment_flux",
              "t_last": "last_visit", "now": "now", "locks": "locks",
              "claim": "claim", "stats": "walk_stats",
              "CELL_COUNT": "cell_count", "N_PARTICLES": "n_particles",
              "WALK_STEPS": "walk_steps", "MFD_EXPONENT": "mfd_exponent",
              "FILL_EPS": "fill_epsilon", "SEED": "seed", "CLOCK": "clock",
              "ERODIBILITY": "erodibility", "M": "m",
              "XI": "transport_coefficient",
          }))

    b.add("seed_rates", seed_rate_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "area": "drainage_area", "rate": "passage_rate",
              "t_last": "last_visit", "CELL_COUNT": "cell_count",
              "LAUNCH_TIME": "launch_time", "N_PARTICLES": "n_particles",
          }))
    b.add("walk_precipitons", precipiton_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "z": "z", "area": "drainage_area", "rate": "passage_rate",
              "t_last": "last_visit", "now": "now", "rec": "rec",
              "locks": "locks",
              "claim": "claim", "stats": "walk_stats",
              "CELL_COUNT": "cell_count", "N_PARTICLES": "n_particles",
              "LAUNCH_TIME": "launch_time", "WALK_STEPS": "walk_steps",
              "MFD_EXPONENT": "mfd_exponent", "FILL_EPS": "fill_epsilon",
              "RATE_TIMESCALE": "rate_timescale", "SEED": "seed",
              "CLOCK": "clock", "ERODIBILITY": "erodibility", "M": "m",
              "XI": "transport_coefficient",
          }))

    b.pipeline("_initialize", (
        "hold_base_level", "route", "resolve_minima", "accumulate", "scale_drainage_area",
        "seed_state",
    ))
    b.pipeline("_launch", ("apply_uplift", "walk_particles"))
    b.pipeline("_precipiton_launch", ("apply_uplift", "walk_precipitons"))
    b.pipeline("_precipiton_graph_launch", (
        "apply_uplift", "route", "resolve_minima", "walk_precipitons",
    ))

    program = b.freeze()

    def initialize(self):
        """Pointer-jump drainage area on the current z and the starting
        state (MFD sends, or precipiton passage rates); resets the clock
        to 0."""
        self._initialize()
        if self._config["scheme"] == "precipiton":
            self.seed_rates()
        self._handle("now").array.fill(0)
        object.__setattr__(self, "_time", 0.0)

    def run(self, n=1):
        """Run ``n`` launches; return their statistics (see the module
        docstring)."""
        totals = dict.fromkeys(_STATS, 0.0)
        clock = getattr(self, "_clock", 0)
        time = getattr(self, "_time", 0.0)
        step = float(self.launch_time.read())
        launch = self._launch
        if self._config["scheme"] == "precipiton":
            launch = (self._precipiton_graph_launch
                      if self._config["pits"] == "graph"
                      else self._precipiton_launch)
        for _ in range(int(n)):
            time += step
            self._handle("now").array.fill(time)
            self._handle("claim").array.fill(0)
            self._handle("walk_stats").array.fill(0)
            self.clock.set(clock)
            launch()
            raw = self._handle("walk_stats").to_numpy()
            for key, value in zip(_STATS, raw):
                totals[key] += float(value)
            clock += 1
        object.__setattr__(self, "_clock", clock)
        object.__setattr__(self, "_time", time)
        return totals

    program.initialize = initialize
    program.run = run
    program.time = property(lambda self: getattr(self, "_time", 0.0))
    program.outlet_mask = property(
        lambda self: GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"),
    )
    program.nodata_mask = property(
        lambda self: GridMaskAccessor(self, "NODATA_MASK", "nodata=True"),
    )
    return program


ParticleLEMProgram = build_particle_lem_program()

__all__ = ["ParticleLEMProgram", "build_particle_lem_program"]
