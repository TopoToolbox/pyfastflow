"""GolemParcelSedProgram: GOLEM erosion with stochastic sediment parcels.

CuPy only. One step:
  1. uplift (outlets keep their captured elevation);
  2. SFD receivers, local minima, pointer-jump drainage area (m2);
  3. entrainment: implicit stream-power incision solved by Newton pointer
     jumping lowers z and writes ``erosion_rate``; its volume
     ``erosion_rate * dt * dx^2`` is the entrained volume of the cell;
  4. deposition by the parcel engine (``_engine``) on the surface z, with
     the transport length L = ``transport_coefficient`` * area / dx
     (``transport_coefficient`` is P / v_s of the transport-length model,
     area the drainage area): the fraction (L / dx) (1 - exp(-dx / L)) of
     the entrained volume leaves its cell as parcels of at most
     ``parcel_volume``, the rest settles back on it. ``hop`` config:
     ``steepest`` (steepest lower neighbour of the current z) or
     ``weighted`` (a lower neighbour drawn with probability proportional
     to the slope towards it); a hop costs dist / L. ``deposition`` config:
       - ``stochastic``: the parcel drops its whole volume once an optical
         depth tau = -ln(u) is consumed by the hop costs;
       - ``gradual``: each hop leaves the fraction 1 - exp(-dist / L) of the
         carried volume on the cell being left; once the remainder would
         fall below ``min_volume`` (m3) it is all left there.
     On a cell with no lower neighbour a parcel leaves just the volume that
     raises it ``fill_epsilon`` (m) above its lowest neighbour and moves
     there (everything, if it carries less); parcels reaching an outlet
     leave the domain;
  5. implicit linear hillslope diffusion.

``run_n_step(n)`` returns the parcel count and the entrained, deposited and
exported volumes (m3) summed over the n steps.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.experimental.golem._fluvial import fluvial_factory, fluvial_plan
from pyfastflow.experimental.golem._hillslope import hillslope_factory, hillslope_plan
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

from ._engine import make_parcel_release, make_parcel_walk, run_parcels


def build_golem_parcel_sed_program() -> type:
    """Build the GolemParcelSedProgram class (see the module docstring)."""
    b = ProgramBuilder("GolemParcelSedProgram")
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
    b.config("diffusion_iterations", default=20)
    b.config("newton_iterations", default=8)
    b.config("threads", default=65536)
    b.config("deposition", choices=("stochastic", "gradual"),
             default="stochastic")
    b.config("hop", choices=("steepest", "weighted"), default="steepest")

    shape = (Dim("ny"), Dim("nx"))
    flat = Dim("ny") * Dim("nx")
    b.param("uplift", "auto", "f32", value=0.0, shape=shape)
    b.param("erodibility", "auto", "f32", value=0.0, shape=shape)
    b.param("hillslope_diffusivity", "auto", "f32", value=0.0, shape=shape)
    b.param("transport_coefficient", "auto", "f32", value=1.0, shape=shape)
    b.param("dt", "auto", "f32", value=1.0, shape=shape)
    b.param("m", "auto", "f32", value=0.4, shape=shape)
    b.param("n", "auto", "f32", value=1.0, shape=shape)
    b.param("slope_floor", "auto", "f32", value=1.0e-6, shape=shape)
    b.param("source", "auto", "f32", value=1.0, shape=shape)
    b.param("parcel_volume", "auto", "f32", value=1.0e4)
    b.param("min_volume", "auto", "f32", value=1.0e2)
    b.param("fill_epsilon", "auto", "f32", value=1.0e-3)
    b.param("cell_count", "const", "i32",
            value=lambda dims: dims["nx"] * dims["ny"])
    b.param("seed", "scalar", "i32", value=1)
    b.param("clock", "scalar", "i32", value=0)
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    b.data("z", "f32", shape, role="state", shape_source=True)
    b.data("outlet_z", "f32", shape, role="state")
    b.data("rec", "i32", shape, role="output")
    b.data("drainage_area", "f32", shape, role="output")
    b.data("erosion_rate", "f32", shape, role="output")
    b.data("filled", "f32", shape, role="output")
    b.data("epsilon_distance", "f32", shape, role="output")
    b.data("entrained", "f32", shape, role="output")
    b.data("transport_length", "f32", shape, role="output")
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

    for name in ("pj_work", "pj_work2", "fluvial_rec_a", "fluvial_rec_b"):
        b.data(name, "i32", (flat,), lifetime="temp")
    for name in (
        "pj_q_work", "fluvial_z0", "fluvial_guess", "fluvial_z_a",
        "fluvial_z_b", "fluvial_alpha_a", "fluvial_alpha_b",
        "diffusion_z0", "diffusion_a", "diffusion_b",
    ):
        b.data(name, "f32", (flat,), lifetime="temp")

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

    def cell_kernel(name, signature, body):
        def factory(be, bundles, config):
            _cupy_only(be)
            n_cells = config["nx"] * config["ny"]
            return KernelBuilder(
                f'''extern "C" __global__ void {name}({signature}) {{
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i >= {n_cells}) return;
                    {body}
                }}''', domain=n_cells,
            ).compose("grid", bundles["grid"]).freeze()
        return factory

    b.add("scale_drainage_area", cell_kernel(
        "sedp_scale_drainage_area", "float* area",
        "if (!$ctx.grid.is_active(i)$) { area[i] = 0.0f; return; }"
        " float dx = $ctx.grid.DX.get(0)$; area[i] *= dx * dx;",
    ), bind={"grid": "grid", "area": "drainage_area"})
    b.add("apply_uplift", cell_kernel(
        "sedp_apply_uplift", "float* z",
        "if (!$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$) return;"
        " z[i] += $ctx.UPLIFT.get(i)$ * fmaxf($ctx.DT.get(i)$, 0.0f);",
    ), bind={"grid": "grid", "z": "z", "UPLIFT": "uplift", "DT": "dt"})
    b.add("capture_outlet_z", cell_kernel(
        "sedp_capture_outlets", "const float* z, float* outlet_z",
        "if ($ctx.grid.can_out(i)$) outlet_z[i] = z[i];",
    ), bind={"grid": "grid", "z": "z", "outlet_z": "outlet_z"})
    b.add("restore_outlet_z", cell_kernel(
        "sedp_restore_outlets", "float* z, const float* outlet_z",
        "if ($ctx.grid.can_out(i)$) z[i] = outlet_z[i];",
    ), bind={"grid": "grid", "z": "z", "outlet_z": "outlet_z"})
    b.add("fluvial", fluvial_factory, bind=fluvial_plan)
    b.add("hillslope", hillslope_factory, bind=hillslope_plan)
    b.add("fluvial_entrainment", cell_kernel(
        "sedp_fluvial_entrainment", "const float* erosion_rate, float* entrained",
        "if (!$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$) return;"
        " float dx = $ctx.grid.DX.get(0)$;"
        " entrained[i] += fmaxf(erosion_rate[i], 0.0f)"
        " * fmaxf($ctx.DT.get(i)$, 0.0f) * dx * dx;",
    ), bind={"grid": "grid", "erosion_rate": "erosion_rate",
             "entrained": "entrained", "DT": "dt"})
    b.add("compute_transport_length", cell_kernel(
        "sedp_transport_length", "const float* area, float* length",
        "length[i] = fmaxf($ctx.XI.get(i)$, 0.0f) * fmaxf(area[i], 0.0f)"
        " / $ctx.grid.DX.get(0)$;",
    ), bind={"grid": "grid", "area": "drainage_area",
             "length": "transport_length", "XI": "transport_coefficient"})

    def release_factory(be, bundles, config):
        return make_parcel_release(be, bundles["grid"],
                                   n_cells=config["nx"] * config["ny"])

    def walk_factory(be, bundles, config):
        return make_parcel_walk(be, bundles["grid"], threads=config["threads"],
                                deposition=config["deposition"],
                                hop=config["hop"])

    b.add("release_parcels", release_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "volume": "entrained", "length": "transport_length",
              "surface": "z", "count": "parcel_count",
              "CELL_COUNT": "cell_count", "PARCEL_VOLUME": "parcel_volume",
          }))
    b.add("walk_parcels", walk_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "cdf": "parcel_cdf", "volume": "entrained",
              "length": "transport_length", "surface": "z",
              "claim": "parcel_claim", "stats": "parcel_stats",
              "CELL_COUNT": "cell_count", "SEED": "seed", "CLOCK": "clock",
              "MIN_VOLUME": "min_volume", "FILL_EPS": "fill_epsilon",
          }))

    b.pipeline("initialize", ("capture_outlet_z", "restore_outlet_z"))
    b.pipeline("_erode", (
        "apply_uplift", "restore_outlet_z",
        "route", "resolve_minima", "accumulate", "scale_drainage_area",
        "fluvial", "restore_outlet_z", "fluvial_entrainment",
        "compute_transport_length",
    ))
    b.pipeline("_release", ("release_parcels",))
    b.pipeline("_walk", ("walk_parcels", "restore_outlet_z"))
    b.pipeline("_diffuse", ("hillslope", "restore_outlet_z"))

    program = b.freeze()

    def run_n_step(self, n=1):
        """Run ``n`` steps; return ``{"parcels", "entrained", "deposited",
        "exported"}`` summed over them (volumes in m3)."""
        totals = {"parcels": 0, "entrained": 0.0, "deposited": 0.0,
                  "exported": 0.0}
        clock = getattr(self, "_clock", 0)
        for _ in range(int(n)):
            self._handle("entrained").array.fill(0)
            self._erode()
            step = run_parcels(self, self._release, self._walk, clock=clock)
            for key in totals:
                totals[key] += step[key]
            self._diffuse()
            clock += 1
        object.__setattr__(self, "_clock", clock)
        return totals

    program.run_n_step = run_n_step
    program.outlet_mask = property(
        lambda self: GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"),
    )
    program.nodata_mask = property(
        lambda self: GridMaskAccessor(self, "NODATA_MASK", "nodata=True"),
    )
    return program


GolemParcelSedProgram = build_golem_parcel_sed_program()

__all__ = ["GolemParcelSedProgram", "build_golem_parcel_sed_program"]
