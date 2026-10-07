"""CuPy MFD flow Program with optional local-minima resolution.

The public pipeline is deliberately small::

    flow = MFDFlowProgram(backend, nx=nx, ny=ny,
                          local_minima="cordonnier_carve")
    flow.z.from_numpy(dem)
    flow.route()
    flow.snapshot_receivers()
    flow.resolve_minima()
    flow.prepare_mfd_surface()
    flow.build_topology()
    flow.accumulate()

``local_minima="none"`` makes the first call a true no-op. Its topology is
the raw downslope MFD graph, so flow may terminate in interior sinks.
Cordonnier carve uses receiver-rank gating. ``fill_cordonnier`` instead derives
a filled elevation from the carved paths, then runs ordinary MFD on that field
with receiver distance only as an epsilon flat ordering. Reconstruction uses
its own filled surface and path-distance tie breaker. ``quantized_weight=True``
(the default) stores eight max-normalized unsigned-byte scores per cell;
accumulation renormalizes their sum so the complete discharge is still
partitioned."""

from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow import depression_binding_plan
from pyfastflow.flow._program_cupy import (
    _cupy_only, _noop_factory, _reconstruct_epsilon_factory,
    _reconstruct_epsilon_plan,
)
from pyfastflow.grid import make_grid_group, make_grid_parameters

from ._mfd_program_steps import (
    _cordonnier_factory,
    _route_factory,
    _snapshot_factory,
    _rank_factory,
    _cordonnier_fill_factory,
    _rank_plan,
    _cordonnier_fill_plan,
    _raw_topology_factory,
    _surface_topology_factory,
    _rank_topology_factory,
    _fill_topology_factory,
    _topology_plan,
    _accumulation_factory,
    _accumulation_plan,
)


def build_mfd_flow_program() -> type:
    """Build the reusable CuPy MFD flow Program class."""
    b = ProgramBuilder("MFDFlowProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx")
    b.config(
        "local_minima",
        choices=(
            "none", "reconstruct_epsilon", "cordonnier_carve",
            "fill_cordonnier",
        ),
        default="cordonnier_carve",
    )
    b.config("dx", default=1.0)
    b.config("quantized_weight", choices=(False, True), default=True)
    b.param("source", "auto", "f32", value=1.0,
            shape=(Dim("ny"), Dim("nx")))
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    flat = Dim("ny") * Dim("nx")
    b.data("z", "f32", (Dim("ny"), Dim("nx")), role="input", shape_source=True)
    b.data("drainage", "f32", (Dim("ny"), Dim("nx")), role="output")
    b.data("filled", "f32", (Dim("ny"), Dim("nx")), role="output")
    b.data("rec", "i32", (Dim("ny"), Dim("nx")), role="output")

    def grid_structure(be, **_):
        _cupy_only(be)
        return make_grid_group(be, topology="D8", boundary="normal", outlet="edge")

    def grid_params(be, pool, *, nx, ny, dx):
        return make_grid_parameters(
            be, pool, nx, ny, dx, topology="D8", outlet="edge"
        )

    b.bundle("grid", grid_structure, grid_params, dims=("nx", "ny"), config=("dx",))

    for name in (
        "rec_initial", "rank_ancestor", "rank_ancestor_alt", "rank", "rank_alt",
        "bid", "rec_jump", "basin_saddlenode", "basin_route", "b_rcv",
        "parent", "epsilon_ancestor", "epsilon_ancestor_work", "mfd_frontier0",
        "mfd_frontier1", "indegree",
    ):
        b.data(name, "i32", (flat,), role="internal")
    for name in ("z_prime", "epsilon_distance", "epsilon_distance_work"):
        b.data(name, "f32", (flat,), role="internal")
    for name in ("is_border", "rerouted", "directions"):
        b.data(name, "u8", (flat,), role="internal")
    for name in ("basin_saddle", "outlet"):
        b.data(name, "i64", (flat,), role="internal")
    b.data(
        "weights",
        lambda config: "u8" if config["quantized_weight"] else "f32",
        (8 * flat,), role="internal",
    )
    b.data("frontier", "i32", (2 * flat,), role="internal")
    b.data("counters", "i32", (flat,), role="internal")
    b.data("queued_gen", "i32", (flat,), role="internal")
    b.data("mfd_count", "i32", (3,), role="internal")
    b.data("mfd_barrier", "u32", (1,), role="internal")

    b.add("route", _route_factory, bind={"grid": "grid", "z": "z", "rec": "rec"})
    b.add("snapshot_receivers", _snapshot_factory, bind={
        "rec": "rec", "rec_initial": "rec_initial",
    })
    b.add("compute_rank", _rank_factory, bind=_rank_plan)
    b.add("compute_cordonnier_fill", _cordonnier_fill_factory,
          bind=_cordonnier_fill_plan)
    b.add("resolve_none", _noop_factory, bind={})
    b.add("resolve_reconstruct_epsilon", _reconstruct_epsilon_factory, bind=_reconstruct_epsilon_plan)
    b.add("resolve_carve", _cordonnier_factory("carve"), bind=lambda f, be: depression_binding_plan(f, method="optimized", reroute="carve"))
    b.dispatch("resolve_minima", on="local_minima", cases={
        "none": "resolve_none",
        "reconstruct_epsilon": "resolve_reconstruct_epsilon",
        "cordonnier_carve": "resolve_carve",
        "fill_cordonnier": "resolve_carve",
    })
    b.dispatch("prepare_mfd_surface", on="local_minima", cases={
        "none": "resolve_none",
        "reconstruct_epsilon": "resolve_none",
        "cordonnier_carve": "compute_rank",
        "fill_cordonnier": "compute_cordonnier_fill",
    })

    b.add("topology_none", _raw_topology_factory, bind=_topology_plan("raw"))
    b.add("topology_reconstruct", _surface_topology_factory, bind=_topology_plan("surface"))
    b.add("topology_rank", _rank_topology_factory, bind=_topology_plan("rank"))
    b.add("topology_fill", _fill_topology_factory, bind=_topology_plan("fill"))
    b.dispatch("build_topology", on="local_minima", cases={
        "none": "topology_none",
        "reconstruct_epsilon": "topology_reconstruct",
        "cordonnier_carve": "topology_rank",
        "fill_cordonnier": "topology_fill",
    })
    b.add("accumulate", _accumulation_factory, bind=_accumulation_plan)
    return b.freeze()


MFDFlowProgram = build_mfd_flow_program()

__all__ = ["MFDFlowProgram", "build_mfd_flow_program"]
