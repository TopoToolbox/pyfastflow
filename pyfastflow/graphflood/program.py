"""GraphFlood program assembly and runtime control."""

import math

import numpy as np

from pyfastflow.core import ProgramError
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow._program_cupy import (
    _cupy_only, _grid_leaf_plan, _noop_factory,
    _reconstruct_epsilon_factory,
)
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.grid._mask import GridMaskAccessor as _GridMaskAccessor
from pyfastflow.ops import make_scan
from ._surface import (
    _make_surface_factory, _refresh_hydraulic_surface_factory,
    _route_factory, _snapshot_factory, _carve_factory, _carve_plan,
    _rank_factory, _rank_plan, _cordonnier_surface_factory,
    _cordonnier_fill_factory, _cordonnier_carve_factory,
    _active_cordonnier_fill_factory, _cordonnier_fill_plan,
    _active_cordonnier_fill_plan, _cordonnier_carve_plan,
    _reset_h_factory, _copy_fill_depth_factory, _merge_fill_depth_factory,
    _merge_active_fill_depth_factory, _fill_surface_plan,
)
from ._routing import (
    _topology_factory, _filled_topology_factory, _carve_topology_factory,
    _reconstructed_topology_factory, _topology_plan, _filled_topology_plan,
    _carve_topology_plan, _reconstructed_topology_plan, _frontier_factory,
    _accumulation_factory, _drainage_area_factory, _drainage_area_plan,
    _subset_accumulation_factory, _subset_accumulation_plan,
)
from ._domain import (
    _copy_flux_factory, _river_distance_init_factory, _river_distance_relax_factory,
    _river_seed_factory, _dynamic_domain_init_factory,
    _dynamic_domain_grow_factory, _dynamic_transient_grow_factory,
    _dynamic_activity_clear_factory, _river_distance_relax_plan,
    _active_work_factory,
)
from ._solvers import (
    _transient_factory, _dynamic_transient_factory, _hybrid_factory,
    _transient_plan, _update_factory, _local_analytical_update_factory,
    _capped_local_analytical_update_factory,
    _dynamic_local_analytical_update_factory,
    _local_analytical_update_plan, _capped_local_analytical_update_plan,
    _dynamic_local_analytical_update_plan,
    _bottom_up_analytical_update_factory,
    _bottom_up_analytical_update_plan,
)


def build_graphflood_program() -> type:
    """Build the configurable regular-grid CuPy GraphFlood Program class."""
    b = ProgramBuilder("GraphFloodProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("topology", choices=("D4", "D8"), default="D8")
    b.config(
        "boundary",
        choices=("normal", "periodic_EW", "periodic_NS"),
        default="normal",
    )
    b.config("outlet", choices=("edge", "mask"), default="edge")
    b.config("nodata", choices=(False, True), default=False)
    b.config("friction_law", choices=("manning",), default="manning")
    b.config("quantized_weight", choices=(False, True), default=True)
    b.config(
        "adaptive_transient_dt", choices=(False, True), default=True,
    )
    b.config(
        "analytical_solver", choices=("local", "bottom_up"),
        default="bottom_up",
    )
    b.config(
        "mfd_local_minima",
        choices=(
            "rank_cordonnier", "fill_cordonnier", "carve_cordonnier",
            "reconstruct_epsilon",
        ),
        default="rank_cordonnier",
    )

    flat = Dim("ny") * Dim("nx")
    shape = (Dim("ny"), Dim("nx"))
    b.param("precipitation", "auto", "f32", value=0.0, shape=shape)
    b.param("friction_coefficient", "auto", "f32", value=0.033, shape=shape)
    b.param("friction_exponent", "auto", "f32", value=2.0 / 3.0, shape=shape)
    b.param("dt", "auto", "f32", value=1.0e-3, shape=shape)
    b.param("analytical_relaxation", "auto", "f32", value=0.1, shape=shape)
    b.param("carve_slope_min", "auto", "f32", value=1.0e-4, shape=shape)
    b.param("transient_dt", "auto", "f32", value=1.0e-3)
    b.param("transient_cfl", "auto", "f32", value=0.5)
    b.param("hybrid_theta", "auto", "f32", value=0.1, shape=shape)
    b.param("flow_area_threshold", "scalar", "f32", value=0.0)
    b.param("precipitation_reference", "scalar", "f32", value=0.0)
    b.param("river_padding", "scalar", "f32", value=0.0)
    b.param("hillslope_depth_cap", "scalar", "f32", value=0.03)
    b.param("growth_residual_threshold", "scalar", "f32", value=1.0e-3)
    b.param("transient_activity_threshold", "scalar", "f32", value=1.0e-3)
    b.param("active_count", "scalar", "i32", value=0)
    b.param("work_count", "scalar", "i32", value=0)
    b.param("dynamic_active_count", "scalar", "i32", value=0)
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    b.data("z", "f32", shape, role="input", shape_source=True)
    b.data("h", "f32", shape, role="state")
    b.data("Qi", "f32", shape, role="output")
    b.data("Qo", "f32", shape, role="output")
    b.data("Qsend", "f32", shape, role="output")
    b.data("surface", "f32", shape, role="internal")
    b.data("hydraulic_surface", "f64", shape, role="internal")
    b.data("rec", "i32", shape, role="output")
    b.data("drainage_area", "f32", shape, role="output")
    b.data("effective_drainage_area", "f32", shape, role="output")
    b.data("river_distance", "f32", shape, role="output")
    b.data("river_distance_work", "f32", shape, role="internal")
    b.data("analytical_residual", "f32", shape, role="output")
    b.data("transient_activity", "f32", shape, role="output")

    def grid_structure(be, *, topology, boundary, outlet, nodata, **_):
        _cupy_only(be)
        return make_grid_group(
            be, topology=topology, boundary=boundary, outlet=outlet,
            nodata=nodata,
        )

    def grid_params(
            be, pool, *, nx, ny, dx, topology, boundary, outlet, nodata):
        del boundary
        return make_grid_parameters(
            be, pool, nx, ny, dx, topology=topology, outlet=outlet,
            nodata=nodata,
        )

    b.bundle("grid", grid_structure, grid_params,
             dims=("nx", "ny"),
             config=("dx", "topology", "boundary", "outlet", "nodata"))

    for name in (
        "rec_initial", "rank_ancestor", "rank_ancestor_alt", "rank", "rank_alt",
        "bid", "basin_saddlenode", "basin_route", "b_rcv",
        "mfd_frontier0", "mfd_frontier1", "indegree",
        "active_indegree", "river_flags",
        "dynamic_flags", "dynamic_claims", "dynamic_active_ids",
        "dynamic_work_flags", "dynamic_work_ids",
    ):
        b.data(name, "i32", (flat,), role="internal")
    for name in ("z_prime", "steepest_slope", "flow_width"):
        b.data(name, "f32", (flat,), role="internal")
    b.data("Qi_boundary", "f32", shape, role="output")
    b.data(
        "weights",
        lambda config: "u8" if config["quantized_weight"] else "f32",
        (8 * flat,), role="internal",
    )
    for name in (
        "is_border", "directions", "hydraulic_conditioned", "hydraulic_cut",
    ):
        b.data(name, "u8", (flat,), role="internal")
    b.data("river_mask", "u8", shape, role="output")
    b.data("dynamic_active_mask", "u8", shape, role="output")
    b.data("dynamic_work_mask", "u8", shape, role="internal")
    for name in ("basin_saddle", "basin_outlet"):
        b.data(name, "i64", (flat,), role="internal")
    b.data("mfd_count", "i32", (3,), role="internal")
    b.data("mfd_barrier", "u32", (1,), role="internal")
    b.data("transient_dt_used", "f32", (1,), role="output")
    b.data("dynamic_count_device", "i32", (1,), role="internal")

    # One-shot fill initialization scratch. Keeping every field temporary
    # releases it back to the program pool as soon as the operation returns.
    for name in ("fill_parent", "fill_epsilon_ancestor", "fill_epsilon_ancestor_work",
                 "fill_counters", "fill_queued_gen"):
        b.data(name, "i32", (flat,), lifetime="temp")
    for name in ("fill_epsilon_distance", "fill_epsilon_distance_work"):
        b.data(name, "f32", (flat,), lifetime="temp")
    b.data("fill_frontier", "i32", (2 * flat,), lifetime="temp")

    b.add("reset_h", _reset_h_factory, bind={"h": "h"})
    b.add("reconstruct_fill_surface", _reconstruct_epsilon_factory,
          bind=_fill_surface_plan("z", "surface"))
    b.add("copy_fill_depth", _copy_fill_depth_factory,
          bind={"z": "z", "filled": "surface", "h": "h"})
    b.pipeline("initialize_h_from_fill",
               ("reconstruct_fill_surface", "copy_fill_depth"))
    b.add("make_surface", _make_surface_factory,
          bind={
              "z": "z", "h": "h", "surface": "surface",
              "hydraulic_surface": "hydraulic_surface",
          })
    b.add("refresh_hydraulic_surface", _refresh_hydraulic_surface_factory,
          bind={
              "z": "z", "h": "h",
              "hydraulic_surface": "hydraulic_surface",
          })
    b.add("reconstruct_hydraulic_surface", _reconstruct_epsilon_factory,
          bind=_fill_surface_plan("surface", "z_prime", distance="surface"))
    b.add("copy_hydraulic_fill_depth", _merge_fill_depth_factory,
          bind={
              "z": "z", "filled": "z_prime", "h": "h",
              "conditioned": "is_border",
          })
    b.pipeline("fill_hydraulic_surface", (
        "make_surface", "reconstruct_hydraulic_surface",
        "copy_hydraulic_fill_depth", "refresh_hydraulic_surface",
    ))
    b.add("route", _route_factory,
          bind={
              "grid": "grid", "surface": "hydraulic_surface",
              "rec": "rec",
          })
    b.add("snapshot_receivers", _snapshot_factory,
          bind={"rec": "rec", "rec_initial": "rec_initial"})
    b.add("skip_local_minima", _noop_factory, bind={})
    b.add("resolve_cordonnier", _carve_factory, bind=_carve_plan)
    b.dispatch("route_local_minima", on="mfd_local_minima", cases={
        "rank_cordonnier": "route",
        "fill_cordonnier": "route",
        "carve_cordonnier": "route",
        "reconstruct_epsilon": "skip_local_minima",
    })
    b.dispatch("snapshot_local_minima", on="mfd_local_minima", cases={
        "rank_cordonnier": "snapshot_receivers",
        "fill_cordonnier": "snapshot_receivers",
        "carve_cordonnier": "snapshot_receivers",
        "reconstruct_epsilon": "skip_local_minima",
    })
    b.dispatch("resolve_minima", on="mfd_local_minima", cases={
        "rank_cordonnier": "resolve_cordonnier",
        "fill_cordonnier": "resolve_cordonnier",
        "carve_cordonnier": "resolve_cordonnier",
        "reconstruct_epsilon": "reconstruct_hydraulic_surface",
    })
    b.add("compute_rank", _rank_factory, bind=_rank_plan)
    b.add("compute_cordonnier_fill", _cordonnier_fill_factory,
          bind=_cordonnier_fill_plan)
    b.add("compute_active_cordonnier_fill", _active_cordonnier_fill_factory,
          bind=_active_cordonnier_fill_plan)
    b.add("compute_cordonnier_carve", _cordonnier_carve_factory,
          bind=_cordonnier_carve_plan)
    b.dispatch("prepare_mfd_surface", on="mfd_local_minima", cases={
        "rank_cordonnier": "compute_rank",
        "fill_cordonnier": "compute_cordonnier_fill",
        "carve_cordonnier": "compute_cordonnier_carve",
        "reconstruct_epsilon": "copy_hydraulic_fill_depth",
    })
    b.add("copy_active_hydraulic_fill_depth", _merge_active_fill_depth_factory,
          bind={
              "z": "z", "filled": "z_prime", "active": "active_mask",
              "h": "h", "conditioned": "is_border",
          })
    b.dispatch("prepare_active_mfd_surface", on="mfd_local_minima", cases={
        "rank_cordonnier": "compute_rank",
        "fill_cordonnier": "compute_active_cordonnier_fill",
        "carve_cordonnier": "compute_cordonnier_carve",
        "reconstruct_epsilon": "copy_active_hydraulic_fill_depth",
    })
    b.add("build_rank_topology", _topology_factory, bind=_topology_plan)
    b.add("build_fill_topology", _filled_topology_factory,
          bind=_filled_topology_plan)
    b.add("build_carve_topology", _carve_topology_factory,
          bind=_carve_topology_plan)
    b.add("build_reconstructed_topology", _reconstructed_topology_factory,
          bind=_reconstructed_topology_plan)
    b.dispatch("build_topology", on="mfd_local_minima", cases={
        "rank_cordonnier": "build_rank_topology",
        "fill_cordonnier": "build_fill_topology",
        "carve_cordonnier": "build_carve_topology",
        "reconstruct_epsilon": "build_reconstructed_topology",
    })
    b.add("prepare_frontier", _frontier_factory, bind={
        "clear.count": "mfd_count", "clear.barrier": "mfd_barrier",
        "compact.indegree": "indegree", "compact.frontier": "mfd_frontier0",
        "compact.count": "mfd_count",
    })
    b.add("accumulate", _accumulation_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "PRECIPITATION": "precipitation", "Qi": "Qi",
              "frontier0": "mfd_frontier0", "frontier1": "mfd_frontier1",
              "count": "mfd_count", "barrier": "mfd_barrier",
              "dirs": "directions", "mfd_w": "weights", "accum": "Qi",
              "indegree": "indegree",
          }))
    b.add("accumulate_drainage_area", _drainage_area_factory,
          bind=_drainage_area_plan)
    b.add("accumulate_active", _subset_accumulation_factory,
          bind=_subset_accumulation_plan)
    b.add("snapshot_boundary_flux", _copy_flux_factory,
          bind={"source": "Qi", "destination": "Qi_boundary"})
    b.add("initialize_river_distance", _river_distance_init_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "drainage_area": "drainage_area",
              "Qi": "Qi", "effective_area": "effective_drainage_area",
              "distance": "river_distance",
              "distance_work": "river_distance_work",
              "AREA_THRESHOLD": "flow_area_threshold",
              "PRECIPITATION_REFERENCE": "precipitation_reference",
          }))
    b.add("relax_river_distance", _river_distance_relax_factory,
          bind=_river_distance_relax_plan)
    b.add("mark_river_seed", _river_seed_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "distance": "river_distance", "river_flags": "river_flags",
              "river_mask": "river_mask",
              "RIVER_PADDING": "river_padding",
          }))
    b.add("initialize_dynamic_domain", _dynamic_domain_init_factory,
          bind={
              "seed": "river_mask", "flags": "dynamic_flags",
              "claims": "dynamic_claims", "active": "dynamic_active_mask",
          })
    b.add("grow_dynamic_domain", _dynamic_domain_grow_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "active_ids": "dynamic_active_ids", "h": "h",
              "residual": "analytical_residual",
              "claims": "dynamic_claims", "active": "dynamic_active_mask",
              "count": "dynamic_count_device",
              "ACTIVE_COUNT": "dynamic_active_count",
              "ANALYTICAL_RELAXATION": "analytical_relaxation",
              "GROWTH_RESIDUAL": "growth_residual_threshold",
              "DEPTH_CAP": "hillslope_depth_cap",
          }))
    b.add("grow_dynamic_domain_transient", _dynamic_transient_grow_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "active_ids": "dynamic_active_ids", "h": "h",
              "activity": "transient_activity",
              "claims": "dynamic_claims", "active": "dynamic_active_mask",
              "count": "dynamic_count_device",
              "ACTIVE_COUNT": "dynamic_active_count",
              "TRANSIENT_ACTIVITY": "transient_activity_threshold",
              "DEPTH_CAP": "hillslope_depth_cap",
          }))
    b.add("clear_dynamic_activity", _dynamic_activity_clear_factory,
          bind={
              "active_ids": "dynamic_active_ids",
              "activity": "transient_activity",
              "ACTIVE_COUNT": "dynamic_active_count",
          })
    b.add("mark_dynamic_work", _active_work_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "active": "dynamic_active_mask",
              "flags": "dynamic_work_flags",
              "work": "dynamic_work_mask",
          }))
    b.add("update_depth", _update_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "h": "h", "Qi": "Qi", "Qo": "Qo",
              "steepest_slope": "steepest_slope", "flow_width": "flow_width",
              "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
              "DT": "dt",
          }))
    b.add("transport_transient", _transient_factory, bind=_transient_plan)
    b.add("transport_dynamic_transient", _dynamic_transient_factory,
          bind=_transient_plan)
    b.add("transport_hybrid", _hybrid_factory, bind=_transient_plan)
    b.add("update_depth_analytical_local", _local_analytical_update_factory,
          bind=_local_analytical_update_plan)
    b.add(
        "update_depth_analytical_bottom_up",
        _bottom_up_analytical_update_factory,
        bind=_bottom_up_analytical_update_plan,
    )
    b.add("update_depth_analytical_capped",
          _capped_local_analytical_update_factory,
          bind=_capped_local_analytical_update_plan)
    b.add("update_dynamic_depth_analytical",
          _dynamic_local_analytical_update_factory,
          bind=_dynamic_local_analytical_update_plan)
    b.dispatch("update_depth_analytical", on="analytical_solver", cases={
        "local": "update_depth_analytical_local",
        "bottom_up": "update_depth_analytical_bottom_up",
    })
    b.pipeline("run_n_step", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "prepare_frontier", "accumulate", "update_depth",
    ))
    b.pipeline("run_n_step_transient", (
        "make_surface", "transport_transient",
    ))
    b.pipeline("run_n_step_hybrid", (
        "make_surface", "transport_hybrid",
    ))
    b.pipeline("run_n_step_analytical", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "prepare_frontier", "accumulate", "update_depth_analytical",
    ))
    b.pipeline("run_n_step_analytical_capped", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "prepare_frontier", "accumulate", "update_depth_analytical_capped",
    ))
    b.pipeline("_prepare_dynamic_seed", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "prepare_frontier", "accumulate_drainage_area",
        # Accumulation consumes indegree, so rebuild it for the initial Qi.
        "build_topology", "prepare_frontier", "accumulate",
        "snapshot_boundary_flux", "initialize_river_distance",
    ))
    b.pipeline("_relax_river_distance", ("relax_river_distance",))
    b.pipeline("_refresh_dynamic_boundary", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_active_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "prepare_frontier", "accumulate", "snapshot_boundary_flux",
    ))
    b.pipeline("_run_dynamic_active_batch", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_active_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "accumulate_active", "update_dynamic_depth_analytical",
    ))
    b.pipeline("_run_dynamic_transient", (
        "make_surface", "transport_dynamic_transient",
    ))
    program = b.freeze()

    def _bind_dynamic_domain(self, count, view):
        self.active_count.set(count)
        self.dynamic_active_count.set(count)
        self._ensure_compiled("accumulate_active")
        self._ensure_compiled("update_dynamic_depth_analytical")
        self._ensure_compiled("grow_dynamic_domain")
        self._ensure_compiled("grow_dynamic_domain_transient")
        self._ensure_compiled("clear_dynamic_activity")
        accumulation = self._states["accumulate_active"].compiled
        accumulation.swap("clear.active_ids", view)
        accumulation.swap("prepare.active_ids", view)
        accumulation.swap(
            "prepare.active", self._handle("dynamic_active_mask"),
        )
        accumulation.swap(
            "accum.active", self._handle("dynamic_active_mask"),
        )
        self._states["update_dynamic_depth_analytical"].compiled.swap(
            "active_ids", view,
        )
        self._states["grow_dynamic_domain"].compiled.swap("active_ids", view)
        self._states["grow_dynamic_domain_transient"].compiled.swap(
            "active_ids", view,
        )
        self._states["clear_dynamic_activity"].compiled.swap(
            "active_ids", view,
        )
        transient = self._states["transport_dynamic_transient"].compiled
        if transient is not None:
            transient.swap("update.active_ids", view)

    def prepare_dynamic_flow_domain(
            self, area_threshold, initial_padding=0.0,
            precipitation_reference=None):
        """Seed a growable river domain from effective drainage area."""
        if self.mfd_local_minima != "carve_cordonnier":
            raise ProgramError(
                "dynamic flow domains currently require "
                "mfd_local_minima='carve_cordonnier'",
            )
        area_threshold = float(area_threshold)
        initial_padding = float(initial_padding)
        if not math.isfinite(area_threshold) or area_threshold <= 0.0:
            raise ProgramError("area_threshold must be finite and > 0")
        if not math.isfinite(initial_padding) or initial_padding < 0.0:
            raise ProgramError("initial_padding must be finite and >= 0")
        if precipitation_reference is None:
            precipitation_reference = 0.0
        else:
            precipitation_reference = float(precipitation_reference)
            if (not math.isfinite(precipitation_reference)
                    or precipitation_reference <= 0.0):
                raise ProgramError(
                    "precipitation_reference must be finite and > 0",
                )
        self.flow_area_threshold.set(area_threshold)
        self.precipitation_reference.set(precipitation_reference)
        self.river_padding.set(initial_padding)
        self._prepare_dynamic_seed()
        edge_steps = int(math.ceil(initial_padding / float(self.dx)))
        if edge_steps:
            self._relax_river_distance((edge_steps + 1) // 2)
        self.mark_river_seed()
        self.initialize_dynamic_domain()
        scan = getattr(self, "_dynamic_domain_scan", None)
        if scan is None:
            scan = make_scan(self._be, self._pool, self.nx * self.ny)
            object.__setattr__(self, "_dynamic_domain_scan", scan)
        count = scan.compact(
            self._handle("dynamic_flags"),
            self._handle("dynamic_active_ids"),
        )
        self.dynamic_active_count.set(count)
        view = self._be.wrap(
            self._handle("dynamic_active_ids").array[:count], owned=False,
        )
        old_view = getattr(self, "_dynamic_domain_view", None)
        object.__setattr__(self, "_dynamic_domain_view", view)
        object.__setattr__(self, "_dynamic_residual_valid", False)
        object.__setattr__(self, "_dynamic_activity_valid", False)
        object.__setattr__(self, "_dynamic_activity_initialized", False)
        object.__setattr__(self, "_dynamic_work_dirty", True)
        _bind_dynamic_domain(self, count, view)
        if old_view is not None:
            old_view.destroy()
        return count

    def grow_dynamic_flow_domain(self, residual_threshold=None):
        """Grow from the last analytical depth residual; return nodes added."""
        if getattr(self, "_dynamic_domain_view", None) is None:
            raise ProgramError("call prepare_dynamic_flow_domain() first")
        if not getattr(self, "_dynamic_residual_valid", False):
            raise ProgramError("run_dynamic_n_step_analytical() before growing")
        if residual_threshold is not None:
            residual_threshold = float(residual_threshold)
            if not math.isfinite(residual_threshold) or residual_threshold < 0.0:
                raise ProgramError(
                    "residual_threshold must be finite and >= 0",
                )
            self.growth_residual_threshold.set(residual_threshold)
        return _append_dynamic_neighbours(self, "grow_dynamic_domain")

    def grow_dynamic_flow_domain_transient(self, activity_threshold=None):
        """Grow from accumulated transient |Δh|, then reset that activity."""
        if getattr(self, "_dynamic_domain_view", None) is None:
            raise ProgramError("call prepare_dynamic_flow_domain() first")
        if not getattr(self, "_dynamic_activity_valid", False):
            raise ProgramError("run_dynamic_n_step_transient() before growing")
        if activity_threshold is not None:
            activity_threshold = float(activity_threshold)
            if not math.isfinite(activity_threshold) or activity_threshold < 0.0:
                raise ProgramError("activity_threshold must be finite and >= 0")
            self.transient_activity_threshold.set(activity_threshold)
        added = _append_dynamic_neighbours(
            self, "grow_dynamic_domain_transient",
        )
        self.clear_dynamic_activity()
        object.__setattr__(self, "_dynamic_activity_valid", False)
        object.__setattr__(self, "_dynamic_activity_initialized", True)
        return added

    def _append_dynamic_neighbours(self, grow_name):
        count = int(self.dynamic_active_count.read())
        self._handle("dynamic_count_device").from_numpy(
            np.asarray([count], dtype=np.int32),
        )
        getattr(self, grow_name)()
        new_count = int(
            self._handle("dynamic_count_device").to_numpy()[0],
        )
        if new_count == count:
            return 0
        view = self._be.wrap(
            self._handle("dynamic_active_ids").array[:new_count], owned=False,
        )
        old_view = getattr(self, "_dynamic_domain_view")
        object.__setattr__(self, "_dynamic_domain_view", view)
        _bind_dynamic_domain(self, new_count, view)
        old_view.destroy()
        object.__setattr__(self, "_dynamic_residual_valid", False)
        object.__setattr__(self, "_dynamic_activity_valid", False)
        object.__setattr__(self, "_dynamic_activity_initialized", False)
        object.__setattr__(self, "_dynamic_work_dirty", True)
        return new_count - count

    def run_dynamic_n_step_analytical(self, n):
        """Solve the current dynamic domain for ``n`` steps without growth."""
        n = int(n)
        if n < 1:
            raise ProgramError("n must be >= 1")
        if getattr(self, "_dynamic_domain_view", None) is None:
            raise ProgramError("call prepare_dynamic_flow_domain() first")
        object.__setattr__(self, "_dynamic_residual_valid", False)
        object.__setattr__(self, "_dynamic_activity_valid", False)
        object.__setattr__(self, "_dynamic_activity_initialized", False)
        self._refresh_dynamic_boundary()
        count = int(self.dynamic_active_count.read())
        _bind_dynamic_domain(self, count, self._dynamic_domain_view)
        if count:
            self._run_dynamic_active_batch(n)
        object.__setattr__(self, "_dynamic_residual_valid", True)

    def _ensure_dynamic_work(self):
        """Compact the active list plus one-cell halo only after growth."""
        if getattr(self, "_dynamic_work_dirty", True):
            self.mark_dynamic_work()
            scan = getattr(self, "_dynamic_work_scan", None)
            if scan is None:
                scan = make_scan(self._be, self._pool, self.nx * self.ny)
                object.__setattr__(self, "_dynamic_work_scan", scan)
            work_count = scan.compact(
                self._handle("dynamic_work_flags"),
                self._handle("dynamic_work_ids"),
            )
            view = self._be.wrap(
                self._handle("dynamic_work_ids").array[:work_count],
                owned=False,
            )
            old_view = getattr(self, "_dynamic_work_view", None)
            object.__setattr__(self, "_dynamic_work_view", view)
            object.__setattr__(self, "_dynamic_work_count", work_count)
            object.__setattr__(self, "_dynamic_work_dirty", False)
            transient = self._states["transport_dynamic_transient"].compiled
            if transient is not None:
                for step in ("topology", "outflow", "limit"):
                    transient.swap(step + ".work_ids", view)
            if old_view is not None:
                old_view.destroy()
        self.work_count.set(self._dynamic_work_count)

    def run_dynamic_n_step_transient(self, n):
        """Run fixed-dt local transient transport on the dynamic domain."""
        n = int(n)
        if n < 1:
            raise ProgramError("n must be >= 1")
        if getattr(self, "_dynamic_domain_view", None) is None:
            raise ProgramError("call prepare_dynamic_flow_domain() first")

        count = int(self.dynamic_active_count.read())
        if not count:
            return
        object.__setattr__(self, "_dynamic_residual_valid", False)
        if not getattr(self, "_dynamic_activity_initialized", False):
            self.clear_dynamic_activity()
            object.__setattr__(self, "_dynamic_activity_initialized", True)
        self.active_count.set(count)
        _ensure_dynamic_work(self)
        self._ensure_compiled("transport_dynamic_transient")
        transport = self._states["transport_dynamic_transient"].compiled
        for step in ("topology", "outflow", "limit"):
            transport.swap(step + ".work_ids", self._dynamic_work_view)
        transport.swap("update.active_ids", self._dynamic_domain_view)
        self._run_dynamic_transient(n)
        object.__setattr__(self, "_dynamic_activity_valid", True)

    def dynamic_max_residual(self):
        """Return max absolute undamped depth residual on the solved domain."""
        if not getattr(self, "_dynamic_residual_valid", False):
            raise ProgramError("run_dynamic_n_step_analytical() first")
        count = int(self.dynamic_active_count.read())
        if count == 0:
            return 0.0
        import cupy as cp

        ids = self._handle("dynamic_active_ids").array[:count]
        residual = self._handle("analytical_residual").array
        return float(cp.max(cp.abs(residual[ids])).item())

    base_close = program.close

    def close(self):
        domain_scan = getattr(self, "_dynamic_domain_scan", None)
        if domain_scan is not None:
            domain_scan.close()
            object.__setattr__(self, "_dynamic_domain_scan", None)
        dynamic_work_scan = getattr(self, "_dynamic_work_scan", None)
        if dynamic_work_scan is not None:
            dynamic_work_scan.close()
            object.__setattr__(self, "_dynamic_work_scan", None)
        base_close(self)
        dynamic_view = getattr(self, "_dynamic_domain_view", None)
        if dynamic_view is not None:
            dynamic_view.destroy()
        object.__setattr__(self, "_dynamic_domain_view", None)
        dynamic_work_view = getattr(self, "_dynamic_work_view", None)
        if dynamic_work_view is not None:
            dynamic_work_view.destroy()
        object.__setattr__(self, "_dynamic_work_view", None)

    program.prepare_dynamic_flow_domain = prepare_dynamic_flow_domain
    program.run_dynamic_n_step_analytical = run_dynamic_n_step_analytical
    program.run_dynamic_n_step_transient = run_dynamic_n_step_transient
    program.grow_dynamic_flow_domain = grow_dynamic_flow_domain
    program.grow_dynamic_flow_domain_analytical = grow_dynamic_flow_domain
    program.grow_dynamic_flow_domain_transient = grow_dynamic_flow_domain_transient
    program.dynamic_max_residual = dynamic_max_residual
    program.close = close
    program.outlet_mask = property(
        lambda self: _GridMaskAccessor(self, "OUTLET_MASK", "outlet='mask'"),
    )
    program.nodata_mask = property(
        lambda self: _GridMaskAccessor(self, "NODATA_MASK", "nodata=True"),
    )
    return program

GraphFloodProgram = build_graphflood_program()

__all__ = ["GraphFloodProgram", "build_graphflood_program"]
