"""CuPy GraphFlood using Cordonnier carving and persistent MFD.

``h`` is persistent model state. Initialise it with ``h.from_numpy(...)``,
call ``reset_h()`` for a dry surface, or call ``initialize_h_from_fill()`` to
start with the reconstructed depression-storage depth before the first
``run_n_step(n)``. Between iterations, ``fill_hydraulic_surface()`` fills the
current ``z + h`` surface while preserving existing water depth.
Precipitation is a depth rate; the accumulation initialiser converts it to
per-cell discharge using the grid cell area. Max-normalized unsigned-byte MFD
scores are enabled by default and may be disabled with
``quantized_weight=False``. ``mfd_local_minima="rank_cordonnier"`` keeps the
carved receiver-rank gate; ``"fill_cordonnier"`` instead converts the carved
paths into a filled hydraulic surface and runs ordinary MFD on that surface;
``"carve_cordonnier"`` uses the same monotonic routing potential without
adding depression storage to ``h`` and cuts physically uphill links from
pressure-head propagation;
``"reconstruct_epsilon"`` uses morphological reconstruction plus its epsilon
flat ordering and does not run Cordonnier.

``run_n_step_analytical(n)`` is an opt-in nonlinear steady-state pipeline.
``analytical_solver="local"`` uses the original pointwise frozen-slope
inversion; ``"bottom_up"`` mirrors the persistent Kahn walk from outlets to
sources, constructs a complete receiver-consistent candidate, then damps that
candidate globally. Repeating the pipeline rebuilds the graph between sweeps.
``run_n_step_transient(n)`` is a conservative local-flux operator. It always
uses raw ``z+h`` MFD without depression conditioning: sinks retain incoming
water until their hydraulic surface develops a physical outlet. Its global
explicit timestep is selected from the local Manning discharge sensitivity;
``transient_dt`` is its upper bound and ``transient_cfl`` its safety factor.
Set ``adaptive_transient_dt=False`` to retain the fixed-step operator.

Receiver routing and unconditioned link gradients use an internal float64
``z + h`` surface. Depth, discharge, terrain, and accumulation remain
float32. Depression-conditioned MFD routing deliberately retains the exact
float32 fill/epsilon potential so its ordering stays globally acyclic; its
physical slope diagnostics use float64 away from conditioned links.

Grid connectivity and boundaries are compile-time program options. ``topology``
accepts ``"D4"`` or ``"D8"``; ``boundary`` accepts ``"normal"``,
``"periodic_EW"``, or ``"periodic_NS"``; and ``outlet`` accepts ``"edge"``
or ``"mask"``. With masked outlets, set ``outlet_mask`` before running. Set
``nodata=True`` to additionally expose ``nodata_mask``.

For ordered regional relaxation, ``prepare_distance_sweep()`` performs one
Cordonnier fill, freezes global boundary discharge, and computes a fixed
outlet-to-source coordinate. ``set_active_band(lower, upper)`` compacts the
selected nodes. Use ``run_active_n_step()``,
``run_active_n_step_analytical()``, or ``run_active_n_step_transient()``;
``refresh_band_boundary()`` updates boundary discharge between full sweeps.
"""

import math

import numpy as np

from pyfastflow.core import (
    KernelBuilder, ProgramError, RoutineBuilder, SequenceBuilder,
)
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow import (
    depression_binding_plan,
    make_accumulation,
    make_depression_solver,
    make_depressions,
    make_mfd_distance,
    make_mfd_topology,
    make_subset_mfd_accumulation,
)
from pyfastflow.flow._cupy_mfd_accum import persistent_grid_block
from pyfastflow.graphflood._cupy_friction import build_friction_velocity
from pyfastflow.grid import make_grid_group, make_grid_parameters
from pyfastflow.ops import make_scan

from ..flow.sfd import (
    BLOCK,
    _cupy_only,
    _grid_leaf_plan,
    _noop_factory,
    _reconstruct_epsilon_factory,
)


class _GridMaskAccessor:
    """Public 2-D view of an optional field parameter owned by the grid."""

    __slots__ = ("_program", "_leaf", "_option")

    def __init__(self, program, leaf, option):
        self._program = program
        self._leaf = leaf
        self._option = option

    def _parameter(self):
        self._program._check_open()
        try:
            return self._program._bundle_params["grid"][self._leaf]
        except KeyError as exc:
            raise ProgramError(
                f"{self._leaf.lower()!r} is unavailable; construct the "
                f"program with {self._option}"
            ) from exc

    def from_numpy(self, array):
        array = np.asarray(array)
        shape = (self._program.ny, self._program.nx)
        if array.shape != shape:
            raise ProgramError(
                f"{self._leaf.lower()!r}: expected shape {shape}, "
                f"got {array.shape}"
            )
        self._parameter().set(array.astype(np.uint8, copy=False).reshape(-1))

    def to_numpy(self):
        return self._parameter().handle().to_numpy().reshape(
            self._program.ny, self._program.nx,
        )

    @property
    def array(self):
        return self._parameter().handle().array

    @property
    def shape(self):
        return (self._program.ny, self._program.nx)

    @property
    def dtype(self):
        return np.dtype(np.uint8)


def _make_surface_factory(be, _bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void make_hydraulic_surface(
                const float* z, const float* h, float* surface,
                double* hydraulic_surface) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) {{
                double eta = (double)z[i] + (double)h[i];
                surface[i] = (float)eta;
                hydraulic_surface[i] = eta;
            }}
        }}''',
        domain=n,
    ).freeze()


def _refresh_hydraulic_surface_factory(be, _bundles, config):
    """Refresh double precision eta after a minima method changes h."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void refresh_hydraulic_surface(
                const float* z, const float* h,
                double* hydraulic_surface) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n})
                hydraulic_surface[i] = (double)z[i] + (double)h[i];
        }}''',
        domain=n,
    ).freeze()


def _route_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_receivers_f64(
                const double* surface, int* rec) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.can_out(i)$) {{ rec[i] = i; return; }}

            int receiver = i;
            double best_slope = 0.0;
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk; ++k) {{
                int j = $ctx.grid.neighbour(i, k)$;
                if (j == -1) continue;
                double slope = (surface[i] - surface[j])
                             / (double)$ctx.grid.dist_from_k(k)$;
                if (slope > best_slope) {{
                    best_slope = slope;
                    receiver = j;
                }}
            }}
            rec[i] = receiver;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()


def _snapshot_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology=config["topology"],
        quantized_weight=config["quantized_weight"],
    )["snapshot_receivers"]


def _carve_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    deps = make_depressions(
        be, bundles["grid"], bundles.param("ndep"), method="optimized",
        reroute="carve", n_flat=n,
    )
    return make_depression_solver(
        be, deps, bundles.bundle_params("grid"), method="optimized",
        reroute="carve", n_flat=n, block_size=BLOCK,
    )[0]


def _carve_plan(frozen, _be):
    plan = depression_binding_plan(frozen, method="optimized", reroute="carve")
    return {target: (
                "surface" if value == "z" else
                "basin_outlet" if value == "outlet" else value
            )
            for target, value in plan.items()}


def _rank_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_topology(
        be, bundles["grid"], method="cordonnier_rank",
        n_flat=config["nx"] * config["ny"], topology=config["topology"],
        quantized_weight=config["quantized_weight"],
    )["receiver_rank"]


def _rank_plan(_frozen, _be):
    return {
        "init.rec": "rec", "init.ancestor": "rank_ancestor", "init.rank": "rank",
        "forward.ancestor_in": "rank_ancestor", "forward.rank_in": "rank",
        "forward.ancestor_out": "rank_ancestor_alt", "forward.rank_out": "rank_alt",
        "backward.ancestor_in": "rank_ancestor_alt", "backward.rank_in": "rank_alt",
        "backward.ancestor_out": "rank_ancestor", "backward.rank_out": "rank",
    }


def _cordonnier_surface_factory(
        be, _bundles, config, *, fill_depth, active_only=False):
    """Build Cordonnier's path-maximum routing potential."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    init = KernelBuilder(
        f'''extern "C" __global__ void graphflood_fill_init(
                const float* surface, const int* rec,
                int* ancestor, int* rank, float* spill) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            ancestor[i] = rec[i];
            rank[i] = rec[i] == i ? 0 : 1;
            spill[i] = surface[i];
        }}''', domain=n,
    ).freeze()
    jump = KernelBuilder(
        f'''extern "C" __global__ void graphflood_fill_jump(
                const int* ancestor_in, const int* rank_in,
                const float* spill_in, int* ancestor_out, int* rank_out,
                float* spill_out) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int a = ancestor_in[i];
            if (a == i) {{
                ancestor_out[i] = i;
                rank_out[i] = rank_in[i];
                spill_out[i] = spill_in[i];
            }} else {{
                ancestor_out[i] = ancestor_in[a];
                rank_out[i] = rank_in[i] + rank_in[a];
                spill_out[i] = fmaxf(spill_in[i], spill_in[a]);
            }}
        }}''', domain=n,
    ).freeze()
    h_argument = "float* h, " if fill_depth else ""
    active_argument = "const unsigned char* active, " if active_only else ""
    active_gate = "active[i] && " if active_only else ""
    h_update = (
        f"if ({active_gate}added_depth > 0.0f) h[i] += added_depth;"
        if fill_depth else ""
    )
    apply_name = "graphflood_apply_fill" if fill_depth else "graphflood_apply_carve"
    apply = KernelBuilder(
        f'''extern "C" __global__ void {apply_name}(
                {h_argument}{active_argument}float* surface, const float* spill,
                const int* rec_initial, const int* rec,
                unsigned char* conditioned) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float filled = spill[i];
            float added_depth = filled - surface[i];
            conditioned[i] = (added_depth > 0.0f
                              || rec_initial[i] != rec[i]) ? 1u : 0u;
            {h_update}
            surface[i] = filled;
        }}''', domain=n,
    ).freeze()
    rounds = math.ceil(math.log2(max(2, n))) + 1
    if rounds % 2:
        rounds += 1
    return (SequenceBuilder()
            .add("init", init).add("forward", jump).add("backward", jump)
            .add("apply", apply).step("init")
            .loop(("forward", "backward"), max_times=rounds // 2)
            .step("apply").freeze())


def _cordonnier_fill_factory(be, bundles, config):
    """Physically fill z+h to Cordonnier's path maximum."""
    return _cordonnier_surface_factory(
        be, bundles, config, fill_depth=True,
    )


def _cordonnier_carve_factory(be, bundles, config):
    """Build the same routing potential without changing water depth."""
    return _cordonnier_surface_factory(
        be, bundles, config, fill_depth=False,
    )


def _active_cordonnier_fill_factory(be, bundles, config):
    return _cordonnier_surface_factory(
        be, bundles, config, fill_depth=True, active_only=True,
    )


def _cordonnier_fill_plan(_frozen, _be):
    return {
        "init.surface": "surface", "init.rec": "rec",
        "init.ancestor": "rank_ancestor", "init.rank": "rank",
        "init.spill": "z_prime",
        "forward.ancestor_in": "rank_ancestor",
        "forward.rank_in": "rank", "forward.spill_in": "z_prime",
        "forward.ancestor_out": "rank_ancestor_alt",
        "forward.rank_out": "rank_alt",
        "forward.spill_out": "steepest_slope",
        "backward.ancestor_in": "rank_ancestor_alt",
        "backward.rank_in": "rank_alt",
        "backward.spill_in": "steepest_slope",
        "backward.ancestor_out": "rank_ancestor",
        "backward.rank_out": "rank", "backward.spill_out": "z_prime",
        "apply.h": "h", "apply.surface": "surface",
        "apply.spill": "z_prime", "apply.rec_initial": "rec_initial",
        "apply.rec": "rec", "apply.conditioned": "is_border",
    }


def _active_cordonnier_fill_plan(frozen, be):
    plan = _cordonnier_fill_plan(frozen, be)
    plan["apply.active"] = "active_mask"
    return plan


def _cordonnier_carve_plan(frozen, be):
    plan = _cordonnier_fill_plan(frozen, be)
    plan.pop("apply.h")
    plan["apply.conditioned"] = "hydraulic_conditioned"
    return plan


def _build_hydraulic_topology(be, grid, n, topology, quantized_weight):
    """Rank-gated MFD weights plus unmodified hydraulic slope diagnostics."""
    weight_type = "unsigned char" if quantized_weight else "float"
    max_decl = "double max_score = 0.0;" if quantized_weight else ""
    max_forced = "max_score = 1.0;" if quantized_weight else ""
    max_update = "if (score > max_score) max_score = score;" if quantized_weight else ""
    weight_write = (
        "weights[i * nk + k] = scores[k] > 0.0 "
        "? (unsigned char)max(1, __double2int_rn(255.0 * scores[k] / max_score)) : 0;"
        if quantized_weight else
        "weights[i * nk + k] = sum_score > 0.0 "
        "? (float)(scores[k] / sum_score) : 0.0f;"
    )
    dirs_weights = KernelBuilder(
        f'''extern "C" __global__ void graphflood_ranked_mfd(
                const double* surface, const int* rec_initial, const int* rec,
                const int* rank, unsigned char* dirs, {weight_type}* weights,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            unsigned char mask = 0;
            double scores[8];
            double sum_score = 0.0;
            {max_decl}
            double best_slope = 0.0;
            float best_width = $ctx.grid.DX.get(0)$;
            for (int k = 0; k < 8; ++k) scores[k] = 0.0;

            if (!$ctx.grid.can_out(i)$ && !$ctx.grid.nodata(i)$) {{
                // Manning uses the true hydraulic slope. MFD partitioning
                // separately uses slope times the link's effective width.
                for (int k = 0; k < nk; ++k) {{
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j == -1) continue;
                    double width = (double)$ctx.grid.dist_from_k(k)$;
                    double slope = (surface[i] - surface[j]) / width;
                    if (slope > best_slope) {{
                        best_slope = slope;
                        best_width = width;
                    }}
                }}

                if (rec_initial[i] != rec[i]) {{
                    for (int k = 0; k < nk; ++k) {{
                        if ($ctx.grid.neighbour(i, k)$ == rec[i]) {{
                            mask = (unsigned char)(1u << k);
                            scores[k] = 1.0;
                            sum_score = 1.0;
                            {max_forced}
                            break;
                        }}
                    }}
                }} else {{
                    for (int k = 0; k < nk; ++k) {{
                        int j = $ctx.grid.neighbour(i, k)$;
                        if (j == -1 || rank[j] >= rank[i]) continue;
                        double width = (double)$ctx.grid.dist_from_k(k)$;
                        double slope = (surface[i] - surface[j]) / width;
                        double score = slope > 0.0 ? slope * width : 0.0;
                        scores[k] = score;
                        if (score > 0.0) {{
                            mask |= (unsigned char)(1u << k);
                            sum_score += score;
                            {max_update}
                        }}
                    }}
                }}
            }}

            for (int k = 0; k < nk; ++k) {{ {weight_write} }}
            dirs[i] = mask;
            steepest_slope[i] = (float)best_slope;
            flow_width[i] = best_width;
        }}''',
        domain=n,
    ).compose("grid", grid).freeze()

    generic = make_mfd_topology(
        be, grid,
        method="cordonnier_rank", n_flat=n, topology=topology,
        quantized_weight=quantized_weight,
    )
    return dirs_weights, generic["indegree_reset"], generic["indegree_count"]


def _build_filled_hydraulic_topology(
        be, grid, n, topology, quantized_weight):
    """MFD on the exact conditioned surface, with precise physical slopes."""
    weight_type = "unsigned char" if quantized_weight else "float"
    max_decl = "float max_score = 0.0f;" if quantized_weight else ""
    max_update = "if (score > max_score) max_score = score;" if quantized_weight else ""
    weight_write = (
        "weights[i * nk + k] = scores[k] > 0.0f "
        "? (unsigned char)max(1, __float2int_rn(255.0f * scores[k] / max_score)) : 0;"
        if quantized_weight else
        "weights[i * nk + k] = sum_score > 0.0f "
        "? scores[k] / sum_score : 0.0f;"
    )
    dirs_weights = KernelBuilder(
        f'''extern "C" __global__ void graphflood_filled_mfd(
                const float* surface, const int* rank,
                unsigned char* dirs, {weight_type}* weights,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            unsigned char mask = 0;
            float scores[8];
            float sum_score = 0.0f;
            {max_decl}
            float best_slope = 0.0f;
            float best_width = $ctx.grid.DX.get(0)$;
            for (int k = 0; k < 8; ++k) scores[k] = 0.0f;

            if (!$ctx.grid.can_out(i)$ && !$ctx.grid.nodata(i)$) {{
                float zi = surface[i];
                float ulp = nextafterf(zi, 1.0e30f) - zi;
                for (int k = 0; k < nk; ++k) {{
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j == -1) continue;
                    float drop = 0.0f;
                    if (zi > surface[j]) {{
                        drop = zi - surface[j];
                    }} else if (zi == surface[j] && rank[i] > rank[j]) {{
                        drop = ulp * (float)(rank[i] - rank[j]);
                    }}
                    if (drop <= 0.0f) continue;
                    float width = $ctx.grid.dist_from_k(k)$;
                    float slope = drop / width;
                    float score = slope * width;
                    scores[k] = score;
                    mask |= (unsigned char)(1u << k);
                    sum_score += score;
                    {max_update}
                    if (slope > best_slope) {{
                        best_slope = slope;
                        best_width = width;
                    }}
                }}
            }}

            for (int k = 0; k < nk; ++k) {{ {weight_write} }}
            dirs[i] = mask;
            steepest_slope[i] = best_slope;
            flow_width[i] = best_width;
        }}''', domain=n,
    ).compose("grid", grid).freeze()

    # Do not perturb the conditioned DAG. Only replace its hydraulic
    # diagnostic where neither endpoint was raised/rerouted and the true
    # double-precision surface has a valid downslope link.
    diagnostics = KernelBuilder(
        f'''extern "C" __global__ void graphflood_filled_diagnostics_f64(
                const double* physical_surface,
                const unsigned char* conditioned,
                const unsigned char* dirs,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n} || conditioned[i]) return;
            double best_slope = 0.0;
            float best_width = flow_width[i];
            unsigned char mask = dirs[i];
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk; ++k) {{
                if (!(mask & (1u << k))) continue;
                int j = $ctx.grid.neighbour(i, k)$;
                if (j == -1 || conditioned[j]) continue;
                double width = (double)$ctx.grid.dist_from_k(k)$;
                double slope = (physical_surface[i] - physical_surface[j])
                             / width;
                if (slope > best_slope) {{
                    best_slope = slope;
                    best_width = (float)width;
                }}
            }}
            if (best_slope > 0.0) {{
                steepest_slope[i] = (float)best_slope;
                flow_width[i] = best_width;
            }}
        }}''', domain=n,
    ).compose("grid", grid).freeze()
    generic = make_mfd_topology(
        be, grid, method="cordonnier_rank", n_flat=n, topology=topology,
        quantized_weight=quantized_weight,
    )
    return (
        dirs_weights, diagnostics,
        generic["indegree_reset"], generic["indegree_count"],
    )


def _topology_factory(be, bundles, config):
    _cupy_only(be)
    dirs, reset, count = _build_hydraulic_topology(
        be, bundles["grid"], config["nx"] * config["ny"],
        config["topology"],
        config["quantized_weight"],
    )
    return (RoutineBuilder().step("dirs_weights", dirs)
            .step("indegree_reset", reset).step("indegree_count", count).freeze())


def _filled_topology_factory(be, bundles, config):
    _cupy_only(be)
    dirs, diagnostics, reset, count = _build_filled_hydraulic_topology(
        be, bundles["grid"], config["nx"] * config["ny"],
        config["topology"],
        config["quantized_weight"],
    )
    return (RoutineBuilder().step("dirs_weights", dirs)
            .step("diagnostics", diagnostics)
            .step("indegree_reset", reset).step("indegree_count", count).freeze())


def _carve_topology_factory(be, bundles, config):
    """MFD on the virtual fill, with physical-or-subgrid hydraulic slope."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    dirs, diagnostics, reset, count = _build_filled_hydraulic_topology(
        be, bundles["grid"], n, config["topology"],
        config["quantized_weight"],
    )
    effective_slope = KernelBuilder(
        f'''extern "C" __global__ void graphflood_carve_effective_slope(
                const double* physical_surface,
                const unsigned char* dirs,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n} || $ctx.grid.nodata(i)$
                    || $ctx.grid.can_out(i)$) return;
            unsigned char mask = dirs[i];
            double best = 0.0;
            float width_best = flow_width[i];
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk; ++k) {{
                if (!(mask & (1u << k))) continue;
                int r = $ctx.grid.neighbour_raw(i, k)$;
                double width = (double)$ctx.grid.dist_from_k(k)$;
                double slope = (physical_surface[i] - physical_surface[r])
                             / width;
                if (slope > best) {{
                    best = slope;
                    width_best = (float)width;
                }}
            }}
            steepest_slope[i] = best > 0.0
                ? (float)best
                : fmaxf($ctx.CARVE_SLOPE.get(i)$, 1.0e-12f);
            flow_width[i] = width_best;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()
    return (RoutineBuilder().step("dirs_weights", dirs)
            .step("diagnostics", diagnostics)
            .step("effective_slope", effective_slope)
            .step("indegree_reset", reset).step("indegree_count", count)
            .freeze())


def _reconstructed_topology_factory(be, bundles, config):
    """MFD and Manning diagnostics on reconstruct+epsilon potential."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    topology = make_mfd_topology(
        be, bundles["grid"], method="surface", n_flat=n,
        topology=config["topology"], diagonal_partition_correction=True,
        quantized_weight=config["quantized_weight"],
    )
    diagnostics = KernelBuilder(
        f'''extern "C" __global__ void graphflood_reconstruct_diagnostics(
                const float* filled, const float* dist,
                const double* physical_surface,
                const unsigned char* conditioned,
                const unsigned char* dirs,
                float* steepest_slope, float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            double best_slope = 0.0;
            float best_width = $ctx.grid.DX.get(0)$;
            unsigned char mask = dirs[i];
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk; ++k) {{
                if (!(mask & (1u << k))) continue;
                int j = $ctx.grid.neighbour(i, k)$;
                double width = (double)$ctx.grid.dist_from_k(k)$;
                double conditioned_drop = ((double)filled[i]
                                           - (double)filled[j])
                                          + ((double)dist[i]
                                           - (double)dist[j]);
                double drop = conditioned_drop;
                if (!conditioned[i] && !conditioned[j]) {{
                    double physical_drop = physical_surface[i]
                                         - physical_surface[j];
                    if (physical_drop > 0.0) drop = physical_drop;
                }}
                double slope = drop / width;
                if (slope > best_slope) {{
                    best_slope = slope;
                    best_width = (float)width;
                }}
            }}
            steepest_slope[i] = (float)best_slope;
            flow_width[i] = best_width;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()
    return (RoutineBuilder()
            .step("dirs_weights", topology["dirs_weights"])
            .step("diagnostics", diagnostics)
            .step("indegree_reset", topology["indegree_reset"])
            .step("indegree_count", topology["indegree_count"]).freeze())


def _topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "surface": "hydraulic_surface",
        "rec_initial": "rec_initial", "rec": "rec",
        "rank": "rank", "dirs": "directions", "weights": "weights",
        "steepest_slope": "steepest_slope", "flow_width": "flow_width",
        "indegree": "indegree",
    })


def _filled_topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "surface": "surface", "physical_surface": "hydraulic_surface",
        "conditioned": "is_border", "rank": "rank",
        "dirs": "directions",
        "weights": "weights", "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "indegree": "indegree",
    })


def _carve_topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "surface": "surface", "physical_surface": "hydraulic_surface",
        "conditioned": "hydraulic_conditioned", "rank": "rank",
        "dirs": "directions", "weights": "weights",
        "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "indegree": "indegree",
        "CARVE_SLOPE": "carve_slope_min",
    })


def _reconstructed_topology_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "filled": "z_prime", "dist": "surface",
        "physical_surface": "hydraulic_surface",
        "conditioned": "is_border", "dirs": "directions",
        "mfd_w": "weights",
        "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "indegree": "indegree",
    })


def _frontier_factory(be, _bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    clear = KernelBuilder(
        '''extern "C" __global__ void graphflood_clear_frontier(
                int* count, unsigned int* barrier) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < 2) count[i] = 0;
            if (i == 0) barrier[0] = 0u;
        }''', domain=2,
    ).freeze()
    compact = KernelBuilder(
        f'''extern "C" __global__ void graphflood_compact_frontier(
                const int* indegree, int* frontier, int* count) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n} && indegree[i] == 0) {{
                int p = atomicAdd(&count[0], 1);
                frontier[p] = i;
            }}
        }}''', domain=n,
    ).freeze()
    return RoutineBuilder().step("clear", clear).step("compact", compact).freeze()


def _accumulation_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    q_init = KernelBuilder(
        f'''extern "C" __global__ void graphflood_q_init(float* Qi) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float dx = $ctx.grid.DX.get(0)$;
            Qi[i] = $ctx.grid.nodata(i)$ ? 0.0f
                : $ctx.PRECIPITATION.get(i)$ * dx * dx;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()
    accum = make_accumulation(
        be, bundles["grid"], method="persistent_mfd", n_flat=n,
        n_neighbours=8 if config["topology"] == "D8" else 4,
        quantized_weight=config["quantized_weight"],
    )["accum"]
    return RoutineBuilder().step("q_init", q_init).step("accum", accum).freeze()


def _subset_accumulation_factory(be, bundles, config):
    _cupy_only(be)
    return make_subset_mfd_accumulation(
        be, bundles["grid"], n_flat=config["nx"] * config["ny"],
        n_neighbours=8 if config["topology"] == "D8" else 4,
        quantized_weight=config["quantized_weight"],
    )


def _subset_accumulation_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "active_ids": "active_ids", "active": "active_mask",
        "dirs": "directions", "mfd_w": "weights",
        "boundary": "Qi_boundary", "accum": "Qi",
        "accumulation": "Qi", "remaining": "active_indegree",
        "frontier": "mfd_frontier0", "frontier0": "mfd_frontier0",
        "frontier1": "mfd_frontier1", "count": "mfd_count",
        "barrier": "mfd_barrier", "SOURCE": "precipitation",
        "ACTIVE_COUNT": "active_count",
    })


def _copy_flux_factory(be, _bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_copy_flux(
                const float* source, float* destination) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) destination[i] = source[i];
        }}''', domain=n,
    ).freeze()


def _distance_factory(be, bundles, config):
    _cupy_only(be)
    return make_mfd_distance(
        be, bundles["grid"], n_flat=config["nx"] * config["ny"],
        n_neighbours=8 if config["topology"] == "D8" else 4,
    )


def _distance_plan(frozen, _be):
    plan = _grid_leaf_plan(frozen, {
        "dirs": "directions", "upstream": "distance_upstream",
        "downstream": "distance_downstream",
        "position": "distance_position", "remaining": "distance_remaining",
        "maximum": "distance_max",
        "frontier": "distance_frontier0", "frontier0": "distance_frontier0",
        "frontier1": "distance_frontier1", "count": "distance_count",
        "barrier": "distance_barrier",
    })
    plan["walk_forward.distance"] = "distance_upstream"
    plan["walk_reverse.distance"] = "distance_downstream"
    return plan


def _active_band_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_mark_active_band(
                const float* position, int* flags, unsigned char* active) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float x = position[i];
            bool selected = !$ctx.grid.nodata(i)$ && x >= $ctx.BAND_MIN.get(0)$
                         && x <= $ctx.BAND_MAX.get(0)$;
            flags[i] = selected ? 1 : 0;
            active[i] = selected ? 1u : 0u;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()


def _active_work_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_mark_active_work(
                const unsigned char* active,
                int* flags, unsigned char* work) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            bool selected = active[i] != 0u;
            int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
            for (int k = 0; k < nk && !selected; ++k) {{
                int j = $ctx.grid.neighbour(i, k)$;
                selected = j != -1 && active[j] != 0u;
            }}
            flags[i] = selected ? 1 : 0;
            work[i] = selected ? 1u : 0u;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()


def _transient_factory(be, bundles, config, *, active_only=False):
    """Conservative local MFD transport with no depression conditioning."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    nk = 8 if config["topology"] == "D8" else 4
    weight_type = "unsigned char" if config["quantized_weight"] else "float"
    adaptive_response = """
                    if (depth > 0.0f && slope > 0.0f && qout > 0.0f) {
                        // Response to both depth and hydraulic-gradient change.
                        float drop = fmaxf(slope * width, 1.0e-12f);
                        float dqdh = qout * (alpha / depth + 0.5f / drop);
                        float dx = $ctx.grid.DX.get(0)$;
                        float area = dx * dx;
                        float cfl = fminf(1.0f, fmaxf(
                            $ctx.TRANSIENT_CFL.get(0)$, 1.0e-6f));
                        candidate = fminf(cap, cfl * area / dqdh);
                    }
    """ if config["adaptive_transient_dt"] else ""
    available_fraction = (
        "1.0f" if config["adaptive_transient_dt"] else
        "fminf(1.0f, fmaxf($ctx.TRANSIENT_CFL.get(0)$, 0.0f))"
    )
    max_decl = "double max_score = 0.0;" if config["quantized_weight"] else ""
    max_update = (
        "if (score > max_score) max_score = score;"
        if config["quantized_weight"] else ""
    )
    weight_write = (
        "weights[i * nk + k] = scores[k] > 0.0 "
        "? (unsigned char)max(1, __double2int_rn("
        "255.0 * scores[k] / max_score)) : 0;"
        if config["quantized_weight"] else
        "weights[i * nk + k] = sum_score > 0.0 "
        "? (float)(scores[k] / sum_score) : 0.0f;"
    )

    work_arg = "const int* work_ids, " if active_only else ""
    work_index = (
        "int p = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "            if (p >= $ctx.WORK_COUNT.get(0)$) return;\n"
        "            int i = work_ids[p];"
        if active_only else
        f"int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        f"            if (i >= {n}) return;"
    )
    outflow_index = (
        "int p = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "            bool valid = p < $ctx.WORK_COUNT.get(0)$;\n"
        "            int i = valid ? work_ids[p] : 0;"
        if active_only else
        f"int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        f"            bool valid = i < {n};"
    )
    work_domain = "work_ids" if active_only else n

    topology = KernelBuilder(
        f'''extern "C" __global__ void graphflood_transient_topology(
                {work_arg}const double* hydraulic_surface, unsigned char* dirs,
                {weight_type}* weights, float* steepest_slope,
                float* flow_width) {{
            {work_index}
            int nk = {nk};
            unsigned char mask = 0u;
            double scores[8];
            double sum_score = 0.0;
            {max_decl}
            double best_slope = 0.0;
            float best_width = $ctx.grid.DX.get(0)$;
            for (int k = 0; k < 8; ++k) scores[k] = 0.0;

            if (!$ctx.grid.nodata(i)$ && !$ctx.grid.can_out(i)$) {{
                for (int k = 0; k < {nk}; ++k) {{
                    int r = $ctx.grid.neighbour(i, k)$;
                    if (r == -1) continue;
                    double width = (double)$ctx.grid.dist_from_k(k)$;
                    double slope = (hydraulic_surface[i]
                                  - hydraulic_surface[r]) / width;
                    if (slope <= 0.0) continue;
                    double score = slope * width;
                    scores[k] = score;
                    sum_score += score;
                    mask |= (unsigned char)(1u << k);
                    {max_update}
                    if (slope > best_slope) {{
                        best_slope = slope;
                        best_width = (float)width;
                    }}
                }}
            }}
            for (int k = 0; k < {nk}; ++k) {{ {weight_write} }}
            dirs[i] = mask;
            steepest_slope[i] = (float)best_slope;
            flow_width[i] = best_width;
        }}''', domain=work_domain,
    ).compose("grid", bundles["grid"]).freeze()

    init_dt = KernelBuilder(
        '''extern "C" __global__ void graphflood_transient_init_dt(
                float* effective_dt) {
            if (blockIdx.x == 0 && threadIdx.x == 0)
                effective_dt[0] = fmaxf(
                    $ctx.TRANSIENT_DT.get(0)$, 1.0e-12f);
        }''', domain=1,
    ).freeze()

    outflow = KernelBuilder(
        f'''extern "C" __global__ void graphflood_transient_outflow(
                {work_arg}const float* h, const unsigned char* dirs,
                const float* steepest_slope, const float* flow_width,
                float* Qo, float* effective_dt) {{
            {outflow_index}
            float cap = fmaxf($ctx.TRANSIENT_DT.get(0)$, 1.0e-12f);
            float candidate = cap;
            if (valid) {{
                Qo[i] = 0.0f;
                if (!$ctx.grid.nodata(i)$ && !$ctx.grid.can_out(i)$
                        && dirs[i] != 0u) {{
                    float depth = fmaxf(h[i], 0.0f);
                    float slope = fmaxf(steepest_slope[i], 0.0f);
                    float width = fmaxf(flow_width[i], 1.0e-9f);
                    float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
                    float alpha = fmaxf(1.0f + $ctx.EXPO.get(i)$, 1.0e-3f);
                    float qout = width / manning * powf(depth, alpha)
                               * sqrtf(slope);
                    Qo[i] = qout;

{adaptive_response}
                }}
            }}

            __shared__ float block_min[256];
            block_min[threadIdx.x] = candidate;
            __syncthreads();
            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {{
                if (threadIdx.x < offset)
                    block_min[threadIdx.x] = fminf(
                        block_min[threadIdx.x],
                        block_min[threadIdx.x + offset]);
                __syncthreads();
            }}
            if (threadIdx.x == 0)
                atomicMin((unsigned int*)effective_dt,
                          __float_as_uint(block_min[0]));
        }}''', domain=work_domain, block=256,
    ).compose("grid", bundles["grid"]).freeze()

    limit = KernelBuilder(
        f'''extern "C" __global__ void graphflood_transient_limit(
                {work_arg}const float* h, float* Qo,
                const float* effective_dt) {{
            {work_index}
            float dx = $ctx.grid.DX.get(0)$;
            float area = dx * dx;
            float dt = fmaxf(effective_dt[0], 1.0e-12f);
            float available = {available_fraction}
                            * area * fmaxf(h[i], 0.0f) / dt
                            + fmaxf($ctx.PRECIPITATION.get(i)$, 0.0f) * area;
            Qo[i] = fminf(Qo[i], available);
        }}''', domain=work_domain,
    ).compose("grid", bundles["grid"]).freeze()

    update_args = "const int* active_ids, " if active_only else ""
    update_index = (
        "int p = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "            if (p >= $ctx.ACTIVE_COUNT.get(0)$) return;\n"
        "            int i = active_ids[p];"
        if active_only else
        f"int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        f"            if (i >= {n}) return;"
    )
    update = KernelBuilder(
        f'''extern "C" __global__ void graphflood_transient_update(
                {update_args}float* h, float* Qi, float* Qo,
                const unsigned char* dirs, const {weight_type}* weights,
                const float* effective_dt) {{
            {update_index}
            if ($ctx.grid.nodata(i)$) {{
                h[i] = 0.0f;
                Qi[i] = 0.0f;
                Qo[i] = 0.0f;
                return;
            }}
            float dx = $ctx.grid.DX.get(0)$;
            float area = dx * dx;
            float qin = fmaxf($ctx.PRECIPITATION.get(i)$, 0.0f) * area;
            for (int k = 0; k < {nk}; ++k) {{
                int donor = $ctx.grid.neighbour(i, k)$;
                if (donor == -1) continue;
                int reverse = {nk - 1} - k;
                unsigned char donor_mask = dirs[donor];
                if (!(donor_mask & (1u << reverse))) continue;
                float selected = (float)weights[donor * {nk} + reverse];
                float total = 0.0f;
                #pragma unroll
                for (int q = 0; q < {nk}; ++q)
                    total += (float)weights[donor * {nk} + q];
                if (total > 0.0f) qin += Qo[donor] * selected / total;
            }}
            Qi[i] = qin;
            if ($ctx.grid.can_out(i)$) {{
                h[i] = 0.0f;
                Qo[i] = qin;
                return;
            }}
            float dt = fmaxf(effective_dt[0], 1.0e-12f);
            h[i] = fmaxf(h[i] + (qin - Qo[i]) * dt / area, 0.0f);
        }}''', domain="active_ids" if active_only else n,
    ).compose("grid", bundles["grid"]).freeze()

    return (RoutineBuilder().step("topology", topology)
            .step("init_dt", init_dt).step("outflow", outflow)
            .step("limit", limit).step("update", update).freeze())


def _active_transient_factory(be, bundles, config):
    return _transient_factory(be, bundles, config, active_only=True)


def _transient_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "hydraulic_surface": "hydraulic_surface", "h": "h",
        "Qi": "Qi", "Qo": "Qo", "dirs": "directions",
        "weights": "weights", "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "MANNING": "friction_coefficient",
        "EXPO": "friction_exponent", "PRECIPITATION": "precipitation",
        "TRANSIENT_DT": "transient_dt", "TRANSIENT_CFL": "transient_cfl",
        "effective_dt": "transient_dt_used",
        "active_ids": "active_ids", "ACTIVE_COUNT": "active_count",
        "work_ids": "work_ids", "WORK_COUNT": "work_count",
    })


def _update_factory(be, bundles, config, *, active_only=False):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    friction = build_friction_velocity(config["friction_law"])
    args = "const int* active_ids, " if active_only else ""
    index = (
        "int p = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "            if (p >= $ctx.ACTIVE_COUNT.get(0)$) return;\n"
        "            int i = active_ids[p];"
        if active_only else
        f"int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        f"            if (i >= {n}) return;"
    )
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_update_depth(
                {args}float* h, const float* Qi, float* Qo,
                const float* steepest_slope, const float* flow_width) {{
            {index}
            if ($ctx.grid.nodata(i)$) {{
                h[i] = 0.0f;
                Qo[i] = 0.0f;
                return;
            }}
            if ($ctx.grid.can_out(i)$) {{
                Qo[i] = Qi[i];
                h[i] = 0.0f;
                return;
            }}
            float qout = $ctx.friction(h[i], steepest_slope[i], i)$
                * h[i] * flow_width[i];
            Qo[i] = qout;
            float dx = $ctx.grid.DX.get(0)$;
            float next = h[i] + (Qi[i] - qout) / (dx * dx) * $ctx.DT.get(i)$;
            h[i] = next > 0.0f ? next : 0.0f;
        }}''', domain="active_ids" if active_only else n,
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()


def _active_update_factory(be, bundles, config):
    return _update_factory(be, bundles, config, active_only=True)


def _local_analytical_update_factory(be, bundles, config, *, active_only=False):
    """Original pointwise inverse of the frozen-slope discharge closure."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    friction = build_friction_velocity(config["friction_law"])
    args = "const int* active_ids, " if active_only else ""
    index = (
        "int p = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "            if (p >= $ctx.ACTIVE_COUNT.get(0)$) return;\n"
        "            int i = active_ids[p];"
        if active_only else
        f"int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        f"            if (i >= {n}) return;"
    )
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_local_analytical(
                {args}float* h, const float* Qi, float* Qo,
                const float* steepest_slope, const float* flow_width) {{
            {index}
            if ($ctx.grid.nodata(i)$) {{
                h[i] = 0.0f;
                Qo[i] = 0.0f;
                return;
            }}
            if ($ctx.grid.can_out(i)$) {{
                h[i] = 0.0f;
                Qo[i] = Qi[i];
                return;
            }}

            float q = fmaxf(Qi[i], 0.0f);
            float slope = fmaxf(steepest_slope[i], 1.0e-5f);
            float width = fmaxf(flow_width[i], 1.0e-9f);
            float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
            float alpha = fmaxf(1.0f + $ctx.EXPO.get(i)$, 1.0e-3f);
            float target = q > 0.0f
                ? powf(q * manning / (width * sqrtf(slope)), 1.0f / alpha)
                : 0.0f;
            float relaxation = fminf(1.0f, fmaxf(0.0f,
                $ctx.ANALYTICAL_RELAXATION.get(i)$));
            float next_h = fmaxf(
                h[i] + relaxation * (target - fmaxf(h[i], 0.0f)), 0.0f);
            h[i] = next_h;
            Qo[i] = $ctx.friction(next_h, slope, i)$ * next_h * width;
        }}''', domain="active_ids" if active_only else n,
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()


def _active_local_analytical_update_factory(be, bundles, config):
    return _local_analytical_update_factory(
        be, bundles, config, active_only=True,
    )


def _local_analytical_update_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "h": "h", "Qi": "Qi", "Qo": "Qo",
        "steepest_slope": "steepest_slope", "flow_width": "flow_width",
        "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
        "ANALYTICAL_RELAXATION": "analytical_relaxation",
        "active_ids": "active_ids", "ACTIVE_COUNT": "active_count",
    })


def _bottom_up_analytical_update_factory(be, bundles, config):
    """Receiver-first persistent Kahn solve of the frozen hydraulic DAG."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    nk = 8 if config["topology"] == "D8" else 4
    persistent_grid, persistent_block = persistent_grid_block(
        blocks_per_sm=1, threads=256,
    )
    resident_threads = persistent_grid[0] * persistent_block[0]
    carve = config["mfd_local_minima"] == "carve_cordonnier"
    cut_assignment = (
        f"hydraulic_cut[i] = (control[i] < {nk} && best <= 0.0f) ? 1u : 0u;"
        if carve else "hydraulic_cut[i] = 0u;"
    )

    clear = KernelBuilder(
        '''extern "C" __global__ void graphflood_reverse_clear(
                int* count, unsigned int* barrier) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < 2) count[i] = 0;
            if (i == 0) barrier[0] = 0u;
        }''', domain=2,
    ).freeze()

    # Freeze one controlling receiver for the nonlinear sweep. Its SFD subset
    # supplies the only dependencies needed by the scalar Manning closure;
    # the full MFD graph is rebuilt by the outer pipeline before the next
    # sweep.
    prepare = KernelBuilder(
        f'''extern "C" __global__ void graphflood_reverse_prepare(
                const float* z, const float* h,
                const unsigned char* directions, int* remaining,
                unsigned char* control, unsigned char* hydraulic_cut,
                int* frontier, int* count) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            unsigned int mask = (unsigned int)directions[i];
            control[i] = 255u;
            float best = -3.402823466e+38f;
            if (!$ctx.grid.nodata(i)$ && mask) {{
                #pragma unroll
                for (int k = 0; k < {nk}; ++k) {{
                    if (!(mask & (1u << k))) continue;
                    int r = $ctx.grid.neighbour_raw(i, k)$;
                    float slope = ((z[i] - z[r]) + (h[i] - h[r]))
                                / $ctx.grid.dist_from_k(k)$;
                    if (slope > best) {{
                        best = slope;
                        control[i] = (unsigned char)k;
                    }}
                }}
            }}
            {cut_assignment}
            // The nonlinear candidate only reads the controlling receiver.
            // Waiting for every MFD receiver adds dependencies without
            // changing the result and makes the reverse walk much deeper.
            remaining[i] = control[i] < {nk} ? 1 : 0;
            if (!$ctx.grid.nodata(i)$ && remaining[i] == 0) {{
                int p = atomicAdd(&count[0], 1);
                frontier[p] = i;
            }}
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    solve = KernelBuilder(
        f'''extern "C" __global__ void graphflood_reverse_hydraulics(
                int* __restrict__ frontier0, int* __restrict__ frontier1,
                int* __restrict__ count, unsigned int* __restrict__ barrier,
                const unsigned char* __restrict__ control,
                const unsigned char* __restrict__ hydraulic_cut,
                int* __restrict__ remaining, const float* __restrict__ z,
                const float* __restrict__ h, const float* __restrict__ Qi,
                float* __restrict__ candidate,
                float* __restrict__ solved_slope) {{
            __shared__ int staged[2048];
            __shared__ int staged_n;
            __shared__ unsigned int staged_base;

            int* frontiers[2] = {{frontier0, frontier1}};
            int parity = 0;
            unsigned int level = 0;
            while (true) {{
                int size_in = *((volatile int*)&count[parity]);
                if (size_in == 0) break;
                int* fin = frontiers[parity];
                int* fout = frontiers[1 - parity];
                if (threadIdx.x == 0) staged_n = 0;
                __syncthreads();

                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int stride = gridDim.x * blockDim.x;
                for (int p = tid; p < size_in; p += stride) {{
                    int u = fin[p];
                    if ($ctx.grid.can_out(u)$) {{
                        candidate[u] = 0.0f;
                        solved_slope[u] = fmaxf(
                            $ctx.CARVE_SLOPE.get(u)$, 1.0e-12f);
                    }} else {{
                        float target_q = fmaxf(Qi[u], 0.0f);
                        float old_h = fmaxf(h[u], 0.0f);
                        float manning = fmaxf(
                            $ctx.MANNING.get(u)$, 1.0e-9f);
                        float alpha = fmaxf(
                            1.0f + $ctx.EXPO.get(u)$, 1.0e-3f);
                        int k = (int)control[u];
                        if (k >= {nk}) {{
                            // A non-outlet graph sink has no physical closure.
                            // It should not occur after local-minimum handling.
                            candidate[u] = old_h;
                            solved_slope[u] = fmaxf(
                                $ctx.CARVE_SLOPE.get(u)$, 1.0e-12f);
                        }} else {{
                            int r = $ctx.grid.neighbour_raw(u, k)$;
                            float width = $ctx.grid.dist_from_k(k)$;
                            bool cut = hydraulic_cut[u] != 0u;
                            float inherited_slope = fmaxf(
                                solved_slope[r],
                                fmaxf($ctx.CARVE_SLOPE.get(u)$, 1.0e-12f));
                            float receiver_h = cut ? 0.0f : candidate[r];
                            float bed_drop = z[u] - z[r];

                            // The lower bracket enforces eta[u] >= eta[r].
                            // Unlike the explicit update, this solve must not
                            // hide an uphill link behind the slope floor.
                            float lo = cut ? 0.0f
                                : fmaxf(receiver_h - bed_drop, 0.0f);
                            float old_slope = cut ? inherited_slope : fmaxf(
                                (bed_drop + old_h - receiver_h) / width,
                                1.0e-12f);
                            float guess = target_q > 0.0f
                                ? powf(target_q * manning
                                       / (width * sqrtf(old_slope)),
                                       1.0f / alpha)
                                : lo;
                            float hi = fmaxf(fmaxf(old_h, guess),
                                             lo + 1.0e-7f);
                            #pragma unroll
                            for (int it = 0; it < 32; ++it) {{
                                float slope = cut ? inherited_slope : fmaxf(
                                    (bed_drop + hi - receiver_h) / width,
                                    0.0f);
                                float qhi = width / manning * powf(hi, alpha)
                                          * sqrtf(slope);
                                if (qhi >= target_q) break;
                                hi = hi * 2.0f + 1.0e-6f;
                            }}

                            float x = fminf(fmaxf(guess, lo), hi);
                            #pragma unroll
                            for (int it = 0; it < 16; ++it) {{
                                float slope = cut ? inherited_slope : fmaxf(
                                    (bed_drop + x - receiver_h) / width,
                                    0.0f);
                                float q = width / manning * powf(x, alpha)
                                        * sqrtf(slope);
                                if (q < target_q) lo = x;
                                else hi = x;
                                float derivative = q * alpha
                                    / fmaxf(x, 1.0e-12f);
                                if (!cut && slope > 0.0f)
                                    derivative += q * 0.5f / (slope * width);
                                float trial = x - (q - target_q)
                                    / fmaxf(derivative, 1.0e-20f);
                                if (!(trial > lo && trial < hi)
                                        || !isfinite(trial))
                                    trial = 0.5f * (lo + hi);
                                x = trial;
                            }}
                            float solved_h = target_q > 0.0f
                                ? x : lo;
                            candidate[u] = solved_h;
                            solved_slope[u] = cut ? inherited_slope : fmaxf(
                                (bed_drop + solved_h - receiver_h) / width,
                                0.0f);
                        }}
                    }}

                    __threadfence();
                    #pragma unroll
                    for (int k = 0; k < {nk}; ++k) {{
                        int donor = $ctx.grid.neighbour(u, k)$;
                        if (donor == -1) continue;
                        if ((int)control[donor] != {nk - 1} - k) continue;
                        int old = atomicAdd(&remaining[donor], -1);
                        if (old == 1) {{
                            int sp = atomicAdd(&staged_n, 1);
                            if (sp < 2048) staged[sp] = donor;
                            else {{
                                int pos = atomicAdd(&count[1 - parity], 1);
                                fout[pos] = donor;
                            }}
                        }}
                    }}
                }}

                __syncthreads();
                int flush_n = min(staged_n, 2048);
                if (threadIdx.x == 0)
                    staged_base = atomicAdd(
                        (unsigned int*)&count[1 - parity],
                        (unsigned int)flush_n);
                __syncthreads();
                for (int i = threadIdx.x; i < flush_n; i += blockDim.x)
                    fout[staged_base + i] = staged[i];
                __threadfence();

                __syncthreads();
                if (threadIdx.x == 0) {{
                    if (blockIdx.x == 0) count[parity] = 0;
                    unsigned int target = (level + 1)
                                        * (unsigned int)gridDim.x;
                    atomicAdd(barrier, 1u);
                    unsigned int wait_ns = 32;
                    while (*((volatile unsigned int*)barrier) < target) {{
#if __CUDA_ARCH__ >= 700
                        __nanosleep(wait_ns);
                        if (wait_ns < 1024) wait_ns <<= 1;
#endif
                    }}
                }}
                __syncthreads();
                level++;
                parity = 1 - parity;
            }}
        }}''', domain=resident_threads, block=256,
    ).compose("grid", bundles["grid"]).freeze()

    apply = KernelBuilder(
        f'''extern "C" __global__ void graphflood_apply_bottom_up_candidate(
                const float* candidate, float* h) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$ || $ctx.grid.can_out(i)$) {{
                h[i] = 0.0f;
                return;
            }}
            float relaxation = fminf(1.0f, fmaxf(0.0f,
                $ctx.ANALYTICAL_RELAXATION.get(i)$));
            float old_h = fmaxf(h[i], 0.0f);
            h[i] = fmaxf(
                old_h + relaxation * (candidate[i] - old_h), 0.0f);
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    diagnostics = KernelBuilder(
        f'''extern "C" __global__ void graphflood_bottom_up_diagnostics(
                const float* z, const float* h, const float* Qi,
                const unsigned char* control,
                const unsigned char* hydraulic_cut,
                const float* solved_slope, float* Qo) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$) {{ Qo[i] = 0.0f; return; }}
            if ($ctx.grid.can_out(i)$) {{ Qo[i] = Qi[i]; return; }}
            int k = (int)control[i];
            if (k >= {nk}) {{ Qo[i] = 0.0f; return; }}
            int r = $ctx.grid.neighbour_raw(i, k)$;
            float width = $ctx.grid.dist_from_k(k)$;
            float slope = hydraulic_cut[i] ? solved_slope[i] : fmaxf(
                ((z[i] - z[r]) + (h[i] - h[r])) / width, 0.0f);
            float depth = fmaxf(h[i], 0.0f);
            float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
            float alpha = fmaxf(1.0f + $ctx.EXPO.get(i)$, 1.0e-3f);
            Qo[i] = width / manning * powf(depth, alpha) * sqrtf(slope);
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    return (RoutineBuilder().step("clear", clear).step("prepare", prepare)
            .step("solve", solve).step("apply", apply)
            .step("diagnostics", diagnostics).freeze())


def _bottom_up_analytical_update_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "z": "z", "h": "h", "Qi": "Qi", "Qo": "Qo",
        "candidate": "z_prime",
        "directions": "directions", "remaining": "indegree",
        "control": "is_border", "hydraulic_cut": "hydraulic_cut",
        "solved_slope": "steepest_slope", "frontier": "mfd_frontier0",
        "frontier0": "mfd_frontier0", "frontier1": "mfd_frontier1",
        "count": "mfd_count", "barrier": "mfd_barrier",
        "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
        "CARVE_SLOPE": "carve_slope_min",
        "ANALYTICAL_RELAXATION": "analytical_relaxation",
    })


def _reset_h_factory(be, _bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_reset_h(float* h) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) h[i] = 0.0f;
        }}''', domain=n,
    ).freeze()


def _copy_fill_depth_factory(be, _bundles, config):
    """Turn the reconstructed surface into initial physical water depth."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_copy_fill_depth(
                const float* z, const float* filled, float* h) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float depth = filled[i] - z[i];
            h[i] = depth > 0.0f ? depth : 0.0f;
        }}''', domain=n,
    ).freeze()


def _merge_fill_depth_factory(be, _bundles, config):
    """Add reconstructed storage without losing sub-ULP existing depth."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_merge_fill_depth(
                const float* z, const float* filled, float* h,
                unsigned char* conditioned) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float depth = filled[i] - z[i];
            conditioned[i] = depth > h[i] ? 1u : 0u;
            if (depth > h[i]) h[i] = depth;
        }}''', domain=n,
    ).freeze()


def _merge_active_fill_depth_factory(be, _bundles, config):
    """Apply reconstructed storage only where the current band is active."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_merge_active_fill(
                const float* z, const float* filled,
                const unsigned char* active, float* h,
                unsigned char* conditioned) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float depth = filled[i] - z[i];
            conditioned[i] = depth > h[i] ? 1u : 0u;
            if (active[i] && depth > h[i]) h[i] = depth;
        }}''', domain=n,
    ).freeze()


def _fill_surface_plan(source, destination, distance="fill_epsilon_distance"):
    def plan_for(frozen, _be):
        values = {
            "z": source, "filled": destination, "parent": "fill_parent",
            "frontier": "fill_frontier", "counters": "fill_counters",
            "queued_gen": "fill_queued_gen", "active": "active.handle",
            "P": "pass_index", "ACTIVE": "active",
            "dist": distance, "anc": "fill_epsilon_ancestor",
            "dist_in": distance,
            "dist_out": "fill_epsilon_distance_work",
            "anc_in": "fill_epsilon_ancestor",
            "anc_out": "fill_epsilon_ancestor_work",
            "rec": "rec",
        }
        plan = _grid_leaf_plan(frozen, values)
        plan.update({
            "hops_forward.dist_in": distance,
            "hops_forward.dist_out": "fill_epsilon_distance_work",
            "hops_forward.anc_in": "fill_epsilon_ancestor",
            "hops_forward.anc_out": "fill_epsilon_ancestor_work",
            "hops_backward.dist_in": "fill_epsilon_distance_work",
            "hops_backward.dist_out": distance,
            "hops_backward.anc_in": "fill_epsilon_ancestor_work",
            "hops_backward.anc_out": "fill_epsilon_ancestor",
        })
        return plan

    return plan_for


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
    b.param("band_min", "scalar", "f32", value=0.0)
    b.param("band_max", "scalar", "f32", value=1.0)
    b.param("active_count", "scalar", "i32", value=0)
    b.param("work_count", "scalar", "i32", value=0)
    b.param("ndep", "scalar", "i32", value=0)
    b.param("pass_index", "scalar", "i32", value=0)
    b.param("active", "scalar", "i32", value=0)

    b.data("z", "f32", shape, role="input", shape_source=True)
    b.data("h", "f32", shape, role="state")
    b.data("Qi", "f32", shape, role="output")
    b.data("Qo", "f32", shape, role="output")
    b.data("surface", "f32", shape, role="internal")
    b.data("hydraulic_surface", "f64", shape, role="internal")
    b.data("rec", "i32", shape, role="output")
    b.data("distance_position", "f32", shape, role="output")

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
        "active_flags", "active_indegree", "work_ids", "work_flags",
        "distance_remaining", "distance_frontier0", "distance_frontier1",
    ):
        b.data(name, "i32", (flat,), role="internal")
    b.data("active_ids", "i32", (flat,), role="output")
    for name in ("z_prime", "steepest_slope", "flow_width"):
        b.data(name, "f32", (flat,), role="internal")
    for name in ("Qi_boundary", "distance_upstream", "distance_downstream"):
        b.data(name, "f32", shape, role="output")
    b.data(
        "weights",
        lambda config: "u8" if config["quantized_weight"] else "f32",
        (8 * flat,), role="internal",
    )
    for name in (
        "is_border", "directions", "hydraulic_conditioned", "hydraulic_cut",
    ):
        b.data(name, "u8", (flat,), role="internal")
    b.data("active_mask", "u8", shape, role="output")
    b.data("work_mask", "u8", shape, role="output")
    for name in ("basin_saddle", "basin_outlet"):
        b.data(name, "i64", (flat,), role="internal")
    b.data("mfd_count", "i32", (2,), role="internal")
    b.data("mfd_barrier", "u32", (1,), role="internal")
    b.data("distance_count", "i32", (2,), role="internal")
    b.data("distance_barrier", "u32", (1,), role="internal")
    b.data("distance_max", "f32", (1,), role="output")
    b.data("transient_dt_used", "f32", (1,), role="output")

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
    b.add("accumulate_active", _subset_accumulation_factory,
          bind=_subset_accumulation_plan)
    b.add("snapshot_boundary_flux", _copy_flux_factory,
          bind={"source": "Qi", "destination": "Qi_boundary"})
    b.add("compute_distance_position", _distance_factory,
          bind=_distance_plan)
    b.add("mark_active_band", _active_band_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "position": "distance_position", "flags": "active_flags",
              "active": "active_mask", "BAND_MIN": "band_min",
              "BAND_MAX": "band_max",
          }))
    b.add("mark_active_work", _active_work_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "active": "active_mask", "flags": "work_flags",
              "work": "work_mask",
          }))
    b.add("update_depth", _update_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "h": "h", "Qi": "Qi", "Qo": "Qo",
              "steepest_slope": "steepest_slope", "flow_width": "flow_width",
              "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
              "DT": "dt",
          }))
    b.add("update_active_depth", _active_update_factory,
          bind=lambda f, be: _grid_leaf_plan(f, {
              "active_ids": "active_ids", "ACTIVE_COUNT": "active_count",
              "h": "h", "Qi": "Qi", "Qo": "Qo",
              "steepest_slope": "steepest_slope", "flow_width": "flow_width",
              "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
              "DT": "dt",
          }))
    b.add("transport_transient", _transient_factory, bind=_transient_plan)
    b.add("transport_active_transient", _active_transient_factory,
          bind=_transient_plan)
    b.add("update_depth_analytical_local", _local_analytical_update_factory,
          bind=_local_analytical_update_plan)
    b.add(
        "update_depth_analytical_bottom_up",
        _bottom_up_analytical_update_factory,
        bind=_bottom_up_analytical_update_plan,
    )
    b.add("update_active_depth_analytical",
          _active_local_analytical_update_factory,
          bind=_local_analytical_update_plan)
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
    b.pipeline("run_n_step_analytical", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "prepare_frontier", "accumulate", "update_depth_analytical",
    ))
    b.pipeline("_prepare_distance_sweep", (
        "make_surface", "route", "snapshot_receivers",
        "resolve_cordonnier", "compute_cordonnier_fill",
        "refresh_hydraulic_surface", "build_fill_topology",
        "prepare_frontier", "accumulate", "snapshot_boundary_flux",
        "compute_distance_position",
    ))
    b.pipeline("refresh_band_boundary", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_active_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "prepare_frontier", "accumulate", "snapshot_boundary_flux",
    ))
    b.pipeline("run_active_n_step", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_active_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "accumulate_active", "update_active_depth",
    ))
    b.pipeline("run_active_n_step_analytical", (
        "make_surface", "route_local_minima", "snapshot_local_minima",
        "resolve_minima", "prepare_active_mfd_surface",
        "refresh_hydraulic_surface", "build_topology",
        "accumulate_active", "update_active_depth_analytical",
    ))
    b.pipeline("run_active_n_step_transient", (
        "make_surface", "transport_active_transient",
    ))
    program = b.freeze()

    def prepare_distance_sweep(self):
        """Fill once, freeze global Qi, and build the fixed drainage coordinate."""
        self._prepare_distance_sweep()
        return self.set_active_band(0.0, 1.0)

    def set_active_band(self, lower, upper):
        """Compact nodes whose fixed outlet-to-source coordinate is in a band."""
        lower, upper = float(lower), float(upper)
        if not (0.0 <= lower <= upper <= 1.0):
            raise ProgramError("active band must satisfy 0 <= lower <= upper <= 1")
        self.band_min.set(lower)
        self.band_max.set(upper)
        self.mark_active_band()
        scan = getattr(self, "_active_scan", None)
        if scan is None:
            scan = make_scan(self._be, self._pool, self.nx * self.ny)
            object.__setattr__(self, "_active_scan", scan)
        count = scan.compact(
            self._handle("active_flags"), self._handle("active_ids"),
        )
        self.active_count.set(count)

        self.mark_active_work()
        work_scan = getattr(self, "_work_scan", None)
        if work_scan is None:
            work_scan = make_scan(self._be, self._pool, self.nx * self.ny)
            object.__setattr__(self, "_work_scan", work_scan)
        work_count = work_scan.compact(
            self._handle("work_flags"), self._handle("work_ids"),
        )
        self.work_count.set(work_count)

        for name in (
            "accumulate_active", "update_active_depth",
            "update_active_depth_analytical", "transport_active_transient",
        ):
            self._ensure_compiled(name)

        active_view = self._be.wrap(
            self._handle("active_ids").array[:count], owned=False,
        )
        work_view = self._be.wrap(
            self._handle("work_ids").array[:work_count], owned=False,
        )
        self._states["accumulate_active"].compiled.swap(
            "clear.active_ids", active_view,
        ).swap("prepare.active_ids", active_view)
        self._states["update_active_depth"].compiled.swap(
            "active_ids", active_view,
        )
        self._states["update_active_depth_analytical"].compiled.swap(
            "active_ids", active_view,
        )
        transient = self._states["transport_active_transient"].compiled
        transient.swap("topology.work_ids", work_view)
        transient.swap("outflow.work_ids", work_view)
        transient.swap("limit.work_ids", work_view)
        transient.swap("update.active_ids", active_view)

        old_views = getattr(self, "_compact_domain_views", ())
        object.__setattr__(self, "_compact_domain_views", (active_view, work_view))
        for view in old_views:
            view.destroy()
        return count

    base_close = program.close

    def close(self):
        scan = getattr(self, "_active_scan", None)
        if scan is not None:
            scan.close()
            object.__setattr__(self, "_active_scan", None)
        work_scan = getattr(self, "_work_scan", None)
        if work_scan is not None:
            work_scan.close()
            object.__setattr__(self, "_work_scan", None)
        base_close(self)
        for view in getattr(self, "_compact_domain_views", ()):
            view.destroy()
        object.__setattr__(self, "_compact_domain_views", ())

    program.prepare_distance_sweep = prepare_distance_sweep
    program.set_active_band = set_active_band
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
