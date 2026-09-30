"""Solvers operations for GraphFlood."""

from pyfastflow.core import KernelBuilder, RoutineBuilder, SequenceBuilder
from pyfastflow.flow._cupy_mfd_accum import persistent_grid_block
from pyfastflow.graphflood._friction import build_friction_velocity
from pyfastflow.flow._program_cupy import BLOCK, _cupy_only, _grid_leaf_plan

def _transient_factory(
        be, bundles, config, *, active_only=False, hybrid=False,
        track_activity=False):
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
            float available = area * fmaxf(h[i], 0.0f) / dt
                            + fmaxf($ctx.PRECIPITATION.get(i)$, 0.0f) * area;
            Qo[i] = fminf(Qo[i], available);
        }}''', domain=work_domain,
    ).compose("grid", bundles["grid"]).freeze()

    mix = None
    if hybrid:
        mix = KernelBuilder(
            f'''extern "C" __global__ void graphflood_hybrid_mix(
                    {work_arg}const float* Qi, const float* Qo,
                    const unsigned char* dirs, float* Qsend) {{
                {work_index}
                if ($ctx.grid.nodata(i)$) {{ Qsend[i] = 0.0f; return; }}
                if ($ctx.grid.can_out(i)$) {{
                    Qsend[i] = fmaxf(Qi[i], 0.0f);
                    return;
                }}
                if (dirs[i] == 0u) {{ Qsend[i] = 0.0f; return; }}
                float theta = fminf(1.0f, fmaxf(
                    $ctx.THETA.get(i)$, 0.0f));
                Qsend[i] = theta * fmaxf(Qo[i], 0.0f)
                         + (1.0f - theta) * fmaxf(Qi[i], 0.0f);
            }}''', domain=work_domain,
        ).compose("grid", bundles["grid"]).freeze()

    update_args = "const int* active_ids, " if active_only else ""
    send_arg = "float* Qsend, " if hybrid else ""
    activity_arg = "float* activity, " if track_activity else ""
    donor_flux = "Qsend[donor]" if hybrid else "Qo[donor]"
    local_flux = "Qsend[i]" if hybrid else "Qo[i]"
    clear_send = "Qsend[i] = 0.0f;" if hybrid else ""
    outlet_send = "Qsend[i] = qin;" if hybrid else ""
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
                {send_arg}{activity_arg}
                const unsigned char* dirs, const {weight_type}* weights,
                const float* effective_dt) {{
            {update_index}
            if ($ctx.grid.nodata(i)$) {{
                h[i] = 0.0f;
                Qi[i] = 0.0f;
                Qo[i] = 0.0f;
                {clear_send}
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
                if (total > 0.0f) qin += {donor_flux} * selected / total;
            }}
            Qi[i] = qin;
            if ($ctx.grid.can_out(i)$) {{
                h[i] = 0.0f;
                Qo[i] = qin;
                {outlet_send}
                return;
            }}
            float dt = fmaxf(effective_dt[0], 1.0e-12f);
            {"float old_h = h[i];" if track_activity else ""}
            h[i] = fmaxf(h[i] + (qin - {local_flux}) * dt / area, 0.0f);
            {"activity[i] += fabsf(h[i] - old_h);" if track_activity else ""}
        }}''', domain="active_ids" if active_only else n,
    ).compose("grid", bundles["grid"]).freeze()

    routine = (RoutineBuilder().step("topology", topology)
               .step("init_dt", init_dt).step("outflow", outflow)
               .step("limit", limit))
    if mix is not None:
        routine.step("mix", mix)
    return routine.step("update", update).freeze()

def _dynamic_transient_factory(be, bundles, config):
    # Dynamic-domain transport always uses transient_dt, regardless of the
    # full-domain adaptive timestep choice.
    return _transient_factory(
        be, bundles, {**config, "adaptive_transient_dt": False},
        active_only=True, track_activity=True,
    )

def _hybrid_factory(be, bundles, config):
    return _transient_factory(be, bundles, config, hybrid=True)

def _transient_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "hydraulic_surface": "hydraulic_surface", "h": "h",
        "Qi": "Qi", "Qo": "Qo", "dirs": "directions",
        "weights": "weights", "steepest_slope": "steepest_slope",
        "flow_width": "flow_width", "MANNING": "friction_coefficient",
        "EXPO": "friction_exponent", "PRECIPITATION": "precipitation",
        "TRANSIENT_DT": "transient_dt", "TRANSIENT_CFL": "transient_cfl",
        "effective_dt": "transient_dt_used",
        "active_ids": "dynamic_active_ids", "ACTIVE_COUNT": "active_count",
        "work_ids": "dynamic_work_ids", "WORK_COUNT": "work_count",
        "Qsend": "Qsend", "THETA": "hybrid_theta",
        "activity": "transient_activity",
    })

def _update_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    friction = build_friction_velocity(config["friction_law"])
    index = (f"int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
             f"            if (i >= {n}) return;")
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_update_depth(
                float* h, const float* Qi, float* Qo,
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
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()

def _local_analytical_update_factory(be, bundles, config):
    """Original pointwise inverse of the frozen-slope discharge closure."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    friction = build_friction_velocity(config["friction_law"])
    index = (f"int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
             f"            if (i >= {n}) return;")
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_local_analytical(
                float* h, const float* Qi, float* Qo,
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
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()

def _capped_local_analytical_update_factory(be, bundles, config):
    """Whole-domain local analytical warm-up with a hard depth ceiling."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    friction = build_friction_velocity(config["friction_law"])
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_capped_analytical(
                float* h, const float* Qi, float* Qo,
                const float* steepest_slope, const float* flow_width) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if ($ctx.grid.nodata(i)$) {{ h[i] = 0.0f; Qo[i] = 0.0f; return; }}
            if ($ctx.grid.can_out(i)$) {{ h[i] = 0.0f; Qo[i] = Qi[i]; return; }}

            float q = fmaxf(Qi[i], 0.0f);
            float slope = fmaxf(steepest_slope[i], 1.0e-5f);
            float width = fmaxf(flow_width[i], 1.0e-9f);
            float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
            float alpha = fmaxf(1.0f + $ctx.EXPO.get(i)$, 1.0e-3f);
            float target = q > 0.0f
                ? powf(q * manning / (width * sqrtf(slope)), 1.0f / alpha)
                : 0.0f;
            float depth_cap = fmaxf($ctx.DEPTH_CAP.get(0)$, 0.0f);
            target = fminf(target, depth_cap);
            float relaxation = fminf(1.0f, fmaxf(0.0f,
                $ctx.ANALYTICAL_RELAXATION.get(i)$));
            float next_h = fminf(fmaxf(
                h[i] + relaxation * (target - fmaxf(h[i], 0.0f)), 0.0f),
                depth_cap);
            h[i] = next_h;
            Qo[i] = $ctx.friction(next_h, slope, i)$ * next_h * width;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()

def _dynamic_local_analytical_update_factory(be, bundles, config):
    """Active local solve recording the undamped depth residual for growth."""
    _cupy_only(be)
    friction = build_friction_velocity(config["friction_law"])
    return KernelBuilder(
        '''extern "C" __global__ void graphflood_dynamic_analytical(
                const int* active_ids, float* h, const float* Qi, float* Qo,
                const float* steepest_slope, const float* flow_width,
                float* residual) {
            int p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= $ctx.ACTIVE_COUNT.get(0)$) return;
            int i = active_ids[p];
            if ($ctx.grid.nodata(i)$) {
                h[i] = 0.0f; Qo[i] = 0.0f; residual[i] = 0.0f; return;
            }
            if ($ctx.grid.can_out(i)$) {
                h[i] = 0.0f; Qo[i] = Qi[i]; residual[i] = 0.0f; return;
            }

            float old_h = fmaxf(h[i], 0.0f);
            float q = fmaxf(Qi[i], 0.0f);
            float slope = fmaxf(steepest_slope[i], 1.0e-5f);
            float width = fmaxf(flow_width[i], 1.0e-9f);
            float manning = fmaxf($ctx.MANNING.get(i)$, 1.0e-9f);
            float alpha = fmaxf(1.0f + $ctx.EXPO.get(i)$, 1.0e-3f);
            float target = q > 0.0f
                ? powf(q * manning / (width * sqrtf(slope)), 1.0f / alpha)
                : 0.0f;
            float change = target - old_h;
            float relaxation = fminf(1.0f, fmaxf(0.0f,
                $ctx.ANALYTICAL_RELAXATION.get(i)$));
            float next_h = fmaxf(old_h + relaxation * change, 0.0f);
            residual[i] = change;
            h[i] = next_h;
            Qo[i] = $ctx.friction(next_h, slope, i)$ * next_h * width;
        }''', domain="active_ids",
    ).compose("grid", bundles["grid"]).compose("friction", friction).freeze()

def _local_analytical_update_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "h": "h", "Qi": "Qi", "Qo": "Qo",
        "steepest_slope": "steepest_slope", "flow_width": "flow_width",
        "MANNING": "friction_coefficient", "EXPO": "friction_exponent",
        "ANALYTICAL_RELAXATION": "analytical_relaxation",
        "active_ids": "dynamic_active_ids", "ACTIVE_COUNT": "active_count",
    })

def _capped_local_analytical_update_plan(frozen, _be):
    plan = _local_analytical_update_plan(frozen, _be)
    plan["DEPTH_CAP"] = "hillslope_depth_cap"
    return plan

def _dynamic_local_analytical_update_plan(frozen, _be):
    plan = _local_analytical_update_plan(frozen, _be)
    plan["residual"] = "analytical_residual"
    return plan

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
            __shared__ int s_size;

            int* frontiers[2] = {{frontier0, frontier1}};
            int parity = 0;
            unsigned int level = 0;
            // Three rotating counters: level L reads count[L % 3], pushes
            // into count[(L + 1) % 3] and clears count[(L + 2) % 3], which
            // no block touches during level L.
            while (true) {{
                int* c_out = &count[(level + 1) % 3];
                if (threadIdx.x == 0)
                    s_size = *((volatile int*)&count[level % 3]);
                __syncthreads();
                int size_in = s_size;
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
                                int pos = atomicAdd(c_out, 1);
                                fout[pos] = donor;
                            }}
                        }}
                    }}
                }}

                __syncthreads();
                int flush_n = min(staged_n, 2048);
                if (threadIdx.x == 0)
                    staged_base = atomicAdd(
                        (unsigned int*)c_out, (unsigned int)flush_n);
                __syncthreads();
                for (int i = threadIdx.x; i < flush_n; i += blockDim.x)
                    fout[staged_base + i] = staged[i];
                __threadfence();

                __syncthreads();
                if (threadIdx.x == 0) {{
                    if (blockIdx.x == 0) {{
                        count[(level + 2) % 3] = 0;
                        __threadfence();
                    }}
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
