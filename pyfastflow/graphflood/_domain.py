"""Domain operations for GraphFlood."""

from pyfastflow.core import KernelBuilder, RoutineBuilder, SequenceBuilder
from pyfastflow.flow._program_cupy import BLOCK, _cupy_only, _grid_leaf_plan

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

def _river_distance_init_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_init_river_distance(
                const float* drainage_area, const float* Qi,
                float* effective_area, float* distance, float* distance_work) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float area = drainage_area[i];
            float reference = $ctx.PRECIPITATION_REFERENCE.get(0)$;
            if (reference > 0.0f)
                area = fmaxf(area, fmaxf(Qi[i], 0.0f) / reference);
            effective_area[i] = area;
            float d = (!$ctx.grid.nodata(i)$ &&
                       area >= $ctx.AREA_THRESHOLD.get(0)$)
                    ? 0.0f : 3.402823466e+38F;
            distance[i] = d;
            distance_work[i] = d;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

def _river_distance_relax_factory(be, bundles, config):
    """Two Jacobi sweeps of multi-source chamfer distance propagation."""
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    nk = 8 if config["topology"] == "D8" else 4

    def sweep(name):
        return KernelBuilder(
            f'''extern "C" __global__ void {name}(
                    const float* source, float* destination) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= {n}) return;
                if ($ctx.grid.nodata(i)$) {{
                    destination[i] = 3.402823466e+38F;
                    return;
                }}
                float best = source[i];
                for (int k = 0; k < {nk}; ++k) {{
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j == -1) continue;
                    float candidate = source[j] + $ctx.grid.dist_from_k(k)$;
                    if (candidate < best) best = candidate;
                }}
                destination[i] = best;
            }}''', domain=n,
        ).compose("grid", bundles["grid"]).freeze()

    return (RoutineBuilder()
            .step("forward", sweep("graphflood_river_distance_forward"))
            .step("backward", sweep("graphflood_river_distance_backward"))
            .freeze())

def _river_seed_factory(be, bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_mark_river_seed(
                const float* distance, int* river_flags,
                unsigned char* river_mask) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float split = fmaxf($ctx.RIVER_PADDING.get(0)$, 0.0f);
            bool valid = !$ctx.grid.nodata(i)$;
            bool river = valid && distance[i] <= split;
            river_flags[i] = river ? 1 : 0;
            river_mask[i] = river ? 1u : 0u;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

def _dynamic_domain_init_factory(be, _bundles, config):
    _cupy_only(be)
    n = config["nx"] * config["ny"]
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_init_dynamic_domain(
                const unsigned char* seed, int* flags, int* claims,
                unsigned char* active) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            int selected = seed[i] != 0u ? 1 : 0;
            flags[i] = selected;
            claims[i] = selected;
            active[i] = selected ? 1u : 0u;
        }}''', domain=n,
    ).freeze()

def _dynamic_domain_grow_factory(be, bundles, config, *, transient=False):
    _cupy_only(be)
    nk = 8 if config["topology"] == "D8" else 4
    signal = "activity" if transient else "residual"
    target = "h[i]" if transient else (
        "h[i] + (1.0f - fminf(1.0f, fmaxf("
        "$ctx.ANALYTICAL_RELAXATION.get(i)$, 0.0f))) * residual[i]"
    )
    threshold = "TRANSIENT_ACTIVITY" if transient else "GROWTH_RESIDUAL"
    return KernelBuilder(
        f'''extern "C" __global__ void graphflood_grow_dynamic_domain(
                int* active_ids, const float* h,
                const float* {signal}, int* claims,
                unsigned char* active, int* count) {{
            int p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= $ctx.ACTIVE_COUNT.get(0)$) return;
            int i = active_ids[p];
            float target = {target};
            if (fabsf({signal}[i]) <= $ctx.{threshold}.get(0)$ ||
                    fmaxf(h[i], target) <= $ctx.DEPTH_CAP.get(0)$)
                return;
            for (int k = 0; k < {nk}; ++k) {{
                int j = $ctx.grid.neighbour(i, k)$;
                if (j == -1 || $ctx.grid.nodata(j)$) continue;
                if (atomicCAS(&claims[j], 0, 1) == 0) {{
                    active[j] = 1u;
                    int out = atomicAdd(&count[0], 1);
                    active_ids[out] = j;
                }}
            }}
        }}''', domain="active_ids",
    ).compose("grid", bundles["grid"]).freeze()

def _dynamic_transient_grow_factory(be, bundles, config):
    return _dynamic_domain_grow_factory(be, bundles, config, transient=True)

def _dynamic_activity_clear_factory(be, _bundles, _config):
    _cupy_only(be)
    return KernelBuilder(
        '''extern "C" __global__ void graphflood_clear_dynamic_activity(
                const int* active_ids, float* activity) {
            int p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p < $ctx.ACTIVE_COUNT.get(0)$)
                activity[active_ids[p]] = 0.0f;
        }''', domain="active_ids",
    ).freeze()

def _river_distance_relax_plan(frozen, _be):
    plan = _grid_leaf_plan(frozen, {})
    plan.update({
        "forward.source": "river_distance",
        "forward.destination": "river_distance_work",
        "backward.source": "river_distance_work",
        "backward.destination": "river_distance",
    })
    return plan

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
