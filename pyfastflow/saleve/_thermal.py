"""Receiver-first finite-time thermal correction on a fixed Salève network."""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.flow._cupy_mfd_accum import persistent_grid_block

from ._speed import HILLSLOPE_MODELS, speed_source


def build_thermal_program():
    b = ProgramBuilder("SaleveThermalProgram")
    b.dim("ny").dim("nx").dim("levels")
    b.config("ny").config("nx").config("levels")
    b.config("m", default=0.4)
    b.config("hack_constant", default=1.5).config("hack_exponent", default=0.6)
    b.config("hillslope_model", choices=HILLSLOPE_MODELS, default="hack")
    b.config("channel_area", default=0.0)
    b.param("active_nx", "scalar", "i32", value=lambda d: d["nx"])
    b.param("active_n", "scalar", "i32", value=lambda d: d["nx"] * d["ny"])
    b.param("active_dx", "scalar", "f32", value=1.0)
    b.param("max_depth", "scalar", "i32", value=0)
    b.param("time", "scalar", "f32", value=0.0)
    b.param("critical_slope", "auto", "f32", value=0.57)
    shape = (Dim("ny"), Dim("nx"))
    b.param("uplift", "auto", "f32", value=1.0, shape=shape)
    b.param("erodibility", "auto", "f32", value=1.0, shape=shape)
    b.param("thermal_erosion", "auto", "f32", value=0.0, shape=shape)
    b.param("hillslope_erosion", "auto", "f32", value=0.0, shape=shape)
    for name in ("z0", "drainage", "slope_correction", "divide_distance"):
        b.data(name, "f32", shape, role="input")
    b.data("rec", "i32", shape, role="input")
    b.data("ancestors", "i32", (Dim("levels"), *shape), role="input")
    for name in ("tau", "phi", "z"):
        b.data(name, "f32", shape, role="output")
    for name in ("order", "starts", "ends"):
        b.data(name, "i32", shape, role="internal")
    b.data("barrier", "u32", (1,), role="internal")

    def solve(_be, _bundles, config):
        cap = config["nx"] * config["ny"]
        levels = config["levels"]
        grid, block = persistent_grid_block(blocks_per_sm=1, threads=256)
        helper = f'''
__device__ float saleve_thermal_elevation(
    int i, float t, const int* rec, const int* ancestors,
    const float* tau, const float* phi, const float* z0) {{
    float target = tau[i] - t;
    if (target <= 0.0f) {{
        int root = ancestors[({levels - 1}) * {cap} + i];
        return z0[root] + phi[i];
    }}
    int high = i;
    for (int k = {levels - 1}; k >= 0; --k) {{
        int candidate = ancestors[k * {cap} + high];
        if (tau[candidate] >= target) high = candidate;
    }}
    int low = rec[high];
    float alpha = (target - tau[low]) / fmaxf(tau[high] - tau[low], 1.0e-20f);
    alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
    float initial = z0[low] + alpha * (z0[high] - z0[low]);
    float source_phi = phi[low] + alpha * (phi[high] - phi[low]);
    return initial + phi[i] - source_phi;
}}'''
        source = f'''
__device__ float saleve_thermal_elevation(
    int i, float t, const int* rec, const int* ancestors,
    const float* tau, const float* phi, const float* z0);

extern "C" __global__ void saleve_thermal_solve(
    const int* rec, const int* ancestors, const int* order,
    const int* starts, const int* ends, unsigned int* barrier,
    const float* z0, const float* drainage, const float* correction,
    const float* divide_distance, float* tau, float* phi, float* z) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int nx = $ctx.ACTIVE_NX.get(0)$;
    float dx = $ctx.ACTIVE_DX.get(0)$;
    float time = $ctx.TIME.get(0)$;
    float critical = $ctx.CRITICAL.get(0)$;
    for (int depth = 0; depth <= $ctx.MAX_DEPTH.get(0)$; ++depth) {{
        for (int p = starts[depth] + tid; p < ends[depth]; p += stride) {{
            int i = order[p];
            int r = rec[i];
            if (r == i) {{
                tau[i] = 0.0f;
                phi[i] = 0.0f;
                z[i] = z0[i];
                continue;
            }}
            int dr = i / nx - r / nx;
            int dc = i % nx - r % nx;
            float distance = (dr && dc) ? 1.41421356237f : 1.0f;
            float width = dx * distance;
            float area = fmaxf(drainage[i] * dx * dx, 1.0e-20f);
            float uplift = $ctx.UPLIFT.get(i)$;
            {speed_source(config)}
            float link_length = width / correction[i];
            float travel = link_length / speed;
            tau[i] = tau[r] + travel;
            phi[i] = phi[r] + travel * uplift;
            float value = fmaxf(saleve_thermal_elevation(
                i, time, rec, ancestors, tau, phi, z0), z[r]);
            float kt = $ctx.THERMAL.get(i)$;
            if (kt > 0.0f && (value - z[r]) / link_length > critical) {{
                speed += kt;
                uplift += kt * critical;
                travel = link_length / speed;
                tau[i] = tau[r] + travel;
                phi[i] = phi[r] + travel * uplift;
                value = fmaxf(saleve_thermal_elevation(
                    i, time, rec, ancestors, tau, phi, z0), z[r]);
            }}
            z[i] = value;
        }}
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {{
            unsigned int target = (depth + 1) * (unsigned int)gridDim.x;
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
    }}
}}''' + helper
        return KernelBuilder(source, domain=grid[0] * block[0], block=256).freeze()

    b.add("solve", solve, bind={
        "rec": "rec", "ancestors": "ancestors", "order": "order",
        "starts": "starts", "ends": "ends", "barrier": "barrier",
        "z0": "z0", "drainage": "drainage", "correction": "slope_correction",
        "divide_distance": "divide_distance",
        "tau": "tau", "phi": "phi", "z": "z",
        "ACTIVE_NX": "active_nx", "ACTIVE_DX": "active_dx",
        "TIME": "time", "CRITICAL": "critical_slope",
        "MAX_DEPTH": "max_depth", "UPLIFT": "uplift",
        "ERODIBILITY": "erodibility", "THERMAL": "thermal_erosion",
        "HILLSLOPE": "hillslope_erosion",
    })
    return b.freeze()


SaleveThermalProgram = build_thermal_program()
