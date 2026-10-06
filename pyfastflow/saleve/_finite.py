"""Finite-time stream-power solution on a frozen receiver forest."""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder

from ._speed import speed_source


def build_finite_program():
    b = ProgramBuilder("SaleveFiniteProgram")
    b.dim("ny").dim("nx").dim("levels")
    b.config("ny").config("nx").config("levels")
    b.config("dx", default=1.0).config("m", default=0.4)
    b.config("epsilon", default=1.0e-3)
    b.param("active_nx", "scalar", "i32", value=lambda dims: dims["nx"])
    b.param("active_n", "scalar", "i32", value=lambda dims: dims["nx"] * dims["ny"])
    b.param("active_dx", "scalar", "f32", value=1.0)
    b.param("uplift", "auto", "f32", value=1.0, shape=(Dim("ny"), Dim("nx")))
    b.param("erodibility", "auto", "f32", value=1.0, shape=(Dim("ny"), Dim("nx")))
    b.param("hillslope_erosion", "auto", "f32", value=0.0, shape=(Dim("ny"), Dim("nx")))
    b.config("hack_constant", default=1.5).config("hack_exponent", default=0.6)
    b.param("time", "scalar", "f32", value=0.0)
    b.param("level", "scalar", "i32", value=0)
    shape = (Dim("ny"), Dim("nx"))
    b.data("rec", "i32", shape, role="input")
    b.data("drainage", "f32", shape, role="input")
    b.data("slope_correction", "f32", shape, role="input")
    b.data("z0", "f32", shape, role="input")
    b.data("z", "f32", shape, role="output")
    b.data("conditioned_z0", "f32", shape, role="internal")
    b.data("candidate_z", "f32", shape, role="internal")
    b.data("ancestors", "i32", (Dim("levels"), *shape), role="internal")
    for suffix in ("a", "b"):
        b.data(f"tau_{suffix}", "f32", shape, role="internal")
        b.data(f"phi_{suffix}", "f32", shape, role="internal")
        b.data(f"depth_{suffix}", "i32", shape, role="internal")
        b.data(f"max_{suffix}", "f32", shape, role="internal")

    def initialize(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_finite_init(
    const int* rec, const float* drainage, const float* slope_correction,
    int* ancestors,
    float* tau, float* phi, int* depth) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int r = rec[i];
    ancestors[i] = r;
    depth[i] = (r != i);
    if (r == i) {{ tau[i] = 0.0f; phi[i] = 0.0f; return; }}
    int nx = $ctx.ACTIVE_NX.get(0)$;
    int dr = i / nx - r / nx;
    int dc = i % nx - r % nx;
    float distance = (dr && dc) ? 1.41421356237f : 1.0f;
    float dx = $ctx.ACTIVE_DX.get(0)$;
    float area = fmaxf(drainage[i] * dx * dx, 1.0e-20f);
    float uplift = $ctx.UPLIFT.get(i)$;
    {speed_source(config)}
    float dt = dx * distance / (speed * slope_correction[i]);
    tau[i] = dt;
    phi[i] = dt * uplift;
}}''', domain=n).freeze()

    def jump(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_finite_jump(
    int* ancestors, const float* tau_in, const float* phi_in,
    float* tau_out, float* phi_out,
    const int* depth_in, int* depth_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int k = $ctx.LEVEL.get(0)$;
    int* current = ancestors + k * {n};
    int* next = current + {n};
    int r = current[i];
    next[i] = current[r];
    tau_out[i] = tau_in[i] + tau_in[r];
    phi_out[i] = phi_in[i] + phi_in[r];
    depth_out[i] = depth_in[i] + depth_in[r];
}}''', domain=n).freeze()

    def max_seed(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        epsilon = float(config["epsilon"])
        return KernelBuilder(f'''
extern "C" __global__ void saleve_max_seed(
    const float* value, const int* depth, float* maxima) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < $ctx.ACTIVE_N.get(0)$) maxima[i] = value[i] - {epsilon:.9e}f * depth[i];
}}''', domain=n).freeze()

    def max_jump(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_max_jump(
    const int* ancestors, const float* maxima_in, float* maxima_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int r = ancestors[$ctx.LEVEL.get(0)$ * {n} + i];
    maxima_out[i] = fmaxf(maxima_in[i], maxima_in[r]);
}}''', domain=n).freeze()

    def max_finish(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        epsilon = float(config["epsilon"])
        return KernelBuilder(f'''
extern "C" __global__ void saleve_max_finish(
    const float* maxima, const int* depth, float* value) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < $ctx.ACTIVE_N.get(0)$) value[i] = maxima[i] + {epsilon:.9e}f * depth[i];
}}''', domain=n).freeze()

    def finish(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        levels = config["levels"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_finite_finish(
    const int* rec, const int* ancestors,
    const float* tau, const float* phi, const float* z0, float* z) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    float time = $ctx.TIME.get(0)$;
    if (time == 0.0f) {{ z[i] = z0[i]; return; }}
    float target = tau[i] - time;
    if (target <= 0.0f) {{
        int root = ancestors[{levels - 1} * {n} + i];
        z[i] = z0[root] + phi[i];
        return;
    }}
    int high = i;
    for (int k = {levels - 1}; k >= 0; --k) {{
        int candidate = ancestors[k * {n} + high];
        if (tau[candidate] >= target) high = candidate;
    }}
    int low = rec[high];
    float alpha = (target - tau[low]) / fmaxf(tau[high] - tau[low], 1.0e-20f);
    alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
    float initial = z0[low] + alpha * (z0[high] - z0[low]);
    float uplift_at_source = phi[low] + alpha * (phi[high] - phi[low]);
    z[i] = initial + phi[i] - uplift_at_source;
}}''', domain=n).freeze()

    b.add("initialize", initialize, bind={
        "rec": "rec", "drainage": "drainage",
        "slope_correction": "slope_correction", "ancestors": "ancestors",
        "tau": "tau_a", "phi": "phi_a", "depth": "depth_a", "UPLIFT": "uplift",
        "ERODIBILITY": "erodibility", "HILLSLOPE": "hillslope_erosion",
        "ACTIVE_NX": "active_nx",
        "ACTIVE_N": "active_n", "ACTIVE_DX": "active_dx",
    })
    for src, dst in (("a", "b"), ("b", "a")):
        b.add(f"jump_{src}_to_{dst}", jump, bind={
            "ancestors": "ancestors", "tau_in": f"tau_{src}",
            "phi_in": f"phi_{src}", "tau_out": f"tau_{dst}",
            "phi_out": f"phi_{dst}", "depth_in": f"depth_{src}",
            "depth_out": f"depth_{dst}", "LEVEL": "level",
            "ACTIVE_N": "active_n",
        })
        b.add(f"max_jump_{src}_to_{dst}", max_jump, bind={
            "ancestors": "ancestors", "maxima_in": f"max_{src}",
            "maxima_out": f"max_{dst}", "LEVEL": "level",
            "ACTIVE_N": "active_n",
        })
    for value, name in (("z0", "seed_initial"),
                        ("candidate_z", "seed_candidate")):
        for depth in ("a", "b"):
            b.add(f"{name}_{depth}", max_seed, bind={
                "value": value, "depth": f"depth_{depth}",
                "maxima": "max_a", "ACTIVE_N": "active_n",
            })
    for src in ("a", "b"):
        b.add(f"finish_{src}", finish, bind={
            "rec": "rec", "ancestors": "ancestors", "tau": f"tau_{src}",
            "phi": f"phi_{src}", "z0": "conditioned_z0",
            "z": "candidate_z", "TIME": "time",
            "ACTIVE_N": "active_n",
        })
        for depth in ("a", "b"):
            for name, value in (("condition_initial", "conditioned_z0"),
                                ("condition_candidate", "z")):
                b.add(f"{name}_{src}_{depth}", max_finish, bind={
                    "maxima": f"max_{src}", "depth": f"depth_{depth}",
                    "value": value, "ACTIVE_N": "active_n",
                })
    return b.freeze()


SaleveFiniteProgram = build_finite_program()
