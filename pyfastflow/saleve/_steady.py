"""Steady stream-power solution on a frozen single-receiver forest."""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder

from ._speed import speed_source


def build_steady_program():
    b = ProgramBuilder("SaleveSteadyProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx").config("dx", default=1.0)
    b.config("m", default=0.4)
    b.param("active_nx", "scalar", "i32", value=lambda dims: dims["nx"])
    b.param("active_n", "scalar", "i32", value=lambda dims: dims["nx"] * dims["ny"])
    b.param("active_dx", "scalar", "f32", value=1.0)
    b.param("uplift", "auto", "f32", value=1.0, shape=(Dim("ny"), Dim("nx")))
    b.param("erodibility", "auto", "f32", value=1.0, shape=(Dim("ny"), Dim("nx")))
    b.param("thermal_erosion", "auto", "f32", value=0.0, shape=(Dim("ny"), Dim("nx")))
    b.param("hillslope_erosion", "auto", "f32", value=0.0, shape=(Dim("ny"), Dim("nx")))
    b.config("hack_constant", default=1.5).config("hack_exponent", default=0.6)
    b.param("critical_slope", "auto", "f32", value=0.57)
    shape = (Dim("ny"), Dim("nx"))
    b.data("rec", "i32", shape, role="input")
    b.data("drainage", "f32", shape, role="input")
    b.data("slope_correction", "f32", shape, role="input")
    b.data("outlet_z", "f32", shape, role="input")
    b.data("z", "f32", shape, role="output")
    for name in ("parent_a", "parent_b"):
        b.data(name, "i32", shape, role="internal")
    for name in ("sum_a", "sum_b"):
        b.data(name, "f32", shape, role="internal")

    def initialize(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_steady_init(
    const int* rec, const float* drainage, const float* slope_correction,
    int* parent, float* sum) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int r = rec[i];
    parent[i] = r;
    if (r == i) {{ sum[i] = 0.0f; return; }}
    int nx = $ctx.ACTIVE_NX.get(0)$;
    int dr = i / nx - r / nx;
    int dc = i % nx - r % nx;
    float distance = (dr && dc) ? 1.41421356237f : 1.0f;
    float dx = $ctx.ACTIVE_DX.get(0)$;
    float area = fmaxf(drainage[i] * dx * dx, 1.0e-20f);
    float uplift = $ctx.UPLIFT.get(i)$;
    {speed_source(config)}
    float kt = $ctx.THERMAL.get(i)$;
    float critical = $ctx.CRITICAL.get(0)$;
    if (kt > 0.0f && uplift / speed > critical) {{
        uplift += kt * critical;
        speed += kt;
    }}
    sum[i] = dx * distance * uplift / (speed * slope_correction[i]);
}}''', domain=n).freeze()

    def jump(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_steady_jump(
    const int* parent_in, const float* sum_in,
    int* parent_out, float* sum_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int r = parent_in[i];
    parent_out[i] = parent_in[r];
    sum_out[i] = sum_in[i] + sum_in[r];
}}''', domain=n).freeze()

    def finish(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_steady_finish(
    const int* parent, const float* sum,
    const float* outlet_z, float* z) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < $ctx.ACTIVE_N.get(0)$) z[i] = outlet_z[parent[i]] + sum[i];
}}''', domain=n).freeze()

    b.add("initialize", initialize, bind={
        "rec": "rec", "drainage": "drainage",
        "slope_correction": "slope_correction", "parent": "parent_a",
        "sum": "sum_a", "UPLIFT": "uplift", "ERODIBILITY": "erodibility",
        "THERMAL": "thermal_erosion", "CRITICAL": "critical_slope",
        "HILLSLOPE": "hillslope_erosion",
        "ACTIVE_NX": "active_nx", "ACTIVE_N": "active_n",
        "ACTIVE_DX": "active_dx",
    })
    for src, dst in (("a", "b"), ("b", "a")):
        b.add(f"jump_{src}_to_{dst}", jump, bind={
            "parent_in": f"parent_{src}", "sum_in": f"sum_{src}",
            "parent_out": f"parent_{dst}", "sum_out": f"sum_{dst}",
            "ACTIVE_N": "active_n",
        })
    for src in ("a", "b"):
        b.add(f"finish_{src}", finish, bind={
            "parent": f"parent_{src}", "sum": f"sum_{src}",
            "outlet_z": "outlet_z", "z": "z",
            "ACTIVE_N": "active_n",
        })
    return b.freeze()


SaleveSteadyProgram = build_steady_program()
