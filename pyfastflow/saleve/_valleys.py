"""Height-above-channel valley floors on a frozen receiver forest.

A channel cell is one whose drainage area reaches ``valley_area``; its
floodplain height is ``h = valley_height * (A / valley_area) **
valley_exponent`` and its floodplain surface ``z + h * (1 +
valley_transition)``. Every cell is paired with the channel cell on its
downstream flow path whose floodplain surface is highest, found by a
pointer-jump path maximum carrying (surface, channel index). That pairing
lets a large river claim cells whose path first meets a small tributary,
which the nearest channel alone would leave untouched. With ``hand`` the
cell's height above its channel and ``d`` its along-path distance to it, the
floor is ``z_channel + valley_slope * d``; cells below ``h`` are lowered
onto it and ``valley_transition`` widens that step into a smooth wall over
``hand`` in ``[h, h * (1 + transition)]``. Cells are only ever lowered and
channel cells are never modified.

Author: B.G (09/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder


def build_valley_program():
    b = ProgramBuilder("SaleveValleyProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx")
    b.config("valley_area", default=1.0e6)
    b.config("valley_height", default=10.0)
    b.config("valley_exponent", default=0.5)
    b.config("valley_slope", default=1.0e-3)
    b.config("valley_transition", default=0.5)
    b.param("active_n", "scalar", "i32", value=lambda d: d["nx"] * d["ny"])
    b.param("active_dx", "scalar", "f32", value=1.0)
    shape = (Dim("ny"), Dim("nx"))
    b.data("rec", "i32", shape, role="input")
    b.data("drainage", "f32", shape, role="input")
    b.data("path_length", "f32", shape, role="input")
    b.data("z", "f32", shape, role="output")
    for name in ("ptr_a", "ptr_b", "idx_a", "idx_b"):
        b.data(name, "i32", shape, role="internal")
    for name in ("val_a", "val_b"):
        b.data(name, "f32", shape, role="internal")

    def height(config):
        area = float(config["valley_area"])
        return (f"{float(config['valley_height']):.9e}f"
                f" * powf(channel_area / {area:.9e}f, {float(config['valley_exponent']):.9e}f)")

    def initialize(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        area = float(config["valley_area"])
        transition = float(config["valley_transition"])
        return KernelBuilder(f'''
extern "C" __global__ void saleve_valley_init(
    const int* rec, const float* drainage, const float* z,
    int* ptr, float* val, int* idx) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    float dx = $ctx.ACTIVE_DX.get(0)$;
    float channel_area = drainage[i] * dx * dx;
    ptr[i] = rec[i];
    if (channel_area >= {area:.9e}f) {{
        val[i] = z[i] + {height(config)} * (1.0f + {transition:.9e}f);
        idx[i] = i;
    }} else {{
        val[i] = -3.0e38f;
        idx[i] = -1;
    }}
}}''', domain=n).freeze()

    def jump(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_valley_jump(
    const int* ptr_in, const float* val_in, const int* idx_in,
    int* ptr_out, float* val_out, int* idx_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int r = ptr_in[i];
    ptr_out[i] = ptr_in[r];
    if (val_in[r] > val_in[i]) {{ val_out[i] = val_in[r]; idx_out[i] = idx_in[r]; }}
    else {{ val_out[i] = val_in[i]; idx_out[i] = idx_in[i]; }}
}}''', domain=n).freeze()

    def apply(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        slope = float(config["valley_slope"])
        transition = float(config["valley_transition"])
        return KernelBuilder(f'''
extern "C" __global__ void saleve_valley_apply(
    const int* idx, const float* drainage, const float* path_length,
    float* z) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int c = idx[i];
    if (c < 0 || c == i) return;
    float dx = $ctx.ACTIVE_DX.get(0)$;
    float zc = z[c];
    float hand = z[i] - zc;
    float channel_area = drainage[c] * dx * dx;
    float h = {height(config)};
    if (hand >= h * (1.0f + {transition:.9e}f)) return;
    float d = fmaxf(path_length[i] - path_length[c], 0.0f);
    float floor_z = zc + {slope:.9e}f * d;
    float t = {transition:.9e}f > 0.0f
        ? fminf(fmaxf((hand - h) / (h * {transition:.9e}f), 0.0f), 1.0f)
        : (hand >= h ? 1.0f : 0.0f);
    t = t * t * (3.0f - 2.0f * t);
    z[i] = fminf(z[i], floor_z + (z[i] - floor_z) * t);
}}''', domain=n).freeze()

    b.add("initialize", initialize, bind={
        "rec": "rec", "drainage": "drainage", "z": "z",
        "ptr": "ptr_a", "val": "val_a", "idx": "idx_a",
        "ACTIVE_N": "active_n", "ACTIVE_DX": "active_dx",
    })
    for src, dst in (("a", "b"), ("b", "a")):
        b.add(f"jump_{src}_to_{dst}", jump, bind={
            "ptr_in": f"ptr_{src}", "val_in": f"val_{src}", "idx_in": f"idx_{src}",
            "ptr_out": f"ptr_{dst}", "val_out": f"val_{dst}", "idx_out": f"idx_{dst}",
            "ACTIVE_N": "active_n",
        })
    for src in ("a", "b"):
        b.add(f"apply_{src}", apply, bind={
            "idx": f"idx_{src}", "drainage": "drainage",
            "path_length": "path_length", "z": "z",
            "ACTIVE_N": "active_n", "ACTIVE_DX": "active_dx",
        })
    return b.freeze()


SaleveValleyProgram = build_valley_program()
