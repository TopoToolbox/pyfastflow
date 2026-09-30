"""Flow-path geometry on a frozen receiver forest.

``path_length`` is the along-path distance from each cell to its root.
``divide_distance`` is the longest upstream flow path ending at the cell,
i.e. its distance from the divide: the maximum of ``path_length`` over the
cell's subtree minus its own ``path_length``. The downstream sum is a
pointer-jump prefix sum; the subtree maximum is a pointer-jump push with
``atomicMax`` on the float bit pattern (valid for non-negative floats).

Author: B.G (09/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder


def build_divide_program():
    b = ProgramBuilder("SaleveDivideProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx")
    b.param("active_nx", "scalar", "i32", value=lambda d: d["nx"])
    b.param("active_n", "scalar", "i32", value=lambda d: d["nx"] * d["ny"])
    b.param("active_dx", "scalar", "f32", value=1.0)
    shape = (Dim("ny"), Dim("nx"))
    b.data("rec", "i32", shape, role="input")
    b.data("path_length", "f32", shape, role="output")
    b.data("divide_distance", "f32", shape, role="output")
    for name in ("parent_a", "parent_b", "ptr_a", "ptr_b", "max_a", "max_b"):
        b.data(name, "i32", shape, role="internal")
    for name in ("sum_a", "sum_b"):
        b.data(name, "f32", shape, role="internal")

    def initialize(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_divide_init(
    const int* rec, int* parent, float* sum) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int r = rec[i];
    parent[i] = r;
    if (r == i) {{ sum[i] = 0.0f; return; }}
    int nx = $ctx.ACTIVE_NX.get(0)$;
    int dr = i / nx - r / nx;
    int dc = i % nx - r % nx;
    float distance = (dr && dc) ? 1.41421356237f : 1.0f;
    sum[i] = $ctx.ACTIVE_DX.get(0)$ * distance;
}}''', domain=n).freeze()

    def jump(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_divide_jump(
    const int* parent_in, const float* sum_in,
    int* parent_out, float* sum_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int r = parent_in[i];
    parent_out[i] = parent_in[r];
    sum_out[i] = sum_in[i] + sum_in[r];
}}''', domain=n).freeze()

    def max_init(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_divide_max_init(
    const int* rec, const float* sum, float* path_length,
    int* ptr, int* maxima) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    path_length[i] = sum[i];
    ptr[i] = rec[i];
    maxima[i] = __float_as_int(sum[i]);
}}''', domain=n).freeze()

    def push_copy(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_divide_push_copy(
    const int* max_in, int* max_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < $ctx.ACTIVE_N.get(0)$) max_out[i] = max_in[i];
}}''', domain=n).freeze()

    def push_core(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_divide_push_core(
    const int* ptr_in, int* ptr_out, const int* max_in, int* max_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    int parent = ptr_in[i];
    ptr_out[i] = parent;
    if (parent != i) {{
        atomicMax(&max_out[parent], max_in[i]);
        int grandparent = ptr_in[parent];
        ptr_out[i] = (grandparent == parent) ? i : grandparent;
    }}
}}''', domain=n).freeze()

    def finish(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_divide_finish(
    const int* maxima, const float* path_length, float* divide_distance) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE_N.get(0)$) return;
    divide_distance[i] = fmaxf(__int_as_float(maxima[i]) - path_length[i], 0.0f);
}}''', domain=n).freeze()

    b.add("initialize", initialize, bind={
        "rec": "rec", "parent": "parent_a", "sum": "sum_a",
        "ACTIVE_NX": "active_nx", "ACTIVE_N": "active_n",
        "ACTIVE_DX": "active_dx",
    })
    for src, dst in (("a", "b"), ("b", "a")):
        b.add(f"jump_{src}_to_{dst}", jump, bind={
            "parent_in": f"parent_{src}", "sum_in": f"sum_{src}",
            "parent_out": f"parent_{dst}", "sum_out": f"sum_{dst}",
            "ACTIVE_N": "active_n",
        })
        b.add(f"push_{src}_to_{dst}_copy", push_copy, bind={
            "max_in": f"max_{src}", "max_out": f"max_{dst}",
            "ACTIVE_N": "active_n",
        })
        b.add(f"push_{src}_to_{dst}_core", push_core, bind={
            "ptr_in": f"ptr_{src}", "ptr_out": f"ptr_{dst}",
            "max_in": f"max_{src}", "max_out": f"max_{dst}",
            "ACTIVE_N": "active_n",
        })
    for src in ("a", "b"):
        b.add(f"max_init_{src}", max_init, bind={
            "rec": "rec", "sum": f"sum_{src}", "path_length": "path_length",
            "ptr": "ptr_a", "maxima": "max_a", "ACTIVE_N": "active_n",
        })
        b.add(f"finish_{src}", finish, bind={
            "maxima": f"max_{src}", "path_length": "path_length",
            "divide_distance": "divide_distance", "ACTIVE_N": "active_n",
        })
    return b.freeze()


SaleveDivideProgram = build_divide_program()
