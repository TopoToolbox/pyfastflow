"""Fixed-network cliff correction in receiver-link elevation differences."""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder
from pyfastflow.grid import make_grid_group, make_grid_parameters


def build_cliff_program():
    b = ProgramBuilder("SaleveCliffProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx")
    b.config("topology", choices=("D4", "D8"), default="D8")
    b.config("boundary", choices=("normal", "periodic_EW", "periodic_NS"),
             default="normal")
    b.param("active_n", "scalar", "i32", value=lambda d: d["nx"] * d["ny"])
    b.param("minimum", "scalar", "f32", value=0.0)
    b.param("scale", "scalar", "f32", value=1.0)
    b.param("learning_rate", "scalar", "f32", value=0.01)
    b.param("river_weight", "scalar", "f32", value=1.0 / 3.0)
    shape = (Dim("ny"), Dim("nx"))
    for name in ("rec",):
        b.data(name, "i32", shape, role="input")
    for name in ("physical_z", "grad_sum"):
        b.data(name, "f32", shape, role="output")
    for name in ("target", "z", "target_diff", "diff", "source", "area",
                 "sum_a", "sum_b"):
        b.data(name, "f32", shape, role="internal")
    for name in ("parent_a", "parent_b"):
        b.data(name, "i32", shape, role="internal")

    def grid_structure(be, *, topology, boundary, **_):
        return make_grid_group(be, topology=topology, boundary=boundary,
                               outlet="edge", nodata=True)

    def grid_params(be, pool, *, nx, ny, topology, **_):
        return make_grid_parameters(be, pool, nx, ny, 1.0, topology=topology,
                                    outlet="edge", nodata=True,
                                    nx_mode="scalar", ny_mode="scalar")

    b.bundle("grid", grid_structure, grid_params, dims=("nx", "ny"),
             config=("topology", "boundary"))

    def initialize(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_cliff_init(
    const float* physical, const int* rec,
    float* target, float* z, float* target_diff, float* diff) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE.get(0)$) return;
    float minimum = $ctx.MINIMUM.get(0)$;
    float scale = $ctx.SCALE.get(0)$;
    int r = rec[i];
    target[i] = z[i] = (physical[i] - minimum) / scale;
    target_diff[i] = diff[i] = r == i ? 0.0f
        : (physical[i] - physical[r]) / scale;
}}''', domain=n).freeze()

    def local_gradient(_be, bundles, config):
        n = config["nx"] * config["ny"]
        directions = (0, 1, 2, 3) if config["topology"] == "D4" else (1, 3, 4, 6)
        return KernelBuilder(f'''
extern "C" __global__ void saleve_cliff_local_gradient(
    const float* z, const float* target_diff,
    const int* rec, float* source) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE.get(0)$) return;
    if (!$ctx.grid.is_active(i)$) {{ source[i] = 0.0f; return; }}
    int r = rec[i];
    float gradient = 0.0f;
    const int directions[4] = {{{', '.join(map(str, directions))}}};
    #pragma unroll
    for (int t = 0; t < 4; ++t) {{
        int j = $ctx.grid.neighbour(i, directions[t])$;
        if (j < 0 || rec[j] == i || r == j) continue;
        float delta = z[i] - z[j];
        if (delta > 0.0f) {{
            gradient += r == i ? delta : fmaxf(delta - target_diff[i], 0.0f);
        }} else {{
            gradient += rec[j] == j ? delta
                : fminf(delta + target_diff[j], 0.0f);
        }}
    }}
    source[i] = gradient;
}}''', domain=n).compose("grid", bundles["grid"]).freeze()

    def update(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_cliff_update(
    const int* rec, const float* z, const float* target_diff,
    const float* grad_sum, const float* area, float* diff) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE.get(0)$) return;
    int r = rec[i];
    if (r == i) {{ diff[i] = 0.0f; return; }}
    float old_diff = z[i] - z[r];
    float w = $ctx.RIVER_WEIGHT.get(0)$;
    float gradient = w * (old_diff - target_diff[i])
        + (1.0f - w) * grad_sum[i] / fmaxf(area[i], 1.0f);
    float lower = fminf(target_diff[i], 0.0f);
    diff[i] = fmaxf(old_diff - $ctx.LEARNING_RATE.get(0)$ * gradient, lower);
}}''', domain=n).freeze()

    def path_init(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_cliff_path_init(
    const int* rec, const float* diff, int* parent, float* sum) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE.get(0)$) return;
    parent[i] = rec[i];
    sum[i] = diff[i];
}}''', domain=n).freeze()

    def jump(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_cliff_jump(
    const int* parent_in, const float* sum_in,
    int* parent_out, float* sum_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.ACTIVE.get(0)$) return;
    int r = parent_in[i];
    parent_out[i] = parent_in[r];
    sum_out[i] = sum_in[i] + sum_in[r];
}}''', domain=n).freeze()

    def finish(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_cliff_finish(
    const int* parent, const float* sum,
    const float* target, float* z) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < $ctx.ACTIVE.get(0)$) z[i] = target[parent[i]] + sum[i];
}}''', domain=n).freeze()

    def export(_be, _bundles, config):
        n = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_cliff_export(
    const float* z, float* physical) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < $ctx.ACTIVE.get(0)$)
        physical[i] = $ctx.MINIMUM.get(0)$ + $ctx.SCALE.get(0)$ * z[i];
}}''', domain=n).freeze()

    b.add("initialize", initialize, bind={
        "physical": "physical_z", "rec": "rec", "target": "target",
        "z": "z", "target_diff": "target_diff", "diff": "diff",
        "ACTIVE": "active_n", "MINIMUM": "minimum", "SCALE": "scale",
    })
    b.add("local_gradient", local_gradient, bind={
        "grid": "grid", "z": "z", "target_diff": "target_diff",
        "rec": "rec", "source": "source", "ACTIVE": "active_n",
    })
    b.add("update", update, bind={
        "rec": "rec", "z": "z", "target_diff": "target_diff",
        "grad_sum": "grad_sum", "area": "area", "diff": "diff",
        "ACTIVE": "active_n", "RIVER_WEIGHT": "river_weight",
        "LEARNING_RATE": "learning_rate",
    })
    b.add("path_init", path_init, bind={
        "rec": "rec", "diff": "diff", "parent": "parent_a",
        "sum": "sum_a", "ACTIVE": "active_n",
    })
    for src, dst in (("a", "b"), ("b", "a")):
        b.add(f"jump_{src}_to_{dst}", jump, bind={
            "parent_in": f"parent_{src}", "sum_in": f"sum_{src}",
            "parent_out": f"parent_{dst}", "sum_out": f"sum_{dst}",
            "ACTIVE": "active_n",
        })
    for src in ("a", "b"):
        b.add(f"finish_{src}", finish, bind={
            "parent": f"parent_{src}", "sum": f"sum_{src}",
            "target": "target", "z": "z", "ACTIVE": "active_n",
        })
    b.add("export", export, bind={
        "z": "z", "physical": "physical_z", "ACTIVE": "active_n",
        "MINIMUM": "minimum", "SCALE": "scale",
    })
    return b.freeze()


SaleveCliffProgram = build_cliff_program()
