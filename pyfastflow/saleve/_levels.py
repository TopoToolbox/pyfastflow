"""Capacity-sized GPU transfers for Salève's active grid level."""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder


def build_level_program():
    b = ProgramBuilder("SaleveLevelProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx")
    b.config("jitter", default=0.25).config("seed", default=0)
    b.config("boundary", choices=("normal", "periodic_EW", "periodic_NS"),
             default="normal")
    b.param("active_nx", "scalar", "i32", value=lambda d: d["nx"])
    b.param("active_ny", "scalar", "i32", value=lambda d: d["ny"])
    b.param("copy_n", "scalar", "i32", value=0)
    b.param("relaxation", "scalar", "f32", value=0.25)
    shape = (Dim("ny"), Dim("nx"))
    for name in ("z", "z0", "uplift", "erodibility", "thermal_erosion",
                 "hillslope_erosion", "work"):
        b.data(name, "f32", shape, role="output")
    for name in ("nodata", "outlet", "mask_work"):
        b.data(name, "u8", shape, role="output")

    def restrict(_be, _bundles, config):
        cap = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_restrict(
    const float* src, float* dst) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nx = $ctx.NX.get(0)$, ny = $ctx.NY.get(0)$;
    int coarse_nx = nx / 2, coarse_n = coarse_nx * (ny / 2);
    if (i >= coarse_n) return;
    int x = i % coarse_nx, y = i / coarse_nx;
    int j = 2 * y * nx + 2 * x;
    dst[i] = 0.25f * (src[j] + src[j + 1]
            + src[j + nx] + src[j + nx + 1]);
}}''', domain=cap).freeze()

    def prolong(_be, _bundles, config):
        cap = config["nx"] * config["ny"]
        periodic_x = int(config["boundary"] == "periodic_EW")
        periodic_y = int(config["boundary"] == "periodic_NS")
        jitter = float(config["jitter"])
        seed = int(config["seed"]) & 0xffffffff
        return KernelBuilder(f'''
extern "C" __global__ void saleve_prolong(
    const float* src, float* dst) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nx = $ctx.NX.get(0)$, ny = $ctx.NY.get(0)$;
    int fine_nx = 2 * nx;
    if (i >= 4 * nx * ny) return;
    int x = i % fine_nx, y = i / fine_nx;
    unsigned int hx = (unsigned int)i ^ {seed}u;
    hx ^= hx >> 16; hx *= 0x7feb352du;
    hx ^= hx >> 15; hx *= 0x846ca68bu; hx ^= hx >> 16;
    unsigned int hy = hx ^ 0x9e3779b9u;
    hy ^= hy >> 16; hy *= 0x7feb352du;
    hy ^= hy >> 15; hy *= 0x846ca68bu; hy ^= hy >> 16;
    float jx = ((float)(hx & 65535u) / 65535.0f - 0.5f) * {2 * jitter:.9e}f;
    float jy = ((float)(hy & 65535u) / 65535.0f - 0.5f) * {2 * jitter:.9e}f;
    float xf = 0.5f * ((float)x + 0.5f + jx) - 0.5f;
    float yf = 0.5f * ((float)y + 0.5f + jy) - 0.5f;
    if (!{periodic_x}) xf = fminf(fmaxf(xf, 0.0f), (float)(nx - 1));
    if (!{periodic_y}) yf = fminf(fmaxf(yf, 0.0f), (float)(ny - 1));
    int x0 = (int)floorf(xf), y0 = (int)floorf(yf);
    float tx = xf - x0, ty = yf - y0;
    int x1, y1;
    if ({periodic_x}) {{
        x0 = ((x0 % nx) + nx) % nx;
        x1 = (x0 + 1) % nx;
    }} else {{
        x0 = max(0, min(x0, nx - 1));
        x1 = min(x0 + 1, nx - 1);
    }}
    if ({periodic_y}) {{
        y0 = ((y0 % ny) + ny) % ny;
        y1 = (y0 + 1) % ny;
    }} else {{
        y0 = max(0, min(y0, ny - 1));
        y1 = min(y0 + 1, ny - 1);
    }}
    float a = src[y0 * nx + x0], b = src[y0 * nx + x1];
    float c = src[y1 * nx + x0], d = src[y1 * nx + x1];
    dst[i] = (1.0f - ty) * ((1.0f - tx) * a + tx * b)
           + ty * ((1.0f - tx) * c + tx * d);
}}''', domain=cap).freeze()

    def copy(_be, _bundles, config):
        cap = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_copy(const float* src, float* dst) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < $ctx.COUNT.get(0)$) dst[i] = src[i];
}}''', domain=cap).freeze()

    def restrict_mask(_be, _bundles, config):
        cap = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_restrict_mask(
    const unsigned char* src, unsigned char* dst) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nx = $ctx.NX.get(0)$, ny = $ctx.NY.get(0)$;
    int coarse_nx = nx / 2, coarse_n = coarse_nx * (ny / 2);
    if (i >= coarse_n) return;
    int j = 2 * (i / coarse_nx) * nx + 2 * (i % coarse_nx);
    dst[i] = src[j] | src[j + 1] | src[j + nx] | src[j + nx + 1];
}}''', domain=cap).freeze()

    def copy_mask(_be, _bundles, config, *, inactive):
        cap = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_copy_mask(
    const unsigned char* src, unsigned char* dst) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {cap}) dst[i] = i < $ctx.COUNT.get(0)$ ? src[i] : {inactive};
}}''', domain=cap).freeze()

    def blend(_be, _bundles, config):
        cap = config["nx"] * config["ny"]
        return KernelBuilder(f'''
extern "C" __global__ void saleve_blend(
    float* z, const float* previous) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.COUNT.get(0)$) return;
    float a = $ctx.RELAXATION.get(0)$;
    z[i] = previous[i] + a * (z[i] - previous[i]);
}}''', domain=cap).freeze()

    for source in ("z", "z0", "uplift", "erodibility", "thermal_erosion",
                   "hillslope_erosion"):
        b.add(f"restrict_{source}", restrict, bind={
            "src": source, "dst": "work", "NX": "active_nx",
            "NY": "active_ny",
        })
        b.add(f"copy_{source}", copy, bind={
            "src": "work", "dst": source, "COUNT": "copy_n",
        })
    b.add("prolong_z", prolong, bind={
        "src": "z", "dst": "work", "NX": "active_nx", "NY": "active_ny",
    })
    b.add("snapshot_z", copy, bind={
        "src": "z", "dst": "work", "COUNT": "copy_n",
    })
    b.add("blend_z", blend, bind={
        "z": "z", "previous": "work", "COUNT": "copy_n",
        "RELAXATION": "relaxation",
    })
    for mask, inactive in (("nodata", 1), ("outlet", 0)):
        b.add(f"restrict_{mask}", restrict_mask, bind={
            "src": mask, "dst": "mask_work", "NX": "active_nx",
            "NY": "active_ny",
        })
        b.add(f"copy_{mask}",
              lambda be, bundles, config, inactive=inactive:
                  copy_mask(be, bundles, config, inactive=inactive),
              bind={"src": "mask_work", "dst": mask, "COUNT": "copy_n"})
    return b.freeze()


SaleveLevelProgram = build_level_program()
