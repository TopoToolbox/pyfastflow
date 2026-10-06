"""Multi-scale erosion of Schott et al. (2024), ported to a CuPy program.

Port of the release code of *Terrain Amplification using Multi-scale
Erosion* (H. Schott, E. Galin, E. Guérin, A. Peytavie, A. Paris, ACM TOG
43(4), 2024; https://github.com/H-Schott/MultiScaleErosion): its three compute
shaders, ``erosion.glsl``, ``thermal.glsl`` and ``deposition.glsl``, each one
explicit, double-buffered step on a vertex grid of spacing ``dx``:

- erosion: drainage (``stream``) carried one cell per step by a multiple-flow
  split weighted by ``slope ** flow_p``, starting from the cell diagonal
  length; stream power ``k * stream**p_sa * clamp(slope**p_sl, 0, 1)``,
  capped at ``k * max_spe``, times ``dt``, never below the steepest receiver.
- thermal: each of the 3x3 neighbours (periodic taps) steeper than the talus
  angle moves ``eps * dx**2`` of height; the angle's tangent is drawn between
  ``noise_min`` and ``noise_max`` from 3D simplex noise of the cell's
  position and height times ``noise_wavelength`` when ``noisified``, else
  ``tan_angle``.
- deposition: water ``rain * dx**2 * 1e-5`` per cell plus the split inflow,
  and sediment routed the same way (kept only in pits); sediment above
  ``stream**0.3 * clamp(slope**2, 0, 1) / deposition_strength`` is deposited
  (a tenth of the excess per step), and a tenth of that stream power is
  picked up.

Each cell's 8 split shares are computed once per step (``weights_*``)
instead of once per receiving neighbour as in the shaders; same arithmetic.
Constants and parameter defaults are the shaders' own. Two shader quirks are
kept: ``deposition.glsl`` ignores its ``p_sa``/``p_sl`` uniforms (exponents
0.3 and 2 written in), and its weighted split takes ``pow`` of negative
slopes (NaN, never counted); both splits here count only positive slopes,
with each shader's own normalisation guard. The shaders' grid origin is
(0, 0), which only matters for the thermal noise position.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.core.context.program import Dim, ProgramBuilder

# next8 of the shaders: (dx, dy) of the 8 neighbours, in their order.
_COMMON = '''
__device__ const int NX8[8] = {0, 1, 1, 1, 0, -1, -1, -1};
__device__ const int NY8[8] = {1, 1, 0, -1, -1, -1, 0, 1};

// Slope(p, q) of the shaders: (z[q] - z[p]) / |q - p|, 0 off the grid or
// for p == q.
__device__ float mse_slope(const float* z, int px, int py, int qx, int qy,
                           int nx, int ny, float dx) {
    if (px < 0 || px >= nx || py < 0 || py >= ny) return 0.0f;
    if (qx < 0 || qx >= nx || qy < 0 || qy >= ny) return 0.0f;
    if (px == qx && py == qy) return 0.0f;
    float ddx = (float)(qx - px) * dx, ddy = (float)(qy - py) * dx;
    float d = sqrtf(ddx * ddx + ddy * ddy);
    return (z[qy * nx + qx] - z[py * nx + px]) / d;
}

// GetFlowSteepest: the first of next8 with the largest positive slope down
// from p, else 8 (no receiver, the shaders' ivec2(0, 0)).
__device__ int mse_steepest(const float* z, int x, int y, int nx, int ny,
                            float dx) {
    int d = 8;
    float max_slope = 0.0f;
    for (int k = 0; k < 8; k++) {
        float s = mse_slope(z, x + NX8[k], y + NY8[k], x, y, nx, ny, dx);
        if (s > max_slope) { max_slope = s; d = k; }
    }
    return d;
}
'''

_SIMPLEX = '''
// snoise(vec3) of the shaders (Ashima Arts / Ian McEwan 3D simplex noise),
// written out per component.
__device__ float mse_mod289(float x) { return x - floorf(x * (1.0f / 289.0f)) * 289.0f; }
__device__ float mse_permute(float x) { return mse_mod289(((x * 34.0f) + 1.0f) * x); }

__device__ float mse_snoise(float vx, float vy, float vz) {
    const float CX = 1.0f / 6.0f, CY = 1.0f / 3.0f;
    float s = (vx + vy + vz) * CY;
    float ix = floorf(vx + s), iy = floorf(vy + s), iz = floorf(vz + s);
    float t = (ix + iy + iz) * CX;
    float x0x = vx - ix + t, x0y = vy - iy + t, x0z = vz - iz + t;
    // g = step(x0.yzx, x0.xyz); l = 1 - g
    float gx = x0x < x0y ? 0.0f : 1.0f;
    float gy = x0y < x0z ? 0.0f : 1.0f;
    float gz = x0z < x0x ? 0.0f : 1.0f;
    float lx = 1.0f - gx, ly = 1.0f - gy, lz = 1.0f - gz;
    // i1 = min(g.xyz, l.zxy); i2 = max(g.xyz, l.zxy)
    float i1x = fminf(gx, lz), i1y = fminf(gy, lx), i1z = fminf(gz, ly);
    float i2x = fmaxf(gx, lz), i2y = fmaxf(gy, lx), i2z = fmaxf(gz, ly);
    float cx[4] = {x0x, x0x - i1x + CX, x0x - i2x + CY, x0x - 0.5f};
    float cy[4] = {x0y, x0y - i1y + CX, x0y - i2y + CY, x0y - 0.5f};
    float cz[4] = {x0z, x0z - i1z + CX, x0z - i2z + CY, x0z - 0.5f};
    ix = mse_mod289(ix); iy = mse_mod289(iy); iz = mse_mod289(iz);
    float ox[4] = {0.0f, i1x, i2x, 1.0f};
    float oy[4] = {0.0f, i1y, i2y, 1.0f};
    float oz[4] = {0.0f, i1z, i2z, 1.0f};
    // ns = n_ * D.wyz - D.xzx with n_ = 1/7, D = (0, 0.5, 1, 2)
    const float n_ = 0.142857142857f;
    const float nsx = n_ * 2.0f, nsy = n_ * 0.5f - 1.0f, nsz = n_;
    float result = 0.0f;
    for (int k = 0; k < 4; k++) {
        float p = mse_permute(mse_permute(mse_permute(iz + oz[k]) + iy + oy[k]) + ix + ox[k]);
        float j = p - 49.0f * floorf(p * nsz * nsz);
        float x_ = floorf(j * nsz);
        float y_ = floorf(j - 7.0f * x_);
        float gxk = x_ * nsx + nsy;
        float gyk = y_ * nsx + nsy;
        float h = 1.0f - fabsf(gxk) - fabsf(gyk);
        // sh = -step(h, 0); a = b + s * sh with s = floor(b) * 2 + 1
        float sh = (0.0f < h) ? 0.0f : -1.0f;
        float px = gxk + (floorf(gxk) * 2.0f + 1.0f) * sh;
        float py = gyk + (floorf(gyk) * 2.0f + 1.0f) * sh;
        float pz = h;
        float norm = 1.79284291400159f - 0.85373472095314f * (px * px + py * py + pz * pz);
        px *= norm; py *= norm; pz *= norm;
        float m = fmaxf(0.6f - (cx[k] * cx[k] + cy[k] * cy[k] + cz[k] * cz[k]), 0.0f);
        m = m * m;
        result += m * m * (px * cx[k] + py * cy[k] + pz * cz[k]);
    }
    return 42.0f * result;
}
'''


def build_mserosion_program():
    b = ProgramBuilder("MultiScaleErosionProgram")
    b.dim("ny").dim("nx")
    b.config("ny").config("nx")
    b.param("dx", "scalar", "f32", value=1.0)
    # erosion.glsl
    b.param("flow_p", "scalar", "f32", value=1.3)
    b.param("k", "scalar", "f32", value=0.0005)
    b.param("p_sa", "scalar", "f32", value=0.8)
    b.param("p_sl", "scalar", "f32", value=2.0)
    b.param("dt", "scalar", "f32", value=1.0)
    b.param("max_spe", "scalar", "f32", value=10000.0)
    # thermal.glsl
    b.param("eps", "scalar", "f32", value=0.00005)
    b.param("tan_angle", "scalar", "f32", value=0.57)
    b.param("noisified", "scalar", "i32", value=1)
    b.param("noise_min", "scalar", "f32", value=0.9)
    b.param("noise_max", "scalar", "f32", value=1.4)
    b.param("noise_wavelength", "scalar", "f32", value=0.0023)
    # deposition.glsl
    b.param("deposition_strength", "scalar", "f32", value=1.0)
    shape = (Dim("ny"), Dim("nx"))
    for name in ("z_a", "z_b", "stream_a", "stream_b", "sed_a", "sed_b"):
        b.data(name, "f32", shape, role="output")
    # The 8 outgoing shares of each cell's multiple-flow split, computed once
    # per step (the shaders recompute them for every receiving neighbour).
    b.data("weights", "f32", (8, Dim("ny"), Dim("nx")), role="internal")

    def weights(_be, _bundles, config, *, variant):
        nx, ny = int(config["nx"]), int(config["ny"])
        if variant == "erosion":
            # GetFlowWeighted of erosion.glsl
            split = """
        if (s > 0.0f) { sn[k] = powf(fabsf(s), flow_p); slope_sum += sn[k]; }
        else sn[k] = -1.0f;
    }
    slope_sum = (slope_sum < 0.00001f) ? 1.0f : slope_sum;"""
        else:
            # GetFlowWeighted of deposition.glsl (positive slopes only, guard == 0)
            split = """
        sn[k] = s > 0.0f ? powf(s, flow_p) : 0.0f;
        if (sn[k] > 0.0f) slope_sum += sn[k];
    }
    slope_sum = (slope_sum == 0.0f) ? 1.0f : slope_sum;"""
        source = (_COMMON
                  + 'extern "C" __global__ void mse_weights_' + variant
                  + "(const float* z, float* weights) {\n"
                  + "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                  + f"    const int nx = {nx}, ny = {ny};\n"
                  + """    if (i >= nx * ny) return;
    int x = i % nx, y = i / nx;
    float dx = $ctx.DX.get(0)$;
    float flow_p = $ctx.FLOW_P.get(0)$;
    float sn[8];
    float slope_sum = 0.0f;
    for (int k = 0; k < 8; k++) {
        float s = mse_slope(z, x + NX8[k], y + NY8[k], x, y, nx, ny, dx);"""
                  + split + """
    for (int k = 0; k < 8; k++) weights[k * nx * ny + i] = sn[k] / slope_sum;
}""")
        return KernelBuilder(source, domain=nx * ny).freeze()

    def erosion(_be, _bundles, config):
        nx, ny = int(config["nx"]), int(config["ny"])
        return KernelBuilder(_COMMON + f'''
// `weights`: the erosion split (weights_erosion on z_in); a neighbour q off
// the grid sends nothing, as its GetFlowWeighted gives only -1 shares.
extern "C" __global__ void mse_erosion(
    const float* z_in, float* z_out,
    const float* stream_in, float* stream_out, const float* weights) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nx = {nx}, ny = {ny};
    if (i >= nx * ny) return;
    int x = i % nx, y = i / nx;
    float dx = $ctx.DX.get(0)$;

    // Flow accumulation at p: q = p + next8[k] sends its share (k + 4) % 8
    float stream = 1.0f * sqrtf(2.0f) * dx;
    for (int k = 0; k < 8; k++) {{
        int qx = x + NX8[k], qy = y + NY8[k];
        if (qx < 0 || qx >= nx || qy < 0 || qy >= ny) continue;
        int q = qy * nx + qx;
        float ss = weights[((k + 4) % 8) * nx * ny + q];
        if (ss > 0.0f) stream += ss * stream_in[q];
    }}

    // steepest slope
    int d = mse_steepest(z_in, x, y, nx, ny, dx);
    int rx = d < 8 ? x + NX8[d] : x, ry = d < 8 ? y + NY8[d] : y;
    float receiver_height = z_in[ry * nx + rx];
    float steepest_slope = fabsf(mse_slope(z_in, rx, ry, x, y, nx, ny, dx));

    // stream power
    float spe = powf(stream, $ctx.P_SA.get(0)$)
              * fminf(fmaxf(powf(steepest_slope, $ctx.P_SL.get(0)$), 0.0f), 1.0f);
    spe = fminf(fmaxf(spe, 0.0f), $ctx.MAX_SPE.get(0)$);
    spe *= $ctx.K.get(0)$;

    // update height
    float new_height = z_in[i] - $ctx.DT.get(0)$ * spe;
    z_out[i] = fmaxf(new_height, receiver_height);
    stream_out[i] = stream;
}}''', domain=nx * ny).freeze()

    def thermal(_be, _bundles, config):
        nx, ny = int(config["nx"]), int(config["ny"])
        return KernelBuilder(_SIMPLEX + f'''
extern "C" __global__ void mse_thermal(const float* z_in, float* z_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nx = {nx}, ny = {ny};
    if (i >= nx * ny) return;
    int x = i % nx, y = i / nx;
    float dx = $ctx.DX.get(0)$;
    float z = z_in[i];

    // Threshold angle from noise between noise_min and noise_max
    float tan_angle = $ctx.TAN_ANGLE.get(0)$;
    if ($ctx.NOISIFIED.get(0)$) {{
        float w = $ctx.NOISE_WAVELENGTH.get(0)$;
        float t = mse_snoise((float)x * dx * w, (float)y * dx * w, z * w) * 0.5f + 0.5f;
        tan_angle = $ctx.NOISE_MIN.get(0)$
                  + ($ctx.NOISE_MAX.get(0)$ - $ctx.NOISE_MIN.get(0)$) * t;
    }}

    // Check stability with the 3x3 taps (periodic), the cell itself included
    float receive = 0.0f, distribute = 0.0f;
    for (int a = 0; a < 3; a++) {{
        for (int b = 0; b < 3; b++) {{
            int tx = (x + a - 1 + nx) % nx, ty = (y + b - 1 + ny) % ny;
            float ddx = (float)(x - tx) * dx, ddy = (float)(y - ty) * dx;
            float d = sqrtf(ddx * ddx + ddy * ddy);
            float sample = z_in[ty * nx + tx];
            if ((sample - z) / d > tan_angle) receive += 1.0f;
            if ((z - sample) / d > tan_angle) distribute += 1.0f;
        }}
    }}

    // Add/Remove matter if necessary
    float matter = $ctx.EPS.get(0)$ * dx * dx;
    z_out[i] = z + matter * (receive - distribute);
}}''', domain=nx * ny).freeze()

    def deposition(_be, _bundles, config):
        nx, ny = int(config["nx"]), int(config["ny"])
        return KernelBuilder(_COMMON + f'''
// `weights`: the deposition split (weights_deposition on z_in).
extern "C" __global__ void mse_deposition(
    const float* z_in, float* z_out,
    const float* stream_in, float* stream_out,
    const float* sed_in, float* sed_out, const float* weights) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nx = {nx}, ny = {ny};
    if (i >= nx * ny) return;
    int x = i % nx, y = i / nx;
    float dx = $ctx.DX.get(0)$;
    const float cell_area = dx * dx * 0.00001f;
    const float rain = 2.6f;

    float height = z_in[i];
    float sed = sed_in[i];

    int d = mse_steepest(z_in, x, y, nx, ny, dx);
    float steepest_slope = d < 8
        ? mse_slope(z_in, x + NX8[d], y + NY8[d], x, y, nx, ny, dx) : 0.0f;

    // Modify water & sediment values: sediment stays only in pits
    bool pit = true;
    for (int k = 0; k < 8; k++)
        if (mse_slope(z_in, x + NX8[k], y + NY8[k], x, y, nx, ny, dx) > 0.0f) pit = false;
    if (!pit) sed = 0.0f;

    // Add sediment and water
    float stream = rain * cell_area;
    for (int k = 0; k < 8; k++) {{
        int qx = x + NX8[k], qy = y + NY8[k];
        if (qx < 0 || qx >= nx || qy < 0 || qy >= ny) continue;
        int q = qy * nx + qx;
        float ss = weights[((k + 4) % 8) * nx * ny + q];
        if (ss > 0.0f) {{
            stream += ss * stream_in[q];
            sed += ss * sed_in[q];
        }}
    }}

    float speed = fminf(fmaxf(powf(steepest_slope, 2.0f), 0.0f), 1.0f);
    float stream_power = powf(stream, 0.3f) * speed;

    // Deposit
    float strength = $ctx.DEPOSITION_STRENGTH.get(0)$;
    if (strength * sed > stream_power) {{
        float deposit = fminf(sed, (strength * sed - stream_power) * 0.1f);
        height += deposit;
        sed = fmaxf(0.0f, sed - deposit);
    }}
    sed += 0.1f * stream_power;

    z_out[i] = height;
    stream_out[i] = stream;
    sed_out[i] = sed;
}}''', domain=nx * ny).freeze()

    for src, dst in (("a", "b"), ("b", "a")):
        for variant in ("erosion", "deposition"):
            b.add(f"weights_{variant}_{src}",
                  lambda be, bundles, config, variant=variant:
                      weights(be, bundles, config, variant=variant),
                  bind={"z": f"z_{src}", "weights": "weights",
                        "DX": "dx", "FLOW_P": "flow_p"})
        b.add(f"erosion_{src}_to_{dst}", erosion, bind={
            "z_in": f"z_{src}", "z_out": f"z_{dst}",
            "stream_in": f"stream_{src}", "stream_out": f"stream_{dst}",
            "weights": "weights",
            "DX": "dx", "P_SA": "p_sa", "P_SL": "p_sl",
            "MAX_SPE": "max_spe", "K": "k", "DT": "dt",
        })
        b.add(f"thermal_{src}_to_{dst}", thermal, bind={
            "z_in": f"z_{src}", "z_out": f"z_{dst}",
            "DX": "dx", "TAN_ANGLE": "tan_angle", "NOISIFIED": "noisified",
            "NOISE_WAVELENGTH": "noise_wavelength", "NOISE_MIN": "noise_min",
            "NOISE_MAX": "noise_max", "EPS": "eps",
        })
        b.add(f"deposition_{src}_to_{dst}", deposition, bind={
            "z_in": f"z_{src}", "z_out": f"z_{dst}",
            "stream_in": f"stream_{src}", "stream_out": f"stream_{dst}",
            "sed_in": f"sed_{src}", "sed_out": f"sed_{dst}",
            "weights": "weights", "DX": "dx",
            "DEPOSITION_STRENGTH": "deposition_strength",
        })
    return b.freeze()


MultiScaleErosionProgram = build_mserosion_program()
