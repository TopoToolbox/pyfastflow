"""CuPy kernels of ShearParcelSedProgram.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.flow._program_cupy import _cupy_only


# Deposition rules of the walk (as in the parcel engine): stochastic drops
# the whole parcel once an optical depth -ln(u) is consumed by the hop costs;
# gradual leaves the fraction 1 - exp(-cost) at each hop, down to MIN_VOLUME.
_STOCHASTIC_INIT = r'''
        unsigned int h = sedp_hash(key ^ sedp_hash((unsigned int)p));
        float tau = -logf(((float)(h >> 8) + 1.0f) * (1.0f / 16777216.0f));
'''
_STOCHASTIC_RULE = r'''
            tau -= cost;
            if (tau <= 0.0f) {
                if (escape) i = r;
                break;
            }
'''
_GRADUAL_RULE = r'''
            if (!escape) {
                float drop = carried * (1.0f - expf(-cost));
                if (carried - drop < $ctx.MIN_VOLUME.get(0)$) break;
                atomicAdd(&z[i], drop * inv_area);
                counts[2] += (double)drop;
                carried -= drop;
            }
'''


def _cells(config):
    return int(config["nx"]) * int(config["ny"])


_SAMPLE = r'''
// Flow depth of the link between cells a and b as the flow solver sees it,
// max(z_a + h_a, z_b + h_b) - max(z_a, z_b); returns the link velocity
// q / depth (0 for a missing link or a depth not above hmin) and adds the
// depth of a wet link to *depth_sum, *wet_links.
__device__ __forceinline__ float sedsh_link_velocity(
        const float* z, const float* h, float q, int a, int b, float hmin,
        float* depth_sum, int* wet_links) {
    if (a < 0 || b < 0) return 0.0f;
    float za = z[a], zb = z[b];
    float depth = fmaxf(za + fmaxf(h[a], 0.0f), zb + fmaxf(h[b], 0.0f))
                - fmaxf(za, zb);
    if (!(depth > hmin)) return 0.0f;
    *depth_sum += depth;
    *wet_links += 1;
    return q / depth;
}

// One hydraulic sample: adds the entrainment rate and the transport length
// of the current flow to rate and length (sedsh_mean divides them by the
// sample count). Link fluxes in the InertialFloodProgram layout: qx on the
// (ny, nx + 1) west faces, +east; qy on the (ny + 1, nx) north faces,
// +south; in D8 also qxy (cell -> south-east) and qyx (cell -> south-west).
// Every link of the cell gets its velocity from its own flow depth
// (sedsh_link_velocity); the cell velocity u averages the two links of each
// direction through the cell, weighted by their widths (dx cardinal, dx / 2
// diagonal). A cell with h below MIN_DEPTH, or without a wet link, is dry
// and adds nothing. With d the mean depth of the wet links:
//   tau   = RHO_W G MANNING^2 |u|^2 / d^(1/3)            (Manning friction)
//   tau_r = THETA_R (RHO_S - RHO_W) G D                  (reference stress)
//   w_s   = R G D^2 / (C1 NU + sqrt(0.75 C2 R G D^3)),  R = RHO_S / RHO_W - 1
//   rate  = ALPHA / (RHO_S w_s) max(tau - tau_r, 0)      (m/s of bed)
//   u*    = sqrt(tau / RHO_W),  T* = max(tau / tau_r - 1, 0)
//   h_s   = min(Z0 + 0.6 D + 0.025 D T*, h)              (saltation height)
//   v_s   = min(u* / KAPPA (ln(h_s / Z0) - 1 + Z0 / h_s), |u|)
//   length = h_s v_s / w_s                                (transport length)
// (Le Minor et al. 2025, ejection velocity = w_s; w_s from Ferguson and
// Church 2004). Inactive cells
// and outlets add nothing.
extern "C" __global__ void sedsh_sample(
        const float* z, const float* h, const float* qx, const float* qy@DIAG_ARGS@,
        float* rate, float* length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    if (!$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$) return;
    float hmin = $ctx.MIN_DEPTH.get(0)$;
    float hi = h[i];
    if (!(hi > hmin)) return;
    int x = i % @NX@;
    int y = i / @NX@;
    int west = $ctx.grid.neighbour(i, @K_W@)$;
    int east = $ctx.grid.neighbour(i, @K_E@)$;
    int north = $ctx.grid.neighbour(i, @K_N@)$;
    int south = $ctx.grid.neighbour(i, @K_S@)$;
    float q_w = qx[y * (@NX@ + 1) + x];
    float q_e = (x == @NX@ - 1 && @PERIODIC_X@)
        ? qx[y * (@NX@ + 1)] : qx[y * (@NX@ + 1) + x + 1];
    float q_n = qy[y * @NX@ + x];
    float q_s = (y == @NY@ - 1 && @PERIODIC_Y@)
        ? qy[x] : qy[(y + 1) * @NX@ + x];
    float depth_sum = 0.0f;
    int wet_links = 0;
    float u_w = sedsh_link_velocity(z, h, q_w, west, i, hmin, &depth_sum, &wet_links);
    float u_e = sedsh_link_velocity(z, h, q_e, i, east, hmin, &depth_sum, &wet_links);
    float u_n = sedsh_link_velocity(z, h, q_n, north, i, hmin, &depth_sum, &wet_links);
    float u_s = sedsh_link_velocity(z, h, q_s, i, south, hmin, &depth_sum, &wet_links);
    float velocity_x = 0.5f * (u_w + u_e);
    float velocity_y = 0.5f * (u_n + u_s);
    @DIAG@
    if (wet_links == 0) return;
    float flow_depth = depth_sum / (float)wet_links;
    float speed = sqrtf(velocity_x * velocity_x + velocity_y * velocity_y);
    float manning = fmaxf($ctx.MANNING.get(i)$, 0.0f);
    float g = $ctx.GRAVITY.get(0)$;
    float rho_w = $ctx.RHO_W.get(0)$;
    float rho_s = $ctx.RHO_S.get(0)$;
    float d = fmaxf($ctx.GRAIN_SIZE.get(i)$, 1.0e-9f);
    float tau = rho_w * g * manning * manning * speed * speed / cbrtf(flow_depth);
    float tau_r = fmaxf($ctx.THETA_R.get(i)$ * fmaxf(rho_s - rho_w, 1.0e-6f) * g * d,
                        1.0e-12f);
    float r = fmaxf(rho_s / rho_w - 1.0f, 1.0e-6f);
    float w_s = r * g * d * d
              / ($ctx.C1.get(0)$ * $ctx.NU.get(0)$
                 + sqrtf(0.75f * $ctx.C2.get(0)$ * r * g * d * d * d));
    float e = fmaxf($ctx.ALPHA.get(i)$, 0.0f) / (rho_s * w_s)
            * fmaxf(tau - tau_r, 0.0f);
    if (isfinite(e) && e > 0.0f) rate[i] += e;

    float z0 = fmaxf($ctx.Z0.get(i)$, 1.0e-9f);
    float ustar = sqrtf(tau / rho_w);
    float stage = fmaxf(tau / tau_r - 1.0f, 0.0f);
    float h_s = fminf(z0 + 0.6f * d + 0.025f * d * stage, hi);
    if (!(h_s > z0)) return;
    float v_s = fminf(ustar / $ctx.KAPPA.get(0)$
                      * (logf(h_s / z0) - 1.0f + z0 / h_s), speed);
    float l = h_s * fmaxf(v_s, 0.0f) / w_s;
    if (isfinite(l) && l > 0.0f) length[i] += l;
}
'''

_SAMPLE_DIAG = r'''
    int nw = $ctx.grid.neighbour(i, 0)$;
    int ne = $ctx.grid.neighbour(i, 2)$;
    int se = $ctx.grid.neighbour(i, 7)$;
    int sw = $ctx.grid.neighbour(i, 5)$;
    float q_nw = nw >= 0 ? qxy[nw] : 0.0f;
    float q_ne = ne >= 0 ? qyx[ne] : 0.0f;
    float u_se = 0.5f * (sedsh_link_velocity(z, h, qxy[i], i, se, hmin, &depth_sum, &wet_links)
                       + sedsh_link_velocity(z, h, q_nw, nw, i, hmin, &depth_sum, &wet_links));
    float u_sw = 0.5f * (sedsh_link_velocity(z, h, qyx[i], i, sw, hmin, &depth_sum, &wet_links)
                       + sedsh_link_velocity(z, h, q_ne, ne, i, hmin, &depth_sum, &wet_links));
    float w = 0.5f * 0.70710678f;
    velocity_x += w * (u_se - u_sw);
    velocity_y += w * (u_se + u_sw);'''

_DIAG_ARGS = ", const float* qxy, const float* qyx"

_EROSION = r'''
// Bed lowering of one sediment step of MORPHO_DT seconds: of the material
// entrained at the mean rate from uniformly distributed points of the cell,
// the fraction (xi / dx) (1 - exp(-dx / xi)) travels past the cell edge
// (xi the mean transport length); the rest settles back in the cell, so
// only the leaving volume is removed, as parcels of at most PARCEL_VOLUME
// (count = ceil(volume / PARCEL_VOLUME)); sedsh_apply lowers z.
extern "C" __global__ void sedsh_erosion(
        const float* rate, const float* length, float* erosion, double* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    float dx = $ctx.grid.DX.get(0)$;
    float xi = length[i];
    float leave = xi > 0.0f ? -(xi / dx) * expm1f(-dx / xi) : 0.0f;
    float e = rate[i] * leave * fmaxf($ctx.MORPHO_DT.get(0)$, 0.0f);
    if (!isfinite(e) || !(e > 0.0f)) e = 0.0f;
    erosion[i] = e;
    count[i] = ceil((double)e * (double)(dx * dx)
                    / (double)fmaxf($ctx.PARCEL_VOLUME.get(0)$, 1.0e-12f));
}
'''

_APPLY = r'''
extern "C" __global__ void sedsh_apply(float* z, const float* erosion) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < $ctx.CELL_COUNT.get(0)$) z[i] -= erosion[i];
}
'''


def _substitute(text, config):
    """Grid size, periodicity and the D4 / D8 neighbour indices."""
    d8 = config["topology"] == "D8"
    k_w, k_e, k_n, k_s = (3, 4, 1, 6) if d8 else (1, 2, 0, 3)
    return (text
            .replace("@K_W@", str(k_w)).replace("@K_E@", str(k_e))
            .replace("@K_N@", str(k_n)).replace("@K_S@", str(k_s))
            .replace("@NX@", str(int(config["nx"])))
            .replace("@NY@", str(int(config["ny"])))
            .replace("@PERIODIC_X@", str(int(config["boundary"] == "periodic_EW")))
            .replace("@PERIODIC_Y@", str(int(config["boundary"] == "periodic_NS")))
            .replace("@DIAG_ARGS@", _DIAG_ARGS if d8 else ""))


def sample_factory(be, bundles, config):
    _cupy_only(be)
    d8 = config["topology"] == "D8"
    text = _substitute(_SAMPLE.replace("@DIAG@", _SAMPLE_DIAG if d8 else ""),
                       config)
    return (KernelBuilder(text, domain=_cells(config))
            .compose("grid", bundles["grid"]).freeze())


def erosion_factory(be, bundles, config):
    _cupy_only(be)
    return (KernelBuilder(_EROSION, domain=_cells(config))
            .compose("grid", bundles["grid"]).freeze())


def apply_factory(be, bundles, config):
    _cupy_only(be)
    return KernelBuilder(_APPLY, domain=_cells(config)).freeze()


_WALK = r'''
__device__ unsigned int sedsh_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
#define sedp_hash sedsh_hash

// Discharge leaving cell i towards its neighbour k (index j), weighted by the
// link width (1 cardinal, 1/2 diagonal); negative when it flows in.
__device__ __forceinline__ float sedsh_outflow(
        const float* qx, const float* qy@DIAG_ARGS@, int i, int k, int j) {
    int x = i % @NX@;
    int y = i / @NX@;
    if (k == @K_W@) return -qx[y * (@NX@ + 1) + x];
    if (k == @K_E@) return (x == @NX@ - 1 && @PERIODIC_X@)
        ? qx[y * (@NX@ + 1)] : qx[y * (@NX@ + 1) + x + 1];
    if (k == @K_N@) return -qy[y * @NX@ + x];
    if (k == @K_S@) return (y == @NY@ - 1 && @PERIODIC_Y@)
        ? qy[x] : qy[(y + 1) * @NX@ + x];
    @DIAG_OUTFLOW@
    return 0.0f;
}

// Persistent parcel walk. Parcel p starts on the cell s with
// cdf[s - 1] <= p < cdf[s] and carries that cell's eroded volume divided by
// its parcel count. At every hop it moves to a neighbour j drawn with
// probability proportional to
//   max(outflow_j, 0) + BED_SLOPE_WEIGHT * max((z_i - z_j) / dist_j, 0),
// outflow_j the mean discharge of the link towards j (sedsh_outflow); the
// hop costs dist / length_i with the mean transport length of the cell it
// leaves (a cell without transport length keeps the parcel). Parcels have
// already crossed the edge of their source cell (sedsh_erosion), so the
// first hop is free; the hop rule (see walk_factory) decides what is
// left on the cell being left; deposits are atomic adds of volume / dx^2 on
// z. On a cell with no candidate the parcel leaves just the volume that
// raises the bed FILL_EPS above its lowest neighbour (what is still missing
// when it writes) and moves there; if it carries less, it leaves everything.
// A parcel reaching an outlet exports what it carries; one still walking
// after CELL_COUNT hops leaves it where it is.
// stats: parcels, eroded volume, deposited volume, exported volume.
extern "C" __global__ void sedsh_walk(
        const double* cdf, const float* erosion, const float* length,
        const float* qx, const float* qy@DIAG_ARGS@, float* z, int* claim,
        double* stats) {
    int n = $ctx.CELL_COUNT.get(0)$;
    double total = cdf[n - 1];
    if (!(total > 0.0)) return;
    int parcels = (int)total;
    float dx = $ctx.grid.DX.get(0)$;
    float inv_area = 1.0f / (dx * dx);
    float beta = fmaxf($ctx.BED_SLOPE_WEIGHT.get(0)$, 0.0f);
    unsigned int key = sedsh_hash((unsigned int)$ctx.SEED.get(0)$
                                  ^ sedsh_hash((unsigned int)$ctx.CLOCK.get(0)$));
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;

    double counts[4] = {0.0, 0.0, 0.0, 0.0};
    while (true) {
        int p = atomicAdd(&claim[0], 1);
        if (p >= parcels) break;
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (cdf[mid] > (double)p) hi = mid;
            else lo = mid + 1;
        }
        int s = lo;
        double share = cdf[s] - (s > 0 ? cdf[s - 1] : 0.0);
        float volume = (float)((double)erosion[s] * (double)(dx * dx) / share);
        @HOP_INIT@
        unsigned int stream = sedsh_hash(key ^ sedsh_hash((unsigned int)p ^ 0x5bd1e995u));
        int i = s;
        float carried = volume;
        bool exported = false;
        for (int hop = 0; hop < n; ++hop) {
            if ($ctx.grid.can_out(i)$) { exported = true; break; }
            float zi = __ldcg(&z[i]);
            float w[8];
            float w_sum = 0.0f;
            for (int k = 0; k < nk; ++k) {
                w[k] = 0.0f;
                int j = $ctx.grid.neighbour(i, k)$;
                if (j < 0 || !$ctx.grid.is_active(j)$) continue;
                float out = fmaxf(sedsh_outflow(qx, qy@DIAG_CALL@, i, k, j), 0.0f);
                float slope = fmaxf((zi - __ldcg(&z[j])) / $ctx.grid.dist_from_k(k)$, 0.0f);
                w[k] = out + beta * slope;
                w_sum += w[k];
            }
            int r = -1, kr = 0;
            bool escape = !(w_sum > 0.0f);
            if (!escape) {
                unsigned int draw = sedsh_hash(stream + (unsigned int)hop * 2654435769u);
                float pick = (float)(draw >> 8) * (1.0f / 16777216.0f) * w_sum;
                for (int k = 0; k < nk; ++k) {
                    if (!(w[k] > 0.0f)) continue;
                    r = $ctx.grid.neighbour(i, k)$;
                    kr = k;
                    pick -= w[k];
                    if (pick <= 0.0f) break;
                }
            } else {
                float z_low = 3.0e38f;
                for (int k = 0; k < nk; ++k) {
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j < 0 || !$ctx.grid.is_active(j)$) continue;
                    float zj = __ldcg(&z[j]);
                    if (zj < z_low) { z_low = zj; r = j; kr = k; }
                }
                if (r < 0) break;
                // Raise z[i] with a compare-and-swap, so parcels filling the
                // same pit at once add only what is still missing.
                float target = z_low + $ctx.FILL_EPS.get(0)$;
                float supply = carried * inv_area;
                float added = 0.0f;
                int* zi_bits = (int*)&z[i];
                int old = __float_as_int(__ldcg(&z[i])), assumed;
                do {
                    assumed = old;
                    float current = __int_as_float(assumed);
                    added = fminf(fmaxf(target - current, 0.0f), supply);
                    if (!(added > 0.0f)) { added = 0.0f; break; }
                    old = atomicCAS(zi_bits, assumed,
                                    __float_as_int(current + added));
                } while (assumed != old);
                if (added >= supply) {
                    counts[2] += (double)carried;
                    carried = 0.0f;
                    break;
                }
                counts[2] += (double)(added * dx * dx);
                carried -= added * dx * dx;
            }
            float cost = hop == 0 ? 0.0f
                : $ctx.grid.dist_from_k(kr)$ / fmaxf(length[i], 1.0e-12f);
            @HOP_RULE@
            i = r;
        }
        counts[0] += 1.0;
        counts[1] += (double)volume;
        if (exported) {
            counts[3] += (double)carried;
        } else {
            atomicAdd(&z[i], carried * inv_area);
            counts[2] += (double)carried;
        }
    }
    for (int c = 0; c < 4; ++c)
        if (counts[c] != 0.0) atomicAdd(&stats[c], counts[c]);
}
'''

_WALK_DIAG_OUTFLOW = r'''
    if (k == 7) return 0.5f * qxy[i];
    if (k == 0) return -0.5f * qxy[j];
    if (k == 5) return 0.5f * qyx[i];
    if (k == 2) return -0.5f * qyx[j];'''


def walk_factory(be, bundles, config):
    """Persistent parcel walk over ``threads`` threads; ``deposition``
    selects the hop rule (``stochastic`` or ``gradual``, as in
    GolemParcelSedProgram)."""
    _cupy_only(be)
    d8 = config["topology"] == "D8"
    gradual = config["deposition"] == "gradual"
    text = _substitute(
        _WALK
        .replace("@DIAG_OUTFLOW@", _WALK_DIAG_OUTFLOW if d8 else "")
        .replace("@DIAG_CALL@", ", qxy, qyx" if d8 else "")
        .replace("@HOP_INIT@", "" if gradual else _STOCHASTIC_INIT)
        .replace("@HOP_RULE@", _GRADUAL_RULE if gradual else _STOCHASTIC_RULE),
        config)
    return (KernelBuilder(text, domain=int(config["threads"]))
            .compose("grid", bundles["grid"]).freeze())
