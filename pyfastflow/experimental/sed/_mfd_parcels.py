"""CuPy kernels of MFDParcelSedProgram.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.flow._program_cupy import _cupy_only


def _cells(config):
    return int(config["nx"]) * int(config["ny"])


_UPLIFT = r'''
// Uplift of the active non-outlet cells over dt; outlets are held at z = 0
// (base level).
extern "C" __global__ void mfdp_uplift(float* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    if (!$ctx.grid.is_active(i)$) return;
    if ($ctx.grid.can_out(i)$) { z[i] = 0.0f; return; }
    z[i] += $ctx.UPLIFT.get(i)$ * fmaxf($ctx.DT.get(i)$, 0.0f);
}
'''

_BASE_LEVEL = r'''
// Outlets at z = 0 (base level).
extern "C" __global__ void mfdp_base_level(float* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    if ($ctx.grid.is_active(i)$ && $ctx.grid.can_out(i)$) z[i] = 0.0f;
}
'''

_ENTRAIN = r'''
// Entrained volume of the step, erosion_rate dt dx^2 (active non-outlet
// cells), split into ceil(volume / PARCEL_VOLUME) parcels.
extern "C" __global__ void mfdp_entrain(
        const float* erosion_rate, float* entrained, double* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    float dx = $ctx.grid.DX.get(0)$;
    float v = 0.0f;
    if ($ctx.grid.is_active(i)$ && !$ctx.grid.can_out(i)$)
        v = fmaxf(erosion_rate[i], 0.0f) * fmaxf($ctx.DT.get(i)$, 0.0f) * dx * dx;
    entrained[i] = v;
    count[i] = v > 0.0f
        ? ceil((double)v / (double)fmaxf($ctx.PARCEL_VOLUME.get(0)$, 1.0e-12f))
        : 0.0;
}
'''

_WALK = r'''
__device__ unsigned int mfdp_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Persistent parcel walk on the MFD graph of the step (dirs, weights: the
// receivers and their fractions after local-minima handling). Threads claim
// parcel indices until cdf[n - 1] (inclusive prefix sum of the parcel
// counts) is exhausted; parcel p starts on the cell s with
// cdf[s - 1] <= p < cdf[s] with the volume v0 = entrained[s] / count[s].
// At every hop it moves to
// @PICK_TEXT@
// The hop covers the travelled distance [x, x + dist] and leaves on the cell
// it departs from the volume of the deposition rule:
// @RULE_TEXT@
// A remainder that would fall below MIN_VOLUME is left whole. Deposits are
// atomic adds of volume / dx^2 on z. A parcel reaching an outlet exports
// what it carries; one on a cell without receiver, or still walking after
// CELL_COUNT hops, leaves it there.
// stats: parcels, deposited volume, exported volume.
extern "C" __global__ void mfdp_walk(
        const double* cdf, const float* entrained, const unsigned char* dirs,
        const float* weights, float* z, int* claim, double* stats) {
    int n = $ctx.CELL_COUNT.get(0)$;
    double total = cdf[n - 1];
    if (!(total > 0.0)) return;
    int parcels = (int)total;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    float dx = $ctx.grid.DX.get(0)$;
    float inv_area = 1.0f / (dx * dx);
    float min_volume = fmaxf($ctx.MIN_VOLUME.get(0)$, 0.0f);
    unsigned int key = mfdp_hash((unsigned int)$ctx.SEED.get(0)$
                                 ^ mfdp_hash((unsigned int)$ctx.CLOCK.get(0)$));

    double counts[3] = {0.0, 0.0, 0.0};
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
        float v0 = (float)((double)entrained[s] / share);
        float v = v0;
        unsigned int stream = mfdp_hash(key ^ mfdp_hash((unsigned int)p));
        @RULE_INIT@
        counts[0] += 1.0;
        float x = 0.0f;
        int i = s;
        for (int hop = 0; hop < n && v > 0.0f; ++hop) {
            if ($ctx.grid.can_out(i)$) {
                counts[2] += (double)v;
                v = 0.0f;
                break;
            }
            unsigned int draw = mfdp_hash(stream + (unsigned int)(hop + 1) * 2654435769u);
            float unit = (float)(draw >> 8) * (1.0f / 16777216.0f);
            int kc = -1;
@PICK@
            if (kc < 0) break;
            float d = $ctx.grid.dist_from_k(kc)$;
            float drop;
            @RULE@
            drop = fminf(fmaxf(drop, 0.0f), v);
            if (v - drop < min_volume) drop = v;
            if (drop > 0.0f) {
                atomicAdd(&z[i], drop * inv_area);
                counts[1] += (double)drop;
                v -= drop;
            }
            x += d;
            i = $ctx.grid.neighbour(i, kc)$;
        }
        if (v > 0.0f) {
            atomicAdd(&z[i], v * inv_area);
            counts[1] += (double)v;
        }
    }
    for (int c = 0; c < 3; ++c)
        if (counts[c] != 0.0) atomicAdd(&stats[c], counts[c]);
}
'''

# gaussian: the deposits of a parcel follow a gaussian of the travelled
# distance centred on L = TRANSPORT_DISTANCE with spread SIGMA =
# TRANSPORT_SPREAD (of the source cell), truncated to x >= 0: the hop over
# [x, x + d] leaves v0 times its share of that distribution.
_GAUSSIAN = {
    "init": r'''
        float g_mu = fmaxf($ctx.TRANSPORT_DISTANCE.get(s)$, 0.0f);
        float g_scale = 1.0f / (fmaxf($ctx.TRANSPORT_SPREAD.get(s)$, 1.0e-6f)
                                * 1.41421356f);
        float g_norm = 0.5f * (1.0f + erff(g_mu * g_scale));''',
    "rule": r'''
            drop = v0 * 0.5f * (erff((x + d - g_mu) * g_scale)
                                - erff((x - g_mu) * g_scale)) / g_norm;''',
    "text": "gaussian: v0 times the share of [x, x + dist] in a gaussian of the\n"
            "// travelled distance centred on TRANSPORT_DISTANCE with spread\n"
            "// TRANSPORT_SPREAD (source cell), truncated to x >= 0;",
}

# exponential: each hop leaves the fraction 1 - exp(-dist / L) of the carried
# volume, L = TRANSPORT_DISTANCE of the source cell.
_EXPONENTIAL = {
    "init": r'''
        float e_length = fmaxf($ctx.TRANSPORT_DISTANCE.get(s)$, 1.0e-6f);''',
    "rule": r'''
            drop = v * -expm1f(-d / e_length);''',
    "text": "exponential: the fraction 1 - exp(-dist / TRANSPORT_DISTANCE) of the\n"
            "// carried volume (TRANSPORT_DISTANCE of the source cell);",
}

_RULES = {"gaussian": _GAUSSIAN, "exponential": _EXPONENTIAL}

# graph: a receiver of the step-start MFD graph (after local-minima
# handling) drawn in proportion to its weight; deposits do not change paths.
_GRAPH_PICK = r'''
            {
                unsigned int mask = (unsigned int)dirs[i];
                float w_sum = 0.0f;
                for (int k = 0; k < nk; ++k)
                    if (mask & (1u << k)) w_sum += weights[i * nk + k];
                float pick = unit * w_sum;
                for (int k = 0; k < nk; ++k) {
                    if (!(mask & (1u << k))) continue;
                    kc = k;
                    pick -= weights[i * nk + k];
                    if (pick <= 0.0f) break;
                }
            }'''

# surface: a lower active neighbour of the current z (every deposit so far
# included, read through L2) drawn in proportion to its slope; a cell
# without lower neighbour falls back to the step-start graph.
_SURFACE_PICK = r'''
            {
                float zi = __ldcg(&z[i]);
                float w[8];
                float s_sum = 0.0f;
                for (int k = 0; k < nk; ++k) {
                    w[k] = 0.0f;
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j < 0 || !$ctx.grid.is_active(j)$) continue;
                    float slope = (zi - __ldcg(&z[j])) / $ctx.grid.dist_from_k(k)$;
                    if (slope > 0.0f) { w[k] = slope; s_sum += slope; }
                }
                float pick = unit * s_sum;
                for (int k = 0; k < nk; ++k) {
                    if (!(w[k] > 0.0f)) continue;
                    kc = k;
                    pick -= w[k];
                    if (pick <= 0.0f) break;
                }
            }
            if (kc < 0)''' + _GRAPH_PICK

_PICKS = {
    "graph": (_GRAPH_PICK,
              "a receiver of the step-start MFD graph drawn with probability\n"
              "// proportional to its weight (paths fixed during the walk)."),
    "surface": (_SURFACE_PICK,
                "a lower active neighbour of the current z (deposits included)\n"
                "// drawn with probability proportional to its slope; a cell\n"
                "// without lower neighbour takes a receiver of the step-start\n"
                "// MFD graph drawn in proportion to its weight."),
}


def _kernel(be, bundles, text, domain):
    _cupy_only(be)
    return (KernelBuilder(text, domain=domain)
            .compose("grid", bundles["grid"]).freeze())


def uplift_factory(be, bundles, config):
    return _kernel(be, bundles, _UPLIFT, _cells(config))


def base_level_factory(be, bundles, config):
    return _kernel(be, bundles, _BASE_LEVEL, _cells(config))


def entrain_factory(be, bundles, config):
    return _kernel(be, bundles, _ENTRAIN, _cells(config))


def walk_factory(be, bundles, config):
    """Persistent parcel walk over ``threads`` threads; ``deposition``
    (gaussian|exponential) selects the rule, ``parcel_path`` (graph|surface)
    the receivers."""
    rule = _RULES[config["deposition"]]
    pick, pick_text = _PICKS[config["parcel_path"]]
    text = (_WALK.replace("@PICK@", pick).replace("@PICK_TEXT@", pick_text)
            .replace("@RULE_INIT@", rule["init"])
            .replace("@RULE@", rule["rule"])
            .replace("@RULE_TEXT@", rule["text"]))
    return _kernel(be, bundles, text, int(config["threads"]))
