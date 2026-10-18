"""Sediment-parcel engine: release and stochastic deposition of entrained
volumes on an evolving surface.

CuPy only. The engine has no notion of time or of the processes that entrain
material. A client program binds:

  - ``volume``: f32 per cell, the volume (m3) entrained this step. Every
    entrainment term of the client lowers its reservoir (z, or a sediment
    layer) by what it takes and adds the same volume here; the client zeroes
    it before the terms run;
  - ``length``: f32 per cell, the transport length xi (m);
  - ``surface``: f32 per cell, the elevation the parcels descend and their
    deposits raise (atomic adds of volume / dx^2);
  - ``count``, ``cdf`` (f64 per cell), ``claim`` (i32, 1), ``stats``
    (f64, 4) as scratch.

``make_parcel_release`` builds the release kernel: of the material entrained
from uniformly distributed points of a cell, the fraction
(xi / dx) (1 - exp(-dx / xi)) crosses the cell edge and the rest settles back
on the cell's surface; ``volume`` is overwritten with the leaving volume and
``count`` with its parcel count ceil(leaving / PARCEL_VOLUME).
``make_parcel_walk`` builds the persistent walk of those parcels (see
``_WALK``), and ``run_parcels`` drives the two plus the prefix sum from the
host. A parcel is a ``sedp_parcel`` record (only its volume so far) created
on the fly from the prefix sum of the counts.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder, ProgramError
from pyfastflow.flow._program_cupy import _cupy_only

_MAX_PARCELS = 2**31 - 1

_RELEASE = r'''
extern "C" __global__ void sedp_release(
        float* volume, const float* length, float* surface, double* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    count[i] = 0.0;
    float v = volume[i];
    if (!(v > 0.0f) || !$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$) {
        volume[i] = 0.0f;
        return;
    }
    float dx = $ctx.grid.DX.get(0)$;
    float xi = length[i];
    float leave = xi > 0.0f ? fminf(-(xi / dx) * expm1f(-dx / xi), 1.0f) : 0.0f;
    float out = v * leave;
    surface[i] += (v - out) / (dx * dx);
    volume[i] = out;
    if (out > 0.0f)
        count[i] = ceil((double)out
                        / (double)fmaxf($ctx.PARCEL_VOLUME.get(0)$, 1.0e-12f));
}
'''

_WALK = r'''
__device__ unsigned int sedp_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct sedp_parcel {
    float volume;
};

// Walk elevation of cell c (the surface, or the surface plus a client field).
#define SEDP_ELEVATION(c) (@ELEVATION@)

// Persistent parcel walk. Threads claim parcel indices until cdf[n - 1]
// (the inclusive prefix sum of the parcel counts) is exhausted; parcel p
// starts on the cell s with cdf[s - 1] <= p < cdf[s] and carries that cell's
// leaving volume divided by its parcel count. At every hop it moves to a
// lower neighbour of the walk elevation (which deposits keep changing), by
// the hop rule of make_parcel_walk; the hop costs dist / length_i of the
// cell it leaves, except the first (the parcel has already crossed the edge
// of its source cell). The deposition rule decides what is left on the cell
// being left; deposits are atomic adds of volume / dx^2 on the surface. On a
// cell with no candidate the parcel raises the surface until the walk
// elevation is FILL_EPS above its lowest neighbour (adding only what is
// still missing when it writes) and moves there; if it carries less, it
// leaves everything. A parcel reaching an outlet exports what it carries;
// one still walking after CELL_COUNT hops leaves it where it is.
// stats: parcels, released volume, deposited volume, exported volume.
extern "C" __global__ void sedp_walk(
        const double* cdf, const float* volume, const float* length,
        float* surface, int* claim, double* stats@ARGS@) {
    int n = $ctx.CELL_COUNT.get(0)$;
    double total = cdf[n - 1];
    if (!(total > 0.0)) return;
    int parcels = (int)total;
    float dx = $ctx.grid.DX.get(0)$;
    float inv_area = 1.0f / (dx * dx);
    unsigned int key = sedp_hash((unsigned int)$ctx.SEED.get(0)$
                                 ^ sedp_hash((unsigned int)$ctx.CLOCK.get(0)$));
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
        sedp_parcel parcel;
        parcel.volume = (float)((double)volume[s] / share);
        unsigned int stream = sedp_hash(key ^ sedp_hash((unsigned int)p));
        @DEPOSIT_INIT@
        int i = s;
        float carried = parcel.volume;
        bool exported = false;
        for (int hop = 0; hop < n; ++hop) {
            if ($ctx.grid.can_out(i)$) { exported = true; break; }
            float si = SEDP_ELEVATION(i);
            int r = -1, kr = 0;
            @HOP@
            bool escape = r < 0;
            if (escape) {
                float s_low = 3.0e38f;
                for (int k = 0; k < nk; ++k) {
                    int j = $ctx.grid.neighbour(i, k)$;
                    if (j < 0 || !$ctx.grid.is_active(j)$) continue;
                    float sj = SEDP_ELEVATION(j);
                    if (sj < s_low) { s_low = sj; r = j; kr = k; }
                }
                if (r < 0) break;
                // Raise surface[i] with a compare-and-swap, so parcels
                // filling the same pit at once add only what is missing.
                float offset = si - __ldcg(&surface[i]);
                float target = s_low + $ctx.FILL_EPS.get(0)$ - offset;
                float supply = carried * inv_area;
                float added = 0.0f;
                int* bits = (int*)&surface[i];
                int old = __float_as_int(__ldcg(&surface[i])), assumed;
                do {
                    assumed = old;
                    float current = __int_as_float(assumed);
                    added = fminf(fmaxf(target - current, 0.0f), supply);
                    if (!(added > 0.0f)) { added = 0.0f; break; }
                    old = atomicCAS(bits, assumed,
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
            @DEPOSIT_RULE@
            i = r;
        }
        counts[0] += 1.0;
        counts[1] += (double)parcel.volume;
        if (exported) {
            counts[3] += (double)carried;
        } else {
            atomicAdd(&surface[i], carried * inv_area);
            counts[2] += (double)carried;
        }
    }
    for (int c = 0; c < 4; ++c)
        if (counts[c] != 0.0) atomicAdd(&stats[c], counts[c]);
}
'''

# steepest: the neighbour of steepest descent of the walk elevation.
_HOP_STEEPEST = r'''
            float best = 0.0f;
            for (int k = 0; k < nk; ++k) {
                int j = $ctx.grid.neighbour(i, k)$;
                if (j < 0 || !$ctx.grid.is_active(j)$) continue;
                float slope = (si - SEDP_ELEVATION(j)) / $ctx.grid.dist_from_k(k)$;
                if (slope > best) { best = slope; r = j; kr = k; }
            }
'''
# weighted: a neighbour drawn with probability proportional to the weight w
# set by the WEIGHT snippet (from i, k, j, si, sj and dist; non-positive
# weights are never drawn).
_HOP_WEIGHTED = r'''
            float w[8];
            float w_sum = 0.0f;
            for (int k = 0; k < nk; ++k) {
                w[k] = 0.0f;
                int j = $ctx.grid.neighbour(i, k)$;
                if (j < 0 || !$ctx.grid.is_active(j)$) continue;
                float sj = SEDP_ELEVATION(j);
                float dist = $ctx.grid.dist_from_k(k)$;
                float weight = 0.0f;
                @WEIGHT@
                if (weight > 0.0f) { w[k] = weight; w_sum += weight; }
            }
            if (w_sum > 0.0f) {
                unsigned int draw = sedp_hash(stream + (unsigned int)hop * 2654435769u);
                float pick = (float)(draw >> 8) * (1.0f / 16777216.0f) * w_sum;
                for (int k = 0; k < nk; ++k) {
                    if (!(w[k] > 0.0f)) continue;
                    r = $ctx.grid.neighbour(i, k)$;
                    kr = k;
                    pick -= w[k];
                    if (pick <= 0.0f) break;
                }
            }
'''
# Weight proportional to the descent slope of the walk elevation.
SLOPE_WEIGHT = "weight = (si - sj) / dist;"

# stochastic: an optical depth tau = -ln(u) is consumed by the hop costs and
# the whole volume drops on the cell where it runs out (on the cell reached,
# for an escape hop, so it does not pile onto the pit just filled).
_STOCHASTIC_INIT = r'''
        unsigned int tau_bits = sedp_hash(stream ^ 0x68e31da4u);
        float tau = -logf(((float)(tau_bits >> 8) + 1.0f) * (1.0f / 16777216.0f));
'''
_STOCHASTIC_RULE = r'''
            tau -= cost;
            if (tau <= 0.0f) {
                if (escape) i = r;
                break;
            }
'''
# gradual: the fraction 1 - exp(-cost) of the carried volume is left on the
# cell; once what remains would fall below MIN_VOLUME, it is all left there.
# An escape hop leaves nothing more on the pit it has just filled.
_GRADUAL_RULE = r'''
            if (!escape) {
                float drop = carried * (1.0f - expf(-cost));
                if (carried - drop < $ctx.MIN_VOLUME.get(0)$) break;
                atomicAdd(&surface[i], drop * inv_area);
                counts[2] += (double)drop;
                carried -= drop;
            }
'''


def make_parcel_release(be, grid, *, n_cells):
    """Release kernel over ``n_cells`` cells (see the module docstring).
    Binds: volume, length, surface, count, CELL_COUNT, PARCEL_VOLUME, grid."""
    _cupy_only(be)
    return (KernelBuilder(_RELEASE, domain=int(n_cells))
            .compose("grid", grid).freeze())


def make_parcel_walk(be, grid, *, threads, deposition="stochastic",
                     hop="steepest", weight=SLOPE_WEIGHT,
                     elevation="__ldcg(&surface[c])", args=""):
    """Persistent parcel walk over ``threads`` threads.

    ``deposition``: ``stochastic`` (the whole parcel drops once an optical
    depth -ln(u) is consumed by the hop costs) or ``gradual`` (the fraction
    1 - exp(-cost) is left at each hop, down to MIN_VOLUME). ``hop``:
    ``steepest`` or ``weighted`` (``weight`` is the CUDA snippet setting
    ``weight`` from i, k, j, si, sj, dist). ``elevation`` is the CUDA
    expression of the walk elevation of cell ``c``; ``args`` the extra
    kernel arguments the snippets read (", const float* water", ...).
    Binds: cdf, volume, length, surface, claim, stats, CELL_COUNT, SEED,
    CLOCK, FILL_EPS, MIN_VOLUME (gradual), grid, and ``args``.
    """
    _cupy_only(be)
    if deposition not in ("stochastic", "gradual"):
        raise ValueError(f"deposition must be stochastic or gradual, got {deposition!r}")
    if hop not in ("steepest", "weighted"):
        raise ValueError(f"hop must be steepest or weighted, got {hop!r}")
    gradual = deposition == "gradual"
    hop_text = (_HOP_STEEPEST if hop == "steepest"
                else _HOP_WEIGHTED.replace("@WEIGHT@", weight))
    text = (_WALK
            .replace("@ARGS@", args)
            .replace("@ELEVATION@", elevation)
            .replace("@HOP@", hop_text)
            .replace("@DEPOSIT_INIT@", "" if gradual else _STOCHASTIC_INIT)
            .replace("@DEPOSIT_RULE@", _GRADUAL_RULE if gradual else _STOCHASTIC_RULE))
    return (KernelBuilder(text, domain=int(threads))
            .compose("grid", grid).freeze())


def run_parcels(program, release, walk, *, volume="entrained",
                count="parcel_count", cdf="parcel_cdf", claim="parcel_claim",
                stats="parcel_stats", clock=0):
    """Release and walk the entrained volume of ``program`` (``release`` and
    ``walk`` its pipelines, the other names its data); return
    ``{"parcels", "entrained", "deposited", "exported"}`` (m3; deposited
    includes what settles back on its source cell)."""
    import cupy as cp

    volumes = program._handle(volume).array
    entrained = float(cp.sum(volumes, dtype=cp.float64).item())
    release()
    leaving = float(cp.sum(volumes, dtype=cp.float64).item())
    cdf_array = program._handle(cdf).array.ravel()
    cp.cumsum(program._handle(count).array.ravel(), out=cdf_array)
    total = float(cdf_array[-1].item())
    if total > _MAX_PARCELS:
        raise ProgramError(
            f"{total:.3g} parcels in one step exceed {_MAX_PARCELS}: "
            "raise parcel_volume")
    program._handle(stats).array.fill(0)
    program._handle(claim).array.fill(0)
    program.clock.set(int(clock))
    walk()
    raw = program._handle(stats).to_numpy()
    return {"parcels": int(raw[0]), "entrained": entrained,
            "deposited": entrained - leaving + float(raw[2]),
            "exported": float(raw[3])}
