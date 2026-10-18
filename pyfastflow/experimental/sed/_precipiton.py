"""CuPy kernels of PrecipitonSedProgram.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.flow._program_cupy import _cupy_only

from ._sed_law import LAW_ARGS, PICK, law_text

_WALK = r'''
#define LOAD(x) __ldcg(&(x))
// Persistent precipiton walk of one launch of LAUNCH_TIME (yr). Threads
// claim precipitons until N_PRECIPITONS. Precipiton p lands on a uniformly
// drawn cell s with the water volume V = precipitation(s) LAUNCH_TIME dx^2
// cells / N_PRECIPITONS and no sediment, then for at most WALK_STEPS cells:
//   - at an outlet it exports its load and stops;
//   - it adds V to the water of the launch of the cell;
//   - it applies the law over the time it represents, dt = V / Q (capped
//     at LAUNCH_TIME; Q of the previous launches), with its load as the
//     incoming volume, on the current z of the cell (compare-and-swap:
//     evaluated again if another precipiton changed z meanwhile; the
//     neighbours are those read on arrival), adds the h_s change
//     atomically and carries the leaving volume;
//   - it moves to the receiver of the current surface (see _sed_law.py);
//     a cell that is its own receiver stops it.
// A precipiton stopped before an outlet leaves its load on its cell.
// stats: precipitons, eroded, deposited, exported (m3), stopped.
extern "C" __global__ void prec_walk(
        const int* rec, const float* spill, const float* Q, float* z,
        float* hs, float* water, int* claim, double* stats) {
    int n = $ctx.CELL_COUNT.get(0)$;
    int total = $ctx.N_PRECIPITONS.get(0)$;
    int steps = $ctx.WALK_STEPS.get(0)$;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    float dx = $ctx.grid.DX.get(0)$;
    float area = dx * dx;
    float launch = $ctx.LAUNCH_TIME.get(0)$;
    unsigned int key = sed_hash((unsigned int)$ctx.SEED.get(0)$
                                ^ sed_hash((unsigned int)$ctx.CLOCK.get(0)$));
    double counts[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    while (true) {
        int p = atomicAdd(&claim[0], 1);
        if (p >= total) break;
        counts[0] += 1.0;
        unsigned int stream = sed_hash(key ^ sed_hash((unsigned int)p));
        int s = (int)(sed_hash(stream) % (unsigned int)n);
        if (!$ctx.grid.is_active(s)$ || $ctx.grid.can_out(s)$) continue;
        {
            int i = s;
            float volume = $ctx.PRECIPITATION.get(i)$ * launch * area
                           * ((float)n / (float)total);
            float load = 0.0f;
            bool stopped = true;
            for (int hop = 0; hop < steps; ++hop) {
                if ($ctx.grid.can_out(i)$) {
                    counts[3] += (double)load;
                    load = 0.0f;
                    stopped = false;
                    break;
                }
                atomicAdd(&water[i], volume);
                unsigned int draw = sed_hash(stream + (unsigned int)(hop + 1) * 2654435769u);
                float unit = (float)(draw >> 8) * (1.0f / 16777216.0f);
@PICK@
                float qi = fmaxf(Q[i], 1.0e-12f);
                float dt = fminf(volume / qi, launch);
                float dz, dhs, vout, ve, vd;
                unsigned int* zbits = (unsigned int*)&z[i];
                unsigned int old = __float_as_uint(zi);
                while (true) {
                    float zc = __uint_as_float(old);
                    sed_law(load, qi, sed_slope(zc, amin, na, dx), LOAD(hs[i]),
                            dt, dx, fmaxf(zc - low, 0.0f), @LAW_ARGS@,
                            &dz, &dhs, &vout, &ve, &vd);
                    unsigned int seen = atomicCAS(zbits, old, __float_as_uint(zc + dz));
                    if (seen == old) break;
                    old = seen;
                }
                if (dhs != 0.0f) atomicAdd(&hs[i], dhs);
                counts[1] += (double)ve;
                counts[2] += (double)vd;
                load = vout;
                if (next == i) break;
                i = next;
            }
            if (load > 0.0f) {
                atomicAdd(&z[i], load / area);
#if SED_LAW != 0
                atomicAdd(&hs[i], load / area);
#endif
                counts[2] += (double)load;
            }
            if (stopped) counts[4] += 1.0;
        }
    }
    for (int c = 0; c < 5; ++c)
        if (counts[c] != 0.0) atomicAdd(&stats[c], counts[c]);
}
'''

_DISCHARGE = r'''
// Discharge of the launch, water / LAUNCH_TIME, floored at the cell's own
// rain precipitation dx^2, relaxed into Q with DISCHARGE_RELAXATION; resets
// the water and clips h_s at 0.
extern "C" __global__ void prec_discharge(float* Q, float* water, float* hs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    float w = water[i];
    water[i] = 0.0f;
    hs[i] = fmaxf(hs[i], 0.0f);
    if (!$ctx.grid.is_active(i)$) { Q[i] = 0.0f; return; }
    float dx = $ctx.grid.DX.get(0)$;
    float qn = fmaxf(w / $ctx.LAUNCH_TIME.get(0)$,
                     $ctx.PRECIPITATION.get(i)$ * dx * dx);
    float r = fminf(fmaxf($ctx.DISCHARGE_RELAXATION.get(0)$, 0.0f), 1.0f);
    Q[i] += r * (qn - Q[i]);
}
'''


def _kernel(be, bundles, text, n):
    _cupy_only(be)
    return (KernelBuilder(text, domain=n)
            .compose("grid", bundles["grid"]).freeze())


def walk_factory(be, bundles, config):
    """Persistent walk over ``threads`` threads; ``law`` selects the law."""
    text = law_text(config["law"], _WALK.replace("@PICK@", PICK)
                    .replace("@LAW_ARGS@", LAW_ARGS))
    return _kernel(be, bundles, text, int(config["threads"]))


def discharge_factory(be, bundles, config):
    return _kernel(be, bundles, _DISCHARGE,
                   int(config["nx"]) * int(config["ny"]))
