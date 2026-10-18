"""CuPy kernels of CellularSedProgram.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.flow import make_accumulation
from pyfastflow.flow._program_cupy import _cupy_only

from ._sed_law import LAW_ARGS, PICK, law_text

_RECEIVERS = r'''
#define LOAD(x) (x)
// Receiver, slope and drop to the lowest neighbour of every active cell on
// the current surface (see _sed_law.py); outlets are their own receiver.
extern "C" __global__ void cell_receivers(
        const float* z, const float* spill, const int* rec,
        const unsigned int* step, int* recv, float* slope_out,
        float* drop_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    if (!$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$) {
        recv[i] = i;
        slope_out[i] = 0.0f;
        drop_out[i] = 0.0f;
        return;
    }
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    float dx = $ctx.grid.DX.get(0)$;
    unsigned int draw = sed_hash(sed_hash((unsigned int)$ctx.SEED.get(0)$ ^ step[0] * 2654435769u)
                                 ^ (unsigned int)i);
    float unit = (float)(draw >> 8) * (1.0f / 16777216.0f);
@PICK@
    recv[i] = next;
    slope_out[i] = slope;
    drop_out[i] = drop;
}
'''

_UPDATE = r'''
// One step of dt. The sediment flux advances one hop: a cell takes Qs from
// the cells whose receiver it is (previous step) and applies the law with
// the incoming volume Qs_in dt. Discharge: DISCHARGE_HOP, one hop as Qs
// plus the cell's rain; otherwise Q_next already holds the accumulation
// over the current receivers.
// With DEPOSITION_CAP the deposit is limited to bring z no higher than the
// Qs-weighted mean elevation of its givers; the excess leaves with Qs.
// Outlets: z = h_s = 0, they export Qs_in dt.
extern "C" __global__ void cell_update(
        const float* z, const float* hs, const float* Q, const float* Qs,
        const int* recv, const float* slope, const float* drop,
        float* z_next, float* hs_next, float* Q_next, float* Qs_next,
        float* exported) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    if (!$ctx.grid.is_active(i)$) {
        z_next[i] = z[i]; hs_next[i] = hs[i];
        Q_next[i] = 0.0f; Qs_next[i] = 0.0f;
        return;
    }
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    float dx = $ctx.grid.DX.get(0)$;
    float area = dx * dx;
    float dt = $ctx.DT.get(0)$;
    float q_in = 0.0f, qs_in = 0.0f, zw = 0.0f, w = 0.0f;
    for (int k = 0; k < nk; ++k) {
        int j = $ctx.grid.neighbour(i, k)$;
        if (j < 0 || j == i || recv[j] != i) continue;
        q_in += Q[j];
        qs_in += Qs[j];
        zw += Qs[j] * z[j];
        w += Qs[j];
    }
#if DISCHARGE_HOP
    float qn = $ctx.PRECIPITATION.get(i)$ * area + q_in;
    Q_next[i] = qn;
#else
    float qn = Q_next[i];
#endif
    if ($ctx.grid.can_out(i)$) {
        z_next[i] = 0.0f; hs_next[i] = 0.0f; Qs_next[i] = 0.0f;
        exported[i] += qs_in * dt;
        return;
    }
    float vin = qs_in * dt;
    float dz, dhs, vout, ve, vd;
    sed_law(vin, qn, slope[i], hs[i], dt, dx, drop[i], @LAW_ARGS@,
            &dz, &dhs, &vout, &ve, &vd);
#if DEPOSITION_CAP
    if (w > 0.0f && vd > 0.0f) {
        float zcap = zw / w;
        float excess = fminf(fmaxf(z[i] + dz - zcap, 0.0f), vd / area);
        dz -= excess;
        dhs -= excess;
        vout += excess * area;
    }
#endif
    z_next[i] = z[i] + dz + $ctx.UPLIFT.get(i)$ * dt;
    hs_next[i] = fmaxf(hs[i] + dhs, 0.0f);
    Qs_next[i] = vout / dt;
}
'''

_COMMIT = r'''
// Next state becomes the current one; thread 0 advances the step counter.
extern "C" __global__ void cell_commit(
        const float* z_next, const float* hs_next, const float* Q_next,
        const float* Qs_next, float* z, float* hs, float* Q, float* Qs,
        unsigned int* step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    z[i] = z_next[i];
    hs[i] = hs_next[i];
    Q[i] = Q_next[i];
    Qs[i] = Qs_next[i];
    if (i == 0) step[0] += 1u;
}
'''


def _kernel(be, bundles, text, n):
    _cupy_only(be)
    return (KernelBuilder(text, domain=n)
            .compose("grid", bundles["grid"]).freeze())


def _cells(config):
    return int(config["nx"]) * int(config["ny"])


def receivers_factory(be, bundles, config):
    text = law_text(config["law"], _RECEIVERS.replace("@PICK@", PICK))
    return _kernel(be, bundles, text, _cells(config))


def update_factory(be, bundles, config):
    """``law``, ``deposition_cap`` and ``discharge`` select the variant."""
    text = (f"#define DEPOSITION_CAP {int(bool(config['deposition_cap']))}\n"
            f"#define DISCHARGE_HOP {int(config['discharge'] == 'hop')}\n"
            + law_text(config["law"], _UPDATE.replace("@LAW_ARGS@", LAW_ARGS)))
    return _kernel(be, bundles, text, _cells(config))


def commit_factory(be, bundles, config):
    return _kernel(be, bundles, _COMMIT, _cells(config))


_SCALE = r'''
// Accumulated rain (sum of precipitation) to discharge (m3/yr).
extern "C" __global__ void cell_scale_discharge(float* Q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    float dx = $ctx.grid.DX.get(0)$;
    Q[i] = $ctx.grid.is_active(i)$ ? Q[i] * dx * dx : 0.0f;
}
'''


def scale_factory(be, bundles, config):
    return _kernel(be, bundles, _SCALE, _cells(config))


def accumulation_factory(be, bundles, config):
    """Pointer-jump accumulation of the precipitation over the receivers."""
    _cupy_only(be)
    return make_accumulation(
        be, bundles["grid"], method="pointer_jump_push",
        n_flat=_cells(config),
    )["sequence"].freeze()
