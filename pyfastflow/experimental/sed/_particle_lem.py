"""CuPy kernels of ParticleLEMProgram.

Author: B.G (10/2026)
"""

from pyfastflow.core import KernelBuilder
from pyfastflow.flow._program_cupy import _cupy_only


def _cells(config):
    return int(config["nx"]) * int(config["ny"])


_INIT = r'''
// Starting MFD state: every cell sends its whole drainage area to its
// steepest receiver; no sediment is sent, nothing has been visited.
extern "C" __global__ void plem_init(
        const int* rec, const float* area, float* a_send, float* qs_send,
        float* qs, double* t_last, int* locks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; ++k) {
        int j = $ctx.grid.neighbour(i, k)$;
        a_send[i * nk + k] = (j >= 0 && j == rec[i] && j != i) ? area[i] : 0.0f;
        qs_send[i * nk + k] = 0.0f;
    }
    qs[i] = 0.0f;
    t_last[i] = 0.0;
    locks[i] = 0;
}
'''

_UPLIFT = r'''
// Uplift of the active non-outlet cells over a launch; outlets are held at
// z = 0 (base level).
extern "C" __global__ void plem_uplift(float* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    if (!$ctx.grid.is_active(i)$) return;
    if ($ctx.grid.can_out(i)$) { z[i] = 0.0f; return; }
    z[i] += $ctx.UPLIFT.get(i)$ * $ctx.LAUNCH_TIME.get(0)$;
}
'''

_BASE_LEVEL = r'''
// Outlets at z = 0 (base level).
extern "C" __global__ void plem_base_level(float* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= $ctx.CELL_COUNT.get(0)$) return;
    if ($ctx.grid.is_active(i)$ && $ctx.grid.can_out(i)$) z[i] = 0.0f;
}
'''

_WALK = r'''
__device__ unsigned int plem_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Particles as processors of a landscape evolution model. a_send[c*nk + k]
// (m2) and qs_send[c*nk + k] (m3/yr) are what cell c sends to its neighbour
// k; the opposite of neighbour k is nk - 1 - k.
// Spawn: a uniformly drawn active, non-outlet cell. Each particle walks at
// most WALK_STEPS cells, stopping at an outlet.
// Visit: the particle tries to lock the cell (moving on if it is taken).
// Holding it:
//   - a cell without lower neighbour is raised FILL_EPS above its lowest
//     one, from the particle's load first (carried), the rest counted as
//     filled volume;
//   - drainage area A = dx^2 + the area sends of its higher neighbours;
//   - its time step is the time since its last visit, dt = now - t_last;
//   - stream-power incision against its steepest receiver r, implicit for
//     n = 1: z <- (z + c z_r) / (1 + c), c = K A^m dt / dist_r;
//   - transport length xi = XI A / dx; of the material entrained from
//     uniformly distributed points of the cell the fraction
//     leave = (xi / dx) (1 - exp(-dx / xi)) crosses its edge, and a flux
//     entering it passes with the fraction exp(-dx / xi); the sediment rule
//     (``sediment`` config) deposits the rest on the cell;
//   - it rewrites its sends in proportion to slope^MFD_EXPONENT towards its
//     lower neighbours.
// Move: to a lower neighbour of the current surface drawn with probability
// proportional to slope^MFD_EXPONENT. A cell left without lower neighbour
// (its lock was taken) is waited for, filled the same way, and left to its
// lowest neighbour.
// stats: particles, processed visits, locked visits, outlet arrivals,
// eroded volume, deposited volume, exported volume (carried), filled volume.
extern "C" __global__ void plem_walk(
        float* z, float* a_send, float* qs_send, float* area, float* qs,
        double* t_last, const double* now, int* locks, int* claim,
        double* stats) {
    int n = $ctx.CELL_COUNT.get(0)$;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    float dx = $ctx.grid.DX.get(0)$;
    float cell_area = dx * dx;
    double T = now[0];
    int count = $ctx.N_PARTICLES.get(0)$;
    int max_hops = $ctx.WALK_STEPS.get(0)$;
    float p_exp = $ctx.MFD_EXPONENT.get(0)$;
    float fill_eps = $ctx.FILL_EPS.get(0)$;
    unsigned int key = plem_hash((unsigned int)$ctx.SEED.get(0)$
                                 ^ plem_hash((unsigned int)$ctx.CLOCK.get(0)$));

    double counts[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    while (true) {
        int p = atomicAdd(&claim[0], 1);
        if (p >= count) break;
        unsigned int stream = plem_hash(key ^ plem_hash((unsigned int)p));
        int i = -1;
        for (int t = 0; t < 16 && i < 0; ++t) {
            int c = (int)(plem_hash(stream + (unsigned int)t * 0x9e3779b9u) % (unsigned int)n);
            if ($ctx.grid.is_active(c)$ && !$ctx.grid.can_out(c)$) i = c;
        }
        if (i < 0) continue;
        counts[0] += 1.0;
        float load = 0.0f;
        for (int hop = 0; hop < max_hops; ++hop) {
            if ($ctx.grid.can_out(i)$) {
                counts[3] += 1.0;
                counts[6] += (double)load;
                load = 0.0f;
                break;
            }
            int nb[8];
            for (int k = 0; k < nk; ++k) {
                int j = $ctx.grid.neighbour(i, k)$;
                nb[k] = (j >= 0 && $ctx.grid.is_active(j)$) ? j : -1;
            }
            if (atomicCAS(&locks[i], 0, 1) == 0) {
                counts[1] += 1.0;
                float zi = __ldcg(&z[i]);
                float zn[8];
                float low = 3.0e38f;
                for (int k = 0; k < nk; ++k) {
                    zn[k] = nb[k] >= 0 ? __ldcg(&z[nb[k]]) : 3.0e38f;
                    low = fminf(low, zn[k]);
                }
                if (low < 3.0e38f) {
                    if (!(zi > low)) {
                        float need = (low + fill_eps - zi) * cell_area;
                        float use = fminf(load, need);
                        load -= use;
                        counts[5] += (double)use;
                        counts[7] += (double)(need - use);
                        zi = low + fill_eps;
                    }
                    double a = (double)cell_area;
                    double q_in = 0.0;
                    for (int k = 0; k < nk; ++k) {
                        if (nb[k] < 0 || !(zn[k] > zi)) continue;
                        int o = nb[k] * nk + (nk - 1 - k);
                        a += (double)__ldcg(&a_send[o]);
                        q_in += (double)__ldcg(&qs_send[o]);
                    }
                    float w[8];
                    float w_sum = 0.0f, best = 0.0f;
                    int kr = -1;
                    for (int k = 0; k < nk; ++k) {
                        w[k] = 0.0f;
                        if (nb[k] < 0) continue;
                        float s = (zi - zn[k]) / $ctx.grid.dist_from_k(k)$;
                        if (!(s > 0.0f)) continue;
                        w[k] = powf(s, p_exp);
                        w_sum += w[k];
                        if (s > best) { best = s; kr = k; }
                    }
                    double dt_d = fmax(T - t_last[i], 0.0);
                    t_last[i] = T;
                    float dt = (float)dt_d;
                    float z_new = zi;
                    if (kr >= 0) {
                        float c = fmaxf($ctx.ERODIBILITY.get(i)$, 0.0f)
                                * powf((float)a, $ctx.M.get(i)$)
                                / $ctx.grid.dist_from_k(kr)$ * dt;
                        z_new = (zi + c * zn[kr]) / (1.0f + c);
                    }
                    float eroded = (zi - z_new) * cell_area;
                    float xi = fmaxf($ctx.XI.get(i)$ * (float)a / dx, 1.0e-12f);
                    float pass = expf(-dx / xi);
                    float leave = fminf(-(xi / dx) * expm1f(-dx / xi), 1.0f);
                    float deposit;
                    @SEDIMENT@
                    z[i] = z_new + deposit / cell_area;
                    area[i] = (float)a;
                    for (int k = 0; k < nk; ++k) {
                        float f = w_sum > 0.0f ? w[k] / w_sum : 0.0f;
                        a_send[i * nk + k] = (float)a * f;
                        @SEND@
                    }
                    counts[4] += (double)eroded;
                    counts[5] += (double)deposit;
                }
                __threadfence();
                atomicExch(&locks[i], 0);
            } else {
                counts[2] += 1.0;
            }

            float zi = __ldcg(&z[i]);
            float w[8];
            float w_sum = 0.0f;
            for (int k = 0; k < nk; ++k) {
                w[k] = 0.0f;
                if (nb[k] < 0) continue;
                float s = (zi - __ldcg(&z[nb[k]])) / $ctx.grid.dist_from_k(k)$;
                if (s > 0.0f) { w[k] = powf(s, p_exp); w_sum += w[k]; }
            }
            if (!(w_sum > 0.0f)) {
                // A pit that another particle held: wait for the cell, fill
                // it the same way and go on to its lowest neighbour.
                while (atomicCAS(&locks[i], 0, 1) != 0) {}
                float zc = __ldcg(&z[i]);
                float low = 3.0e38f;
                int lowest = -1;
                for (int k = 0; k < nk; ++k) {
                    if (nb[k] < 0) continue;
                    float zj = __ldcg(&z[nb[k]]);
                    if (zj < low) { low = zj; lowest = nb[k]; }
                }
                if (lowest >= 0 && !(zc > low)) {
                    float need = (low + fill_eps - zc) * cell_area;
                    float use = fminf(load, need);
                    load -= use;
                    counts[5] += (double)use;
                    counts[7] += (double)(need - use);
                    z[i] = low + fill_eps;
                }
                __threadfence();
                atomicExch(&locks[i], 0);
                if (lowest < 0) break;
                i = lowest;
                continue;
            }
            unsigned int draw = plem_hash(stream + (unsigned int)(hop + 1) * 2654435769u);
            float pick = (float)(draw >> 8) * (1.0f / 16777216.0f) * w_sum;
            int next = i;
            for (int k = 0; k < nk; ++k) {
                if (!(w[k] > 0.0f)) continue;
                next = nb[k];
                pick -= w[k];
                if (pick <= 0.0f) break;
            }
            i = next;
        }
        if (load > 0.0f) {
            // What a particle still carries when it stops stays where it is
            // (waiting for the cell's lock, so the deposit is not lost).
            while (atomicCAS(&locks[i], 0, 1) != 0) {}
            z[i] = __ldcg(&z[i]) + load / cell_area;
            __threadfence();
            atomicExch(&locks[i], 0);
            counts[5] += (double)load;
        }
    }
    for (int c = 0; c < 8; ++c)
        if (counts[c] != 0.0) atomicAdd(&stats[c], counts[c]);
}
'''

# pulled: the sediment flux is a field like the drainage area. The cell
# takes q_in (m3/yr) from the sends of its higher neighbours, adds its
# entrainment rate e = eroded / dt, passes q_out = q_in pass + e leave and
# deposits (q_in + e - q_out) dt; q_out goes out on its sends.
_PULLED = r'''
                    float e_rate = dt > 0.0f ? eroded / dt : 0.0f;
                    float q_out = (float)q_in * pass + e_rate * leave;
                    deposit = fmaxf(((float)q_in + e_rate - q_out) * dt, 0.0f);
                    qs[i] = q_out;
'''
_PULLED_SEND = "qs_send[i * nk + k] = q_out * f;"

# carried: the particle carries a volume. On every processed cell it drops
# the fraction 1 - pass of what it brings and takes the leaving fraction of
# the cell's erosion; the rest of that erosion settles back on the cell.
_CARRIED = r'''
                    deposit = load * (1.0f - pass) + eroded * (1.0f - leave);
                    load = load * pass + eroded * leave;
                    qs[i] = load;
'''


def init_factory(be, bundles, config):
    _cupy_only(be)
    return (KernelBuilder(_INIT, domain=_cells(config))
            .compose("grid", bundles["grid"]).freeze())


def uplift_factory(be, bundles, config):
    _cupy_only(be)
    return (KernelBuilder(_UPLIFT, domain=_cells(config))
            .compose("grid", bundles["grid"]).freeze())


def base_level_factory(be, bundles, config):
    _cupy_only(be)
    return (KernelBuilder(_BASE_LEVEL, domain=_cells(config))
            .compose("grid", bundles["grid"]).freeze())


def walk_factory(be, bundles, config):
    """Persistent particle walk over ``threads`` threads; ``sediment``
    selects ``pulled`` or ``carried``."""
    _cupy_only(be)
    pulled = config["sediment"] == "pulled"
    text = (_WALK
            .replace("@SEDIMENT@", _PULLED if pulled else _CARRIED)
            .replace("@SEND@", _PULLED_SEND if pulled else ""))
    return (KernelBuilder(text, domain=int(config["threads"]))
            .compose("grid", bundles["grid"]).freeze())



_SEED_RATE = r'''
// Starting passage rate of the precipiton scheme: a cell of drainage area A
// is crossed on average A / (dx^2 tau) times per year, tau = LAUNCH_TIME
// CELL_COUNT / N_PARTICLES being the mean time between two precipitons
// landing on one cell; its last passage is placed one mean interval ago.
extern "C" __global__ void plem_seed_rate(
        const float* area, float* rate, double* t_last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = $ctx.CELL_COUNT.get(0)$;
    if (i >= n) return;
    float dx = $ctx.grid.DX.get(0)$;
    double tau = (double)$ctx.LAUNCH_TIME.get(0)$ * (double)n
               / (double)max($ctx.N_PARTICLES.get(0)$, 1);
    double r = fmax((double)area[i], (double)(dx * dx)) / ((double)(dx * dx) * tau);
    rate[i] = (float)r;
    t_last[i] = -1.0 / r;
}
'''

_PRECIPITON = r'''
__device__ unsigned int plem_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ bool plem_acquire(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0) {}
    return true;
}

// Precipitons. The launch covers (now - LAUNCH_TIME, now]; precipiton p of
// N_PARTICLES lands at t_p = now - LAUNCH_TIME + (p + 0.5) / N_PARTICLES
// LAUNCH_TIME on a uniformly drawn cell (dropped if inactive) and walks at
// most WALK_STEPS cells to a lower neighbour drawn with probability
// proportional to slope^MFD_EXPONENT, stopping at an outlet (exporting its
// load). tau = LAUNCH_TIME CELL_COUNT / N_PARTICLES is the mean time between
// two landings on one cell, so a cell of drainage area A is crossed every
// dx^2 tau / A on average.
// Passage on a non-outlet cell, holding its lock (@CONTENTION@):
//   - gap = t_p - t_last; t_last moves to t_p if gap > 0, else the passage
//     is late; the incision time is dt = max(gap, 0);
//   - passage rate: @RATE_RULE@; drainage area A = dx^2 tau rate;
@PIT_TEXT@
//   - @INCISION_TEXT@
//   - transport length xi = XI A / dx: the load passes with the fraction
//     exp(-dx / xi), the fraction (xi / dx)(1 - exp(-dx / xi)) of the
//     eroded volume leaves the cell; the rest of both settles on it.
// Move: @MOVE_TEXT@
// The load left when the walk ends settles on its last cell.
// stats: particles, processed passages, locked passages, outlet arrivals,
// eroded, deposited, exported, filled volumes, late passages.
extern "C" __global__ void plem_precipiton(
        float* z, float* area, float* rate, double* t_last, const double* now,
        const int* rec, int* locks, int* claim, double* stats) {
    int n = $ctx.CELL_COUNT.get(0)$;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    float dx = $ctx.grid.DX.get(0)$;
    float cell_area = dx * dx;
    int count = $ctx.N_PARTICLES.get(0)$;
    double T = (double)$ctx.LAUNCH_TIME.get(0)$;
    double t0 = now[0] - T;
    double tau = T * (double)n / (double)max(count, 1);
    int max_hops = $ctx.WALK_STEPS.get(0)$;
    float p_exp = $ctx.MFD_EXPONENT.get(0)$;
    float fill_eps = $ctx.FILL_EPS.get(0)$;
    double ts = fmax((double)$ctx.RATE_TIMESCALE.get(0)$, 1.0e-12);
    unsigned int key = plem_hash((unsigned int)$ctx.SEED.get(0)$
                                 ^ plem_hash((unsigned int)$ctx.CLOCK.get(0)$));

    double counts[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    while (true) {
        int p = atomicAdd(&claim[0], 1);
        if (p >= count) break;
        double tp = t0 + ((double)p + 0.5) / (double)count * T;
        unsigned int stream = plem_hash(key ^ plem_hash((unsigned int)p));
        int i = (int)(plem_hash(stream) % (unsigned int)n);
        if (!$ctx.grid.is_active(i)$) continue;
        counts[0] += 1.0;
        float load = 0.0f;
        bool lm = false;
        float lm_floor = 0.0f;
@WINDOW_DECL@
        for (int hop = 0; hop < max_hops; ++hop) {
            if ($ctx.grid.can_out(i)$) {
@FLUSH@
                counts[3] += 1.0;
                counts[6] += (double)load;
                load = 0.0f;
                break;
            }
            int nb[8];
            for (int k = 0; k < nk; ++k) {
                int j = $ctx.grid.neighbour(i, k)$;
                nb[k] = (j >= 0 && $ctx.grid.is_active(j)$) ? j : -1;
            }
@LM_ARRIVE@
            bool held = @ACQUIRE@;
            if (!held) counts[2] += 1.0;
            if (held) {
                counts[1] += 1.0;
                double gap = tp - t_last[i];
                if (gap > 0.0) t_last[i] = tp;
                else counts[8] += 1.0;
                float dt = (float)fmax(gap, 0.0);
                float r = @RATE_RULE@;
                rate[i] = r;
                float a = fmaxf(cell_area * (float)tau * r, cell_area);
                area[i] = a;
                float zi = __ldcg(&z[i]);
                float zn[8];
                float low = 3.0e38f;
                for (int k = 0; k < nk; ++k) {
                    zn[k] = nb[k] >= 0 ? __ldcg(&z[nb[k]]) : 3.0e38f;
                    low = fminf(low, zn[k]);
                }
                if (!in_lm && low < 3.0e38f && !(zi > low)) {
                    zi += load / cell_area;
                    counts[5] += (double)load;
                    load = 0.0f;
                    if (!(zi > low)) {
                        counts[7] += (double)((low + fill_eps - zi) * cell_area);
                        zi = low + fill_eps;
                    }
                }
@INCISE@
                float xi = fmaxf($ctx.XI.get(i)$ * a / dx, 1.0e-12f);
                float pass = expf(-dx / xi);
                float leave = fminf(-(xi / dx) * expm1f(-dx / xi), 1.0f);
                float drop = load * (1.0f - pass) + eroded * (1.0f - leave);
                load = load * pass + eroded * leave;
                z[i] = z_new + drop / cell_area;
                counts[4] += (double)eroded;
                counts[5] += (double)drop;
                __threadfence();
                atomicExch(&locks[i], 0);
            }
@FLUSH_FULL@

@LM_MOVE@
            float zi = __ldcg(&z[i]);
            float w[8];
            float w_sum = 0.0f;
            for (int k = 0; k < nk; ++k) {
                w[k] = 0.0f;
                if (nb[k] < 0) continue;
                float s = (zi - __ldcg(&z[nb[k]])) / $ctx.grid.dist_from_k(k)$;
                if (s > 0.0f) { w[k] = powf(s, p_exp); w_sum += w[k]; }
            }
            if (!(w_sum > 0.0f)) {
@STUCK@
            }
            unsigned int draw = plem_hash(stream + (unsigned int)(hop + 1) * 2654435769u);
            float pick = (float)(draw >> 8) * (1.0f / 16777216.0f) * w_sum;
            int next = i;
            for (int k = 0; k < nk; ++k) {
                if (!(w[k] > 0.0f)) continue;
                next = nb[k];
                pick -= w[k];
                if (pick <= 0.0f) break;
            }
            i = next;
        }
@FLUSH@
        if (load > 0.0f) {
            while (atomicCAS(&locks[i], 0, 1) != 0) {}
            z[i] = __ldcg(&z[i]) + load / cell_area;
            __threadfence();
            atomicExch(&locks[i], 0);
            counts[5] += (double)load;
        }
    }
    for (int c = 0; c < 9; ++c)
        if (counts[c] != 0.0) atomicAdd(&stats[c], counts[c]);
}
'''

_ACQUIRE = {
    "wait": "plem_acquire(&locks[i])",
    "skip": "(atomicCAS(&locks[i], 0, 1) == 0)",
}
_CONTENTION_TEXT = {
    "wait": "waited for if another precipiton holds it",
    "skip": "skipped, without update, if another precipiton holds it",
}
# Incision on the way down: each passage incises its cell against the
# receiver's current elevation.
_INCISE_DOWN = r'''
                float best = 0.0f, zr = zi, dist = dx;
                for (int k = 0; k < nk; ++k) {
                    if (nb[k] < 0) continue;
                    float d = $ctx.grid.dist_from_k(k)$;
                    float s = (zi - zn[k]) / d;
                    if (s > best) { best = s; zr = zn[k]; dist = d; }
                }
                float z_new = zi;
                if (best > 0.0f && !in_lm) {
                    float c = fmaxf($ctx.ERODIBILITY.get(i)$, 0.0f)
                            * powf(a, $ctx.M.get(i)$) / dist * dt;
                    z_new = (zi + c * zr) / (1.0f + c);
                }
                float eroded = (zi - z_new) * cell_area;'''
_INCISION_DOWN_TEXT = """stream-power incision against its steepest receiver r over dt,
//     implicit for n = 1: z <- (z + c z_r) / (1 + c), c = K A^m dt / dist_r;"""

# Windowed incision: the passage only records (cell, dt, A); every WINDOW
# recorded cells, at an outlet and at the end of the walk, the window is
# incised from its downstream end up, so each cell's receiver on the path is
# processed before it, then its eroded volumes are carried down the window
# with the same transport length rule and join the load.
_WINDOW_DECL = r'''
        int wc[@WINDOW@];
        float wdt[@WINDOW@], wa[@WINDOW@], we[@WINDOW@];
        int wn = 0;'''
_INCISE_WINDOW = r'''
                float z_new = zi;
                float eroded = 0.0f;
                if (dt > 0.0f && !in_lm) {
                    wc[wn] = i;
                    wdt[wn] = dt;
                    wa[wn] = a;
                    ++wn;
                }'''
_INCISION_WINDOW_TEXT = """the cell is recorded (cell, dt, A) when dt > 0; each WINDOW = @WINDOW@
//     recorded cells, at an outlet and at the end of the walk, the recorded
//     cells are incised from the most downstream one up, waiting for each
//     lock: stream-power incision against the current steepest receiver r
//     over its dt, implicit for n = 1: z <- (z + c z_r) / (1 + c),
//     c = K A^m dt / dist_r; the eroded volumes are then carried down the
//     recorded cells with the transport length rule below (deposits under
//     the cells' locks) and what leaves the last one joins the load;"""
_FLUSH = r'''
        if (wn > 0) {
            for (int q = wn - 1; q >= 0; --q) {
                int cc = wc[q];
                while (atomicCAS(&locks[cc], 0, 1) != 0) {}
                float zc = __ldcg(&z[cc]);
                float best = 0.0f, zr = zc, dist = dx;
                for (int k = 0; k < nk; ++k) {
                    int l = $ctx.grid.neighbour(cc, k)$;
                    if (l < 0 || !$ctx.grid.is_active(l)$) continue;
                    float d = $ctx.grid.dist_from_k(k)$;
                    float s = (zc - __ldcg(&z[l])) / d;
                    if (s > best) { best = s; zr = __ldcg(&z[l]); dist = d; }
                }
                float z_new = zc;
                if (best > 0.0f) {
                    float c = fmaxf($ctx.ERODIBILITY.get(cc)$, 0.0f)
                            * powf(wa[q], $ctx.M.get(cc)$) / dist * wdt[q];
                    z_new = (zc + c * zr) / (1.0f + c);
                }
                we[q] = (zc - z_new) * cell_area;
                z[cc] = z_new;
                __threadfence();
                atomicExch(&locks[cc], 0);
                counts[4] += (double)we[q];
            }
            float carry = 0.0f;
            for (int q = 0; q < wn; ++q) {
                int cc = wc[q];
                float xi = fmaxf($ctx.XI.get(cc)$ * wa[q] / dx, 1.0e-12f);
                float pass = expf(-dx / xi);
                float leave = fminf(-(xi / dx) * expm1f(-dx / xi), 1.0f);
                float drop = carry * (1.0f - pass) + we[q] * (1.0f - leave);
                carry = carry * pass + we[q] * leave;
                if (drop > 0.0f) {
                    while (atomicCAS(&locks[cc], 0, 1) != 0) {}
                    z[cc] = __ldcg(&z[cc]) + drop / cell_area;
                    __threadfence();
                    atomicExch(&locks[cc], 0);
                    counts[5] += (double)drop;
                }
            }
            load += carry;
            wn = 0;
        }'''

# Pits. fill: a pit receives the whole load and is raised above its lowest
# neighbour. graph: the receivers of the step-start surface after local
# minima resolution (rec) route a precipiton out of a depression.
_STUCK_FILL = r'''
                while (atomicCAS(&locks[i], 0, 1) != 0) {}
                float zc = __ldcg(&z[i]);
                float low = 3.0e38f;
                int lowest = -1;
                for (int k = 0; k < nk; ++k) {
                    if (nb[k] < 0) continue;
                    float zj = __ldcg(&z[nb[k]]);
                    if (zj < low) { low = zj; lowest = nb[k]; }
                }
                if (lowest >= 0 && !(zc > low)) {
                    zc += load / cell_area;
                    counts[5] += (double)load;
                    load = 0.0f;
                    if (!(zc > low)) {
                        counts[7] += (double)((low + fill_eps - zc) * cell_area);
                        zc = low + fill_eps;
                    }
                    z[i] = zc;
                }
                __threadfence();
                atomicExch(&locks[i], 0);
                if (lowest < 0) break;
                i = lowest;
                continue;
'''
_PIT_TEXT = {
    "fill": """//   - a cell without lower neighbour (pit) receives the whole load, then is
//     raised FILL_EPS above its lowest neighbour if still needed (counted
//     as filled volume);""",
    "graph": """//   - a lake passage (see below) neither fills nor incises;""",
}
_MOVE_TEXT = {
    "fill": """as above on the updated surface; a pit met without processing it
// (late passage, skipped lock) is waited for and treated as above.""",
    "graph": """a precipiton arriving on a pit (no lower neighbour) enters lake mode
// and follows rec, the receivers of the launch-start surface after local
// minima resolution, until it reaches a cell lower than that pit (outside
// the pit's basin, every cell of which is above it); there it moves on as
// above. A cell left without lower neighbour at the move also enters lake
// mode (its elevation the floor). A cell without receiver (rec[i] == i)
// ends the walk.""",
}
_LM_ARRIVE = r'''
            bool in_lm;
            {
                float za = __ldcg(&z[i]);
                bool pit = true;
                for (int k = 0; k < nk; ++k)
                    if (nb[k] >= 0 && __ldcg(&z[nb[k]]) < za) { pit = false; break; }
                if (pit && !lm) lm_floor = za;
                in_lm = pit || (lm && !(za < lm_floor));
                lm = in_lm;
            }'''
_LM_MOVE = r'''
            if (in_lm) {
                int r = rec[i];
                if (r < 0 || r == i) break;
                i = r;
                continue;
            }'''
_STUCK_GRAPH = r'''
                if (!lm) lm_floor = zi;
                lm = true;
                int r = rec[i];
                if (r < 0 || r == i) break;
                i = r;
                continue;'''

_RATE = {
    False: ("(float)(1.0 / fmax(fabs(gap), T / (double)count))",
            "rate = 1 / max(|gap|, LAUNCH_TIME / N_PARTICLES), the interval "
            "to the nearest passage, at least one precipiton interval (A at "
            "most CELL_COUNT dx^2)"),
    True: ("(float)(gap > 0.0 ? (double)rate[i] * exp(-gap / ts) + 1.0 / ts"
           " : (double)rate[i] + exp(gap / ts) / ts)",
           "exponentially weighted passage count, rate = sum over the "
           "passages of exp(-age / RATE_TIMESCALE) / RATE_TIMESCALE (a late "
           "passage adds its decayed weight)"),
}


def seed_rate_factory(be, bundles, config):
    _cupy_only(be)
    return (KernelBuilder(_SEED_RATE, domain=_cells(config))
            .compose("grid", bundles["grid"]).freeze())


def precipiton_factory(be, bundles, config):
    """Persistent precipiton walk over ``threads`` threads; ``contention``
    (wait|skip), ``smoothing`` (bool), ``window`` (0: incision on the way
    down, else the window length) and ``pits`` (fill|graph) select the
    variants."""
    _cupy_only(be)
    rule, rule_text = _RATE[bool(config["smoothing"])]
    window = int(config["window"])
    if window > 0:
        flush = _FLUSH
        parts = {
            "@WINDOW_DECL@": _WINDOW_DECL, "@INCISE@": _INCISE_WINDOW,
            "@INCISION_TEXT@": _INCISION_WINDOW_TEXT,
            "@FLUSH_FULL@": "            if (wn == @WINDOW@) {" + flush + "\n            }",
            "@FLUSH@": flush,
        }
    else:
        parts = {
            "@WINDOW_DECL@": "", "@INCISE@": _INCISE_DOWN,
            "@INCISION_TEXT@": _INCISION_DOWN_TEXT,
            "@FLUSH_FULL@": "", "@FLUSH@": "",
        }
    graph = config["pits"] == "graph"
    parts.update({
        "@PIT_TEXT@": _PIT_TEXT[config["pits"]],
        "@MOVE_TEXT@": _MOVE_TEXT[config["pits"]],
        "@LM_ARRIVE@": _LM_ARRIVE if graph else "            const bool in_lm = false;",
        "@LM_MOVE@": _LM_MOVE if graph else "",
        "@STUCK@": _STUCK_GRAPH if graph else _STUCK_FILL,
    })
    text = _PRECIPITON
    for marker, snippet in parts.items():
        text = text.replace(marker, snippet)
    text = text.replace("@WINDOW@", str(window))
    text = (text
            .replace("@ACQUIRE@", _ACQUIRE[config["contention"]])
            .replace("@CONTENTION@", _CONTENTION_TEXT[config["contention"]])
            .replace("@RATE_RULE@;", rule_text + ";", 1)
            .replace("@RATE_RULE@", rule))
    return (KernelBuilder(text, domain=int(config["threads"]))
            .compose("grid", bundles["grid"]).freeze())
