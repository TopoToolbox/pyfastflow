"""Local erosion/deposition law and receiver rule shared by
CellularSedProgram and PrecipitonSedProgram (CUDA text).

The surface is z = bedrock + h_s (sediment layer h_s). Over a time dt a cell
of discharge Q (m3/yr), slope S and cover h_s receives the sediment volume
vin (m3). With q = Q / dx, the transport length xi = q / v_s (Davy and Lague
2009), the stream power p = q^m S^n and the eroded volume V_e:

  - of vin, the fraction exp(-dx / xi) crosses the cell, the rest deposits;
  - of V_e (entrained from uniformly distributed points of the cell), the
    fraction (xi / dx)(1 - exp(-dx / xi)) leaves it, the rest deposits;

and the erosion depths over dt follow ``law``:

  - ``davy_lague``: one layer, bedrock detachment e_b = K_b p dt; deposits
    go to z, h_s stays 0;
  - ``cover``: cover fraction x_s = max(min(qs / qs_inf, 1),
    1 - exp(-h_s / H*)) with qs = vin / (dt dx) and the transport capacity
    qs_inf = xi K_s p (h_s > 0 counts as covered when H* = 0); lift of the
    sediment layer e_s = min(x_s K_s p dt, h_s), bedrock detachment
    e_b = (1 - x_s) K_b p dt; deposits go to h_s;
  - ``lift_impact`` (Davy et al. 2026): ``cover`` plus the impact of the
    depositing grains on the exposed bedrock, e_b += (1 - x_s) alpha d_in,
    d_in = vin (1 - exp(-dx / xi)) / dx^2.

e_b + e_s is capped at ``drop`` (the drop to the lowest neighbour), so
erosion does not open a pit.

Receivers (``PICK``): a lower neighbour j of the current z with a spill
level (Cordonnier path maximum of the last routing) strictly below the
cell's, drawn with probability proportional to its slope; without one, the
receiver ``rec`` of the last routing. Every edge lowers the spill level or
follows ``rec``, which never raises it and is acyclic, so the receivers are
acyclic. ``slope`` is the upwind gradient on the cardinal axes,
sqrt(sum over axes of max(z_i - z_a, z_i - z_b, 0)^2) / dx.

Author: B.G (10/2026)
"""

LAWS = ("davy_lague", "cover", "lift_impact")

LAW_DEVICE = r'''
__device__ unsigned int sed_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Upwind slope of a cell at elevation zi whose cardinal axes have the
// lowest neighbours amin[0..na).
__device__ float sed_slope(float zi, const float* amin, int na, float dx) {
    float g2 = 0.0f;
    for (int a = 0; a < na; ++a) {
        float d = fmaxf(zi - amin[a], 0.0f);
        g2 += d * d;
    }
    return sqrtf(g2) / dx;
}

// Local erosion/deposition over dt (see _sed_law.py). Returns the change of
// z and h_s (m), the leaving volume vout (m3), the eroded volume and the
// deposited volume (m3).
__device__ void sed_law(
        float vin, float Q, float S, float hs, float dt, float dx, float drop,
        float kb, float ks, float m, float n, float vs, float hstar,
        float alpha, float* dz, float* dhs, float* vout, float* eroded,
        float* deposited) {
    float area = dx * dx;
    float q = fmaxf(Q, 1.0e-12f) / dx;
    float xi = fmaxf(q / fmaxf(vs, 1.0e-12f), 1.0e-12f);
    float pass = expf(-dx / xi);
    float leave = fminf(-(xi / dx) * expm1f(-dx / xi), 1.0f);
    float p = S > 0.0f ? powf(q, m) * powf(S, n) : 0.0f;
    float dep_in = vin * (1.0f - pass);
    float eb, es = 0.0f;
#if SED_LAW == 0
    eb = kb * p * dt;
#else
    float qs = vin / (fmaxf(dt, 1.0e-12f) * dx);
    float qs_inf = xi * ks * p;
    float xs = qs_inf > 0.0f ? fminf(qs / qs_inf, 1.0f) : 1.0f;
    if (hstar > 0.0f) xs = fmaxf(xs, -expm1f(-hs / hstar));
    else if (hs > 0.0f) xs = 1.0f;
    es = fminf(xs * ks * p * dt, hs);
    eb = (1.0f - xs) * kb * p * dt;
#if SED_LAW == 2
    eb += (1.0f - xs) * alpha * dep_in / area;
#endif
#endif
    float etot = eb + es;
    float cap = fmaxf(drop, 0.0f);
    if (etot > cap) {
        float f = cap / etot;
        eb *= f;
        es *= f;
        etot = cap;
    }
    float ve = etot * area;
    float dep = dep_in + ve * (1.0f - leave);
    *vout = vin * pass + ve * leave;
    *dz = dep / area - etot;
#if SED_LAW == 0
    *dhs = 0.0f;
#else
    *dhs = dep / area - es;
#endif
    *eroded = ve;
    *deposited = dep;
}
'''

# Receiver, slope and drop of cell i. Needs: i, nk, dx, z (read with
# LOAD), spill, rec, unit in [0, 1). Defines zi, next, slope, drop and, to
# evaluate them again for another zi, low (lowest neighbour) and amin/na
# (lowest neighbour on each cardinal axis).
PICK = r'''
            float zi = LOAD(z[i]);
            float si = spill[i];
            float w[8];
            float w_sum = 0.0f;
            float low = 3.0e38f;
            for (int k = 0; k < nk; ++k) {
                w[k] = 0.0f;
                int j = $ctx.grid.neighbour(i, k)$;
                if (j < 0 || !$ctx.grid.is_active(j)$) continue;
                float zj = LOAD(z[j]);
                low = fminf(low, zj);
                if (zj < zi && spill[j] < si) {
                    w[k] = (zi - zj) / $ctx.grid.dist_from_k(k)$;
                    w_sum += w[k];
                }
            }
            float amin[2] = {3.0e38f, 3.0e38f};
            int na = 0;
            for (int k = 0; k < nk / 2 && na < 2; ++k) {
                if ($ctx.grid.dist_from_k(k)$ > 1.001f * dx) continue;
                int j1 = $ctx.grid.neighbour(i, k)$;
                int j2 = $ctx.grid.neighbour(i, nk - 1 - k)$;
                if (j1 >= 0 && $ctx.grid.is_active(j1)$) amin[na] = fminf(amin[na], LOAD(z[j1]));
                if (j2 >= 0 && $ctx.grid.is_active(j2)$) amin[na] = fminf(amin[na], LOAD(z[j2]));
                ++na;
            }
            float slope = sed_slope(zi, amin, na, dx);
            float drop = fmaxf(zi - low, 0.0f);
            int next = rec[i];
            if (w_sum > 0.0f) {
                float pick = unit * w_sum;
                for (int k = 0; k < nk; ++k) {
                    if (!(w[k] > 0.0f)) continue;
                    next = $ctx.grid.neighbour(i, k)$;
                    pick -= w[k];
                    if (pick <= 0.0f) break;
                }
            }
'''

# Law parameters read at cell i, as the arguments after ``drop`` of sed_law.
LAW_ARGS = (
    "$ctx.ERODIBILITY.get(i)$, $ctx.SEDIMENT_ERODIBILITY.get(i)$, "
    "$ctx.M.get(i)$, $ctx.N.get(i)$, $ctx.SETTLING_VELOCITY.get(i)$, "
    "$ctx.COVER_DEPTH.get(i)$, $ctx.IMPACT.get(i)$"
)

LAW_BIND = {
    "ERODIBILITY": "erodibility", "SEDIMENT_ERODIBILITY": "sediment_erodibility",
    "M": "m", "N": "n", "SETTLING_VELOCITY": "settling_velocity",
    "COVER_DEPTH": "cover_depth", "IMPACT": "impact",
}


def law_text(law, text):
    """``text`` preceded by the law selection and LAW_DEVICE."""
    return f"#define SED_LAW {LAWS.index(law)}\n" + LAW_DEVICE + text


def law_params(b, shape):
    """Declare the law parameters on ProgramBuilder ``b``."""
    b.param("erodibility", "auto", "f32", value=0.0, shape=shape)
    b.param("sediment_erodibility", "auto", "f32", value=0.0, shape=shape)
    b.param("m", "auto", "f32", value=0.45, shape=shape)
    b.param("n", "auto", "f32", value=1.0, shape=shape)
    b.param("settling_velocity", "auto", "f32", value=1.0, shape=shape)
    b.param("cover_depth", "auto", "f32", value=0.0, shape=shape)
    b.param("impact", "auto", "f32", value=0.0, shape=shape)
