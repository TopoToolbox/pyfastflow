"""Implicit SFD stream-power incision solved with Newton pointer jumping.

Each Newton linearization gives every active non-outlet cell one affine
downstream relation, ``z[i] = beta[i] + alpha[i] * z[rec[i]]``.  Pointer
jumping composes those relations to their outlet in logarithmically many
passes, without a grid-sized sequence of frontier launches.
"""

import math

from pyfastflow.core import KernelBuilder, SequenceBuilder
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan


def fluvial_factory(be, bundles, config):
    """Build a nonlinear implicit SFD stream-power solve.

    The number of pointer-jump passes is logarithmic in the longest possible
    receiver path. ``newton_iterations`` controls the outer nonlinear solve.
    """
    _cupy_only(be)
    n = int(config["nx"]) * int(config["ny"])
    outer_iterations = int(config["newton_iterations"])
    if outer_iterations < 1:
        raise ValueError("newton_iterations must be positive")
    rounds = math.ceil(math.log2(max(1, n)))

    prepare = KernelBuilder(
        f'''extern "C" __global__ void golem_fluvial_prepare(
                const float* z, float* z0, float* guess) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) {{ z0[i] = z[i]; guess[i] = z[i]; }}
        }}''', domain=n,
    ).freeze()
    linearize = KernelBuilder(
        f'''extern "C" __global__ void golem_fluvial_linearize(
                const float* z0, const float* guess, const int* rec,
                const float* area, float* z_a, float* z_b,
                int* rec_a, int* rec_b, float* alpha_a, float* alpha_b) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            float original = z0[i];
            if (!$ctx.grid.is_active(i)$ || $ctx.grid.can_out(i)$) {{
                z_a[i] = z_b[i] = original;
                alpha_a[i] = alpha_b[i] = 0.0f;
                rec_a[i] = rec_b[i] = i;
                return;
            }}
            int r = rec[i];
            if (r < 0 || r >= {n} || r == i || !$ctx.grid.is_active(r)$) {{
                z_a[i] = z_b[i] = original;
                alpha_a[i] = alpha_b[i] = 0.0f;
                rec_a[i] = rec_b[i] = i;
                return;
            }}
            float length = fmaxf($ctx.grid.dist_between_nodes(i, r)$, 1.0e-12f);
            float floor_slope = fmaxf($ctx.SLOPE_FLOOR.get(i)$, 0.0f);
            float slope = fmaxf((guess[i] - guess[r]) / length, floor_slope);
            float K = fmaxf($ctx.ERODIBILITY.get(i)$, 0.0f);
            float dt = fmaxf($ctx.DT.get(i)$, 0.0f);
            float contributing_area = fmaxf(area[i], 0.0f);
            float m = fmaxf($ctx.MEXP.get(i)$, 0.0f);
            float exponent = fmaxf($ctx.NEXP.get(i)$, 1.0e-4f);
            float c = K * dt * powf(contributing_area, m);
            float e = c * exponent * powf(slope, exponent - 1.0f) / length;
            float denominator = 1.0f + e;
            float alpha = (isfinite(denominator) && denominator > 0.0f)
                ? e / denominator : 0.0f;
            float beta = (isfinite(denominator) && denominator > 0.0f)
                ? (original + c * (exponent - 1.0f) * powf(slope, exponent)) / denominator
                : original;
            z_a[i] = z_b[i] = fminf(original, beta);
            alpha_a[i] = alpha_b[i] = fmaxf(alpha, 0.0f);
            rec_a[i] = rec_b[i] = r;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    def jump(name):
        return KernelBuilder(
            f'''extern "C" __global__ void golem_fluvial_{name}(
                    const int* rec_original, const int* rec_current,
                    int* rec_next, const float* z_current, float* z_next,
                    const float* alpha_current, float* alpha_next) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= {n}) return;
                if (!$ctx.grid.is_active(i)$) {{
                    z_next[i] = z_current[i];
                    alpha_next[i] = alpha_current[i];
                    rec_next[i] = rec_current[i];
                    return;
                }}
                int rj = rec_current[i];
                int valid_jump = rj >= 0 && rj < {n} && rj != i
                    && $ctx.grid.is_active(rj)$;
                float value = z_current[i];
                float alpha = alpha_current[i];
                int next = i;
                if (valid_jump) {{
                    value += alpha * z_current[rj];
                    alpha *= alpha_current[rj];
                    int rg = rec_current[rj];
                    next = (rg >= 0 && rg < {n} && rg != rj
                            && $ctx.grid.is_active(rg)$) ? rg : rj;
                }} else {{
                    alpha = 0.0f;
                }}
                int r0 = rec_original[i];
                int valid_original = !$ctx.grid.can_out(i)$
                    && r0 >= 0 && r0 < {n} && r0 != i
                    && $ctx.grid.is_active(r0)$;
                if (valid_original) {{
                    float lower = z_current[r0]
                        + fmaxf($ctx.SLOPE_FLOOR.get(i)$, 0.0f)
                        * $ctx.grid.dist_between_nodes(i, r0)$;
                    value = fmaxf(value, lower);
                }}
                z_next[i] = value;
                alpha_next[i] = alpha;
                rec_next[i] = next;
            }}''', domain=n,
        ).compose("grid", bundles["grid"]).freeze()

    jump_a, jump_b = jump("jump_a"), jump("jump_b")
    copy_a = KernelBuilder(
        f'''extern "C" __global__ void golem_fluvial_copy_a(
                const float* z_a, float* guess) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) guess[i] = z_a[i];
        }}''', domain=n,
    ).freeze()
    copy_b = KernelBuilder(
        f'''extern "C" __global__ void golem_fluvial_copy_b(
                const float* z_b, float* guess) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < {n}) guess[i] = z_b[i];
        }}''', domain=n,
    ).freeze()
    finish = KernelBuilder(
        f'''extern "C" __global__ void golem_fluvial_finish(
                const float* z0, const float* guess, float* z,
                float* erosion_rate) {{
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= {n}) return;
            if (!$ctx.grid.is_active(i)$) {{ erosion_rate[i] = 0.0f; return; }}
            z[i] = guess[i];
            float dt = $ctx.DT.get(i)$;
            erosion_rate[i] = dt > 0.0f
                ? fmaxf((z0[i] - guess[i]) / dt, 0.0f) : 0.0f;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    sequence = SequenceBuilder()
    for name, node in (("prepare", prepare), ("linearize", linearize),
                       ("finish", finish)):
        sequence.add(name, node)
    if rounds >= 1:
        sequence.add("jump_a", jump_a)
    if rounds >= 2:
        sequence.add("jump_b", jump_b)
    sequence.add("copy_b" if rounds % 2 else "copy_a",
                 copy_b if rounds % 2 else copy_a)
    sequence.step("prepare")
    for _ in range(outer_iterations):
        sequence.step("linearize")
        # Alternate the two work buffers while composing affine downstream
        # relations. Each pass doubles the represented receiver-path length.
        for _ in range(rounds // 2):
            sequence.step("jump_a").step("jump_b")
        if rounds % 2:
            sequence.step("jump_a").step("copy_b")
        else:
            sequence.step("copy_a")
    sequence.step("finish")
    return sequence.freeze()


def fluvial_plan(frozen, _be):
    plan = _grid_leaf_plan(frozen, {
        "z": "z", "z0": "fluvial_z0", "guess": "fluvial_guess",
        "rec": "rec", "rec_original": "rec", "area": "drainage_area",
        "z_a": "fluvial_z_a", "z_b": "fluvial_z_b",
        "rec_a": "fluvial_rec_a", "rec_b": "fluvial_rec_b",
        "alpha_a": "fluvial_alpha_a", "alpha_b": "fluvial_alpha_b",
        "rec_current": "fluvial_rec_a", "rec_next": "fluvial_rec_b",
        "z_current": "fluvial_z_a", "z_next": "fluvial_z_b",
        "alpha_current": "fluvial_alpha_a", "alpha_next": "fluvial_alpha_b",
        "ERODIBILITY": "erodibility", "DT": "dt", "MEXP": "m", "NEXP": "n",
        "SLOPE_FLOOR": "slope_floor", "erosion_rate": "erosion_rate",
    })
    bound = frozen.build()
    try:
        addresses = set(bound.addresses())
    finally:
        bound.close()
    if ("jump_b", "rec_current") in addresses:
        plan.update({
            "jump_b.rec_current": "fluvial_rec_b", "jump_b.rec_next": "fluvial_rec_a",
            "jump_b.z_current": "fluvial_z_b", "jump_b.z_next": "fluvial_z_a",
            "jump_b.alpha_current": "fluvial_alpha_b", "jump_b.alpha_next": "fluvial_alpha_a",
        })
    return plan
