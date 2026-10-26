"""Implicit SFD stream-power incision.

Two solvers of ``z - z0 + K dt A^m ((z - z_r) / L)^n = 0`` on an SFD
receiver forest:

- :func:`fluvial_factory`, Newton pointer jumping. Each Newton linearization
  gives every active non-outlet cell one affine downstream relation,
  ``z[i] = beta[i] + alpha[i] * z[rec[i]]``.  Pointer jumping composes those
  relations to their outlet in logarithmically many passes, without a
  grid-sized sequence of frontier launches. The outer Newton loop runs a fixed
  ``newton_iterations`` count, or, when ``newton_tolerance`` is configured,
  until the largest change of the whole field between two outer iterations is
  at most ``NEWTON_TOL`` (``newton_iterations`` is then the cap).
- :func:`persistent_fluvial_factory`, outlet-to-source persistent kernel. A
  work queue is seeded with the receiver-forest roots; a resident worker pool
  solves a node's own scalar Newton problem once its receiver is final (local
  tolerance ``NEWTON_TOL``, at most ``newton_iterations`` steps), then queues
  its donors, found through grid neighbours.
"""

import math

from pyfastflow.core import HostBlockBuilder, KernelBuilder, SequenceBuilder
from pyfastflow.flow._program_cupy import _cupy_only, _grid_leaf_plan


def fluvial_factory(be, bundles, config):
    """Build a nonlinear implicit SFD stream-power solve.

    The number of pointer-jump passes is logarithmic in the longest possible
    receiver path. ``newton_iterations`` controls the outer nonlinear solve:
    a fixed count, or the cap of a convergence loop when
    ``config.get("newton_tolerance")`` is not None. The convergence loop reads
    one changed flag (``CHANGED``) back on the host per outer iteration.
    """
    _cupy_only(be)
    n = int(config["nx"]) * int(config["ny"])
    outer_iterations = int(config["newton_iterations"])
    if outer_iterations < 1:
        raise ValueError("newton_iterations must be positive")
    converge = config.get("newton_tolerance") is not None
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
            z_a[i] = z_b[i] = beta;
            alpha_a[i] = alpha_b[i] = fmaxf(alpha, 0.0f);
            rec_a[i] = rec_b[i] = r;
        }}''', domain=n,
    ).compose("grid", bundles["grid"]).freeze()

    def jump(name):
        return KernelBuilder(
            f'''extern "C" __global__ void golem_fluvial_{name}(
                    const int* rec_current,
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
    def copy_check(name, src):
        return KernelBuilder(
            f'''extern "C" __global__ void golem_fluvial_{name}(
                    const float* {src}, float* guess, int* changed) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= {n}) return;
                float delta = fabsf({src}[i] - guess[i]);
                guess[i] = {src}[i];
                if (delta > $ctx.NEWTON_TOL.get(0)$) *changed = 1;
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
    if converge:
        def zero_changed(ctx):
            ctx.CHANGED.set(0)

        def converged(ctx):
            return int(ctx.CHANGED.read()) == 0

        check_name, check_src = ("check_b", "z_b") if rounds % 2 else ("check_a", "z_a")
        sequence.add(check_name, copy_check(check_name, check_src))
        sequence.add("zero_changed", HostBlockBuilder(zero_changed).freeze())
        sequence.add("converged", HostBlockBuilder(converged).freeze())
        # Same buffer alternation as the fixed-count path below.
        body = ["zero_changed", "linearize"] + ["jump_a", "jump_b"] * (rounds // 2)
        body += ["jump_a", check_name] if rounds % 2 else [check_name]
        sequence.loop(body=body, max_times=outer_iterations, until="converged")
    else:
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
        "rec": "rec", "area": "drainage_area",
        "z_a": "fluvial_z_a", "z_b": "fluvial_z_b",
        "rec_a": "fluvial_rec_a", "rec_b": "fluvial_rec_b",
        "alpha_a": "fluvial_alpha_a", "alpha_b": "fluvial_alpha_b",
        "rec_current": "fluvial_rec_a", "rec_next": "fluvial_rec_b",
        "z_current": "fluvial_z_a", "z_next": "fluvial_z_b",
        "alpha_current": "fluvial_alpha_a", "alpha_next": "fluvial_alpha_b",
        "ERODIBILITY": "erodibility", "DT": "dt", "MEXP": "m", "NEXP": "n",
        "SLOPE_FLOOR": "slope_floor", "erosion_rate": "erosion_rate",
        "NEWTON_TOL": "newton_tolerance", "CHANGED": "fluvial_changed",
        "changed": "fluvial_changed.handle",
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


def persistent_fluvial_factory(be, bundles, config):
    """Build the outlet-to-source persistent implicit SFD stream-power solve.

    ``reset`` clears the queue, ``prepare`` copies ``z`` to ``z0`` and queues
    every active root (outlet, or no valid active receiver), ``solve`` is the
    resident worker pool. A queued node is solved from its receiver's final
    elevation by scalar Newton on ``f(h) = h - z0 + K dt A^m ((h - z_r) / L)^n``,
    started at ``z0`` and kept at or above ``z_r`` (no erosion when
    ``z0 <= z_r``); it then queues its donors. ``finish`` writes
    ``erosion_rate``.
    """
    _cupy_only(be)
    n = int(config["nx"]) * int(config["ny"])
    nk = 8 if config.get("topology", "D8") == "D8" else 4
    max_newton = int(config["newton_iterations"])
    if max_newton < 1:
        raise ValueError("newton_iterations must be positive")
    workers, threads = 4, 256
    resident = workers * threads
    grid = bundles["grid"]

    reset = KernelBuilder(f'''
extern "C" __global__ void golem_pfluvial_reset(int* queue, int* count) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {n}) queue[i] = -1;
    if (i < 4) count[i] = 0;
}}''', domain=max(n, 4)).freeze()

    prepare = KernelBuilder(f'''
extern "C" __global__ void golem_pfluvial_prepare(
        const int* rec, const float* z, float* z0, int* queue, int* count) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    z0[i] = z[i];
    if (!$ctx.grid.is_active(i)$) return;
    atomicAdd(&count[3], 1);
    int r = rec[i];
    if ($ctx.grid.can_out(i)$ || r < 0 || r >= {n} || r == i || !$ctx.grid.is_active(r)$) {{
        int pos = atomicAdd(&count[1], 1);
        queue[pos] = i;
    }}
}}''', domain=n).compose("grid", grid).freeze()

    solve = KernelBuilder(f'''
extern "C" __global__ void golem_pfluvial_solve(
        const int* rec, const float* area, const float* z0, float* z,
        int* queue, int* count) {{
    int lane = threadIdx.x & 31;
    while (true) {{
        int base = -1;
        int batch = 0;
        if (lane == 0) while (base < 0) {{
            volatile int* counters = (volatile int*)count;
            if (counters[2] >= counters[3]) break;
            int head = counters[0];
            int available = counters[1] - head;
            if (available > 0) {{
                int take = min(available, 32);
                if (atomicCAS(&count[0], head, head + take) == head) {{
                    base = head;
                    batch = take;
                }}
            }} else {{
#if __CUDA_ARCH__ >= 700
                __nanosleep(256);
#endif
            }}
        }}
        base = __shfl_sync(0xffffffffu, base, 0);
        batch = __shfl_sync(0xffffffffu, batch, 0);
        if (base < 0) return;
        if (lane < batch) {{
        int slot = base + lane;
        volatile int* ready = (volatile int*)queue;
        int i;
        while ((i = ready[slot]) < 0) {{
#if __CUDA_ARCH__ >= 700
            __nanosleep(256);
#endif
        }}
        int r = rec[i];
        if (!$ctx.grid.can_out(i)$ && r >= 0 && r < {n} && r != i && $ctx.grid.is_active(r)$) {{
            float hr = ((volatile float*)z)[r];
            float h0 = z0[i];
            float h = h0;
            if (h0 > hr) {{
                float length = fmaxf($ctx.grid.dist_between_nodes(i, r)$, 1.0e-12f);
                float exponent = fmaxf($ctx.NEXP.get(i)$, 1.0e-4f);
                float c = fmaxf($ctx.ERODIBILITY.get(i)$, 0.0f) * fmaxf($ctx.DT.get(i)$, 0.0f)
                        * powf(fmaxf(area[i], 0.0f), fmaxf($ctx.MEXP.get(i)$, 0.0f))
                        / powf(length, exponent);
                float tol = $ctx.NEWTON_TOL.get(0)$;
                for (int it = 0; it < {max_newton}; ++it) {{
                    float d = h - hr;
                    float f = h - h0 + c * powf(d, exponent);
                    float fp = 1.0f + c * exponent * powf(d, exponent - 1.0f);
                    float next = fmaxf(h - f / fp, hr);
                    float step = fabsf(next - h);
                    h = next;
                    if (step <= tol) break;
                }}
            }}
            z[i] = h;
        }}
        __threadfence();
        #pragma unroll
        for (int k = 0; k < {nk}; ++k) {{
            int j = $ctx.grid.neighbour(i, k)$;
            if (j < 0 || j == i || rec[j] != i) continue;
            bool seen = false;
            for (int earlier = 0; earlier < k; ++earlier)
                if ($ctx.grid.neighbour(i, earlier)$ == j) seen = true;
            if (seen) continue;
            int pos = atomicAdd(&count[1], 1);
            queue[pos] = j;
        }}
        __threadfence();
        }}
        __syncwarp();
        if (lane == 0) atomicAdd(&count[2], batch);
    }}
}}''', domain=resident, block=threads).compose("grid", grid).freeze()

    finish = KernelBuilder(f'''
extern "C" __global__ void golem_pfluvial_finish(
        const float* z0, const float* z, float* erosion_rate) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n}) return;
    if (!$ctx.grid.is_active(i)$) {{ erosion_rate[i] = 0.0f; return; }}
    float dt = $ctx.DT.get(i)$;
    erosion_rate[i] = dt > 0.0f ? fmaxf((z0[i] - z[i]) / dt, 0.0f) : 0.0f;
}}''', domain=n).compose("grid", grid).freeze()

    sequence = SequenceBuilder()
    for name, node in (("reset", reset), ("prepare", prepare), ("solve", solve), ("finish", finish)):
        sequence.add(name, node).step(name)
    return sequence.freeze()


def persistent_fluvial_plan(frozen, _be):
    return _grid_leaf_plan(frozen, {
        "z": "z", "z0": "fluvial_z0", "rec": "rec", "area": "drainage_area",
        "queue": "fluvial_queue", "count": "fluvial_count",
        "ERODIBILITY": "erodibility", "DT": "dt", "MEXP": "m", "NEXP": "n",
        "NEWTON_TOL": "newton_tolerance", "erosion_rate": "erosion_rate",
    })
