"""
cupy (CUDA source) block templates behind make_fill_reconstruct/
make_fill_reconstruct_solver - grayscale morphological reconstruction
against elevation, ported from
experimental/LM/fill_reconstruct_optimised.py's Round 4/5.

Split out of a single _cupy_blocks.py that used to hold every flow algorithm
- see _cupy_receivers.py/_cupy_accum.py/_cupy_depressions.py for the others.
See that script's module docstring for the full derivation of the
direct-elevation-space formulation (filled[i] = max(z[i], min over
neighbours filled), decreasing from a +inf interior sentinel to a fixed
point) and of every optimisation below (queued_gen dedup instead of a
per-pass reset, gated pushes instead of unconditional ones, four directional
sweeps seeding the frontier, counters[] replacing two scalars reset every
pass).

Framework-specific departure from that script: frontier_a/frontier_b are
not two separate n_flat buffers here. A Sequence's kernel_step binds its
data handles once at compile time (see sequence.py's contract) - it cannot
re-select "the other buffer" between loop iterations the way the script's
own host loop did (`frontier_bufs[p % 2]`). One buffer, "frontier", shape
(2*n_flat,), replaces the pair: `base = (p % 2) * n_flat` selects the input
half, `(1 - p % 2) * n_flat` the output half, both computed inside the
kernel from the bound `P` Parameter - ordinary runtime pointer arithmetic
into one already-bound array, no rebinding needed. `p` itself is `P`, a
caller-allocated scalar i32 Parameter (mode "scalar") bumped by a host_step
between passes, exactly the role `iteration_p` plays for rake_compress
(_cupy_accum.py's build_ping_pong_helpers/build_rake_compress) - required,
not built here, since this factory takes no pool.

atomicExch's dedup ("first writer to claim queued_gen[j] this pass wins")
is unchanged from the script - CUDA has it natively.

Author: B.G (07/2026)
"""

from ..core.context.need import Kind, Need
from ..core.pool.base import new_uid

_POS_SENTINEL = 1.0e9


def build_fill_reconstruct_init(KernelCls, *, grid, n_flat: int):
    """
    init_filled KernelBuilder, data args (z, filled, parent): on a can_out
    node, filled[i] = z[i] and parent[i] = i (self-receiving, the base-level
    convention); elsewhere filled[i] = +inf sentinel, parent[i] = -1 (never
    yet claimed) - the seed state every sweep/relax pass decreases from.

    Author: B.G (07/2026)
    """
    t = f"pfi{new_uid()}"
    return KernelCls().bind("grid", grid).ingest(
        f"""
__global__ void {t}_init_filled(const float* z, float* filled, int* parent) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if ($grid.can_out(i)$) {{
        filled[i] = z[i];
        parent[i] = i;
    }} else {{
        filled[i] = {_POS_SENTINEL}f;
        parent[i] = -1;
    }}
}}
"""
    )


def build_fill_reconstruct_sweeps(KernelCls, *, nx: int, ny: int):
    """
    Four KernelBuilders, each data args (z, filled, parent): one raster sweep
    per direction (row left-to-right, row right-to-left, column top-to-bottom,
    column bottom-to-top), one thread per row/column walking it serially - no
    atomics needed, since distinct rows/columns never touch the same cell.
    Seeds the frontier before any relax pass runs; see the module docstring
    for why this can never produce a wrong value or reintroduce a cycle.

    Keyed "row_lr", "row_rl", "col_tb", "col_bt" by the caller.

    Author: B.G (07/2026)
    """
    t = f"pfs{new_uid()}"

    row_lr = KernelCls().ingest(
        f"""
__global__ void {t}_sweep_row_lr(const float* z, float* filled, int* parent) {{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= {ny}) return;
    int base = r * {nx};
    for (int c = 1; c < {nx}; c++) {{
        int i = base + c;
        int left = i - 1;
        float cand = z[i] > filled[left] ? z[i] : filled[left];
        if (cand < filled[i]) {{
            filled[i] = cand;
            parent[i] = left;
        }}
    }}
}}
"""
    )
    row_rl = KernelCls().ingest(
        f"""
__global__ void {t}_sweep_row_rl(const float* z, float* filled, int* parent) {{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= {ny}) return;
    int base = r * {nx};
    for (int c = {nx} - 2; c >= 0; c--) {{
        int i = base + c;
        int right = i + 1;
        float cand = z[i] > filled[right] ? z[i] : filled[right];
        if (cand < filled[i]) {{
            filled[i] = cand;
            parent[i] = right;
        }}
    }}
}}
"""
    )
    col_tb = KernelCls().ingest(
        f"""
__global__ void {t}_sweep_col_tb(const float* z, float* filled, int* parent) {{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= {nx}) return;
    for (int r = 1; r < {ny}; r++) {{
        int i = r * {nx} + c;
        int up = i - {nx};
        float cand = z[i] > filled[up] ? z[i] : filled[up];
        if (cand < filled[i]) {{
            filled[i] = cand;
            parent[i] = up;
        }}
    }}
}}
"""
    )
    col_bt = KernelCls().ingest(
        f"""
__global__ void {t}_sweep_col_bt(const float* z, float* filled, int* parent) {{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= {nx}) return;
    for (int r = {ny} - 2; r >= 0; r--) {{
        int i = r * {nx} + c;
        int down = i + {nx};
        float cand = z[i] > filled[down] ? z[i] : filled[down];
        if (cand < filled[i]) {{
            filled[i] = cand;
            parent[i] = down;
        }}
    }}
}}
"""
    )
    return {"row_lr": row_lr, "row_rl": row_rl, "col_tb": col_tb, "col_bt": col_bt}


def build_fill_reconstruct_frontier_init(KernelCls, *, n_flat: int):
    """
    frontier_init KernelBuilder, data args (z, filled, frontier, counters):
    every cell not yet sealed after the sweeps (filled[i] > z[i]) is pushed
    into `frontier`'s first half (indices [0, n_flat)) and counted into
    `counters[0]` - the seed frontier the relax loop's pass 0 reads.

    `frontier` is the combined (2*n_flat,) buffer - see this section's
    module note; pass 0 always reads the [0, n_flat) half, so this writes
    there unconditionally rather than computing a parity.

    Author: B.G (07/2026)
    """
    t = f"pff{new_uid()}"
    return KernelCls().ingest(
        f"""
__global__ void {t}_frontier_init(const float* z, const float* filled, int* frontier, int* counters) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    if (filled[i] > z[i]) {{
        int pos = atomicAdd(&counters[0], 1);
        frontier[pos] = i;
    }}
}}
"""
    )


def build_fill_reconstruct_relax(KernelCls, *, grid, pass_p: Need, n_flat: int):
    """
    relax KernelBuilder, data args (z, filled, parent, frontier, counters,
    queued_gen): one grid-stride pass over the `counters[$P.get(0)$]`-sized
    input half of `frontier`, relaxing each active cell against its
    neighbours and pushing any neighbour whose candidate could still improve
    into the output half, deduplicated per pass via `queued_gen` +
    atomicExch. See ../../../experimental/LM/fill_reconstruct_optimised.py's
    module docstring for the push-gate correctness argument (an update at i
    only pushes a neighbour j when i's own contribution alone could still
    improve j - never a missed real activation, only a provably-futile push
    pruned). That script additionally caches each neighbour's `filled` value
    in local arrays to avoid re-reading it for the gate check (its own
    Round 2); this version re-reads `filled[j]` directly instead - same
    values, one more global read per neighbour, no correctness difference,
    since a stale cached value could only ever be stale-high there too.

    `pass_p` is the caller's already-bound `Need("pass_p", kind=Kind.PARAM)`
    (see make_fill_reconstruct) - a fresh, internally-named
    `Need("P", ...)`, matching this template's own `$P.get(0)$` span, is
    bound here to the same underlying Parameter and declared on this
    KernelBuilder via `.need()`. Read only here, via `$P.get(0)$` - bumping
    it between passes is the caller's job (a host_step in the Sequence the
    solver builds), the same division of labour `iteration_p` has for
    rake_compress.

    Author: B.G (07/2026)
    """
    t = f"pfr{new_uid()}"
    p_need = Need("P", kind=Kind.PARAM, dtype=pass_p.dtype, modes=pass_p.modes)
    p_need.bind(pass_p.value)
    return KernelCls().bind("grid", grid).need(p_need).ingest(
        f"""
__global__ void {t}_relax(const float* z, float* filled, int* parent, int* frontier,
                           int* counters, int* queued_gen) {{
    int p = $P.get(0)$;
    int par = p % 2;
    int in_base = par * {n_flat};
    int out_base = (1 - par) * {n_flat};
    int count = counters[p];
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += stride) {{
        int i = frontier[in_base + idx];
        int nk = $grid.n_neighbours.get(0)$;

        float best = {_POS_SENTINEL}f;
        int best_j = -1;
        for (int k = 0; k < nk; k++) {{
            int j = $grid.neighbour(i, k)$;
            if (j != -1) {{
                float v = filled[j];
                if (v < best) {{ best = v; best_j = j; }}
            }}
        }}
        float candidate = z[i] > best ? z[i] : best;

        if (candidate < filled[i]) {{
            filled[i] = candidate;
            parent[i] = best_j;
            for (int k = 0; k < nk; k++) {{
                int j = $grid.neighbour(i, k)$;
                if (j != -1) {{
                    float cand_j = z[j] > candidate ? z[j] : candidate;
                    if (cand_j < filled[j]) {{
                        int old = atomicExch(&queued_gen[j], p);
                        if (old != p) {{
                            int pos = atomicAdd(&counters[p + 1], 1);
                            frontier[out_base + pos] = j;
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
"""
    )
