"""
cupy-only persistent-kernel MFD accumulation ("persistent_mfd" method of
make_accumulation).

No Taichi/Quadrants counterpart, and there never should be one: the
mechanism is a hand-rolled, monotonic grid-wide barrier (a global atomic
counter every block increments then spins on, level by level) around a
persistent kernel that is launched exactly once and loops internally over
levels - built on CUDA's raw `__global__`/`__shared__`/`atomicAdd`/
`__threadfence()` primitives, none of which the closure backends' kernel
model (one `ti.kernel`/Quadrants kernel per launch, no persistent-thread
loop, no portable grid-wide barrier inside one kernel) can express.

Algorithm (level-synchronous, double-buffered frontier, shared-memory
staging): each level, every resident thread of a fixed, small grid pulls
node indices from `frontier[p]` (size `count[p]`, grid-stride loop). For
node u, the receiver mask `dirs[u]` (bit k set == direction k, this grid's
own D8 numbering) picks which of `mfd_w[u*NN + k]` weights to atomic-add
`accum[u]` into `accum[neighbour]` (`neighbour_raw(u, k)` - the grid's own
raw neighbour arithmetic, trusted the same way the mask itself is trusted
to have only ever set bits for directions the topology already validated).
A `__threadfence()` publishes every one of those writes before any thread
decrements `indegree[neighbour]`; a decrement that lands the count exactly
on zero stages that neighbour into a per-block shared buffer (`s_buf`,
capacity `fr_stage`), spilling to a direct `atomicAdd` into `count[1-p]`
past that capacity. After the grid-stride loop, each block flushes its
staged cells into `frontier[1-p]` through one reserved contiguous range
(one `atomicAdd` per block, not per cell), fences again, then every block
increments the global `barrier` counter and spins until it reads
`(level+1) * gridDim.x` - the point every block has published everything
for this level - before moving on. The loop exits once `count[p]`, reloaded
through a volatile pointer each iteration (not just once, and not just via
the atomics' own side effects - nothing else in this kernel forces that
reload), is zero.

`accum` is seeded through a `source` Need (any Parameter mode) by a separate
`q_init` kernel, not hardcoded to 1.0: the persistent kernel's very first
level reads `accum[u]` for every cell already in the initial frontier
before any atomic_add has landed on it, so that seed must be a prior, real,
finished launch - the same reasoning `_cupy_accum.py`'s `build_atomic`
splits `q_init`/`accum` on.

`dirs`, `mfd_w`, `indegree`, `frontier0`, `frontier1`, `count`, `barrier`
are all caller-supplied data args, exactly like `rec` is for the SFD
methods in `_cupy_accum.py` - this module does not build MFD topology
(mask/weights/indegree computation), only accumulation over an
already-built one. `count` is 2 int32 (`count[0]`, `count[1]`, the two
ping-pong frontier sizes) and `barrier` is 1 uint32; initializing them
(`count[p] = n0` from the ready-cell count, `count[1-p] = 0`,
`barrier[0] = 0`, `frontier[p][:n0] = <ready cell indices>`) is the
caller's job before every call - see `init_frontier_mfd` below for the
host-side compaction step, kept as a plain function rather than a Bag
member since it is pure host/cupy indexing, not a kernel.

Author: B.G (08/2026)
"""

import numpy as np
import cupy as cp

from ..core.context.backends import helper_need
from ..core.context.need import Kind, Need
from ..core.pool.base import new_uid


def persistent_grid_block(*, blocks_per_sm: int = 2, threads: int = 256) -> tuple:
    """
    (grid, block) launch dims for the persistent kernel: `blocks_per_sm *
    <this device's SM count>` blocks, never more than can be co-resident,
    of `threads` threads each - queried from the current cupy device, not
    sized off n_flat the way every other launch in this package is (the
    frontier itself, not the whole node range, bounds how much work a
    level does).

    Author: B.G (08/2026)
    """
    sm_count = cp.cuda.Device().attributes["MultiProcessorCount"]
    return (blocks_per_sm * sm_count,), (threads,)


def init_frontier_mfd(indegree_data, frontier_data) -> int:
    """
    Host-side frontier compaction: writes the flat indices of every cell
    with indegree 0 into the front of `frontier_data` (a raw cupy ndarray,
    e.g. a CupyDataHandle's `.data`) and returns how many there were - the
    `count[p]` the caller must then store before the first launch.

    Plain cupy indexing, not a kernel: `cp.nonzero` has no equivalent
    device-side primitive this package's span/template mechanism reaches,
    and the reference implementation this ports does the same compaction
    as an ordinary host op rather than a custom kernel.

    Author: B.G (08/2026)
    """
    ready = cp.nonzero(indegree_data == 0)[0].astype(cp.int32)
    n = int(ready.size)
    frontier_data[:n] = ready
    return n


def build_persistent_mfd(
    KernelCls,
    *,
    grid,
    source: Need,
    n_flat: int,
    fr_stage: int = 2048,
):
    """
    Two KernelBuilders: "q_init" (data arg (accum,), ordinary grid-stride
    over n_flat: accum[i] = source.get(i)) and "accum" (data args
    (frontier0, frontier1, count, barrier, dirs, mfd_w, accum, indegree),
    the persistent kernel described in the module docstring). Both are bare
    KernelBuilders, not a Routine - "q_init" is one ordinary n_flat-sized
    launch, "accum" is one persistent launch on
    `persistent_grid_block(...)`'s dims; there is no per-round host loop
    for a Routine to sequence, unlike rake_compress/pointer_jump_push.

    `source` is the caller's already-bound `Need("source", kind=Kind.PARAM)`
    (see make_accumulation) - a fresh, internally-named `Need("source", ...)`
    is bound here to the same underlying Parameter and declared on "q_init"
    via `.need()`, alongside its own kind=DATA `accum` need below.

    `fr_stage` sizes the per-block shared staging buffer (`s_buf`) baked
    into "accum"'s generated source as a compile-time array length - a
    smaller value uses less shared memory per block at the cost of more
    direct-scatter spills past capacity.

    Every data argument is declared as a kind=DATA Need (see need.py) with
    its expected dtype, so each real call's positional argument is
    dtype-checked at the point it is passed rather than trusted silently -
    the declared contract this whole module's caller-supplied-buffer
    convention was always implicitly relying on, now enforced. The two
    KernelBuilders still return data args in exactly the same names/order/
    count as before, and a caller passing correctly-dtyped buffers (as every
    existing caller already does) sees no behavioural difference at all.

    Both KernelBuilders are constructed strict_needs=True; "accum"'s
    `neighbour_raw=grid.neighbour_raw` bind goes through helper_need,
    mirroring every other converted flow block module.

    Author: B.G (08/2026)
    """
    NN = grid.n_neighbours.get()
    t = f"pm{new_uid()}"

    source_need = Need("source", kind=Kind.PARAM, dtype=source.dtype, modes=source.modes)
    source_need.bind(source.value)
    q_init_accum_need = Need("accum", kind=Kind.DATA, dtype=np.float32)
    q_init = KernelCls(strict_needs=True).need(source_need, q_init_accum_need).ingest(
        f"""
__global__ void {t}_q_init(float* accum) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {n_flat}) return;
    accum[i] = $source.get(i)$;
}}
"""
    )

    accum_data_needs = (
        Need("frontier0", kind=Kind.DATA, dtype=np.int32),
        Need("frontier1", kind=Kind.DATA, dtype=np.int32),
        Need("count", kind=Kind.DATA, dtype=np.int32),
        Need("barrier", kind=Kind.DATA, dtype=np.uint32),
        Need("dirs", kind=Kind.DATA, dtype=np.uint8),
        Need("mfd_w", kind=Kind.DATA, dtype=np.float32),
        Need("accum", kind=Kind.DATA, dtype=np.float32),
        Need("indegree", kind=Kind.DATA, dtype=np.int32),
    )
    neighbour_raw_need = helper_need("neighbour_raw", grid.neighbour_raw)
    accum = (
        KernelCls(strict_needs=True)
        .need(neighbour_raw_need)
        .bind("neighbour_raw", neighbour_raw_need.value)
        .need(*accum_data_needs)
        .ingest(
        f"""
__global__ void {t}_persistent_mfd(
    int* __restrict__ frontier0, int* __restrict__ frontier1,
    int* __restrict__ count, unsigned int* __restrict__ barrier,
    const unsigned char* __restrict__ dirs, const float* __restrict__ mfd_w,
    float* __restrict__ accum, int* __restrict__ indegree)
{{
    __shared__ int s_buf[{fr_stage}];
    __shared__ int s_n;
    __shared__ unsigned int s_base;

    int* frontiers[2] = {{ frontier0, frontier1 }};
    int p = 0;
    unsigned int level = 0;

    while (true) {{
        int size_in = *((volatile int*)&count[p]);
        if (size_in == 0) break;
        int* fin  = frontiers[p];
        int* fout = frontiers[1 - p];

        if (threadIdx.x == 0) s_n = 0;
        __syncthreads();

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = gridDim.x * blockDim.x;
        for (int idx = tid; idx < size_in; idx += stride) {{
            int u = fin[idx];
            float au = accum[u];
            unsigned int mask = (unsigned int)dirs[u];
            int base = u * {NN};
            #pragma unroll
            for (int k = 0; k < {NN}; k++) {{
                if (!(mask & (1u << k))) continue;
                int r = $neighbour_raw(u, k)$;
                atomicAdd(&accum[r], au * mfd_w[base + k]);
            }}
            __threadfence();
            #pragma unroll
            for (int k = 0; k < {NN}; k++) {{
                if (!(mask & (1u << k))) continue;
                int r = $neighbour_raw(u, k)$;
                int old = atomicAdd(&indegree[r], -1);
                if (old == 1) {{
                    int sp = atomicAdd(&s_n, 1);
                    if (sp < {fr_stage}) s_buf[sp] = r;
                    else {{ int pos = atomicAdd(&count[1 - p], 1); fout[pos] = r; }}
                }}
            }}
        }}

        __syncthreads();
        int n_flush = min(s_n, {fr_stage});
        if (threadIdx.x == 0)
            s_base = atomicAdd((unsigned int*)&count[1 - p], (unsigned int)n_flush);
        __syncthreads();
        for (int i = threadIdx.x; i < n_flush; i += blockDim.x)
            fout[s_base + i] = s_buf[i];
        __threadfence();

        __syncthreads();
        if (threadIdx.x == 0) {{
            if (blockIdx.x == 0) count[p] = 0;
            unsigned int target = (level + 1) * (unsigned int)gridDim.x;
            atomicAdd(barrier, 1u);
            unsigned int ns = 32;
            while (*((volatile unsigned int*)barrier) < target) {{
#if __CUDA_ARCH__ >= 700
                __nanosleep(ns);
                if (ns < 1024) ns <<= 1;
#endif
            }}
        }}
        __syncthreads();

        level++;
        p = 1 - p;
    }}
}}
"""
        )
    )

    return {"q_init": q_init, "accum": accum}
