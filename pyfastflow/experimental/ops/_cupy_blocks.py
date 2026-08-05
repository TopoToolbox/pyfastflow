"""
cupy (CUDA source) block templates behind ops's make_bitpack_group/make_scan/
make_reduce, on the new builder/frozen/bound stack (core/context/builder.py,
frozen.py, bound.py). Mirrors _closure_blocks.py's split and
../grid/_cupy_blocks.py's own conventions: every span reaching a PARAM is
spelled `$ctx.NAME.get(...)$`/`$ctx.NAME.set_node(...)$` in full, every span
reaching a composed HELPER is spelled `$ctx.name(args)$` - the old bare-span
shorthand (`$flip(f)$`) is gone. Every device/global function name is
prefixed with this build's own tag (a fresh new_uid()), matching grid/noise/
visu's own belt-and-braces convention (compile_cupy.py already mangles by
address - see its own module docstring - this is redundant safety, not load
bearing).

`.inclusive()` on cupy stays `cp.cumsum` (ops/__init__.py's own module
docstring: CUB's DeviceScan is already the accelerator cupy dispatches to by
default) - no RoutineBuilder involved for that half. Compaction's count-read
and scatter, previously a bare host-side numpy slice-copy plus one directly-
launched kernel, are ported as a two-step FrozenRoutine (routine_v2.py)
instead: "read_count" (a 1-thread kernel writing scan_out[n-1] into the COUNT
PARAM) and "scatter", each composed with its own `launch=` override
(routine_v2.py's `RoutineBuilder.compose(name, frozen, launch=...)`) - a
genuinely different, meaningfully-sized grid/block per step, which is what
actually exercises the per-step launch mechanism on a backend where launch
dims mean anything (see ops/__init__.py's module docstring for the fuller
design-fork note this resolves).

Author: B.G (08/2026)
"""

from ..core.context.builder import GroupBuilder, HelperBuilder, KernelBuilder
from ..core.context.contract import extract_cupy_contract
from ..core.pool.base import new_uid

# ---------------------------------------------------------------------------
# bitpack: pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32
# ---------------------------------------------------------------------------


def _helper(template, *, helpers=None):
    """PARAM slots auto-derived from the template's own contract, exactly like grid/_cupy_blocks.py's `_helper`."""
    b = HelperBuilder()
    for chain in extract_cupy_contract(template).chains:
        if (not helpers) or chain[0] not in helpers:
            b.wire_param(chain[0])
    if helpers:
        for name, frozen in helpers.items():
            b.compose(name, frozen)
    return b.ingest(template)


def build_bitpack_group() -> "FrozenGroup":
    """
    pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32, same
    IEEE-754 bit-flip trick as _closure_blocks.build_bitpack_group, using
    CUDA's __float_as_uint/__uint_as_float. No PARAM slots anywhere in this
    tree.

    Author: B.G (08/2026)
    """
    t = f"pf{new_uid()}"
    flip = _helper(
        f"""
__device__ unsigned int {t}_flip(float f) {{
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? (u ^ 0x80000000u) : (~u);
}}
"""
    )
    unflip = _helper(
        f"""
__device__ float {t}_unflip(unsigned int u) {{
    unsigned int restored = (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
    return __uint_as_float(restored);
}}
"""
    )
    pack = _helper(
        f"""
__device__ long long {t}_pack(float f, int i) {{
    unsigned int f_enc = $ctx.flip(f)$;
    unsigned int i_enc = (unsigned int)i;
    long long packed = ((long long)f_enc << 32) | (long long)i_enc;
    long long flipped_upper = (~packed) & (0xFFFFFFFFLL << 32);
    long long unchanged_lower = packed & 0xFFFFFFFFLL;
    return flipped_upper | unchanged_lower;
}}
""",
        helpers={"flip": flip},
    )
    unpack_raw = _helper(
        f"""
__device__ long long {t}_unpack_raw(long long packed) {{
    long long flipped_upper = (~packed) & (0xFFFFFFFFLL << 32);
    long long unchanged_lower = packed & 0xFFFFFFFFLL;
    return flipped_upper | unchanged_lower;
}}
"""
    )
    unpack_value = _helper(
        f"""
__device__ float {t}_unpack_value(long long packed) {{
    long long u = $ctx.unpack_raw(packed)$;
    unsigned int f_enc = (unsigned int)(u >> 32);
    return $ctx.unflip(f_enc)$;
}}
""",
        helpers={"unpack_raw": unpack_raw, "unflip": unflip},
    )
    unpack_index = _helper(
        f"""
__device__ int {t}_unpack_index(long long packed) {{
    long long u = $ctx.unpack_raw(packed)$;
    unsigned int i_enc = (unsigned int)(u & 0xFFFFFFFFLL);
    return (int)i_enc;
}}
""",
        helpers={"unpack_raw": unpack_raw},
    )

    group = GroupBuilder()
    group.wire_helper("pack").compose("pack", pack)
    group.wire_helper("unpack_value").compose("unpack_value", unpack_value)
    group.wire_helper("unpack_index").compose("unpack_index", unpack_index)
    return group.close()


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def build_math_group() -> "FrozenGroup":
    """
    atan(x) via atan2f(x, 1); nextafter(x, y), one ULP of f32 towards y via
    the same bit-twiddling as _closure_blocks.build_math_group - composed
    onto a fresh GroupBuilder under those two public names. No PARAM slots.

    Author: B.G (08/2026)
    """
    t = f"pf{new_uid()}"
    atan = _helper(f"__device__ float {t}_atan(float x) {{ return atan2f(x, 1.0f); }}")
    nextafter = _helper(
        f"""
__device__ float {t}_nextafter(float x, float y) {{
    float result = y;
    if (x != y) {{
        unsigned int sign_mask = 0x80000000u;
        unsigned int ix = __float_as_uint(x);
        if (x == 0.0f) {{
            ix = (__float_as_uint(y) & sign_mask) | 1u;
        }} else if ((x > 0.0f) == (y > x)) {{
            ix += 1u;
        }} else {{
            ix -= 1u;
        }}
        result = __uint_as_float(ix);
    }}
    return result;
}}
"""
    )

    group = GroupBuilder()
    group.wire_helper("atan").compose("atan", atan)
    group.wire_helper("nextafter").compose("nextafter", nextafter)
    return group.close()


# ---------------------------------------------------------------------------
# elementwise (kernels, returned unbuilt) - `n` is baked as a python int at
# build time (this build's own closure), not a data argument - see
# _closure_blocks.build_elementwise's own docstring; the pre-rewrite cupy
# text carried `n` as a real kernel argument instead, which this port
# tightens to match every other backend's already-closed-over `n`.
# ---------------------------------------------------------------------------


def build_elementwise(n: int) -> dict:
    """
    swap/add_B_to_A/add_B_to_weighted_A/weighted_mean_B_in_A/arange/
    multiply_by_scalar over a flat f32 buffer of length `n`, as unbuilt
    FrozenKernels.

    Author: B.G (08/2026)
    """
    t = f"pf{new_uid()}"

    def _k(template, names):
        b = KernelBuilder()
        for name in names:
            b.wire_data(name)
        return b.ingest(template)

    return {
        "swap": _k(
            f"""
extern "C" __global__ void {t}_swap(float* array1, float* array2) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    float temp = array1[i];
    array1[i] = array2[i];
    array2[i] = temp;
}}
""",
            ["array1", "array2"],
        ),
        "add_B_to_A": _k(
            f"""
extern "C" __global__ void {t}_add_B_to_A(float* array1, const float* array2) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    array1[i] += array2[i];
}}
""",
            ["array1", "array2"],
        ),
        "add_B_to_weighted_A": _k(
            f"""
extern "C" __global__ void {t}_add_B_to_weighted_A(float* array1, const float* array2, float weight) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    array1[i] += array2[i] * weight;
}}
""",
            ["array1", "array2", "weight"],
        ),
        "weighted_mean_B_in_A": _k(
            f"""
extern "C" __global__ void {t}_weighted_mean_B_in_A(float* array1, const float* array2, float weight) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    array1[i] = array2[i] * weight + array1[i] * (1.0f - weight);
}}
""",
            ["array1", "array2", "weight"],
        ),
        "arange": _k(
            f"""
extern "C" __global__ void {t}_arange(float* array) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    array[i] = (float)i;
}}
""",
            ["array"],
        ),
        "multiply_by_scalar": _k(
            f"""
extern "C" __global__ void {t}_multiply_by_scalar(float* A, float scalar) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    A[i] *= scalar;
}}
""",
            ["A", "scalar"],
        ),
    }


# ---------------------------------------------------------------------------
# slope (grid-aware) - the first ops/ case of a nested FrozenGroup-in-Frozen
# Group child on cupy, mirroring _closure_blocks.build_slope_group
# ---------------------------------------------------------------------------


def _find_param_paths(frozen, leaf_name: str, prefix: tuple = ()) -> list:
    """Identical to _closure_blocks.py's own `_find_param_paths`."""
    from ..core.context.slot import SlotKind

    paths = []
    if leaf_name in frozen.slots.names(SlotKind.PARAM):
        paths.append(".".join(prefix + (leaf_name,)))
    for name, child in frozen.composed.items():
        paths.extend(_find_param_paths(child, leaf_name, prefix + (name,)))
    return paths


def _share_leaf(group: GroupBuilder, canonical: str) -> None:
    """Identical to _closure_blocks.py's own `_share_leaf`."""
    paths = []
    for name, child in group.composed.items():
        paths.extend(_find_param_paths(child, canonical, (name,)))
    if paths:
        group.share(canonical, *paths)


def build_slope_group(grid) -> "FrozenGroup":
    """
    sumslope_downstream(z, i) / slope_dir(z, i, k), same arithmetic as
    _closure_blocks.build_slope_group, walking `grid`'s neighbour/dx/
    n_neighbours surface through `$ctx.grid...$` spans - `grid` composed
    independently as each helper's own child, same nested-FrozenGroup shape
    as the closure port (see that module's own docstring).

    Author: B.G (08/2026)
    """
    from ..core.context.slot import SlotKind

    t = f"pf{new_uid()}"
    sumslope_downstream = _helper(
        f"""
__device__ float {t}_sumslope_downstream(const float* z, int i) {{
    float sumslope = 0.0f;
    int nk = $ctx.grid.N_NEIGHBOURS.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $ctx.grid.neighbour(i, k)$;
        if (j > -1) {{
            if (z[j] < z[i]) {{
                sumslope += (z[i] - z[j]) / $ctx.grid.DX.get(0)$;
            }}
        }}
    }}
    return sumslope;
}}
""",
        helpers={"grid": grid},
    )
    slope_dir = _helper(
        f"""
__device__ float {t}_slope_dir(const float* z, int i, int k) {{
    int j = $ctx.grid.neighbour(i, k)$;
    float slope = 0.0f;
    if (j > -1) {{
        slope = (z[i] - z[j]) / $ctx.grid.DX.get(0)$;
    }}
    return slope;
}}
""",
        helpers={"grid": grid},
    )

    group = GroupBuilder()
    grid_param_names = grid.slots.names(SlotKind.PARAM)
    for name in grid_param_names:
        group.wire_param(name)
    group.wire_helper("sumslope_downstream").compose("sumslope_downstream", sumslope_downstream)
    group.wire_helper("slope_dir").compose("slope_dir", slope_dir)

    for name in grid_param_names:
        _share_leaf(group, name)

    return group.close()


# ---------------------------------------------------------------------------
# block_reduce (cub::BlockReduce wrapper) - cupy only
# ---------------------------------------------------------------------------


def build_block_reduce_group(block_size: int = 128) -> "FrozenGroup":
    """
    sum(val): one cub::BlockReduce<float, block_size>::Sum() per calling
    block, returning the block-wide total to thread 0 (undefined on other
    threads - cub's own contract), composed under the public name "sum". The
    first compile that reaches this triggers a one-time jitify header cache
    warm-up for <cub/block/block_reduce.cuh>, roughly two minutes; that is
    expected, not a hang.

    Author: B.G (08/2026)
    """
    t = f"pf{new_uid()}"
    sum_helper = _helper(
        f"""
#include <cub/block/block_reduce.cuh>
__device__ float {t}_block_reduce_sum(float val) {{
    typedef cub::BlockReduce<float, {int(block_size)}> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    return BlockReduceT(temp_storage).Sum(val);
}}
"""
    )
    group = GroupBuilder()
    group.wire_helper("sum").compose("sum", sum_helper)
    return group.close()


# ---------------------------------------------------------------------------
# scan compaction: read_count + scatter, as a 2-step FrozenRoutine
# ---------------------------------------------------------------------------


def _kernel(template, *, data=(), helpers=None):
    b = KernelBuilder()
    for d in data:
        b.wire_data(d)
    for chain in extract_cupy_contract(template).chains:
        if (not helpers) or chain[0] not in helpers:
            b.wire_param(chain[0])
    if helpers:
        for name, frozen in helpers.items():
            b.compose(name, frozen)
    return b.ingest(template)


def build_count_and_scatter_routine(n: int, *, block: int = 256) -> "FrozenRoutine":
    """
    A 2-step FrozenRoutine (routine_v2.py): "read_count" (one thread, writes
    scan_out[n-1] into the wired PARAM slot "COUNT") then "scatter" (one
    thread per node, `ids[scan_out[i]-1] = i` wherever `flags[i] != 0`) - the
    compaction half of scan-based stream compaction. Each step is composed
    with its own `launch=` override (routine_v2.py's `RoutineBuilder.compose(
    ..., launch=...)`) sized to that step's own real thread count - "
    read_count" is one thread regardless of `n`, "scatter" needs
    ceil(n/block) blocks of `block` threads - see the module docstring for
    why this, not the inclusive scan itself, is what exercises per-step
    launch on cupy.

    Author: B.G (08/2026)
    """
    from ..core.context.routine_v2 import RoutineBuilder

    t = f"pf{new_uid()}"
    read_count = _kernel(
        f"""
extern "C" __global__ void {t}_read_count(const int* scan_out) {{
    $ctx.COUNT.set_node(0, scan_out[{int(n)} - 1])$;
}}
""",
        data=["scan_out"],
    )
    scatter = _kernel(
        f"""
extern "C" __global__ void {t}_scatter(const int* flags, const int* scan_out, int* ids) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {int(n)}) return;
    if (flags[i] != 0) {{
        ids[scan_out[i] - 1] = i;
    }}
}}
""",
        data=["flags", "scan_out", "ids"],
    )

    grid_dim = (n + block - 1) // block
    rb = RoutineBuilder()
    rb.compose("read_count", read_count, launch={"grid": 1, "block": 1})
    rb.compose("scatter", scatter, launch={"grid": grid_dim, "block": block})
    return rb.freeze()
