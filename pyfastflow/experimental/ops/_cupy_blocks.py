"""
cupy (CUDA source) block templates behind ops's make_* factories.

Mirrors ../grid/_cupy_blocks.py block for block: same private/public split as
_closure_blocks.py, written as CUDA text instead of python defs. Every
`__device__`/`__global__` symbol is prefixed with this build's own tag (a
fresh new_uid()) so two calls into this module in one process never collide
inside a single compiled cupy module even if both end up bound into the same
kernel - see ../core/context/cupy_backend.py's module docstring.

make_scan and make_reduce need no CUDA text of their own beyond a small
scatter kernel (scan) - `cp.cumsum`/`cp.sum`/`cp.min`/`cp.max`/`cp.argmin`
already do the rest, and CUB's own DeviceScan/reduction accelerators are the
default on this build (see ops/__init__.py's module docstring).

Author: B.G (07/2026)
"""

import functools

from ..core.context.backends import make_helper
from ..core.pool.base import new_uid

# ---------------------------------------------------------------------------
# bitpack
# ---------------------------------------------------------------------------


def build_bitpack(HelperCls):
    """
    pack(f, i) -> i64, unpack_value(p) -> f32, unpack_index(p) -> i32, same
    IEEE-754 bit-flip trick as _closure_blocks.build_bitpack, using CUDA's
    __float_as_uint/__uint_as_float instead of Taichi's bit_cast.

    Author: B.G (07/2026)
    """
    t = f"pf{new_uid()}"
    mk = functools.partial(make_helper, HelperCls)

    flip = mk(
        f"""
__device__ unsigned int {t}_flip(float f) {{
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? (u ^ 0x80000000u) : (~u);
}}
"""
    )
    unflip = mk(
        f"""
__device__ float {t}_unflip(unsigned int u) {{
    unsigned int restored = (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
    return __uint_as_float(restored);
}}
"""
    )
    pack = mk(
        f"""
__device__ long long {t}_pack(float f, int i) {{
    unsigned int f_enc = $flip(f)$;
    unsigned int i_enc = (unsigned int)i;
    long long packed = ((long long)f_enc << 32) | (long long)i_enc;
    long long flipped_upper = (~packed) & (0xFFFFFFFFLL << 32);
    long long unchanged_lower = packed & 0xFFFFFFFFLL;
    return flipped_upper | unchanged_lower;
}}
""",
        flip=flip,
    )
    unpack_raw = mk(
        f"""
__device__ long long {t}_unpack_raw(long long packed) {{
    long long flipped_upper = (~packed) & (0xFFFFFFFFLL << 32);
    long long unchanged_lower = packed & 0xFFFFFFFFLL;
    return flipped_upper | unchanged_lower;
}}
"""
    )
    unpack_value = mk(
        f"""
__device__ float {t}_unpack_value(long long packed) {{
    long long u = $unpack_raw(packed)$;
    unsigned int f_enc = (unsigned int)(u >> 32);
    return $unflip(f_enc)$;
}}
""",
        unpack_raw=unpack_raw,
        unflip=unflip,
    )
    unpack_index = mk(
        f"""
__device__ int {t}_unpack_index(long long packed) {{
    long long u = $unpack_raw(packed)$;
    unsigned int i_enc = (unsigned int)(u & 0xFFFFFFFFLL);
    return (int)i_enc;
}}
""",
        unpack_raw=unpack_raw,
    )
    return {"pack": pack, "unpack_value": unpack_value, "unpack_index": unpack_index}


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def build_math(HelperCls):
    """
    atan(x) via atan2f(x, 1); nextafter(x, y), one ULP of f32 towards y via
    the same bit-twiddling as _closure_blocks.build_math.

    Author: B.G (07/2026)
    """
    t = f"pf{new_uid()}"
    mk = functools.partial(make_helper, HelperCls)

    atan = mk(f"__device__ float {t}_atan(float x) {{ return atan2f(x, 1.0f); }}")
    nextafter = mk(
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
    return {"atan": atan, "nextafter": nextafter}


# ---------------------------------------------------------------------------
# elementwise (kernel builders, returned unbuilt) - `n` is an explicit data
# argument here since a RawModule kernel has no auto-ranging (see
# cupy_backend.py's module docstring)
# ---------------------------------------------------------------------------


def build_elementwise(KernelCls):
    """
    swap/add_B_to_A/add_B_to_weighted_A/weighted_mean_B_in_A/arange/
    multiply_by_scalar over a flat f32 buffer, as unbuilt KernelBuilders.
    Every kernel takes the buffer length `n` as its own last argument.

    Author: B.G (07/2026)
    """
    t = f"pf{new_uid()}"
    return {
        "swap": KernelCls().ingest(
            f"""
extern "C" __global__ void {t}_swap(float* array1, float* array2, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float temp = array1[i];
    array1[i] = array2[i];
    array2[i] = temp;
}}
"""
        ),
        "add_B_to_A": KernelCls().ingest(
            f"""
extern "C" __global__ void {t}_add_B_to_A(float* array1, const float* array2, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    array1[i] += array2[i];
}}
"""
        ),
        "add_B_to_weighted_A": KernelCls().ingest(
            f"""
extern "C" __global__ void {t}_add_B_to_weighted_A(float* array1, const float* array2, float weight, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    array1[i] += array2[i] * weight;
}}
"""
        ),
        "weighted_mean_B_in_A": KernelCls().ingest(
            f"""
extern "C" __global__ void {t}_weighted_mean_B_in_A(float* array1, const float* array2, float weight, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    array1[i] = array2[i] * weight + array1[i] * (1.0f - weight);
}}
"""
        ),
        "arange": KernelCls().ingest(
            f"""
extern "C" __global__ void {t}_arange(float* array, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    array[i] = (float)i;
}}
"""
        ),
        "multiply_by_scalar": KernelCls().ingest(
            f"""
extern "C" __global__ void {t}_multiply_by_scalar(float* A, float scalar, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[i] *= scalar;
}}
"""
        ),
    }


# ---------------------------------------------------------------------------
# slope
# ---------------------------------------------------------------------------


def build_slope(HelperCls, grid):
    """
    sumslope_downstream(z, i) / slope_dir(z, i, k), same arithmetic as
    _closure_blocks.build_slope, walking `grid`'s neighbour/dx/n_neighbours
    surface through $...$ spans.

    Author: B.G (07/2026)
    """
    t = f"pf{new_uid()}"
    mk = functools.partial(make_helper, HelperCls)

    sumslope_downstream = mk(
        f"""
__device__ float {t}_sumslope_downstream(const float* z, int i) {{
    float sumslope = 0.0f;
    int nk = $grid.n_neighbours.get(0)$;
    for (int k = 0; k < nk; k++) {{
        int j = $grid.neighbour(i, k)$;
        if (j > -1) {{
            if (z[j] < z[i]) {{
                sumslope += (z[i] - z[j]) / $grid.dx.get(0)$;
            }}
        }}
    }}
    return sumslope;
}}
""",
        grid=grid,
    )
    slope_dir = mk(
        f"""
__device__ float {t}_slope_dir(const float* z, int i, int k) {{
    int j = $grid.neighbour(i, k)$;
    float slope = 0.0f;
    if (j > -1) {{
        slope = (z[i] - z[j]) / $grid.dx.get(0)$;
    }}
    return slope;
}}
""",
        grid=grid,
    )
    return {"sumslope_downstream": sumslope_downstream, "slope_dir": slope_dir}


# ---------------------------------------------------------------------------
# scan compaction scatter kernel (inclusive scan itself is cp.cumsum - see
# ops/__init__.py)
# ---------------------------------------------------------------------------


def build_scatter_kernel(KernelCls):
    """
    ids[scan_out[i]-1] = i for every i where flags[i]!=0 - the scatter half
    of scan-based stream compaction.

    Author: B.G (07/2026)
    """
    t = f"pf{new_uid()}"
    return KernelCls().ingest(
        f"""
extern "C" __global__ void {t}_scatter(const int* flags, const int* scan_out, int* ids, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (flags[i] != 0) {{
        ids[scan_out[i] - 1] = i;
    }}
}}
"""
    )


# ---------------------------------------------------------------------------
# block_reduce (cub::BlockReduce wrapper)
# ---------------------------------------------------------------------------


def build_block_reduce(HelperCls, block_size: int = 128):
    """
    sum(val): one cub::BlockReduce<float, block_size>::Sum() per calling
    block, returning the block-wide total to thread 0 (undefined on other
    threads - cub's own contract). The first compile that reaches this
    warms up jitify's header cache for <cub/block/block_reduce.cuh> - budget
    roughly two minutes for that one-time cost.

    Author: B.G (07/2026)
    """
    t = f"pf{new_uid()}"
    mk = functools.partial(make_helper, HelperCls)
    sum_helper = mk(
        f"""
#include <cub/block/block_reduce.cuh>
__device__ float {t}_block_reduce_sum(float val) {{
    typedef cub::BlockReduce<float, {int(block_size)}> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    return BlockReduceT(temp_storage).Sum(val);
}}
"""
    )
    return {"sum": sum_helper}
