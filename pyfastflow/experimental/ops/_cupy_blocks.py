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
