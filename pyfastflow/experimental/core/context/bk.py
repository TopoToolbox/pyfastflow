"""
`ctx.bk`: the reserved backend-intrinsics namespace.

Why this exists
----------------
grid/ never needed a bound backend module - its own private blocks are
built entirely out of plain arithmetic and the python builtins Taichi and
Quadrants both trace natively (abs, min - see grid/_closure_blocks.py's
module docstring). noise/'s Perlin lattice noise and visu/'s hillshade
formula are not that lucky: they need real transcendental functions
(sqrt/atan2/cos/sin/floor) and one typed cast (a u32 cast of an oversized
integer literal, in particular - Taichi rejects `0x846CA68B` outright as a
default-i32 literal without one), and there is no python builtin standing in
for any of those. Confirmed empirically before this module existed: calling
bare `math.sqrt`/`math.floor` inside a `ti.func` raises `TaichiTypeError:
must be real number, not Taichi Expression` - Taichi's AST transformer only
special-cases a short, fixed list of python builtins (abs/min/max/int/
float/...), never the `math` module.

The old stack solved the equivalent problem by splicing a free `_BK` name
into a template's globals (`_closure_backend.py`'s `specialize_closure`,
`extra_globals={"_BK": self._backend}`). That mechanism is deliberately not
reintroduced here: a free global name silently available regardless of
whether a template declared any need for it is exactly what the whole
`ctx`-rooted grammar (ctx.py, contract.py) exists to eliminate - every
reference a template makes should be visible in its derived Contract, not
smuggled in through `__globals__`.

Instead, `bk` is a reserved, always-present member of `ctx` itself, on the
closure backends only: `ctx.bk.sqrt(x)`, `ctx.bk.atan2(y, x)`,
`ctx.bk.u32(0x846CA68B)`. One template text works against both Taichi and
Quadrants because `make_closure_bk(backend)` resolves the same short
attribute surface against whichever module (`ti` or `qd`) `backend` is -
mirroring how every other closure template already reads uniformly across
backends.

Reserved, not a slot
---------------------
`ctx.bk.*` is a builtin the grammar recognises structurally, not a
capability a template's Contract requires satisfied - contract.py's python
AST walk drops any chain rooted at `RESERVED_BK_NAME` before it ever reaches
`Contract.chains`, so `ctx.bk.sqrt(...)` never shows up in `inspect()`,
never appears in `unmet()`, and ingest() never demands a wired "bk" slot for
it. Symmetrically, builder.py's `_Builder._wire()`/`compose()` (the only two
places a name becomes a slot or a composed root) raise if a caller ever
tries to declare a slot or compose a sub-structure named "bk" - the name is
reserved for this namespace and can never mean anything else. This is also
why `bk` is exposed on every level of the ctx tree compile_closure.py builds
(the kernel's own root node and every composed helper's node down to the
leaves), not just the root: a private block many levels deep is exactly
where noise's/visu's own use of it lives.

cupy is unaffected and deliberately excluded: its templates are raw CUDA
text where the native C spelling (`sqrtf`, `atan2f`, `floorf`, a plain
`0x846CA68Bu` literal) already is the natural way to write this, and the
python/cupy template surfaces are already a different grammar by design
(contract.py's module docstring) - `ctx.bk` is never resolved against a
cupy compile, and cupy blocks are not expected to reference it.

Surface
-------
Started narrow - exactly what noise/ and visu/ needed: `sqrt`, `atan2`, `cos`,
`sin`, `floor` (visu's hillshade formula, noise's Perlin lattice math) and
`u32` (noise's hash - an oversized u32 literal has nowhere else to become a
correctly-typed value, see below). Extended for ops/'s bitpack/reduce port
(08/2026) with `bit_cast`, `select`, `cast` (the IEEE-754 bit-flip trick
bitpack needs - reinterpreting a float's bits as u32 and back, and a
ternary-style branchless select over that) and `atomic_min`/`atomic_max`
(reduce's per-thread accumulation into a 0-d field) plus the `i32`/`i64`/`f32`
dtype tokens alongside the existing `u32`, needed as `ctx.bk.cast(x, ctx.bk.
i64)`'s second argument and, for `i64`, the same oversized-literal exemption
`u32` already gets (`ctx.bk.i64(0xFFFFFFFF)` - see below). Extended again for
ops/'s elementwise port (08/2026) with `grouped` (`ti.grouped`/`qd.grouped` -
`swap`'s dimensionality-agnostic iteration over a possibly multi-dimensional
field/ndarray, `for idx in ctx.bk.grouped(array1)`), a plain pass-through with
no oversized-literal or identity concern of its own. Extended again for
flow/'s accumulation port (08/2026) with `atomic_add` (accum_downstream_
atomic's own per-node downstream accumulation into a DATA-typed `q` buffer -
the same "genuinely concurrent write, so DATA not PARAM" shape reduce's
`atomic_min`/`atomic_max` already cover, just the third of the three atomic
ops Taichi/Quadrants both expose that this surface had not yet needed).

`abs`/`min`/`max` stay plain python builtins, as grid already established;
`int()`/`float()` join them here rather than becoming `ctx.bk` members -
confirmed empirically to trace exactly like `abs`/`min` do (Taichi/Quadrants
special-case these python builtins for casting too), so Perlin's own
int<->float lattice-cell conversions use the plain builtins, not a `ctx.bk`
member - there is nothing dtype-specific about `int(x)`/`float(x)` the way
there is about an oversized integer literal, and (confirmed the same way)
`ti.f32`/`ti.i32` do NOT get the same oversized-literal exemption `ti.u32`
does: `ti.f32(37)` raises `Integer literals must be annotated with a integer
type. For type casting, use ti.cast.` even though `ti.u32(37)` does not, so
there would be nothing gained by adding `f32`/`i32` members here anyway -
plain `float()`/`int()` are both simpler and strictly more capable (they
work uniformly whether the operand is a bare literal or an already-traced
Expr, which `ti.f32`/`ti.i32` do not).

`u32`/`i64` are exposed as the backend's own dtype objects themselves (`ti.
u32`, `ti.i64`, never a `lambda x: ti.cast(x, ti.u32)` wrapper around either)
- `ctx.bk.u32(0x846CA68B)`, `ctx.bk.i64(0xFFFFFFFF)`, called exactly as
`ti.u32(0x846CA68B)`/`ti.i64(0xFFFFFFFF)` would be. This is load-bearing, not
a style choice: confirmed empirically before settling on it, wrapping the
cast in an ordinary python callable breaks Taichi's own oversized-literal
handling - `ti.u32(2221713035)` compiles (Taichi's frontend recognises a
call whose callee resolves, by identity, to one of its own dtype objects,
and special-cases the literal argument's own type inference accordingly,
even reached through an attribute chain), while `(lambda x: ti.cast(x,
ti.u32))(2221713035)` raises `Integer literal 2221713035 exceeded the range
of default_ip: i32` - the literal is type-inferred as a bare argument to a
generic python callable *before* the lambda body ever runs `ti.cast`, and
the generic path defaults every bare int literal to i32 regardless of what
the callable eventually does with it. So `ctx.bk.u32(...)`/`ctx.bk.i64(...)`
must resolve to the real `ti.u32`/`ti.i64` (or `qd.` equivalent) object one
attribute hop away, not to a function that happens to produce the same
value. `i32`/`f32` are exposed the same uniform way for symmetry (`ctx.bk.
cast(x, ctx.bk.i32)`, ...) even though no template so far needs their own
oversized-literal exemption - `ctx.bk.cast`'s second argument is always a
dtype token regardless, and there is no reason for `i32`/`f32` to be the odd
ones out of the four.

Author: B.G (08/2026)
"""

from typing import Any

RESERVED_BK_NAME = "bk"
"""The reserved ctx member name for the backend-intrinsics namespace - see
the module docstring. Never wirable as a slot, never composable as a root."""


class BkError(Exception):
    """
    Raised by an unknown `ctx.bk.*` attribute - naming it and listing what is
    actually available, rather than letting a typo fall through to a bare
    AttributeError deep inside backend trace machinery.

    Author: B.G (08/2026)
    """


_BK_METHOD_NAMES = (
    "sqrt", "atan2", "cos", "sin", "floor", "u32",
    "bit_cast", "select", "cast", "atomic_min", "atomic_max", "atomic_add", "i32", "i64", "f32",
    "grouped",
)
"""Every name `ctx.bk` resolves - see the module docstring's "Surface" section."""


class ClosureBkNode:
    """
    `ctx.bk` itself, for one closure backend module (`ti` or `qd`) - see the
    module docstring. Every entry in `_BK_METHOD_NAMES` is resolved once, at
    construction, straight to the backend's own object (`ti.sqrt`, `ti.u32`,
    ...) - never wrapped, so a closure backend's own trace-time recognition
    of e.g. `ti.sqrt` as a builtin op, or `ti.u32` as a dtype-cast callee that
    exempts its own literal argument from default-int-type inference (see
    the module docstring), is identity-based and applies exactly as it would
    to a template calling `ti.sqrt`/`ti.u32` directly.

    Author: B.G (08/2026)
    """

    __slots__ = ("_backend", "_fns")

    def __init__(self, backend: Any):
        self._backend = backend
        self._fns = {
            "sqrt": backend.sqrt,
            "atan2": backend.atan2,
            "cos": backend.cos,
            "sin": backend.sin,
            "floor": backend.floor,
            "u32": backend.u32,
            "bit_cast": backend.bit_cast,
            "select": backend.select,
            "cast": backend.cast,
            "atomic_min": backend.atomic_min,
            "atomic_max": backend.atomic_max,
            "atomic_add": backend.atomic_add,
            "i32": backend.i32,
            "i64": backend.i64,
            "f32": backend.f32,
            "grouped": backend.grouped,
        }

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._fns[name]
        except KeyError:
            raise BkError(
                f"ctx.bk.{name} is not a recognised backend intrinsic - available: "
                f"{', '.join(_BK_METHOD_NAMES)}"
            ) from None

    def __repr__(self) -> str:
        return f"ClosureBkNode(backend={self._backend.__name__}, provides={_BK_METHOD_NAMES})"


def make_closure_bk(backend: Any) -> ClosureBkNode:
    """
    `ctx.bk` for one closure compile - `backend` is the `taichi` or
    `quadrants` module (`BoundKernel.compile()`'s own `backend` argument,
    compile_closure.py). Stateless and cheap; built once per compile() and
    shared by every node in that compile's own ctx tree.

    Author: B.G (08/2026)
    """
    return ClosureBkNode(backend)
