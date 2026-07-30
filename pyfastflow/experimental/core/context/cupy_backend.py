"""
cupy implementations of Parameter, Kernel and their builders, plus
CupyHelperBuilder - the recipe for a device helper, specialized as part of
whichever kernel binds it (see compile.py, HelperBuilder).

Here a template is CUDA source text rather than a python function, since
cp.RawModule compiles source and there is no function whose globals could be
patched. Bound objects are written into that source as `$...$` spans holding a
dotted path, which keeps the in-kernel spelling the same as on the other
backends:

    $p.get(i)$        read parameter p at flat index i
    $p.set_node(i,v)$ write parameter p at flat index i
    $grid.nx.get(i)$  reach a bag member
    $helper(a, b)$    call a bound device helper

Compiling substitutes each span according to the parameter's mode - a CUDA
literal for const, or a read/write through a pointer for scalar and field.
That pointer never travels as a kernel argument: every scalar/field Parameter
a compilation unit reaches - the kernel's own bindings plus, recursively, its
helpers' - is collected once, deduplicated by uid, into a module-scope
constant block:

    struct pf_params_t { float* p_<idx>; const float* p_<idx2>; ... };
    __constant__ pf_params_t pf_params;

`<idx>` is a per-compilation-unit local index (0, 1, 2, ...) assigned the
first time this compile's traversal reaches a given Parameter, not its
process-global `uid` - `uid` still identifies the Parameter for dedup, cycle
detection and the ptr registry's keys, but never appears in emitted text, so
an unrelated allocation upstream that shifts every uid does not change this
source at all. See _SpanParser._register_ptr.

uploaded once per compile() via cp.RawModule.get_global. A member is `const`
when nothing in the unit writes that parameter, `T*` otherwise. Every
`__global__` and `__device__` function in the module sees the same block, so a
helper reaches a bound Parameter exactly the way its caller does - there is no
argument to thread through and no call site to rewrite. At the top of each
function body one local is declared per pointer that function's own spans
reference:

    const float* __restrict__ p_<idx> = pf_params.p_<idx>;

read (or written, dropping `const`) through for the rest of that body. This is
what keeps a function's own accesses provably non-aliasing to the compiler,
the same guarantee a `__restrict__` kernel argument used to carry - reading
`pf_params.p_<idx>` directly, span by span, would lose it.

A const parameter can also be used bare, outside any span, in which case it
arrives as a `#define`. Only names the source actually mentions are defined,
which keeps macros for common identifiers - N, DIM, EPS, min - from silently
rewriting unrelated code in the translation unit.

One `cp.RawModule` is built per compilation unit: the constant block, every
`__device__` helper the unit reaches (each emitted once, however many call
sites share it - see _SpecializeCtx), and the unit's `__global__` kernel.
`CupyKernel._raw_cache` keys the module on its final source text, same as the
single-kernel cache did before - the whole specialization still reduces to
that text, local-index-qualified struct members included, so it remains a
sound key, and since that index no longer depends on uid the key (and so the
cache hit rate) is stable across process restarts for an unchanged program.
A cache hit skips recompilation but the constant block is re-uploaded
regardless, since the pointers a compile's bindings currently resolve to are
not part of what the cache key captures.

Author: B.G (07/2026)
"""

import re
from typing import Any

import cupy as cp
import numpy as np

from .bag import Bag
from .compile import HelperBuilder, Kernel, KernelBuilder, _SpecializedHelper, _SpecializeCtx
from .parameter import MODES, Parameter
from .routine import Routine, RoutineBuilder, _CompiledStep

_KERNEL_NAME_RE = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
# the return type is one-or-more tokens, matched non-greedily so the LAST one
# before the parameter list is the function name - `__device__ unsigned int f(`
# names f, not int.
_DEVICE_NAME_RE = re.compile(r"__device__\s+(?:[\w:\*&]+\s+)+?(\w+)\s*\(")
_KERNEL_SIG_RE = re.compile(r"(__global__\s+void\s+\w+\s*\()(.*?)(\))", re.S)
_SPAN_RE = re.compile(r"\$(.*?)\$", re.S)
_CALL_RE = re.compile(r"([\w.]+)\s*(?:\((.*)\))?\s*$", re.S)

_CTYPE = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.int32): "int",
    np.dtype(np.int64): "long long",
    np.dtype(np.uint8): "unsigned char",
    np.dtype(np.uint32): "unsigned int",
}


def _ctype(dtype) -> str:
    """
    CUDA scalar type name for a (numpy) dtype.

    Author: B.G (07/2026)
    """
    return _CTYPE[np.dtype(dtype)]


def _cuda_literal(value) -> str:
    """
    Format a resolved const value as a CUDA literal.

    Author: B.G (07/2026)
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value)}f"
    return str(value)


def _extract_name(pattern: re.Pattern, template: str, kind: str) -> str:
    """
    The `__global__`/`__device__` function's own name, read out of the source
    text - that is the entry point cp.RawModule.get_function is looked up by.

    Author: B.G (07/2026)
    """
    match = pattern.search(template)
    if not match:
        raise ValueError(f"could not find a {kind} function name in template source")
    return match.group(1)


def _split_args(argstr: str) -> list[str]:
    """
    Split a call-argument string on top-level commas (respecting nesting).

    Author: B.G (07/2026)
    """
    parts, depth, cur = [], 0, ""
    for ch in argstr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _declared_arity(template: str) -> int:
    """
    How many data arguments a kernel template's `__global__` signature
    declares - the exact number its compiled form must be launched with, and
    what CupyRoutineBuilder maps a step's data_handle_ref onto.

    Author: B.G (07/2026)
    """
    match = _KERNEL_SIG_RE.search(template)
    if not match:
        raise ValueError("could not find a __global__ signature in template source")
    argstr = match.group(2).strip()
    return len(_split_args(argstr)) if argstr else 0


def _walk(path: list[str], bindings: dict[str, Any]):
    """
    Resolve a dotted path against bindings, descending Bag members.

    Author: B.G (07/2026)
    """
    obj = bindings[path[0]]
    for seg in path[1:]:
        obj = obj[seg] if isinstance(obj, Bag) else getattr(obj, seg)
    return obj


def _param_argname(param: Parameter, local_index: dict[int, int]) -> str:
    """
    The struct member / local variable name a Parameter's pointer is reached
    through - stable for the object's whole lifetime *within this compile*
    since it is derived from `local_index[param.uid]`, a per-compilation-unit
    index assigned in first-encounter order (see _SpanParser._register_ptr),
    not from `uid` itself. `uid` still identifies the Parameter for dedup (two
    spans reaching the same Parameter under two different handles look up the
    same local index and therefore compute the same argname/struct member),
    but the emitted name no longer carries the process-global uid, which is
    what keeps generated source byte-stable across runs regardless of
    allocation order upstream.

    Author: B.G (07/2026)
    """
    return f"p_{local_index[param.uid]}"


class _SpanParser:
    """
    Expands the `$...$` spans in a CUDA template body.

    A span reaching a scalar or field Parameter registers it into `ctx`'s
    shared pointer registry (`ctx.cupy_ptr_registry`, one dict per compile()
    - see CupyKernelBuilder.compile) rather than generating a call argument:
    the registry is what becomes the module's `pf_params` constant block.
    `local_ptrs` tracks, separately, only the pointers *this* body actually
    used - CupyKernelBuilder.compile and CupyHelperBuilder._specialize use it
    to prepend that function's own `__restrict__` locals, so a function
    declares locals for what it reads or writes and nothing else. `ctx` is the
    compile this parse belongs to - a span reaching a CupyHelperBuilder
    specializes it against `ctx` (memoized there - see _SpecializeCtx), so a
    helper reached twice in one compile is specialized once, and any pointer
    it registers lands in the same shared registry as the kernel's own.

    A span reaching a CupyHelper does not collect its source here: that would
    mean two parents sharing one leaf helper each carry a full copy of the
    leaf's text in their own body, so the translation unit ends up with that
    `__device__` function defined twice the moment both parents are bound
    into one kernel. Every reachable helper's own (non-nested) source is
    instead registered once, by name, into `ctx.cupy_device_srcs` - see
    CupyHelperBuilder._specialize and CupyKernelBuilder.compile, which is
    where the deduplicated, dependency-ordered set for the whole unit is
    assembled.

    Author: B.G (07/2026)
    """

    def __init__(self, bindings: dict[str, Any], *, ctx: _SpecializeCtx):
        self.bindings = bindings
        self.ctx = ctx
        self.local_ptrs: dict[int, dict] = {}   # uid -> {ctype, write}

    def _register_ptr(self, param: Parameter, write: bool) -> str:
        registry = getattr(self.ctx, "cupy_ptr_registry", None)
        if registry is None:
            registry = self.ctx.cupy_ptr_registry = {}
        local_index = getattr(self.ctx, "cupy_local_index", None)
        if local_index is None:
            local_index = self.ctx.cupy_local_index = {}
        uid = param.uid
        entry = registry.get(uid)
        if entry is None:
            entry = {"ctype": _ctype(param.dtype), "write": False, "array": param.get().data}
            registry[uid] = entry
        if write:
            entry["write"] = True
        if uid not in local_index:
            # first encounter of this Parameter in this compile - assign it
            # the next local index, in traversal order (see _param_argname).
            local_index[uid] = len(local_index)
        local = self.local_ptrs.get(uid)
        if local is None:
            self.local_ptrs[uid] = {"ctype": entry["ctype"], "write": write}
        elif write:
            local["write"] = True
        return _param_argname(param, local_index)

    def _expand_param(self, param: Parameter, method: str, call_args: list[str]) -> str:
        if method == "get":
            if param.mode == "const":
                return _cuda_literal(param.get())
            argname = self._register_ptr(param, write=False)
            if param.mode == "scalar":
                return f"{argname}[0]"
            idx = call_args[0] if call_args else "0"
            return f"{argname}[{idx}]"
        # set_node
        if param.mode == "const":
            raise ValueError(f"{param.name}: const parameter is read-only")
        if len(call_args) != 2:
            raise ValueError(f"{param.name}: set_node(node, value) takes two arguments")
        argname = self._register_ptr(param, write=True)
        node, val = call_args
        return f"{argname}[0] = {val}" if param.mode == "scalar" else f"{argname}[{node}] = {val}"

    def _repl(self, match: re.Match) -> str:
        cm = _CALL_RE.match(match.group(1).strip())
        if cm is None:
            raise ValueError(f"malformed span: ${match.group(1)}$")
        path = cm.group(1).split(".")
        argstr = cm.group(2)
        call_args = _split_args(argstr) if argstr is not None else []

        if path[-1] in ("get", "set_node"):
            target = _walk(path[:-1], self.bindings)
            if isinstance(target, Parameter):
                return self._expand_param(target, path[-1], call_args)

        target = _walk(path, self.bindings)
        if isinstance(target, CupyHelperBuilder):
            target = self.ctx.specialize(target)
        if isinstance(target, CupyHelper):
            return f"{target.name}({argstr if argstr is not None else ''})"
        if isinstance(target, Parameter):
            return self._expand_param(target, "get", ["0"])
        return _cuda_literal(target)

    def parse(self, body: str) -> str:
        """
        Expand every `$...$` span in `body`, accumulating local_ptrs (and,
        transitively, `ctx.cupy_device_srcs` - see CupyHelperBuilder._specialize)
        as a side effect, then prepend the `__restrict__` locals this body's
        own spans implied - see _insert_locals.

        Author: B.G (07/2026)
        """
        expanded = _SPAN_RE.sub(self._repl, body)
        local_index = getattr(self.ctx, "cupy_local_index", None) or {}
        return _insert_locals(expanded, self.local_ptrs, local_index)


def _insert_locals(body: str, local_ptrs: dict[int, dict], local_index: dict[int, int]) -> str:
    """
    Prepend one `__restrict__` local per pointer `body` itself references,
    reading through the module's `pf_params` constant block, right after the
    function's opening brace.

    Declared `const` unless this body writes that parameter anywhere - kept
    per function rather than read off the struct member (which is `const`
    only when *no* function in the whole unit writes it), so a function that
    only reads a parameter another function in the same unit writes still
    gets the non-aliasing benefit of a const-qualified local.

    Ordered by local index ascending (first-encounter order for this compile,
    see _SpanParser._register_ptr) rather than by uid, so this declaration
    block's text does not depend on the process-global uid values a run
    happened to assign upstream.

    Author: B.G (07/2026)
    """
    if not local_ptrs:
        return body
    idx = body.find("{")
    if idx == -1:
        raise ValueError("could not find a function body to insert parameter locals into")
    decls = "".join(
        f"    {'' if e['write'] else 'const '}{e['ctype']}* __restrict__ {_argname_for(local_index[uid])} = pf_params.{_argname_for(local_index[uid])};\n"
        for uid, e in sorted(local_ptrs.items(), key=lambda kv: local_index[kv[0]])
    )
    return f"{body[: idx + 1]}\n{decls}{body[idx + 1 :]}"


def _argname_for(local_idx: int) -> str:
    """
    The struct member / local name for a pointer already assigned local index
    `local_idx` in this compile - see _param_argname, which this must stay in
    lockstep with.

    Author: B.G (07/2026)
    """
    return f"p_{local_idx}"


def _const_defines(bindings: dict[str, Any], body: str) -> list[str]:
    """
    A `#define NAME literal` for each top-level const Parameter whose name
    appears in `body`, matched on word boundaries.

    Pass the span-expanded text, not the raw template: by then a span such as
    `$phys.dx.get(0)$` is a numeric literal and no longer reads as a bare
    mention of a const name. Names the source never mentions
    are left undefined, keeping macros for identifiers like N, DIM, EPS or min
    out of the translation unit, where they would rewrite unrelated code.

    Author: B.G (07/2026)
    """
    return [
        f"#define {name} {_cuda_literal(obj.get())}"
        for name, obj in bindings.items()
        if isinstance(obj, Parameter) and obj.mode == "const" and re.search(rf"\b{re.escape(name)}\b", body)
    ]


def _param_block_source(registry: dict[int, dict], local_index: dict[int, int]) -> str:
    """
    The `pf_params_t` struct and its `__constant__` instance for one
    compilation unit's pointer registry - empty when the unit reaches no
    scalar/field Parameter, so a unit with only consts and bare helpers emits
    no block at all.

    Member order is by local index, ascending - i.e. first-encounter order
    during this compile's traversal (see _SpanParser._register_ptr), not by
    uid. This is what keeps the struct's text (and therefore the whole
    generated source) independent of the process-global uid values, so an
    unrelated allocation upstream that shifts every uid does not change this
    text. _upload_param_block writes pointers in the same order.

    Author: B.G (07/2026)
    """
    if not registry:
        return ""
    members = "".join(
        f"    {'' if e['write'] else 'const '}{e['ctype']}* {_argname_for(local_index[uid])};\n"
        for uid, e in sorted(registry.items(), key=lambda kv: local_index[kv[0]])
    )
    return f"struct pf_params_t {{\n{members}}};\n__constant__ pf_params_t pf_params;\n"


def _upload_param_block(module: "cp.RawModule", registry: dict[int, dict], local_index: dict[int, int]) -> None:
    """
    Copy the current pointer for every registered Parameter into the module's
    `pf_params` constant block, in the same local-index order the struct was
    emitted in (see _param_block_source).

    Runs once per compile(), synchronously - safe as an ordinary host->device
    copy anywhere a kernel launch would be, but not inside CUDA graph capture
    (see CupyRoutineBuilder.compile, which compiles every step - and so
    performs every upload - before capture starts).

    Author: B.G (07/2026)
    """
    if not registry:
        return
    global_ptr = module.get_global("pf_params")
    ptrs = np.array(
        [e["array"].data.ptr for _, e in sorted(registry.items(), key=lambda kv: local_index[kv[0]])],
        dtype=np.uint64,
    )
    view = cp.ndarray(ptrs.shape, dtype=np.uint64, memptr=global_ptr)
    view.set(ptrs)


class CupyParameter(Parameter):
    """
    Parameter backed by a const python value or a pooled CupyDataHandle.

    dtypes are numpy dtypes throughout, so they need no translation. There is
    no device_view() either: a parameter reaches device code when the span
    parser substitutes it into the source.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, *, dtype, mode: str, value, pool, n_flat: int | None = None):
        """
        Declare and initialize one parameter. "scalar"/"field" modes allocate
        pooled storage immediately via `pool`; "const" stays a plain python
        value, read bare in a template body as a #define.

        Author: B.G (07/2026)
        """
        if mode not in MODES:
            raise ValueError(f"{name}: mode must be one of {sorted(MODES)}, got {mode!r}")

        super().__init__()
        self.name = name
        self.dtype = dtype
        self.mode = mode
        self._pool = pool
        self._const_value: Any = None
        self._handle = None

        if mode == "scalar":
            self._handle = pool.get_data(dtype, ())
        elif mode == "field":
            if n_flat is None:
                raise ValueError(f"{name}: field mode requires n_flat")
            self._handle = pool.get_data(dtype, (n_flat,))

        self._store(value)

    def get(self):
        """
        The python value for const mode, the backing CupyDataHandle otherwise.

        Author: B.G (07/2026)
        """
        return self._const_value if self.mode == "const" else self._handle

    def set(self, value) -> None:
        """
        Overwrite the whole value: a device write for scalar, a full
        host->device copy for field. const is immutable - see Parameter.set.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            raise ValueError(
                f"{self.name}: const parameter is immutable; build a new Parameter and "
                f"replace() it into the bag, then recompile"
            )
        self._store(value)

    def _store(self, value) -> None:
        """
        Write `value` according to the mode, with no immutability check - the
        one path that may set a const, used by __init__ to place its initial
        value.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            self._const_value = np.dtype(self.dtype).type(value).item()
        elif self.mode == "scalar":
            self._handle.data[...] = value
        else:  # field
            arr = np.asarray(value, dtype=self.dtype).reshape(-1)
            self._handle.from_numpy(arr)

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            raise ValueError(f"{self.name}: const parameter is read-only")
        if self.mode == "scalar":
            self._handle.data[...] = value
        else:  # field
            self._handle.data[node] = value

    def destroy(self) -> None:
        """
        Return any pooled storage to the pool. const mode owns none, so this
        is a no-op there.

        Author: B.G (07/2026)
        """
        if self._handle is not None:
            self._pool.release_data(self._handle)
            self._handle = None


class CupyHelper(_SpecializedHelper):
    """
    A device helper's specialization, held as CUDA `__device__` source text.
    Produced by a CupyHelperBuilder as part of an enclosing kernel's
    compile(); see HelperBuilder.

    Nothing is compiled at this stage: RawModule compiles whole translation
    units, so `.compiled` hands back source, and the helper is compiled as part
    of each kernel that splices it in.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, source: str):
        super().__init__()
        self.name = name
        self._compiled_source = source

    @property
    def compiled(self):
        """
        The spliced `__device__` source text, for pasting into whatever
        kernel or helper binds this one.

        Author: B.G (07/2026)
        """
        return self._compiled_source

    def __call__(self, *args, **kwargs):
        """
        CUDA source is not a python callable - a helper only runs inside a
        compiled kernel's device code.

        Author: B.G (07/2026)
        """
        raise RuntimeError(
            f"Helper '{self.name}' is CUDA source, only callable from a compiled kernel's device code"
        )


class CupyKernel(Kernel):
    """
    A launchable kernel, backed by a function pulled from a compiled
    cp.RawModule (see CupyKernelBuilder.compile).

    Launch dimensions are explicit: a RawModule function has nothing like
    Taichi's auto-ranging, so __call__ takes grid and block. The positional
    arguments are exactly the template's own declared data arguments - every
    bound scalar/field Parameter reaches this kernel through the module's
    constant block instead of a launch argument, so there is nothing else to
    append here the way an earlier version of this class needed to.

    Author: B.G (07/2026)
    """

    _raw_cache: dict[str, "cp.RawModule"] = {}

    def __init__(self, name: str, compiled, module: "cp.RawModule", arity: int):
        super().__init__()
        self.name = name
        self._compiled = compiled
        self._module = module
        self._arity = arity

    @property
    def compiled(self):
        """
        The underlying RawModule function this Kernel's __call__ launches.

        Author: B.G (07/2026)
        """
        return self._compiled

    @property
    def module(self) -> "cp.RawModule":
        """
        The cp.RawModule this kernel's function was pulled from - shared with
        every other kernel compiled from the same final source text (see
        CupyKernel._raw_cache).

        Author: B.G (07/2026)
        """
        return self._module

    def __call__(self, *args, grid, block, **kwargs):
        """
        Launches the compiled kernel. `grid`/`block` are int or tuple launch
        dims, required since a RawModule function has no default range.

        The argument count is checked against the `__global__` signature
        first. A RawModule launch does no such check itself: the wrong number
        of arguments reads whatever follows the packed argument buffer on the
        device and takes the process down with it, so a plain int comparison
        here - against a ~5 us launch - buys a named error instead of a
        segfault with no diagnostic.

        Author: B.G (07/2026)
        """
        if len(args) != self._arity:
            raise TypeError(
                f"kernel '{self.name}' declares {self._arity} data argument(s), got {len(args)}"
            )
        grid = (grid,) if isinstance(grid, int) else tuple(grid)
        block = (block,) if isinstance(block, int) else tuple(block)
        return self._compiled(grid, block, tuple(args))


class CupyHelperBuilder(HelperBuilder):
    """
    Recipe for a device helper compiled from CUDA `__device__` source. A span
    inside one may reach any Parameter mode and any other device helper,
    exactly like a kernel's own spans - see the module docstring's constant
    block. Specialized only as part of an enclosing kernel's compile() - see
    HelperBuilder; compile() itself raises.

    Author: B.G (07/2026)
    """

    def _specialize(self, ctx: _SpecializeCtx) -> CupyHelper:
        """
        Expand the template's spans against `ctx` and prepend the const
        #defines it turned out to need, giving this helper's own `__device__`
        text - its own body only, not any dependency's. Any scalar/field
        Parameter a span here reaches is registered into `ctx.cupy_ptr_registry`
        exactly as one reached from the enclosing kernel would be (see
        _SpanParser) - the block that results is shared, so this helper and
        its caller read the same pointer.

        This helper's own text is also registered, by name, into
        `ctx.cupy_device_srcs` - the deduplicated, dependency-ordered set of
        every `__device__` function the whole compile reaches (see
        CupyKernelBuilder.compile). parser.parse() above resolves this
        helper's own spans first, which is what recursively specializes (and
        so registers) every helper *this* one calls before this line runs -
        so by the time this helper registers itself, its own dependencies are
        already in the registry, ahead of it. A helper already reached once in
        this compile - by this call site or an earlier one - keeps its first
        registration; `setdefault` leaves it untouched rather than duplicating
        or reordering it.

        Author: B.G (07/2026)
        """
        template = self._template
        name = _extract_name(_DEVICE_NAME_RE, template, "__device__")
        parser = _SpanParser(self._bindings, ctx=ctx)
        body = parser.parse(template)
        source = "\n".join(_const_defines(self._bindings, body) + [body])
        device_srcs = getattr(ctx, "cupy_device_srcs", None)
        if device_srcs is None:
            device_srcs = ctx.cupy_device_srcs = {}
        device_srcs.setdefault(name, source)
        return CupyHelper(name, source)


class CupyKernelBuilder(KernelBuilder):
    """
    Builds a CupyKernel from CUDA `__global__` source. Spans expand and, for
    scalar/field params, auto-generate the matching pointer args + launch
    arrays.

    Author: B.G (07/2026)
    """

    def compile(self) -> CupyKernel:
        """
        Expand the template's spans - the kernel's own and, recursively,
        every CupyHelperBuilder they reach - prepend the whole unit's
        `__device__` helper sources (deduplicated by name, dependency-first -
        see CupyHelperBuilder._specialize and `ctx.cupy_device_srcs`), the
        kernel's own const #defines, and the `pf_params` constant block the
        whole unit's scalar/field Parameters collected into, then build (or
        reuse, keyed by final source text) the cp.RawModule and pull this
        kernel's function out of it.

        Opens a fresh _SpecializeCtx for this compile, so every
        CupyHelperBuilder this kernel's spans reach is specialized once,
        against these bindings, and every scalar/field Parameter any of them
        binds lands in one shared pointer registry
        (`ctx.cupy_ptr_registry`) - see _SpanParser and _param_block_source.
        The same ctx is what lets a helper reachable from two of this
        kernel's own bindings - or from two different helpers this kernel
        reaches - contribute its `__device__` text to `ctx.cupy_device_srcs`
        exactly once: emitting a shared leaf's definition once per
        translation unit, rather than once per parent that calls it, is what
        the module docstring's "reaches it exactly the way its caller does"
        promise requires, and repeated emission is a compile error a CUDA
        translation unit does not tolerate the way a repeated `#define` of
        an identical macro does.

        The registry - and so the set of pointers to upload - is rebuilt by
        parsing on every call, cache hit or not, since the final source text
        (the cache key) does not capture what a Parameter's storage currently
        points at. A cache hit therefore skips recompilation but never skips
        the upload; see _upload_param_block.

        Author: B.G (07/2026)
        """
        ctx = _SpecializeCtx()
        ctx.cupy_ptr_registry = {}
        ctx.cupy_local_index = {}
        ctx.cupy_device_srcs = {}
        template = self._template
        name = _extract_name(_KERNEL_NAME_RE, template, "__global__")
        parser = _SpanParser(self._bindings, ctx=ctx)
        body = parser.parse(template)
        # extern "C" linkage so cp.RawModule finds the entry by its plain name
        # (C++ would name-mangle it); helper __device__ funcs stay mangled and
        # link fine within the same translation unit.
        if 'extern "C"' not in body:
            body = body.replace("__global__", 'extern "C" __global__', 1)
        registry = ctx.cupy_ptr_registry
        local_index = ctx.cupy_local_index
        source = "\n".join(
            [_param_block_source(registry, local_index)]
            + list(ctx.cupy_device_srcs.values())
            + _const_defines(self._bindings, body)
            + [body]
        )

        module = CupyKernel._raw_cache.get(source)
        if module is None:
            module = cp.RawModule(code=source)
            CupyKernel._raw_cache[source] = module
        _upload_param_block(module, registry, local_index)
        raw = module.get_function(name)
        krn = CupyKernel(name, raw, module, arity=_declared_arity(template))
        krn._final_source = source
        return krn


class _CapturedRoutine(Routine):
    """
    A Routine whose steps have been recorded into a CUDA graph; calling it
    replays that graph instead of re-issuing each step's kernel launch.

    Built by CupyRoutineBuilder.compile(captured=True) (the default there).
    Holds the same steps/data_names/defaults an uncaptured Routine would, so
    introspection agrees between the two, plus the captured cp.cuda.Graph and
    the private stream the capture was recorded on.

    Capture needs a stream of its own - the default stream cannot be put in
    capture mode - but replay does not, and the graph is launched on the
    caller's current stream. A routine that replayed on its private stream
    would order against nothing the caller had queued, so reading a result
    straight after a call would need a device-wide synchronize to be correct,
    which no other launch in this package asks for. Launching on the current
    stream keeps a captured Routine interchangeable with an uncaptured one and
    with a plain Kernel: queue work, call, read.

    A captured graph bakes in the device pointers it was captured with -
    every launch it holds already has its bulk-data argument list resolved,
    the same arguments a step's `data_handle_ref` maps onto. A step's bound
    scalar/field Parameters are not part of that argument list at all: they
    reach the kernel through its module's `pf_params` constant block (see the
    module docstring), read fresh by the device on every execution rather
    than captured as part of the launch. Replay therefore sees whatever that
    block currently holds - which is exactly what the block held at the last
    compile(), since nothing but a compile()'s upload ever writes to it - so
    the staleness rules below apply the same way whether a pointer reached a
    step through its launch arguments or through its constant block. Call-time
    data handle overrides (the `rout(A, B)` form) cannot be honoured against
    a captured graph's already-resolved bulk-data arguments: doing so would
    either use the wrong buffers silently or require a fresh capture on a
    call site that looks like an ordinary launch, hiding a real performance
    cliff behind what reads as a cheap replay. This raises instead -
    compile(captured=False) for a routine that overrides are meant to work
    against.

    See the module docstring's "Contract: no set()/destroy() mid-routine" for
    the staleness rules a Routine has always had; capture adds one more way a
    compiled Routine can go stale without anything checking for it:
    - a write to a scalar or field Parameter reached by this routine's bag
      goes through the same storage both the graph's launch arguments and its
      steps' constant blocks already point at, so replay keeps seeing the new
      value - this is the intended way to feed a captured routine changing
      data, same as for an uncaptured one;
    - set() on a const Parameter changes generated source, which the graph
      never re-reads - the routine (and the graph baked into it) must be
      recompiled;
    - destroy() on any Parameter or data handle this routine reaches, or
      anything else that returns a buffer to the pool, invalidates a pointer
      the graph's launches were captured with or a step's constant block was
      last uploaded with. Recompile.
    None of this is enforced at runtime; it is exactly the discipline a
    Kernel or an uncaptured Routine already asks for, just extended to a
    graph's baked-in pointers - launch arguments and constant blocks alike -
    as an added way "recompile after this" applies.

    Author: B.G (07/2026)
    """

    def __init__(self, steps: list, data_names: tuple, defaults: dict, graph, stream):
        super().__init__(steps, data_names, defaults)
        self._graph = graph
        self._stream = stream

    def __call__(self, *args) -> None:
        """
        Replay the captured graph on the caller's current stream, so the call
        orders against surrounding work exactly as an uncaptured Routine's
        launches would. Takes no arguments - see the class docstring for why
        call-time data handle overrides are rejected rather than honoured or
        silently re-captured.

        Author: B.G (07/2026)
        """
        if args:
            raise RuntimeError(
                "Routine: this routine was compiled with captured=True; call-time data "
                "handle overrides are not supported against a captured CUDA graph, since "
                "the graph's launches already have their pointers baked in. Compile with "
                "captured=False for a routine meant to be called with overrides, or build "
                "a second routine over the override handles and capture that one."
            )
        self._graph.launch()


class CupyRoutineBuilder(RoutineBuilder):
    """
    Compiles an ordered sequence of compiled-kernel launches sharing one bag
    into a Routine.

    A step's data arity is read off the `__global__` signature as written in
    the ingested source - the template author's own declared data arguments,
    the same ones data_handle_ref maps onto. Every bound scalar/field
    Parameter a step's spans reach travels through its module's constant
    block instead (see the module docstring), so nothing is appended to this
    count at compile time the way an earlier version of this class needed to
    account for.

    `grid`/`block` have no auto-ranging equivalent on cupy the way Taichi and
    Quadrants derive one from the template, so they are resolved once per
    step: whatever add_kernel(..., grid=..., block=...) gave that step, else
    the default passed to this builder's own constructor. Neither given
    raises at compile time, naming the step.

    Author: B.G (07/2026)
    """

    def __init__(self, *, grid=None, block=None):
        super().__init__()
        self._default_grid = grid
        self._default_block = block

    def _data_arity(self, kernel_builder: KernelBuilder) -> int:
        template = kernel_builder.template
        if template is None:
            raise ValueError("add_kernel: kernel_builder has no ingested template")
        try:
            return _declared_arity(template)
        except ValueError as exc:
            raise ValueError(f"add_kernel: {exc}") from exc

    def _make_caller(self, compiled_kernel, grid, block):
        grid = grid if grid is not None else self._default_grid
        block = block if block is not None else self._default_block
        if grid is None or block is None:
            raise ValueError(
                f"CupyRoutineBuilder: no grid/block for step '{compiled_kernel.name}' - "
                "pass grid=/block= to the builder or to this step's add_kernel"
            )

        def caller(*args):
            return compiled_kernel(*args, grid=grid, block=block)

        return caller

    def compile(self, captured: bool = True, dump_source: str | None = None) -> Routine:
        """
        Validate (RoutineBuilder._validate), compile every step's kernel, and
        either return a Routine that launches them in order (captured=False)
        or capture that same sequence of launches into a CUDA graph and
        return a Routine that replays it (captured=True, the default here).

        `dump_source` is accepted for signature parity with the closure
        backend's fused compile() and ignored - there is no generated source
        on this backend either way.

        captured=False is exactly RoutineBuilder.compile's base behaviour:
        one host-side launch per step, every call. It is the reference the
        captured path is diffed against, and stays reachable as a runtime
        switch - in particular it is the only way to get a Routine that
        accepts call-time data handle overrides (see _CapturedRoutine).

        captured=True compiles every step exactly as captured=False does -
        deduplicated within this call, keyed on id(kernel_builder), so a
        begin_repeat()/end_repeat() block that unrolls the same
        KernelBuilder into several steps compiles it once, not once per
        repetition - then:
        1. Warms up each step by launching it once, for real, on the
           default stream. Each step's compile() (just above, in this same
           method) already built its cp.RawModule and uploaded its constant
           block before this point, so the warmup is not covering a lazy
           module load the way it would have to for a JIT that only happened
           on first launch; it remains cheap insurance against any other
           first-launch cost CUDA's capture machinery would rather not see
           mid-capture (a launch that fails or behaves unusually the first
           time either fails the capture outright or bakes in a broken graph
           node).
        2. Restores every data buffer this routine reaches (add_data's
           handles) to the values it captured a copy of before warming up
           - a warmup launch is a real launch and actually computes into
           those buffers, and compile() otherwise would not be the
           side-effect-free operation every other compile() in this
           package is.
        3. Captures the same step sequence again, on a dedicated
           non-blocking stream, via cp.cuda.Stream.begin_capture() /
           end_capture(). Nothing on that stream executes during capture -
           only the graph is built - so this pass leaves the (already
           restored) buffers untouched. Each step's constant block (see the
           module docstring) was already uploaded synchronously by its own
           compile() call, above, well before this point - a synchronous
           copy issued while a stream is being captured is illegal, so that
           upload could not happen here even if it needed to.
        4. Checks the default cupy memory pool's used_bytes() is the same
           before and after capture and raises if not. Every data handle a
           routine launches with is already allocated by the time compile()
           runs (add_data takes an existing handle), so nothing captured
           here should need the pool; growth here means some step
           allocated during capture, which CUDA graph capture does not
           support - this is caught rather than silently producing an
           unusable graph.

        The returned _CapturedRoutine keeps the dedicated stream and the
        cp.cuda.Graph alive; __call__ replays the graph with no arguments -
        overriding data handles at call time is rejected, see
        _CapturedRoutine.__call__.

        Author: B.G (07/2026)
        """
        if not captured:
            return super().compile(fused=False)

        self._validate()

        compiled_steps: list[_CompiledStep] = []
        data_names: list[str] = []
        compiled_cache: dict[int, Any] = {}
        for step in self._steps:
            key = id(step.kernel_builder)
            compiled = compiled_cache.get(key)
            if compiled is None:
                compiled = step.kernel_builder.compile()
                compiled_cache[key] = compiled
            caller = self._make_caller(compiled, step.grid, step.block)
            compiled_steps.append(_CompiledStep(caller, step.canonical_refs))
            for name in step.canonical_refs:
                if name not in data_names:
                    data_names.append(name)

        defaults = {name: self._data[name] for name in data_names}

        def _launch_all():
            for step in compiled_steps:
                step.caller(*(defaults[name] for name in step.canonical_refs))

        # 1. warm up every kernel with one real launch, so any remaining
        # first-launch cost happens before capture starts, not during it -
        # the module itself is already built and its constant block already
        # uploaded, by compile() just above.
        snapshots = {name: buf.copy() for name, buf in defaults.items()}
        _launch_all()
        cp.cuda.Device().synchronize()

        # 2. undo the warmup launch's real effect - compile() must not leave
        # the caller's buffers different from how it found them.
        for name, buf in defaults.items():
            buf[...] = snapshots[name]
        cp.cuda.Device().synchronize()

        # 3. capture the same sequence on a dedicated stream.
        mempool = cp.get_default_memory_pool()
        used_before = mempool.used_bytes()
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            _launch_all()
            graph = stream.end_capture()

        # 4. no allocation should have happened while capturing.
        used_after = mempool.used_bytes()
        if used_after != used_before:
            raise RuntimeError(
                "CupyRoutineBuilder.compile(captured=True): the default memory pool's "
                f"used_bytes() changed during capture ({used_before} -> {used_after}); a "
                "step allocated instead of using an already-initialised data handle, which "
                "CUDA graph capture does not support"
            )

        return _CapturedRoutine(compiled_steps, tuple(data_names), defaults, graph, stream)
