"""
cupy implementations of Parameter, Kernel and their builders, plus
CupyHelperBuilder - the recipe for a device helper, specialized as part of
whichever kernel binds it (see base.py, HelperBuilder).

Here a template is CUDA source text rather than a python function, since
cp.RawKernel compiles source and there is no function whose globals could be
patched. Bound objects are written into that source as `$...$` spans holding a
dotted path, which keeps the in-kernel spelling the same as on the other
backends:

    $p.get(i)$        read parameter p at flat index i
    $p.set_node(i,v)$ write parameter p at flat index i
    $grid.nx.get(i)$  reach a bag member
    $helper(a, b)$    call a bound device helper

Compiling substitutes each span according to the parameter's mode - a CUDA
literal for const, NAME[0] for scalar, NAME[i] for field - and, for scalar and
field, also generates the pointer argument that expansion implies. Those
arguments are appended to the __global__ signature and their arrays supplied at
launch, so a template never declares them by hand. A parameter only read is
declared `const T*`; one written anywhere in the kernel is declared `T*`.
Writing to a const parameter is an error.

A const parameter can also be used bare, outside any span, in which case it
arrives as a `#define`. Only names the source actually mentions are defined,
which keeps macros for common identifiers - N, DIM, EPS, min - from silently
rewriting unrelated code in the translation unit.

Spans inside a device helper may only reach const parameters and other
device helpers. That is the framework rule from base.py rather than anything
specific to cupy: a helper is spliced into its caller and cannot take on a
pointer argument of its own.

Author: B.G (07/2026)
"""

import re
from typing import Any

import cupy as cp
import numpy as np

from .base import HelperBuilder, Kernel, KernelBuilder, Parameter, _SpecializedHelper, _SpecializeCtx, attach_meta
from .base import Bag
from .routine import Routine, RoutineBuilder, _CompiledStep

_KERNEL_NAME_RE = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
_DEVICE_NAME_RE = re.compile(r"__device__\s+[\w:\*&]+\s+(\w+)\s*\(")
_KERNEL_SIG_RE = re.compile(r"(__global__\s+void\s+\w+\s*\()(.*?)(\))", re.S)
_SPAN_RE = re.compile(r"\$(.*?)\$", re.S)
_CALL_RE = re.compile(r"([\w.]+)\s*(?:\((.*)\))?\s*$", re.S)

_CTYPE = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.int32): "int",
    np.dtype(np.int64): "long long",
    np.dtype(np.uint8): "unsigned char",
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
    text - that is the entry point cp.RawKernel is looked up by.

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


def _walk(path: list[str], bindings: dict[str, Any]):
    """
    Resolve a dotted path against bindings, descending Bag members.

    Author: B.G (07/2026)
    """
    obj = bindings[path[0]]
    for seg in path[1:]:
        obj = obj[seg] if isinstance(obj, Bag) else getattr(obj, seg)
    return obj


class _SpanParser:
    """
    Expands the `$...$` spans in a CUDA template body.

    Expansion has side effects the builder needs afterwards: `ptr_params`
    collects the pointer arguments the spans implied, and `helper_srcs` the
    source of every device helper they called. Set allow_arrays=False when
    parsing a device helper, which may not reach scalar or field parameters
    and so has no use for either. `ctx` is the compile this parse belongs to
    - a span reaching a CupyHelperBuilder specializes it against `ctx`
    (memoized there - see _SpecializeCtx), so a helper reached twice in one
    compile is specialized once.

    Author: B.G (07/2026)
    """

    def __init__(self, bindings: dict[str, Any], *, allow_arrays: bool, ctx: _SpecializeCtx):
        self.bindings = bindings
        self.allow_arrays = allow_arrays
        self.ctx = ctx
        self.ptr_params: dict[str, dict] = {}   # argname -> {ctype, write, array}
        self.helper_srcs: dict[str, str] = {}   # name -> source

    def _register_ptr(self, argname: str, param: Parameter, write: bool) -> None:
        entry = self.ptr_params.get(argname)
        if entry is None:
            self.ptr_params[argname] = {"ctype": _ctype(param.dtype), "write": write, "array": param.get().data}
        elif write:
            entry["write"] = True

    def _expand_param(self, param: Parameter, method: str, argname: str, call_args: list[str]) -> str:
        if method == "get":
            if param.mode == "const":
                return _cuda_literal(param.get())
            if not self.allow_arrays:
                raise ValueError(f"{param.name}: scalar/field param cannot be used in a device helper (no pointer arg)")
            self._register_ptr(argname, param, write=False)
            if param.mode == "scalar":
                return f"{argname}[0]"
            idx = call_args[0] if call_args else "0"
            return f"{argname}[{idx}]"
        # set_node
        if param.mode == "const":
            raise ValueError(f"{param.name}: const parameter is read-only")
        if not self.allow_arrays:
            raise ValueError(f"{param.name}: set_node cannot be used in a device helper (no pointer arg)")
        if len(call_args) != 2:
            raise ValueError(f"{param.name}: set_node(node, value) takes two arguments")
        self._register_ptr(argname, param, write=True)
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
                return self._expand_param(target, path[-1], "_".join(path[:-1]), call_args)

        target = _walk(path, self.bindings)
        if isinstance(target, CupyHelperBuilder):
            target = self.ctx.specialize(target)
        if isinstance(target, CupyHelper):
            self.helper_srcs[target.name] = target.compiled
            return f"{target.name}({argstr if argstr is not None else ''})"
        if isinstance(target, Parameter):
            return self._expand_param(target, "get", "_".join(path), ["0"])
        return _cuda_literal(target)

    def parse(self, body: str) -> str:
        """
        Expand every `$...$` span in `body`, accumulating ptr_params and
        helper_srcs as a side effect.

        Author: B.G (07/2026)
        """
        return _SPAN_RE.sub(self._repl, body)


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


def _inject_signature(kernel_src: str, ptr_params: dict[str, dict]) -> str:
    """
    Append the generated pointer params to the __global__ signature.

    Author: B.G (07/2026)
    """
    if not ptr_params:
        return kernel_src
    decls = [
        f"{'' if e['write'] else 'const '}{e['ctype']}* {name}"
        for name, e in ptr_params.items()
    ]
    joined = ", ".join(decls)

    def _sub(m: re.Match) -> str:
        existing = m.group(2).strip()
        sep = ", " if existing else ""
        return f"{m.group(1)}{m.group(2)}{sep}{joined}{m.group(3)}"

    return _KERNEL_SIG_RE.sub(_sub, kernel_src, count=1)


class CupyParameter(Parameter):
    """
    Parameter backed by a const python value or a pooled CupyDataHandle.

    dtypes are numpy dtypes throughout, so they need no translation. There is
    no device_view() either: a parameter reaches device code when the span
    parser substitutes it into the source.

    Author: B.G (07/2026)
    """

    SUPPORTED_MODES = frozenset({"const", "scalar", "field"})

    def __init__(self, name: str, *, dtype, mode: str, value, pool, n_flat: int | None = None, solo: bool = False):
        """
        Declare and initialize one parameter. "scalar"/"field" modes allocate
        pooled storage immediately via `pool`; "const" stays a plain python
        value. solo=True (const only) lets the parameter be read bare in a
        template body - it becomes a #define rather than a span expansion.

        Author: B.G (07/2026)
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"{name}: mode must be one of {sorted(self.SUPPORTED_MODES)}, got {mode!r}")
        if solo and mode != "const":
            raise ValueError(f"{name}: solo access is const-only, got mode {mode!r}")

        super().__init__()
        self.name = name
        self.dtype = dtype
        self.mode = mode
        self.solo = solo
        self._pool = pool
        self._const_value: Any = None
        self._handle = None

        if mode == "scalar":
            self._handle = pool.get_data(dtype, ())
        elif mode == "field":
            if n_flat is None:
                raise ValueError(f"{name}: field mode requires n_flat")
            self._handle = pool.get_data(dtype, (n_flat,))

        self.set(value)

    def get(self):
        """
        The python value for const mode, the backing CupyDataHandle otherwise.

        Author: B.G (07/2026)
        """
        return self._const_value if self.mode == "const" else self._handle

    def set(self, value) -> None:
        """
        Overwrite the whole value: a cast python scalar for const, a device
        write for scalar, a full host->device copy for field.

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

    Nothing is compiled at this stage: RawKernel compiles whole translation
    units, so `.compiled` hands back source, and the helper is compiled as part
    of each kernel that splices it in.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, source: str):
        super().__init__()
        self.name = name
        # note: distinct from Specializable._source (the raw template, set by
        # attach_meta) - this is the spliced __device__ source `.compiled` serves.
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
    A launchable kernel, backed by a compiled cp.RawKernel.

    Launch dimensions are explicit: RawKernel has nothing like Taichi's
    auto-ranging, so __call__ takes grid and block. The positional arguments
    are the template's own data arguments, and the arrays behind the generated
    pointer parameters follow them.

    Compiled kernels are cached on the class by final source text, which is
    what the whole specialization reduces to and therefore a sound key.

    Author: B.G (07/2026)
    """

    _raw_cache: dict[str, "cp.RawKernel"] = {}

    def __init__(self, name: str, compiled, bound_arrays: list):
        super().__init__()
        self.name = name
        self._compiled = compiled
        self._bound_arrays = bound_arrays

    @property
    def compiled(self):
        """
        The underlying cp.RawKernel this Kernel's __call__ launches.

        Author: B.G (07/2026)
        """
        return self._compiled

    def __call__(self, *args, grid, block, **kwargs):
        """
        Launches the compiled kernel. `grid`/`block` are int or tuple launch
        dims, required since RawKernel has no default range.

        Author: B.G (07/2026)
        """
        grid = (grid,) if isinstance(grid, int) else tuple(grid)
        block = (block,) if isinstance(block, int) else tuple(block)
        return self._compiled(grid, block, tuple(args) + tuple(self._bound_arrays))


class CupyHelperBuilder(HelperBuilder):
    """
    Recipe for a device helper compiled from CUDA `__device__` source. Spans
    may only reference const params / other device helpers (no pointer
    args). Specialized only as part of an enclosing kernel's compile() - see
    HelperBuilder; compile() itself raises.

    Author: B.G (07/2026)
    """

    def _specialize(self, ctx: _SpecializeCtx) -> CupyHelper:
        """
        Expand the template's spans against `ctx` and prepend the helper
        sources and const #defines it turned out to need, giving the final
        `__device__` text.

        Author: B.G (07/2026)
        """
        template = self._template
        name = _extract_name(_DEVICE_NAME_RE, template, "__device__")
        parser = _SpanParser(self._bindings, allow_arrays=False, ctx=ctx)
        body = parser.parse(template)
        source = "\n".join(list(parser.helper_srcs.values()) + _const_defines(self._bindings, body) + [body])
        fn = CupyHelper(name, source)
        attach_meta(fn, template, self._bindings)
        return fn


class CupyKernelBuilder(KernelBuilder):
    """
    Builds a CupyKernel from CUDA `__global__` source. Spans expand and, for
    scalar/field params, auto-generate the matching pointer args + launch
    arrays.

    Author: B.G (07/2026)
    """

    def compile(self) -> CupyKernel:
        """
        Expand the template's spans, inject the pointer args they implied into
        the __global__ signature, prepend helpers + const #defines, then build
        (or reuse, keyed by final source text) the cp.RawKernel.

        Opens a fresh _SpecializeCtx for this compile, so every
        CupyHelperBuilder this kernel's spans reach is specialized once,
        against these bindings.

        Author: B.G (07/2026)
        """
        ctx = _SpecializeCtx()
        template = self._template
        name = _extract_name(_KERNEL_NAME_RE, template, "__global__")
        parser = _SpanParser(self._bindings, allow_arrays=True, ctx=ctx)
        body = parser.parse(template)
        body = _inject_signature(body, parser.ptr_params)
        # extern "C" linkage so cp.RawKernel finds the entry by its plain name
        # (C++ would name-mangle it); helper __device__ funcs stay mangled and
        # link fine within the same translation unit.
        if 'extern "C"' not in body:
            body = body.replace("__global__", 'extern "C" __global__', 1)
        source = "\n".join(list(parser.helper_srcs.values()) + _const_defines(self._bindings, body) + [body])
        bound_arrays = [e["array"] for e in parser.ptr_params.values()]

        raw = CupyKernel._raw_cache.get(source)
        if raw is None:
            raw = cp.RawKernel(source, name)
            CupyKernel._raw_cache[source] = raw
        krn = CupyKernel(name, raw, bound_arrays)
        krn._final_source = source
        attach_meta(krn, template, self._bindings)
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
    every launch it holds already has its argument list resolved. Call-time
    data handle overrides (the `rout(A, B)` form) cannot be honoured against
    that: doing so would either use the wrong buffers silently or require a
    fresh capture on a call site that looks like an ordinary launch, hiding a
    real performance cliff behind what reads as a cheap replay. This raises
    instead - compile(captured=False) for a routine that overrides are meant
    to work against.

    See the module docstring's "Contract: no set()/destroy() mid-routine" for
    the staleness rules a Routine has always had; capture adds one more way a
    compiled Routine can go stale without anything checking for it:
    - a write to a scalar or field Parameter reached by this routine's bag
      goes through the same storage the graph's launches point at, so replay
      keeps seeing the new value - this is the intended way to feed a
      captured routine changing data, same as for an uncaptured one;
    - set() on a const Parameter changes generated source, which the graph
      never re-reads - the routine (and the graph baked into it) must be
      recompiled;
    - destroy() on any Parameter or data handle this routine reaches, or
      anything else that returns a buffer to the pool, invalidates the
      pointers the graph's launches were captured with. Recompile.
    None of this is enforced at runtime; it is exactly the discipline a
    Kernel or an uncaptured Routine already asks for, just extended to a
    graph's baked-in pointers as an added way "recompile after this" applies.

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
    Compiles an ordered sequence of cp.RawKernel launches sharing one bag
    into a Routine.

    A step's data arity is read off the `__global__` signature as written in
    the ingested source, before compile() appends the pointer arguments
    spans imply - so it counts exactly the arguments the template author
    declared, the same ones data_handle_ref maps onto.

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
        match = _KERNEL_SIG_RE.search(template)
        if not match:
            raise ValueError("add_kernel: could not find a __global__ signature in template source")
        argstr = match.group(2).strip()
        return len(_split_args(argstr)) if argstr else 0

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

        captured=True compiles every step exactly as captured=False does,
        then:
        1. Warms up each step by launching it once, for real, on the
           default stream. cp.RawKernel compiles its module lazily, on
           first launch; that JIT must not happen while capturing (a
           RawKernel launched for the first time mid-capture either fails
           the capture outright or bakes in a broken graph node - CUDA's
           capture machinery assumes the module is already resident).
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
           restored) buffers untouched.
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
        for step in self._steps:
            compiled = step.kernel_builder.compile()
            caller = self._make_caller(compiled, step.grid, step.block)
            compiled_steps.append(_CompiledStep(caller, step.canonical_refs))
            for name in step.canonical_refs:
                if name not in data_names:
                    data_names.append(name)

        defaults = {name: self._data[name] for name in data_names}

        def _launch_all():
            for step in compiled_steps:
                step.caller(*(defaults[name] for name in step.canonical_refs))

        # 1. warm up every kernel with one real launch, so first-launch
        # module JIT happens before capture starts, not during it.
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
