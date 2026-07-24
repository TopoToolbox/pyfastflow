"""
cupy implementations of Parameter, DeviceFunction, Kernel and their builders.

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

Spans inside a device function may only reach const parameters and other
device helpers. That is the framework rule from base.py rather than anything
specific to cupy: a helper is spliced into its caller and cannot take on a
pointer argument of its own.

Author: B.G (07/2026)
"""

import re
from typing import Any

import cupy as cp
import numpy as np

from .base import DeviceFunction, DeviceFunctionBuilder, Kernel, KernelBuilder, Parameter, attach_meta
from .base import Bag

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
    parsing a device function, which may not reach scalar or field parameters
    and so has no use for either.

    Author: B.G (07/2026)
    """

    def __init__(self, bindings: dict[str, Any], *, allow_arrays: bool):
        self.bindings = bindings
        self.allow_arrays = allow_arrays
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
                raise ValueError(f"{param.name}: scalar/field param cannot be used in a device function (no pointer arg)")
            self._register_ptr(argname, param, write=False)
            if param.mode == "scalar":
                return f"{argname}[0]"
            idx = call_args[0] if call_args else "0"
            return f"{argname}[{idx}]"
        # set_node
        if param.mode == "const":
            raise ValueError(f"{param.name}: const parameter is read-only")
        if not self.allow_arrays:
            raise ValueError(f"{param.name}: set_node cannot be used in a device function (no pointer arg)")
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
        if isinstance(target, CupyDeviceFunction):
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


class CupyDeviceFunction(DeviceFunction):
    """
    A device helper, held as CUDA `__device__` source text.

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
        kernel or device function binds this helper.

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
            f"DeviceFunction '{self.name}' is CUDA source, only callable from a compiled kernel's device code"
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


class CupyDeviceFunctionBuilder(DeviceFunctionBuilder):
    """
    Builds a CupyDeviceFunction from CUDA `__device__` source. Spans may only
    reference const params / other device functions (no pointer args).

    Author: B.G (07/2026)
    """

    def compile(self) -> CupyDeviceFunction:
        """
        Expand the template's spans and prepend the helper sources and const
        #defines it turned out to need, giving the final `__device__` text.

        Author: B.G (07/2026)
        """
        template = self._template
        name = _extract_name(_DEVICE_NAME_RE, template, "__device__")
        parser = _SpanParser(self._bindings, allow_arrays=False)
        body = parser.parse(template)
        source = "\n".join(list(parser.helper_srcs.values()) + _const_defines(self._bindings, body) + [body])
        fn = CupyDeviceFunction(name, source)
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

        Author: B.G (07/2026)
        """
        template = self._template
        name = _extract_name(_KERNEL_NAME_RE, template, "__global__")
        parser = _SpanParser(self._bindings, allow_arrays=True)
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
