"""
Machinery shared by the two backends whose templates are python functions:
Taichi and Quadrants.

Specialization works by rebuilding the template function around a globals dict
that carries the bound objects, so a name like `phys` in the template body
resolves to the bound Bag when the backend traces it. The rebuilt function
is then decorated with ti.func/qd.func or ti.kernel/qd.kernel.

The two backends can share all of this because the pieces used here - func,
kernel, static, u8, i32, i64 - carry the same names and the same behaviour in
both modules. A backend subclass therefore only pins `_backend` to the ti or qd
module; nothing else varies.

cupy does not appear here: CUDA source text has no globals to patch, and that
backend substitutes into the source directly instead.

Author: B.G (07/2026)
"""

import ast
import copy
import hashlib
import inspect
import linecache
from types import FunctionType
from typing import Any, ClassVar

import numpy as np

from .compile import (
    HelperBuilder,
    Kernel,
    KernelBuilder,
    _SpecializedHelper,
    _SpecializeCtx,
    capture_template_meta,
    filter_bindings,
    resolve_binding,
)
from .parameter import MODES, Parameter
from .routine import Routine, RoutineBuilder, _CompiledStep, _template_label
from .sequence import SequenceBuilder


def specialize_closure(template, bindings: dict[str, Any], ctx: _SpecializeCtx) -> FunctionType:
    """
    Rebuild `template` as a new function whose globals carry the resolved
    bindings, leaving the original untouched.

    The code object is reused as-is; only the globals differ, which is what
    makes a name in the template body resolve to a bound object. Defaults,
    annotations and the rest are copied over so the result still introspects
    like the template it came from. `ctx` is the compile this specialization
    belongs to - it is what lets a bound HelperBuilder be specialized here,
    against these same bindings, rather than standing for a stale one.

    Author: B.G (07/2026)
    """
    resolved = {name: resolve_binding(value, ctx) for name, value in bindings.items()}
    source = getattr(template, "__wrapped__", template)
    func_globals = dict(source.__globals__)
    func_globals.update(resolved)

    specialised = FunctionType(
        source.__code__,
        func_globals,
        source.__name__,
        source.__defaults__,
        source.__closure__,
    )
    specialised.__kwdefaults__ = source.__kwdefaults__
    specialised.__annotations__ = dict(source.__annotations__)
    specialised.__doc__ = source.__doc__
    specialised.__qualname__ = source.__qualname__
    return specialised


class ClosureParamDeviceView:
    """
    What a Parameter looks like from inside device code.

    `.get` and `.set_node` are compiled device funcs, so a template body reads
    `p.get(i)` and writes `p.set_node(i, v)` the same way whatever the
    parameter's mode. A const parameter is read-only and carries no `.set_node`
    at all, which turns a write to one into a trace-time error.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, get_fn, set_fn=None):
        self._name = name
        self.get = get_fn
        if set_fn is not None:
            self.set_node = set_fn


class ClosureBackendParameter(Parameter):
    """
    Parameter backed by a const python value or by pooled device storage.

    Concrete backends subclass this and pin `_backend` to their module; the
    dtype mapping and the device view are written once here against the names
    both modules share.

    Author: B.G (07/2026)
    """

    _backend: ClassVar[Any]

    def __init__(self, name: str, *, dtype, mode: str, value, pool, n_flat: int | None = None):
        """
        Declare one parameter and give it its initial value.

        scalar and field take pooled storage straight away; const stays a
        plain python value.

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
        self._device_view: "ClosureParamDeviceView | None" = None

        if mode == "scalar":
            self._handle = pool.get_data(dtype, ())
        elif mode == "field":
            if n_flat is None:
                raise ValueError(f"{name}: field mode requires n_flat")
            self._handle = pool.get_data(dtype, (n_flat,))

        self._store(value)

    @classmethod
    def _numpy_dtype(cls, dtype):
        """
        Map a backend dtype (`ti.*`/`qd.*`) to the numpy dtype used for
        host-side (de)serialization.

        Author: B.G (07/2026)
        """
        backend = cls._backend
        if dtype == backend.u8:
            return np.uint8
        if dtype == backend.i32:
            return np.int32
        if dtype == backend.i64:
            return np.int64
        return np.float32

    def get(self):
        """
        The python value for const mode, the backing DataHandle otherwise.

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
            self._const_value = self._numpy_dtype(self.dtype)(value).item()
        elif self.mode == "scalar":
            self._handle.data[None] = value
        else:  # field
            arr = np.asarray(value, dtype=self._numpy_dtype(self.dtype)).reshape(-1)
            self._handle.data.from_numpy(arr)

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            raise ValueError(f"{self.name}: const parameter is read-only")
        if self.mode == "scalar":
            self._handle.data[None] = value
        else:  # field
            self._handle.data[node] = value

    def read(self):
        """
        Host-side scalar read - see Parameter.read for the contract.

        Author: B.G (07/2026)
        """
        if self.mode == "const":
            return self._const_value
        if self.mode == "field":
            raise ValueError(
                f"{self.name}: read() is for scalar/const only; a field is not meant to be "
                f"read back to the host as a whole"
            )
        return self._numpy_dtype(self.dtype)(self._handle.data.to_numpy()).item()

    def destroy(self) -> None:
        """
        Return any pooled storage to the pool. const mode owns none, so this
        is a no-op there.

        Author: B.G (07/2026)
        """
        if self._handle is not None:
            self._pool.release_data(self._handle)
            self._handle = None
            self._device_view = None  # a cached view closes over the released handle

    def device_view(self) -> ClosureParamDeviceView:
        """
        This parameter's device accessor, built on first use and kept.

        The compiled funcs come out identical every time, so one view serves
        every kernel that binds this parameter, for the parameter's whole
        life. A const's literal is fixed at construction, and a scalar or
        field set() writes through the very storage the view already reads, so
        neither can stale it. Only destroy() drops the view, having released
        that storage - and that does not reach kernels compiled earlier, which
        still hold it (see parameter.py, "Lifetime of a compiled object").

        Author: B.G (07/2026)
        """
        if self._device_view is None:
            self._device_view = self._build_device_view()
        return self._device_view

    def _build_device_view(self) -> ClosureParamDeviceView:
        """
        Compile this parameter's device accessors as backend funcs.

        get(node) branches on the mode through `_backend.static`, which
        resolves at trace time, so only one arm survives into the generated
        code: a baked literal for const, HANDLE[None] for scalar, HANDLE[node]
        for field. set_node is built for scalar and field only. MODE, VALUE and
        HANDLE are ordinary python values spliced in as globals.

        Author: B.G (07/2026)
        """
        backend = self._backend
        mode = self.mode
        value = self._const_value
        handle = self._handle.data if self._handle is not None else None

        def get_template(node):
            if STATIC(MODE == "const"):
                return VALUE
            elif STATIC(MODE == "scalar"):
                return HANDLE[None]
            else:
                return HANDLE[node]

        # No Parameter/HelperBuilder/Bag ever appears in these two bindings
        # dicts, so the ctx each specialize_closure call needs is never
        # actually consulted; a throwaway one per call is enough.
        get_fn = backend.func(
            specialize_closure(
                get_template, {"MODE": mode, "VALUE": value, "HANDLE": handle, "STATIC": backend.static}, _SpecializeCtx()
            )
        )

        set_fn = None
        if mode != "const":

            def set_node_template(node, val):
                if STATIC(MODE == "scalar"):
                    HANDLE[None] = val
                else:
                    HANDLE[node] = val

            set_fn = backend.func(
                specialize_closure(set_node_template, {"MODE": mode, "HANDLE": handle, "STATIC": backend.static}, _SpecializeCtx())
            )

        return ClosureParamDeviceView(self.name, get_fn, set_fn)


class ClosureHelper(_SpecializedHelper):
    """
    A device helper's specialization, compiled to a ti.func or qd.func.
    Produced by a ClosureHelperBuilder as part of an enclosing kernel's
    compile(); see HelperBuilder.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled):
        super().__init__()
        self.name = name
        self._compiled = compiled

    @property
    def compiled(self):
        """
        The raw ti.func/qd.func, for binding into another template's body.

        Author: B.G (07/2026)
        """
        return self._compiled

    def __call__(self, *args, **kwargs):
        """
        A ti.func/qd.func only runs inside kernel/func scope; callers use
        `.compiled` there.

        Author: B.G (07/2026)
        """
        raise RuntimeError(f"Helper '{self.name}' is only callable from kernel/func scope, not host Python")


class ClosureKernel(Kernel):
    """
    A launchable kernel compiled to a ti.kernel or qd.kernel.

    Its call signature is the template's own, which declares data arguments
    only - `def template(out: ti.template()): ...` - since bound objects reach
    the body through globals instead. See compile.py.

    Author: B.G (07/2026)
    """

    def __init__(self, name: str, compiled, data_needs: tuple = ()):
        super().__init__()
        self.name = name
        self._compiled = compiled
        # declared kind=DATA Needs (see need.py), positional, from the
        # KernelBuilder that produced this Kernel - empty for a builder that
        # never declared any (__call__ then behaves exactly as before Need
        # existed, relying on ti.kernel/qd.kernel's own signature checking).
        self._data_needs = data_needs

    @property
    def compiled(self):
        """
        The raw ti.kernel/qd.kernel behind this Kernel's __call__.

        Author: B.G (07/2026)
        """
        return self._compiled

    def __call__(self, *args, **kwargs):
        """
        Launches the compiled kernel. Args are data fields only.

        Author: B.G (07/2026)
        """
        if self._data_needs:
            # dtype-validates each positional arg against its declared Need
            # (see need.py); re-binding a kind=DATA Need never freezes it, so
            # this runs every call, not just the first.
            for need, arg in zip(self._data_needs, args):
                need.bind(arg)
        return self._compiled(*args, **kwargs)


class ClosureHelperBuilder(HelperBuilder):
    """
    Recipe for a device helper compiled to a ti.func/qd.func. Subclasses pin
    `_backend`. Specialized only as part of an enclosing kernel's compile()
    - see HelperBuilder; compile() itself raises.

    Author: B.G (07/2026)
    """

    _backend: ClassVar[Any]

    def _specialize(self, ctx: _SpecializeCtx) -> ClosureHelper:
        """
        Splice the referenced bindings into the template's globals against
        `ctx`, and compile the result as a device func.

        Author: B.G (07/2026)
        """
        self._resolve_needs()
        specialised = specialize_closure(self._template, filter_bindings(self._template, self._bindings), ctx)
        return ClosureHelper(specialised.__name__, self._backend.func(specialised))


class ClosureKernelBuilder(KernelBuilder):
    """
    Compiles an ingested def into a launchable kernel. Subclasses pin
    `_backend`.

    Author: B.G (07/2026)
    """

    _backend: ClassVar[Any]

    def compile(self) -> ClosureKernel:
        """
        Splice the referenced bindings into the template's globals - each
        compile() opening a fresh _SpecializeCtx, so every HelperBuilder
        reachable from this kernel's bindings is specialized once, against
        these bindings - and compile the result as a launchable kernel.

        Author: B.G (07/2026)
        """
        self._resolve_needs()
        ctx = _SpecializeCtx()
        specialised = specialize_closure(self._template, filter_bindings(self._template, self._bindings), ctx)
        return ClosureKernel(specialised.__name__, self._backend.kernel(specialised), data_needs=self.data_needs)


class _RenameNames(ast.NodeTransformer):
    """
    Substitute every `ast.Name` whose id is a key of `mapping` with a fresh
    Name carrying the mapped id, leaving everything else - including the
    ctx (Load/Store/Del) of the node being replaced - untouched.

    Used to retarget a step's template argument names (`T_out`, `T_in`) onto
    the routine's own data names (`T1`, `T0`) inside that step's body only;
    bound names (`heat`, `N`, ...) are never touched since they are never
    template arguments.

    Author: B.G (07/2026)
    """

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:
        new_id = self._mapping.get(node.id)
        if new_id is None:
            return node
        return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)


def _top_level_assigned_names(stmt: ast.stmt) -> set[str]:
    """
    Names a top-level Assign/AnnAssign/AugAssign statement binds.

    Only `ast.Name` targets count - `x = 1` binds `x`, including through
    tuple/list unpacking (`a, b = ...`); `arr[i] = v` and `obj.attr = v`
    write through a subscript or attribute and bind no new top-level python
    name, so they are not collected.

    Author: B.G (07/2026)
    """
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        targets = [stmt.target]
    else:
        return set()

    names: set[str] = set()

    def collect(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                collect(elt)

    for target in targets:
        collect(target)
    return names


def _check_fusable(label: str, func_def: ast.FunctionDef) -> None:
    """
    Enforce the two structural fusable-template constraints on one step's
    body, raising with `label` naming the offending step.

    No return anywhere: a return inside a spliced body would exit the whole
    fused kernel, not just this step's contribution to it. Flat splice only:
    once a top-level `for` loop has started, every remaining top-level
    statement must also be a `for` loop - an optional preamble is allowed
    before the first one, but nothing may follow the loops or sit between
    them, since wrapping or interleaving would stop Taichi parallelizing
    them as independent loops.

    Author: B.G (07/2026)
    """
    for node in ast.walk(func_def):
        if isinstance(node, ast.Return):
            raise ValueError(f"fuse: step {label!r} contains a return statement, which would exit the whole fused kernel")

    seen_loop = False
    for stmt in func_def.body:
        if isinstance(stmt, ast.For):
            seen_loop = True
        elif seen_loop:
            raise ValueError(
                f"fuse: step {label!r} has a statement after a top-level for loop; a fusable template "
                "must be an optional preamble followed only by top-level for loops, spliced flat"
            )


def _synthesize_tensor_annotation(backend: Any) -> tuple[ast.expr, str]:
    """
    Build a fresh data-argument annotation node for `backend` (the bound ti
    or qd module) rather than reading one out of a step's AST, plus the bare
    name the node's root refers to.

    Taichi: `ti.template()`, an `ast.Call` on `ast.Attribute(Name("ti"),
    "template")`. Quadrants: `qd.Tensor`, an `ast.Attribute(Name("qd"),
    "Tensor")` (no call - see quadrants_backend.py). Distinguishing the two
    by `backend.__name__` rather than `backend is ti`/`backend is qd` keeps
    this free of a direct import of either module.

    Author: B.G (07/2026)
    """
    if backend.__name__ == "taichi":
        alias = "ti"
        node = ast.Call(
            func=ast.Attribute(value=ast.Name(id=alias, ctx=ast.Load()), attr="template", ctx=ast.Load()),
            args=[],
            keywords=[],
        )
    else:
        alias = "qd"
        node = ast.Attribute(value=ast.Name(id=alias, ctx=ast.Load()), attr="Tensor", ctx=ast.Load())
    return node, alias


def _step_freevars(template) -> dict[str, Any]:
    """
    `template`'s own closure, name -> cell contents. Empty for a module-level
    def (no enclosing function scope to capture from) and for anything with
    no `__code__`/`__closure__` of its own.

    `__wrapped__` is unwrapped first so a decorated template's freevars are
    read off the underlying function rather than the wrapper.

    Author: B.G (07/2026)
    """
    tmpl = getattr(template, "__wrapped__", template)
    code = getattr(tmpl, "__code__", None)
    if code is None:
        return {}
    return dict(zip(code.co_freevars, (c.cell_contents for c in (tmpl.__closure__ or ()))))


class ClosureRoutineBuilder(RoutineBuilder):
    """
    Compiles a linear sequence of Taichi/Quadrants kernels sharing one bag.

    A step's data arity is read straight off its template's own python
    signature - `def diffuse(T_out, T_in)` declares two - since bound objects
    never appear there. Launching a compiled step is just calling it: Taichi
    and Quadrants kernels derive their own launch range from the template, so
    `grid`/`block` (cupy-only - see CupyRoutineBuilder) are accepted and
    ignored.

    compile() defaults to fused=True: consecutive steps (up to a split()
    boundary) are spliced into one generated kernel rather than launched as
    separate ones - see compile() and _fuse_group().

    Author: B.G (07/2026)
    """

    def _data_arity(self, kernel_builder: KernelBuilder) -> int:
        template = kernel_builder.template
        if template is None:
            raise ValueError("add_kernel: kernel_builder has no ingested template")
        return len(inspect.signature(template).parameters)

    def _make_caller(self, compiled_kernel, grid, block):
        return compiled_kernel

    def compile(self, fused: bool = True, dump_source: str | None = None) -> "Routine":
        """
        Validate (RoutineBuilder._validate) and compile every step.

        fused=False falls back to the base implementation: one kernel per
        step, each an ordinary specialize_closure compile, exactly as
        before fusion existed. This is the reference the fused path is
        diffed against, and stays reachable as a runtime switch.

        fused=True (the default) compiles each split()-delimited group of
        steps into one generated kernel: every step's top-level `for` loops
        are concatenated flat into a single generated `def`, in the order
        the steps were added, and that `def` is compiled as one
        ti.kernel/qd.kernel. See _fuse_group for the mechanics and the
        constraints this enforces.

        A group holding a single step has nothing to splice, so it skips
        _fuse_group and compiles that step directly. Beyond saving the
        work, this is what lets a routine that split()s after every step -
        one whose steps each need a global barrier, so fusing them would be
        wrong - compile on this path at all: _fuse_group's constraints are
        the splicer's, and a lone step is subject to none of them. A lone
        step's compile is deduplicated across such groups, keyed on
        id(kernel_builder) - see RoutineBuilder.compile. Fused groups (more
        than one step) need no such dedup: _fuse_group already unions every
        step in the group into one generated kernel, so an unrolled repeat
        block with no split() inside it compiles to a single kernel already,
        never once per repetition.

        `dump_source`, fused mode only, is a file path; when given, every
        generated group's source is appended to it (truncated first),
        separated by a header naming the group. No file is written when
        `dump_source` is None.

        Author: B.G (07/2026)
        """
        if not fused:
            return super().compile(fused=False)

        self._validate()

        compiled_steps: list[_CompiledStep] = []
        data_names: list[str] = []
        compiled_cache: dict[int, Any] = {}
        for group_index, group in enumerate(self._grouped_steps()):
            if len(group) == 1:
                step = group[0]
                key = id(step.kernel_builder)
                compiled = compiled_cache.get(key)
                if compiled is None:
                    compiled = step.kernel_builder.compile()
                    compiled_cache[key] = compiled
                caller = self._make_caller(compiled, None, None)
                compiled_steps.append(_CompiledStep(caller, step.canonical_refs))
                for name in step.canonical_refs:
                    if name not in data_names:
                        data_names.append(name)
                continue
            kernel, group_data_names = self._fuse_group(group, group_index, dump_source)
            caller = self._make_caller(kernel, None, None)
            compiled_steps.append(_CompiledStep(caller, tuple(group_data_names)))
            for name in group_data_names:
                if name not in data_names:
                    data_names.append(name)

        defaults = {name: self._data[name] for name in data_names}
        return Routine(compiled_steps, tuple(data_names), defaults, self._data_needs_tuple(data_names))

    def _fuse_group(self, group: list, group_index: int, dump_source: "str | None"):
        """
        Splice one split()-delimited group of steps into a single generated
        kernel.

        One `_SpecializeCtx` covers the whole group, so a HelperBuilder
        reachable from two of these steps is specialized once and both call
        sites in the generated body share the specialized object - the same
        guarantee a single ordinary kernel compile gives within itself.

        Per step, in order: the template's own AST is deep-copied out of the
        capture_template_meta cache (never mutated in place - that cache is
        shared with every other compile of the same template; the source it
        parsed is dedented first, so a nested def's indented body is
        recoverable exactly like a module-level one - see
        capture_template_meta); the two structural fusable-template
        constraints are checked (_check_fusable); its top-level assignments
        are checked against every earlier step in this group for a name
        collision (_top_level_assigned_names); its template argument names
        are substituted for this step's canonical_refs, positionally
        (_RenameNames) - the same mapping data_handle_ref set up at
        add_kernel time; its freevars, if any, are substituted for
        step-index-mangled names (`_fv{s}_{NAME}`) by a second, independent
        _RenameNames pass over the already-data-renamed body, so a data
        rename and a freevar rename can never collide; and the resulting
        body statements are appended to the generated function's body, flat.

        The generated function's parameters are this group's data names, in
        first-appearance order, each carrying a data-argument annotation
        synthesized fresh for this group's backend (`ti.template()` or
        `qd.Tensor` - see _synthesize_tensor_annotation), not one read out of
        any step's AST: a nested-def template's own annotation is a bare
        alias name (`T`) closed over from its factory, which has no useful
        source form to copy. The result is unparsed to source, registered in
        linecache under a synthetic filename - required for Taichi/Quadrants
        to re-parse it via inspect.getsource when they run their own AST
        transform - exec'd, and decorated as a kernel with the group's
        resolved bindings, every step's mangled freevars, plus every step's
        own template module globals injected.

        Author: B.G (07/2026)
        """
        backend = group[0].kernel_builder._backend
        ctx = _SpecializeCtx()
        body_stmts: list[ast.stmt] = []
        data_names: list[str] = []
        annotations: dict[str, ast.expr] = {}
        assigned_by: dict[str, str] = {}
        module_globals: dict[str, Any] = {}
        resolved_globals: dict[str, Any] = {}

        annotation_node, annotation_alias = _synthesize_tensor_annotation(backend)

        for step_index, step in enumerate(group):
            kernel_builder = step.kernel_builder
            # a fused group reads kernel_builder.bindings directly below
            # (there is no per-step compile() call on this path to do it) -
            # so any .need()-declared PARAM/HELPER need must be resolved into
            # that dict here first, exactly as compile() does for the
            # unfused/single-step path (see compile.py, CompileBuilder.
            # _resolve_needs).
            kernel_builder._resolve_needs()
            template = kernel_builder.template
            label = f"group{group_index}/step{step_index}:{_template_label(template)}"

            _, tree = capture_template_meta(template)
            if tree is None or not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
                raise ValueError(f"fuse: step {label!r} has no recoverable source to fuse")
            func_def = copy.deepcopy(tree.body[0])

            _check_fusable(label, func_def)

            for stmt in func_def.body:
                for name in _top_level_assigned_names(stmt):
                    prior = assigned_by.get(name)
                    if prior is not None:
                        raise ValueError(
                            f"fuse: top-level name '{name}' is assigned by both {prior!r} and {label!r}"
                        )
                    assigned_by[name] = label

            params = [a.arg for a in func_def.args.args]
            if len(params) != len(step.canonical_refs):
                raise ValueError(
                    f"fuse: step {label!r} declares {len(params)} data argument(s), "
                    f"routine gives it {len(step.canonical_refs)}"
                )
            rename_map = dict(zip(params, step.canonical_refs))

            for canon in step.canonical_refs:
                if canon not in data_names:
                    data_names.append(canon)
                    annotations[canon] = copy.deepcopy(annotation_node)

            renamer = _RenameNames(rename_map)
            renamed_body = [renamer.visit(stmt) for stmt in func_def.body]

            freevars = _step_freevars(template)
            if freevars:
                fv_rename_map = {name: f"_fv{step_index}_{name}" for name in freevars}
                fv_renamer = _RenameNames(fv_rename_map)
                renamed_body = [fv_renamer.visit(stmt) for stmt in renamed_body]
                resolved_globals.update(
                    {f"_fv{step_index}_{name}": value for name, value in freevars.items()}
                )

            body_stmts.extend(renamed_body)

            filtered = filter_bindings(template, kernel_builder.bindings)
            module_globals.update(dict(getattr(template, "__globals__", {})))
            resolved_globals.update({name: resolve_binding(value, ctx) for name, value in filtered.items()})

        if not body_stmts:
            raise ValueError(f"fuse: group {group_index} produced an empty body")

        # module_globals is a base layer (ti/qd/np/builtins/plain helper functions
        # each step's own module happens to define) applied once, in step order;
        # resolved_globals - the actual bound objects this group's steps
        # reference, plus every step's mangled freevars - is applied last so a
        # name a later step's unrelated module globals happen to share (e.g.
        # two templates in one file both seeing `heat` in their module
        # namespace) never clobbers an earlier step's resolved binding.
        exec_globals: dict[str, Any] = {}
        exec_globals.update(module_globals)
        exec_globals.update(resolved_globals)
        exec_globals[annotation_alias] = backend

        # Name the fused function deterministically from its own content
        # rather than a process-global uid: build it under a stable
        # placeholder name first, unparse, hash that text, then rename to
        # `_fused_group{group_index}_{hash}` and re-unparse so the source,
        # linecache entry, and compiled filename all agree on the real name.
        # Identical routine -> identical hash -> identical name -> the
        # backend's own offline cache (Taichi/Quadrants re-parse the kernel
        # via inspect.getsource off this source+filename) hits across process
        # restarts; an unrelated upstream allocation shift no longer touches
        # this name since it never depended on uid in the first place.
        placeholder_name = f"_fused_group{group_index}_placeholder"
        args_node = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name, annotation=annotations.get(name)) for name in data_names],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        fused_def = ast.FunctionDef(
            name=placeholder_name, args=args_node, body=body_stmts, decorator_list=[], returns=None
        )
        module = ast.fix_missing_locations(ast.Module(body=[fused_def], type_ignores=[]))
        placeholder_source = ast.unparse(module)
        digest = hashlib.sha256(placeholder_source.encode()).hexdigest()[:12]
        func_name = f"_fused_group{group_index}_{digest}"
        fused_def.name = func_name
        source = ast.unparse(module)

        filename = f"<fused-routine:{func_name}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)

        if dump_source:
            mode = "w" if group_index == 0 else "a"
            with open(dump_source, mode) as fh:
                fh.write(f"# --- group {group_index} ({filename}) ---\n{source}\n\n")

        code = compile(source, filename, "exec")
        exec(code, exec_globals)
        fused_fn = exec_globals[func_name]

        krn = ClosureKernel(func_name, backend.kernel(fused_fn))
        return krn, data_names


class ClosureSequenceBuilder(SequenceBuilder):
    """
    Sequences Taichi/Quadrants blocks under host-driven control.

    A kernel block's data arity is read straight off its template's own
    python signature, and launching a compiled kernel is just calling it -
    the same two facts ClosureRoutineBuilder rests on.

    Author: B.G (07/2026)
    """

    def _data_arity(self, kernel_builder: KernelBuilder) -> int:
        template = kernel_builder.template
        if template is None:
            raise ValueError("add_kernel: kernel_builder has no ingested template")
        return len(inspect.signature(template).parameters)

    def _make_caller(self, compiled_kernel, grid, block):
        return compiled_kernel
