"""
Taichi/Quadrants compile phase (1c): turns a BoundKernel into a CompiledKernel
by emitting real `ti.func`/`ti.kernel` (or `qd.func`/`qd.kernel`) objects.
Both backends share every line here - only which module `backend` points to
differs - mirroring `_closure_backend.py`'s own Taichi/Quadrants split.

The `ctx` problem and how this solves it
-----------------------------------------
1a/1b settled `ctx` as a template's literal first parameter - `def tmpl(ctx,
i): return ctx.grad(ctx.z.get(i), i)` - not a name spliced into the
template's globals the way every earlier compile path in this package
(`_closure_backend.py`'s `specialize_closure`) works. That distinction
matters here: `ctx` being a real parameter means it is read via `LOAD_FAST`
bytecode, not `LOAD_GLOBAL` - splicing a value into `__globals__` under the
name `ctx`, the `specialize_closure` technique, would have **no effect at
all**, since a local parameter's bytecode never consults globals for that
name regardless of what sits there.

The fix used here is a source-level transform, not a globals trick:
`_compile_dropping_ctx` takes the template's own AST (via `capture_template_
meta`, compile.py's cached inspect.getsource + ast.parse - a free function,
no Need/CompileBuilder coupling), deletes `ctx` from the FunctionDef's own
parameter list, unparses the result, and `exec()`s it with `ctx` now bound in
that exec's globals. The body is untouched - every `ctx.foo` reference in it
still reads the name `ctx`, which now resolves via `LOAD_GLOBAL` since the
compiled code object no longer declares it as a local/parameter. Registering
the unparsed source in `linecache` under a synthetic filename before `exec()`
is what lets Taichi/Quadrants re-inspect it via their own `inspect.getsource`
during a later inlined trace (a ti.func is re-traced, not compiled once, each
place it is inlined) - the exact technique `_closure_backend.py`'s
`_fuse_group` already relies on for the same reason, proven working here the
same way.

The ctx tree
------------
What `ctx` resolves to, at every level, is a plain python object built once
per compile() by `_build_ctx_node`, walking the FrozenKernel/FrozenHelper's
own composition tree in lock-step with `BoundKernel`'s address tree:

  - one attribute per PARAM slot at that level, holding the bound
    Parameter's `device_view()` - `ctx.z.get(i)` reaches it exactly as any
    other Taichi/Quadrants-compiled template already does.
  - one attribute per composed HELPER slot, holding the **raw compiled
    `ti.func`/`qd.func` object itself** - not a wrapper - with that helper's
    own children (its PARAM device views and its own composed HELPER
    children, recursively) attached as extra attributes on that same
    function object. A compiled closure-backend function is an ordinary
    python object and accepts arbitrary attribute assignment freely, so
    `ctx.grid` is simultaneously callable (`ctx.grid(...)`, invoking the
    compiled func directly - matching how `compile.py`'s own
    `_LazyBagView`/`resolve_binding` already hand a bound HelperBuilder's
    caller the *raw* `.compiled` func, never a wrapper) and further
    attribute-navigable (`ctx.grid.neighbour(...)`, since `neighbour` was
    attached onto the same object as `.grid`'s own attribute). This is built
    bottom-up: a composed helper's own children are compiled and attached
    before that helper's own template is compiled, since its body may
    reference them.

Every composed FrozenHelper reachable from a BoundKernel is compiled
unconditionally as part of that BoundKernel's compile(), whether or not the
immediate parent's own contract calls it bare - a `ctx.grid.neighbour(...)`
reference needs `neighbour` compiled regardless of whether `grid` itself is
ever called directly, and helper call signatures are fully trusted (no
attempt is made here to prune what nothing in this particular tree happens to
call).

`ctx.bk`, the reserved backend-intrinsics namespace (bk.py) - `ctx.bk.sqrt`,
`ctx.bk.atan2`, `ctx.bk.cast_u32`, ... - is attached to every node this
module builds, at every level of the ctx tree, not just the root: `bk` is
built once per compile() (`make_closure_bk(backend)`) and threaded through
`_build_ctx_node`'s own recursion, so it is reachable from a deeply composed
private block exactly as it is from the kernel's own template. See bk.py's
module docstring for why this namespace exists and contract.py for why it
never appears as a slot requirement.

Author: B.G (08/2026)
"""

import ast
import copy
import linecache
from types import FunctionType
from typing import Any

from ..pool.base import new_uid
from .bk import make_closure_bk
from .bound import Address, BoundKernel, format_address
from .compile import capture_template_meta
from .compile_shared import CompiledKernel, CompileError, check_data_signature, check_legal_accessors, check_unmet
from .ctx import CTX_PARAM_NAME
from .frozen import FrozenGroup, _Frozen
from .slot import SlotKind


def _drop_ctx_param(func_def: ast.FunctionDef, label: str) -> None:
    """
    Remove `ctx` from `func_def`'s own parameter list in place - see the
    module docstring for why this, not a globals splice, is what makes `ctx`
    resolve as a global inside the body that is left untouched.

    Author: B.G (08/2026)
    """
    if func_def.args.posonlyargs and func_def.args.posonlyargs[0].arg == CTX_PARAM_NAME:
        func_def.args.posonlyargs = func_def.args.posonlyargs[1:]
    elif func_def.args.args and func_def.args.args[0].arg == CTX_PARAM_NAME:
        func_def.args.args = func_def.args.args[1:]
    else:
        raise CompileError(f"template {label!r}: first parameter must be {CTX_PARAM_NAME!r}")


def _compile_dropping_ctx(template, ctx_obj: Any, label: str) -> FunctionType:
    """
    Rebuild `template` with `ctx` removed from its own signature and bound
    instead as a global (`ctx_obj`) the body's untouched `ctx.*` references
    now resolve against. Registers the rebuilt source in `linecache` under a
    synthetic filename, `exec()`s it, and returns the resulting function -
    not yet decorated with `backend.func`/`backend.kernel`, see the callers
    below.

    Author: B.G (08/2026)
    """
    _, tree = capture_template_meta(template)
    if tree is None:
        raise CompileError(f"template {label!r}: no recoverable source to compile")
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not body:
        raise CompileError(f"template {label!r}: source is not a function definition")
    func_def = copy.deepcopy(body[0])
    _drop_ctx_param(func_def, label)

    module = ast.fix_missing_locations(ast.Module(body=[func_def], type_ignores=[]))
    source = ast.unparse(module)
    filename = f"<pf-compile:{label}:{new_uid()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)

    exec_globals: dict[str, Any] = dict(getattr(template, "__globals__", {}))
    exec_globals["ctx"] = ctx_obj
    code = compile(source, filename, "exec")
    exec(code, exec_globals)
    return exec_globals[func_def.name]


class _CtxNode:
    """
    What `ctx` (or one of its composed-helper children) resolves to inside a
    specialized template body - a plain attribute bag. See the module
    docstring's "The ctx tree" section for what gets attached and why a
    composed HELPER child is the raw compiled func itself rather than an
    instance of this class.

    Author: B.G (08/2026)
    """


def _build_ctx_node(prefix: Address, frozen: _Frozen, bound: BoundKernel, backend: Any, bk: Any) -> _CtxNode:
    """
    Recursively build the ctx tree rooted at `frozen` (found at `prefix` in
    `bound`'s address tree), compiling every composed HELPER child - bottom
    up, so a child's own compiled func exists before its parent's template
    (which may call it) is compiled - and attaching each as both a callable
    and a further-navigable node on the returned object. See the module
    docstring.

    `bk` (bk.py's `make_closure_bk(backend)`, built once per compile() and
    threaded through every recursive call) is attached to every node at
    every level - the reserved `ctx.bk` namespace is reachable from the
    kernel's own root template and from any composed helper's, however deep,
    since a private block many levels down is exactly where noise's/visu's
    own use of it lives. See bk.py's module docstring.

    Author: B.G (08/2026)
    """
    node = _CtxNode()
    node.bk = bk
    for name in frozen.slots.names(SlotKind.PARAM):
        addr = prefix + (name,)
        param = bound.value_at(addr)
        setattr(node, name, param.device_view())

    for name in frozen.slots.names(SlotKind.HELPER) | set(frozen.composed):
        child_addr = prefix + (name,)
        child_frozen = frozen.composed[name]
        child_node = _build_ctx_node(child_addr, child_frozen, bound, backend, bk)
        if isinstance(child_frozen, FrozenGroup):
            # A FrozenGroup has no template of its own to compile - it is a
            # passive, non-callable composite (frozen.py). `ctx.<name>` is
            # attached exactly as built: navigable (`ctx.<name>.<member>`),
            # never callable.
            setattr(node, name, child_node)
            continue
        label = format_address(child_addr)
        raw = _compile_dropping_ctx(child_frozen.template, child_node, label)
        compiled = backend.func(raw)
        # child_node's own attributes (its PARAM device views, its own
        # composed HELPER children) are copied onto the compiled func object
        # itself, so `ctx.<name>` is simultaneously callable (invokes this
        # func) and navigable (`ctx.<name>.<grandchild>`) - see the module
        # docstring.
        for attr_name, attr_val in vars(child_node).items():
            setattr(compiled, attr_name, attr_val)
        setattr(node, name, compiled)

    return node


def compile_kernel(bound: BoundKernel, backend: Any) -> CompiledKernel:
    """
    Compile `bound` (a BoundKernel) against `backend` (the `taichi` or
    `quadrants` module). Checks unmet slots and legal PARAM accessors first
    (compile_shared.py), then builds the whole ctx tree and compiles the
    kernel's own template as `backend.kernel(...)`.

    Author: B.G (08/2026)
    """
    check_unmet(bound)
    check_legal_accessors(bound)

    frozen = bound.frozen
    data_names = check_data_signature(frozen.template, frozen.slots.names(SlotKind.DATA))
    bk = make_closure_bk(backend)
    root_node = _build_ctx_node((), frozen, bound, backend, bk)
    raw = _compile_dropping_ctx(frozen.template, root_node, "root")
    compiled = backend.kernel(raw)

    data_order = [(name,) for name in data_names]
    return CompiledKernel(bound, compiled, data_order, needs_launch_dims=False)
