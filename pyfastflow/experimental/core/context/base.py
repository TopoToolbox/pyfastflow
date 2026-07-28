"""
Backend-agnostic building blocks for describing GPU work once and compiling it
against Taichi, Quadrants or cupy (or any future one).

What this is for
----------------
A physics model rarely needs a new numerical scheme just because one of its
parameters changed shape. Take heat diffusion: the update is identical whether
the diffusion coefficient K is a spatially variable field, a single value the
host retunes between steps, or a constant fixed for the whole run. That choice
matters enormously on a GPU - a compile-time constant costs no memory traffic
and can be folded into the generated code, a field costs a fetch per node - but
it does not change the maths. Boundary conditions and stencils behave the same
way: making a grid periodic alters the neighbour logic, not the scheme built on
top of it.

Writing one kernel per combination is the obvious way to handle this, and it
becomes unmanageable fast. So instead a template reads a Parameter the same way
whatever its mode, and calls a neighbour helper without knowing which topology
implements it. Which mode, and which helper, is settled at compile time - where
it can still turn into a literal or a specialised routine - and the kernel code
never changes.

Jargon
----------
Parameter       One named, typed value. Its `mode` says where the value lives:
                "const" (a compile-time constant, not modifiable mid-run),
                "scalar" (a single value, modifiable) or "field" (a device
                array, one value per node).
HelperBuilder   The recipe for a device-side helper: a small routine callable
                only from other device code (ti.func, qd.func, CUDA
                __device__). Bind it into a kernel - flat or inside a Bag -
                and the kernel's own compile() specializes it; there is no
                standalone compiled Helper object to hold onto.
Kernel          A compiled entry point - what the host launches (ti.kernel,
                qd.kernel, CUDA __global__).
Bag             A named collection of any of the above, mixed freely, so a
                group travels as one object and is reached in-kernel by dotted
                path: phys.dx.get(i), ops.neighbour(i).

A "context" is any concrete class - GridContext, FlowContext, ... - that groups
Parameters and registers Helpers. There is deliberately no base Context
class: a context needing another context's parameters binds them explicitly,
rather than reaching through a registry of stored connections.

Compiling something
-------------------
Templates are written once, generically, and specialized by a builder:

    kernel = (TaichiKernelBuilder()
              .bind("phys", phys)        # a Bag of parameters
              .bind("ops", ops)          # a Bag of HelperBuilders
              .ingest(update_height)     # the template
              .compile())
    kernel(h_new, h_old)                 # bulk data passed at call time

bind(name, obj) makes `obj` visible inside the template body under `name`.
ingest() takes the template - a python def for Taichi/Quadrants, a CUDA source
string for cupy. compile() returns a Kernel. Only the abstract HelperBuilder /
KernelBuilder live here; the concrete Taichi*, Quadrants* and Cupy* builders
sit alongside this module.

A HelperBuilder bound anywhere in a KernelBuilder's bindings - directly under
a name, or as a member of a bound Bag - is specialized as part of that
kernel's compile(), against that same compile's bindings. This is what lets a
helper reading a const Parameter pick up a different value after the const is
swapped and the *kernel* is recompiled, with the helper's own builder never
touched. Reaching the same HelperBuilder from two places in one kernel - bound
flat and inside a Bag, or under two different names - specializes it once; the
same specialized object is shared at both call sites. A HelperBuilder has no
compiled form of its own to keep between compiles: it is a recipe, always
specialized fresh as part of whatever kernel currently binds it.

The builder is the recipe: its template and bindings can be inspected, and
compile() may be called again after a bind() edit, each call producing a new,
independent callable. Nothing about compile() consumes or mutates the
builder - recompiling a builder that has not changed since its last compile()
just repeats work for an equivalent result, which is pointless and best
avoided, though harmless if it happens.

Data at call time, configuration at compile time
------------------------------------------------
Bound objects are injected into the template body and never appear in the call
signature. A compiled Kernel takes exactly the arguments its template declares,
and that is where bulk data travels - the buffers read and written each step.
Everything that *describes* the problem rather than *being* it - grid spacing,
timestep, gravity, which helper implements the neighbour lookup - is bound.

Reading a Parameter in device code is uniform across modes: p.get(node) to
read, p.set_node(node, value) to write.

What a device helper may bind
-----------------------------
A helper binds whatever a kernel binds, in any mode, on every backend.

On Taichi and Quadrants, bound objects reach device code as globals, and a
helper is traced as part of the kernel that calls it, so alpha.get(i) reads
the same inside a helper as it does in the kernel body.

On cupy, every scalar/field Parameter a compilation unit reaches - the
kernel's own bindings plus, recursively, every helper's - is collected into
one module-scope `__constant__` block, uploaded once per compile(). Every
`__global__` and `__device__` function compiled into that module sees the
same block, so a helper reaches a bound Parameter exactly the way its caller
does, with no pointer argument to thread through and no call site to rewrite.
See cupy_backend.py's module docstring for the block's exact shape.

Lifetime of a compiled object
-----------------------------
compile() freezes what it was given: const Parameters are baked in as literals,
scalar and field Parameters as the storage behind their DataHandle. So:

  - Writing to a scalar or field Parameter *is* visible to already-compiled
    kernels, which hold that same storage. This is the normal way to feed
    changing data.
  - set() on a const Parameter is not. It drops the parameter's cached device
    view so the next compile() picks the new value up, but kernels compiled
    before it keep the old literal. Recompile them.
  - destroy() returns storage to the pool, which may hand the same buffer out
    again. Never destroy a Parameter that a live kernel still binds.

None of this is enforced at runtime.

Author: B.G (07/2026)
"""

import ast
import inspect
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, ClassVar

from ..pool.base import DataHandle, new_uid


class Parameter(ABC):
    """
    One named, typed value owned by a context.

    `mode` decides where the value lives - "const" in the generated code,
    "scalar" in a single device cell, "field" in a device array - and every
    backend must offer all three (REQUIRED_MODES). A backend may widen
    SUPPORTED_MODES with further storage kinds; the check runs when the
    subclass is defined, so an incomplete backend fails at import.

    Two surfaces. From the host: get(), set(value), set_node(node, value).
    From device code: device_view(), which returns a backend object whose
    .get(node) / .set_node(node, val) let a kernel read and write the
    parameter identically whatever its mode.

    Author: B.G (07/2026)
    """

    REQUIRED_MODES: ClassVar[frozenset[str]] = frozenset({"const", "scalar", "field"})
    SUPPORTED_MODES: ClassVar[frozenset[str]] = REQUIRED_MODES

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = Parameter.REQUIRED_MODES - cls.SUPPORTED_MODES
        if missing:
            raise TypeError(f"{cls.__name__} must support modes {sorted(missing)}")

    name: str
    dtype: Any
    solo: bool = False

    def __init__(self):
        """
        Assign this parameter's process-wide uid and open its `mode` slot.
        Concrete backends call this first, then set `self.mode = ...` once as
        part of their own __init__ - see the `mode` property below.

        Author: B.G (07/2026)
        """
        self._uid = new_uid()
        self._mode: str | None = None

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction, from the same counter
        as every other Parameter, Bag, Helper and pool data handle. Two
        references to one Parameter share a uid; two different Parameters
        never do, even if they hold equal values. Not stable across processes
        and never meant to appear in generated code or a cache key - see the
        module docstring, "uid vs handle".

        Author: B.G (07/2026)
        """
        return self._uid

    @property
    def mode(self) -> str:
        """
        Where the value lives - "const", "scalar" or "field". Set once, by
        the backend's __init__; reassigning it raises. To change a
        parameter's mode, construct a new Parameter and swap it into the bag
        in place of this one.

        Author: B.G (07/2026)
        """
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if self._mode is not None:
            raise AttributeError(
                f"{getattr(self, 'name', '?')}: Parameter.mode is immutable once set (already "
                f"{self._mode!r}); construct a new Parameter and swap it into the bag instead"
            )
        self._mode = value

    @abstractmethod
    def get(self):
        """
        Host-side value: a python scalar for const mode, a DataHandle for scalar/field.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def set(self, value) -> None:
        """
        Update the whole parameter value in place, according to its mode. On
        const mode this does not reach already-compiled kernels - see the
        module docstring, "Lifetime of a compiled object".

        Author: B.G (07/2026)
        """
        ...

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.
        Overridden by concrete backends; device-side writes go through
        device_view().set_node instead.

        Author: B.G (07/2026)
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement host set_node")

    def device_view(self):
        """
        An object whose .get(node) / .set_node(node, val) work inside device
        code. Taichi and Quadrants compile one out of ti/qd funcs. cupy leaves
        this unimplemented, having no use for it: its parser substitutes
        parameters into the source directly.

        Author: B.G (07/2026)
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement device_view")

    @abstractmethod
    def destroy(self) -> None:
        """
        Release any backing storage owned by this parameter. Unsafe while a
        compiled kernel still binds it - see the module docstring, "Lifetime
        of a compiled object".

        Author: B.G (07/2026)
        """
        ...


class Specializable(ABC):
    """
    Anything produced by specializing a template against bindings: a
    launchable Kernel, or the internal object a HelperBuilder specializes
    into as part of an enclosing kernel's compile().

    Beyond the call contract, each instance keeps a read-only snapshot of the
    raw material it was made from - template, source text, AST, and the real
    bindings dict as it stood at this object's own compile time. Nothing in
    this module reads those back to drive a later compile; they are here so a
    higher layer (kernel fusion, a graph compiler) can work from the originals
    instead of re-deriving them. The builder that produced this object stays
    authoritative for anything that needs to change - see CompileBuilder.

    Author: B.G (07/2026)
    """

    name: str
    _template: Any = None
    _source: str | None = None
    _ast: ast.AST | None = None
    _dependencies: dict[str, Any] | None = None

    def __init__(self):
        """
        Assign this object's process-wide uid. Concrete backends call this
        first in their own __init__.

        Author: B.G (07/2026)
        """
        self._uid = new_uid()

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction. See Parameter.uid.

        Author: B.G (07/2026)
        """
        return self._uid

    @property
    def template(self):
        """
        The template object this was compiled from - a python def for
        Taichi/Quadrants, a CUDA source string for cupy. Read-only: this is a
        snapshot for introspection, not a handle to recompile from. Change and
        recompile through the builder instead.

        Author: B.G (07/2026)
        """
        return self._template

    @property
    def source(self) -> str | None:
        """
        The template's source text, captured at compile time. See `template`.

        Author: B.G (07/2026)
        """
        return self._source

    @property
    def ast(self) -> "ast.AST | None":
        """
        The template's parsed AST, captured at compile time. None for a
        template with no recoverable source (e.g. cupy's CUDA text). See
        `template`.

        Author: B.G (07/2026)
        """
        return self._ast

    @property
    def bindings(self) -> dict[str, Any]:
        """
        The real bound objects - not type names - as they stood at this
        object's own compile time. Read-only snapshot; see `template`.

        Author: B.G (07/2026)
        """
        return self._dependencies

    @property
    @abstractmethod
    def compiled(self):
        """
        Raw backend callable/source (ti.func / qd.func object, CUDA text),
        for injection into another template's bindings.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs): ...


class _SpecializedHelper(Specializable):
    """
    A device helper's specialized backend object (e.g. a ti.func
    specialization), produced by a HelperBuilder as part of an enclosing
    kernel's compile(). No public class holds this between compiles - it
    lives only inside a _SpecializeCtx.compiled and the Kernel body being
    built alongside it; see HelperBuilder and _SpecializeCtx.

    Not necessarily callable from host Python - backends where device
    helpers can only run inside kernel/func scope raise on __call__.

    Author: B.G (07/2026)
    """


class Kernel(Specializable):
    """
    Compiled entry point (e.g. a ti.kernel specialization).

    Unlike a helper's specialization, __call__ works from host Python - that
    is how compute gets launched. Its arguments are the template's own
    declared data arguments; see the module docstring on data at call time.

    Author: B.G (07/2026)
    """


class _SpecializeCtx:
    """
    The state shared by every resolution happening inside one compile().

    A HelperBuilder is a recipe, not something compiled ahead of time - it is
    specialized here, on demand, the first time this compile reaches it.
    `specialize` memoizes on the builder's uid so a helper reachable from two
    places in one compile - bound flat and inside a Bag, or under two
    different names - is specialized exactly once and both call sites share
    the same object. The memo lives only as long as this ctx, i.e. one
    compile(): a later compile against different bindings gets its own ctx
    and specializes afresh, which is what lets a recompiled kernel pick up a
    changed const in a helper it binds.

    `_active` catches a helper cycle - builder A binding builder B which
    (directly or transitively) binds A back - by raising instead of
    recursing forever.

    Author: B.G (07/2026)
    """

    def __init__(self):
        self._memo: dict[int, Any] = {}
        self._active: set[int] = set()

    def specialize(self, builder: "HelperBuilder") -> Any:
        """
        This builder's specialized object for the compile this ctx belongs
        to, specializing it on first request and returning the memoized
        result on every later one.

        Author: B.G (07/2026)
        """
        uid = builder.uid
        cached = self._memo.get(uid)
        if cached is not None:
            return cached
        if uid in self._active:
            name = getattr(builder.template, "__name__", builder.template)
            raise RecursionError(f"helper cycle detected while specializing '{name}' (uid {uid})")
        self._active.add(uid)
        try:
            specialized = builder._specialize(self)
        finally:
            self._active.discard(uid)
        self._memo[uid] = specialized
        return specialized


class _LazyBagView:
    """
    What a Bag looks like from inside a template body.

    `phys.g` resolves member `g` on first access and caches the result in the
    instance dict, so later lookups skip __getattr__ altogether. Resolving a
    member is not free - for a Parameter it compiles a device view, for a
    HelperBuilder it specializes the helper - and a template usually touches
    only a few members of the bag it binds, so resolution is deferred to the
    members actually named. Members that are themselves Bags resolve to
    another _LazyBagView, keeping nested bags lazy all the way down, and
    carry the same ctx so a HelperBuilder reached through a nested Bag
    specializes against the same compile as one bound flat.

    Author: B.G (07/2026)
    """

    def __init__(self, bag: "Bag", ctx: "_SpecializeCtx"):
        object.__setattr__(self, "_bag", bag)
        object.__setattr__(self, "_ctx", ctx)

    def __getattr__(self, name: str) -> Any:
        # only called on a genuine miss (cached hits never reach here)
        bag = object.__getattribute__(self, "_bag")
        ctx = object.__getattribute__(self, "_ctx")
        if name not in bag:
            raise AttributeError(name)
        resolved = resolve_binding(bag[name], ctx)
        self.__dict__[name] = resolved
        return resolved


def resolve_binding(value, ctx: "_SpecializeCtx"):
    """
    Turn a bound object into what a template body should see in its place.

    Parameter      a bare literal when solo, otherwise device_view() - the
                   carrier of .get / .set_node for in-kernel access.
    HelperBuilder  specialized against `ctx` (memoized - see _SpecializeCtx)
                   and replaced with its .compiled backend callable or
                   source.
    Specializable  its .compiled backend callable or source.
    Bag            a _LazyBagView carrying the same `ctx`, so a dotted path
                   like grid.nx.get(i) traces as plain attribute lookups and
                   a helper reached that way shares the ctx's memo.
    anything else  passed through untouched.

    `ctx` is the compile this resolution belongs to - see _SpecializeCtx. It
    threads through every nested Bag and every helper-calling-helper
    resolution so the whole compile shares one memo.

    Author: B.G (07/2026)
    """
    if isinstance(value, Parameter):
        if getattr(value, "solo", False):
            resolved = value.get()
            return resolved.data if isinstance(resolved, DataHandle) else resolved
        return value.device_view()
    if isinstance(value, HelperBuilder):
        return ctx.specialize(value).compiled
    if isinstance(value, Specializable):
        return value.compiled
    if isinstance(value, Bag):
        return _LazyBagView(value, ctx)
    return value


@lru_cache(maxsize=256)
def capture_template_meta(template) -> tuple[str | None, ast.AST | None]:
    """
    Return (source_text, ast) for a template. A python def is introspected; a
    raw string (CUDA source) is kept verbatim and has no AST.

    Cached because every compile() asks twice - once to filter bindings, once
    for attach_meta - and a miss costs an inspect.getsource plus a parse. The
    tree handed back is therefore shared by every Specializable built from that
    template: treat it as read-only.

    The cache key is the template object itself, so the bound size matters -
    unbounded, it would pin every dynamically generated template and every CUDA
    source string for the life of the process. An eviction only costs one
    re-parse.

    Author: B.G (07/2026)
    """
    if isinstance(template, str):
        return template, None
    try:
        source = inspect.getsource(template)
    except (OSError, TypeError):
        return None, None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    return source, tree


def _used_bindings(template, bindings: dict[str, Any]) -> dict[str, Any]:
    """
    The subset of `bindings` whose name appears in the template body.

    Collecting every ast.Name id in the tree is enough: the root of an
    attribute chain - `phys` in `phys.g.get(0)` - is itself an ast.Name.

    With no AST to consult (a CUDA source string, or a def whose source cannot
    be recovered) this returns `bindings` unchanged, rather than dropping a
    binding it cannot prove is unused.

    Author: B.G (07/2026)
    """
    _, tree = capture_template_meta(template)
    if tree is None:
        return bindings
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return {name: value for name, value in bindings.items() if name in used}


def filter_bindings(template, bindings: dict[str, Any]) -> dict[str, Any]:
    """
    The bindings a template actually references, ready to inject.

    Anything bound but never referenced is reported in a single warning per
    compile, which is what catches a misspelled bind() name in a context with
    many parameters. Nothing is reported when there is no AST to check
    against, since _used_bindings then treats every binding as used.

    Author: B.G (07/2026)
    """
    filtered = _used_bindings(template, bindings)
    unused = sorted(set(bindings) - set(filtered))
    if unused:
        warnings.warn(
            f"template '{getattr(template, '__name__', '?')}': bound but unused: {unused}",
            UserWarning,
            stacklevel=3,
        )
    return filtered


def attach_meta(obj: Specializable, template, bindings: dict[str, Any]) -> None:
    """
    Record on a freshly built Specializable what it was made from: the
    template itself, its source, its AST, and the real bound objects (not
    type names, so a caller can inspect and reuse what was actually bound).
    See Specializable.

    Author: B.G (07/2026)
    """
    obj._template = template
    obj._source, obj._ast = capture_template_meta(template)
    obj._dependencies = dict(bindings)


class CompileBuilder(ABC):
    """
    Collects dependencies and a template, and compiles them into one
    Specializable.

    A builder is used as a chain: any number of bind() calls, one ingest(),
    then compile(). Bound objects are not inspected as they arrive - what
    each one is (Parameter, Helper, Bag, handle, plain value) is worked out
    when the template is specialized, so bind() accepts anything.

    The builder stays authoritative for the recipe throughout its life:
    `template` and `bindings` below are a read-only view onto the same state
    compile() reads, so a later layer can inspect a builder without reaching
    into its private attributes. compile() does not consume or mutate that
    state - bind() again, ingest() a different template, or just call
    compile() again, and every callable made earlier stays exactly as it was.
    Recompiling a builder that has not changed since its last compile()
    produces an equivalent callable; there is no reason to do it, though
    nothing breaks if it happens.

    Everything here is backend-independent. A backend supplies compile(), and
    may override ingest() if its templates need different handling.

    Author: B.G (07/2026)
    """

    def __init__(self):
        self._uid = new_uid()
        self._bindings: dict[str, Any] = {}
        self._bag_names: set[str] = set()
        self._template = None

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction. See Parameter.uid.

        Author: B.G (07/2026)
        """
        return self._uid

    @property
    def template(self):
        """
        The currently ingested template. Read-only - go through ingest() to
        change it.

        Author: B.G (07/2026)
        """
        return self._template

    @property
    def bindings(self) -> dict[str, Any]:
        """
        The current name -> object bindings, in bind() order. Read-only - go
        through bind() / bind_bag() to change them. This is a live view onto
        the builder's own dict, not a copy of a frozen snapshot; a compiled
        object's own `.bindings` is the frozen copy, taken at its compile
        time.

        Author: B.G (07/2026)
        """
        return self._bindings

    def bind(self, name: str, obj: Any) -> "CompileBuilder":
        """
        Register `obj` under `name` for injection into the template body.

        Binding a name a second time replaces what it pointed to - handy for
        editing a builder in place before recompiling. The one case this
        refuses is rebinding a name that arrived through bind_bag(): which
        bag member is meant is ambiguous once the bag itself may have
        changed, so this raises instead of guessing.

        Author: B.G (07/2026)
        """
        if name in self._bag_names:
            raise KeyError(f"'{name}' was bound via bind_bag() and cannot be rebound directly")
        self._bindings[name] = obj
        return self

    def bind_bag(self, bag: "Bag") -> "CompileBuilder":
        """
        Bind every member of `bag` at top level under its own name (flat), for
        when a kernel refers to members directly rather than via a bag path.

        Author: B.G (07/2026)
        """
        for name, item in bag.items():
            self.bind(name, item)
            self._bag_names.add(name)
        return self

    def ingest(self, template) -> "CompileBuilder":
        """
        Take the generic template (a python def for closure backends, a CUDA
        source string for cupy). Backends may override with their own handling.

        Author: B.G (07/2026)
        """
        self._template = template
        return self

    def as_bag(self) -> "Bag":
        """
        This builder's current bindings, regrouped as a Bag under their
        existing names.

        This is extraction for reuse, not a rewrite: the template body still
        reads the same names, so the returned bag's members are exactly what
        the builder already binds. Merge it into another bag and rebind()
        against the result to move a builder's dependencies around without
        touching the template.

        Author: B.G (07/2026)
        """
        return from_builder(self)

    def rebind(self, bag: "Bag") -> "CompileBuilder":
        """
        Re-resolve every name this builder currently binds against `bag`,
        replacing each binding with what `bag` holds under that name.

        Every bound name must be present in `bag`; if any are missing this
        raises once, listing all of them rather than stopping at the first.
        A name whose current binding is itself a Bag (bound with bind() as a
        nested group) requires `bag` to carry a Bag under that name too - a
        template reaching it by dotted path needs the same shape on the
        other end.

        This replaces bindings regardless of how they arrived, including
        names bound via bind_bag() - superseding those is the point, so it
        does not go through bind() and does not hit its bag-name raise.
        Which names came from bind_bag() is unchanged by this call: rebind
        swaps values, not the origin bookkeeping that governs future bind()
        calls.

        Author: B.G (07/2026)
        """
        missing = [name for name in self._bindings if name not in bag]
        if missing:
            raise KeyError(f"rebind: not found in bag: {sorted(missing)}")
        for name, old in self._bindings.items():
            new = bag[name]
            if isinstance(old, Bag) and not isinstance(new, Bag):
                raise TypeError(
                    f"rebind: '{name}' is bound to a nested Bag; replacement must "
                    f"also be a Bag, got {type(new).__name__}"
                )
            self._bindings[name] = new
        return self

    @abstractmethod
    def compile(self) -> Specializable:
        """
        Produce a compiled Kernel from the builder's current template and
        bindings. Does not consume or mutate the builder - the same builder
        may be compiled again, with or without edits in between, and every
        callable produced this way is independent of the others.

        On HelperBuilder this raises instead - see HelperBuilder.compile.

        Author: B.G (07/2026)
        """
        ...


class HelperBuilder(CompileBuilder):
    """
    Builds a device helper: template plus bindings, held purely as a recipe.

    A helper has no independent compiled form to keep between compiles - bind
    this builder into a KernelBuilder, directly under a name or as a member
    of a bound Bag, and the enclosing kernel's compile() specializes it
    against that same compile's bindings. The same builder reached twice in
    one compile is specialized once and shared; a later compile against
    different bindings specializes it afresh. See the module docstring and
    _SpecializeCtx.

    compile() therefore raises here: there is no standalone Helper for it to
    return. _specialize(ctx) is the real entry point, called only by
    _SpecializeCtx.specialize.

    Author: B.G (07/2026)
    """

    def compile(self) -> Specializable:
        """
        Always raises - a HelperBuilder is never specialized on its own.
        Bind it into a KernelBuilder (flat or inside a Bag) and call
        compile() on that instead; the kernel's compile() specializes every
        HelperBuilder it can reach as part of producing the kernel.

        Author: B.G (07/2026)
        """
        raise TypeError(
            "HelperBuilder.compile() is not supported: a device helper is specialized "
            "by the kernel that binds it, not on its own. Bind this builder into a "
            "KernelBuilder (directly or inside a Bag) and call compile() on that builder."
        )

    @abstractmethod
    def _specialize(self, ctx: "_SpecializeCtx") -> Specializable:
        """
        Produce this helper's specialized backend object for the compile
        `ctx` belongs to. Called at most once per compile, by
        _SpecializeCtx.specialize, which memoizes the result on this
        builder's uid - not meant to be called directly.

        Author: B.G (07/2026)
        """
        ...


class KernelBuilder(CompileBuilder):
    """
    Builds a Kernel. compile() -> host-callable Kernel.

    Author: B.G (07/2026)
    """


class Bag:
    """
    A named collection that can be handed to a builder in one go.

    A bag holds whatever a template might want to reach under one name -
    Parameters, Helpers, further Bags, plain python values - mixed
    freely. Nothing dispatches on what a bag contains: each member is resolved
    on its own type when the template is specialized, so a bag grouping a
    quantity with the helpers that act on it works exactly like one holding
    parameters alone.

    Two ways to use one: bind it whole - bind("grid", bag) - and reach its
    members by dotted path in the template body (grid.nx.get(i), grid.nbr(i)),
    or bind_bag(bag) to merge every member in at top level under its own name.

    Build it, grow it, bind it. There is no removal or reassignment: to change
    the contents, build another bag.

    Author: B.G (07/2026)
    """

    def __init__(self, items: dict[str, Any] | None = None):
        self._uid = new_uid()
        self._items: dict[str, Any] = {}
        for name, item in (items or {}).items():
            self.add(name, item)

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction, from the same counter
        as Parameters, Helpers and pool data handles. See Parameter.uid.

        Author: B.G (07/2026)
        """
        return self._uid

    def add(self, name: str, item: Any) -> None:
        """
        Register `item` under `name`. Raises if `name` is already taken.

        Author: B.G (07/2026)
        """
        if name in self._items:
            raise KeyError(f"'{name}' is already registered in this bag")
        self._items[name] = item

    def __getattr__(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name: str) -> Any:
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items)

    def items(self):
        return self._items.items()

    def walk(self, prefix: str = ""):
        """
        Yield (dotted_handle, obj) for every member, descending into nested
        Bags depth-first.

        A nested Bag produces two things: an entry for the Bag itself, at its
        own dotted path, then one entry per member underneath it. So
        `Bag({"at": Bag({"i": p1, "j": p2}), "r": p3})` walks as
        `("at", <Bag>)`, `("at.i", p1)`, `("at.j", p2)`, `("r", p3)` - the
        parent Bag's entry always precedes its members'.

        Author: B.G (07/2026)
        """
        for name, item in self._items.items():
            handle = f"{prefix}.{name}" if prefix else name
            if isinstance(item, Bag):
                yield handle, item
                yield from item.walk(handle)
            else:
                yield handle, item


def _uid_of(obj: Any) -> int | None:
    """
    An object's uid if it has one, else None. Handles bound without a uid
    (plain python values, unwrapped bindings) are simply skipped by
    check_handles rather than treated as a conflict.

    Author: B.G (07/2026)
    """
    uid = getattr(obj, "uid", None)
    return uid if isinstance(uid, int) else None


def check_handles(units: dict[str, dict[str, Any]]) -> None:
    """
    Verify that a handle means the same object everywhere it is used.

    `units` maps a unit name (a kernel, a routine step - whatever the caller
    is checking) to that unit's own {handle: obj} map, typically built from
    Bag.walk(). Across every unit given, the same handle string must resolve
    to objects sharing one uid; if two units bind the same handle to objects
    with different uids, this raises naming the handle and both owning units.

    The converse is fine and common: two different handles pointing at the
    same uid (an alias, or one Parameter reused under two names) is not a
    conflict and is not reported.

    Objects with no `uid` attribute are ignored - there is nothing to compare.

    Author: B.G (07/2026)
    """
    seen: dict[str, tuple[int, str]] = {}
    for unit_name, handles in units.items():
        for handle, obj in handles.items():
            uid = _uid_of(obj)
            if uid is None:
                continue
            prior = seen.get(handle)
            if prior is None:
                seen[handle] = (uid, unit_name)
            elif prior[0] != uid:
                raise ValueError(
                    f"handle '{handle}' is bound to different objects: "
                    f"uid {prior[0]} in '{prior[1]}' vs uid {uid} in '{unit_name}'"
                )


def _resolve_path(bag: "Bag", path: str) -> Any:
    """
    Walk a dotted path through nested Bags and return what it names.

    Raises if any segment is missing or if a non-terminal segment does not
    resolve to a Bag, naming the exact prefix that failed.

    Author: B.G (07/2026)
    """
    obj = bag
    parts = path.split(".")
    for depth, part in enumerate(parts):
        if not isinstance(obj, Bag) or part not in obj:
            failed = ".".join(parts[: depth + 1])
            raise KeyError(f"'{path}' not found in bag (no '{failed}')")
        obj = obj[part]
    return obj


def merge(*bags: "Bag") -> "Bag":
    """
    Union of every member across `bags`, into one new Bag.

    Members are taken in argument order; nesting is kept rather than
    flattened, so where two bags carry a Bag under the same name, those two
    are merged recursively instead of one replacing the other.

    A same-name collision between two non-Bag members is allowed silently
    when both share a uid - the same object reached through two bags - and
    raises when they don't, naming the member and both uids. A collision
    where either side has no uid (a plain python value) cannot be resolved
    this way and always raises, since there is nothing to compare.

    No input bag is read from twice or mutated; the result is a fresh Bag.

    Author: B.G (07/2026)
    """
    merged: dict[str, Any] = {}
    for bag in bags:
        for name, item in bag.items():
            if name not in merged:
                merged[name] = item
                continue
            existing = merged[name]
            if isinstance(existing, Bag) and isinstance(item, Bag):
                merged[name] = merge(existing, item)
                continue
            euid, iuid = _uid_of(existing), _uid_of(item)
            if euid is None or iuid is None:
                raise ValueError(
                    f"merge: '{name}' collides between bags and at least one side has "
                    f"no uid to compare, so they cannot be proven to be the same object"
                )
            if euid != iuid:
                raise ValueError(f"merge: '{name}' collides between bags: uid {euid} vs uid {iuid}")
    return Bag(merged)


def extract(bag: "Bag", names) -> "Bag":
    """
    A new Bag holding just the named members of `bag`.

    Each entry in `names` may be a plain name or a dotted path
    (`"stove.at.i"`); a dotted path is resolved through nested Bags and
    reconstructed as nesting in the result, so extracting `"at.i"` and
    `"at.j"` yields a result with an `at` sub-bag holding `i` and `j`, not
    two flat members. Raises if any path does not resolve.

    Author: B.G (07/2026)
    """
    tree: dict[str, Any] = {}
    for path in names:
        resolved = _resolve_path(bag, path)
        parts = path.split(".")
        cursor = tree
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = resolved
    return _tree_to_bag(tree)


def _tree_to_bag(tree: dict[str, Any]) -> "Bag":
    """
    Convert the nested-dict scaffolding built by extract()/trim() into
    actual Bags, leaves left untouched.

    Author: B.G (07/2026)
    """
    result = Bag()
    for name, value in tree.items():
        result.add(name, _tree_to_bag(value) if isinstance(value, dict) else value)
    return result


def trim(bag: "Bag", names) -> "Bag":
    """
    `bag` minus the named members, as a new Bag.

    Accepts the same plain-name or dotted-path entries as extract(). Removing
    `"at.i"` drops just that member, leaving `at` in the result with whatever
    else it held; removing a bare name drops that member (and, if it names a
    nested Bag, everything under it) whole. Raises if any path does not
    resolve in `bag`.

    Author: B.G (07/2026)
    """
    removal: dict[str, Any] = {}
    for path in names:
        _resolve_path(bag, path)  # validates the path exists; raises otherwise
        parts = path.split(".")
        cursor = removal
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = None

    def _copy_minus(b: "Bag", rem: dict[str, Any]) -> "Bag":
        result = Bag()
        for name, item in b.items():
            if name not in rem:
                result.add(name, item)
                continue
            sub = rem[name]
            if sub is None:
                continue
            if not isinstance(item, Bag):
                raise KeyError(f"trim: cannot descend into '{name}': not a Bag")
            result.add(name, _copy_minus(item, sub))
        return result

    return _copy_minus(bag, removal)


def replace(bag: "Bag", name: str, obj: Any) -> "Bag":
    """
    `bag` with the member at `name` swapped for `obj`, as a new Bag.

    `name` may be a dotted path into nested Bags. This is how a Parameter's
    mode is changed: mode is fixed at construction (see Parameter.mode), so
    changing it means building a new Parameter and using replace() to swap it
    into the bag in place of the old one.

    Author: B.G (07/2026)
    """
    parts = name.split(".")

    def _rebuild(b: "Bag", remaining: list[str]) -> "Bag":
        head = remaining[0]
        if head not in b:
            raise KeyError(f"'{name}' not found in bag (no '{head}')")
        result = Bag()
        for iname, item in b.items():
            if iname != head:
                result.add(iname, item)
                continue
            if len(remaining) == 1:
                result.add(iname, obj)
            else:
                if not isinstance(item, Bag):
                    raise KeyError(f"replace: cannot descend into '{head}': not a Bag")
                result.add(iname, _rebuild(item, remaining[1:]))
        return result

    return _rebuild(bag, parts)


def from_builder(builder: "CompileBuilder") -> "Bag":
    """
    A new Bag holding a builder's current bindings under their existing
    names.

    A snapshot at call time: later bind() / rebind() calls on `builder` do
    not retroactively change the returned Bag, and adding to the Bag does
    not reach back into the builder.

    Author: B.G (07/2026)
    """
    return Bag(dict(builder.bindings))
