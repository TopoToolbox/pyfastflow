"""
Turning a template plus its bindings into something the backend can run.

A builder collects bindings and a template; compile() specializes them into a
Specializable - a launchable Kernel, or, for a helper, the internal object an
enclosing kernel's compile() produces. resolve_binding is the hub of it: it
decides what a bound object becomes inside a template body, dispatching on
whether it is a Parameter, a HelperBuilder, an already-specialized object, or a
Bag.

Everything here is one interlocking piece - _SpecializeCtx specializes a
HelperBuilder, which resolves its own bindings back through resolve_binding,
which may reach another HelperBuilder through a _LazyBagView. It reads
Parameter (parameter.py) and Bag (bag.py) but neither reads it back.

Only the abstract builders live here; the concrete Taichi*, Quadrants* and
Cupy* ones sit alongside in their own modules. See parameter.py's module docstring
for what the whole scheme is for.

Author: B.G (07/2026)
"""

import ast
import inspect
import textwrap
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from ..pool.base import new_uid
from .bag import Bag, from_builder
from .need import Kind, Need
from .parameter import Parameter


class Specializable(ABC):
    """
    Anything produced by specializing a template against bindings: a
    launchable Kernel, or the internal object a HelperBuilder specializes
    into as part of an enclosing kernel's compile().

    The builder that produced this object stays authoritative for its recipe
    - template and bindings - see CompileBuilder.

    Author: B.G (07/2026)
    """

    name: str

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

    Parameter      device_view() - the carrier of .get / .set_node for
                   in-kernel access.
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
    raw string (CUDA source) is kept verbatim and has no AST. The source
    returned as `source_text` is `inspect.getsource`'s own indentation
    (whatever the def's nesting produced); the source parsed into `tree` is
    dedented first, so a nested def's indented body parses instead of
    raising an IndentationError - a no-op for a module-level def, which is
    already at column 0. A def with no recoverable source at all (a lambda,
    an exec'd function) still comes back with `tree = None`.

    Cached because every compile() asks once to filter bindings, and a miss
    costs an inspect.getsource plus a parse. The tree handed back is
    therefore shared by every Specializable built from that template: treat
    it as read-only.

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
        tree = ast.parse(textwrap.dedent(source))
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

    def __init__(self, *needs: Need):
        self._uid = new_uid()
        self._bindings: dict[str, Any] = {}
        self._bag_names: set[str] = set()
        self._template = None
        self._needs: dict[str, Need] = {}
        if needs:
            self.need(*needs)

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
        the builder's own dict, not a snapshot - the builder stays
        authoritative for its recipe (see the class docstring).

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

    def need(self, *needs: Need) -> "CompileBuilder":
        """
        Declare that this builder's template references each of `needs` by
        its own `.name` - the single contract PARAM/DATA/HELPER bindings all
        go through now (see need.py's module docstring). Declaring a Need
        does not bind it; call `.bind()` on the Need object itself,
        independently of this builder, whenever the concrete object is
        available - two builders declaring the *same* Need object share its
        binding, by identity, not by matching name strings.

        Author: B.G (08/2026)
        """
        for n in needs:
            self._needs[n.name] = n
        return self

    @property
    def needs(self) -> dict[str, Need]:
        """
        This builder's own declared needs, by name. Read-only - go through
        need() to add more.

        Author: B.G (08/2026)
        """
        return self._needs

    @property
    def data_needs(self) -> tuple[Need, ...]:
        """
        This builder's own declared kind=DATA needs, in declaration order -
        the contract a compiled Kernel/Routine validates call-time arguments
        against. Does not include a bound HELPER need's own data needs: a
        helper never takes call-time data arguments of its own (only a
        Kernel/Routine's top-level template does), so there is nothing to
        flatten here the way unmet_needs() flattens PARAM/HELPER.

        Author: B.G (08/2026)
        """
        return tuple(n for n in self._needs.values() if n.kind is Kind.DATA)

    def unmet_needs(self) -> list[Need]:
        """
        Every currently-unbound need this builder requires, flattened with
        whatever a bound kind=HELPER need's own HelperBuilder still needs
        (recursively - automatic flattening, all the way down). Empty means
        compile() may proceed.

        Author: B.G (08/2026)
        """
        unmet: list[Need] = []
        for n in self._needs.values():
            unmet.extend(n.unmet_needs())
        return unmet

    def _resolve_needs(self) -> None:
        """
        Raise, listing every unmet need by name and kind, if this builder (or
        anything it transitively needs through a bound HELPER need) is not
        fully bound. Otherwise copy every bound PARAM/HELPER need's value
        into `self._bindings` under its own name, so the existing
        resolve_binding/AST-scan/cupy-span pipeline sees exactly what it
        always has - a plain name -> object dict - without needing to know
        Need exists. kind=DATA needs are deliberately never copied into
        `self._bindings`: they stay call-time arguments, validated instead
        via `data_needs` by whichever concrete compile() wires that check in.

        Every concrete compile() must call this first.

        Author: B.G (08/2026)
        """
        unmet = self.unmet_needs()
        if unmet:
            listing = ", ".join(f"{n.name!r} ({n.kind.value})" for n in unmet)
            raise ValueError(f"cannot compile: unmet needs: {listing}")
        for n in self._needs.values():
            if n.kind is not Kind.DATA:
                self._bindings[n.name] = n.value

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


