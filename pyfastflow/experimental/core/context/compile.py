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

CompileBuilder._bind_raw() is the actual binding implementation, for a
factory's own internal, same-breath wiring - build a Parameter, bind it into
a template two lines later, done. `.bind(name, obj)` is the older alias to
this same method, predating Need (need.py) and kept working for the many
existing call sites across grid/, noise/, ops/ and flow/ that already use
that name; not the name to prefer in new code. Need is the contract for the
other shape - a caller in a different module builds an object and hands it
to a factory that binds it at a distance; see need.py's module docstring.

CompileBuilder's own `bind_bag()`/`_bind_bag_raw()` - binding every member of
a Bag flat, under its own name, in one call - had no real callers left once
every factory's internal wiring went through Need (see the `bind(bag)`
paragraph below) and were removed; `rebind()`, below, is the one piece of
that older all-or-nothing mechanism still in real use, by
RoutineBuilder/SequenceBuilder (routine.py/sequence.py).

`CompileBuilder(strict_needs=True)` makes that contract mandatory rather than
optional: `_bind_raw`/`bind` then refuse a name with no `.need()`-declared
Need and forward to that Need's own `.bind()` instead of writing into
`_bindings` unchecked - see `_bind_raw`'s docstring. This applies to a
Parameter, a HelperBuilder or a Bag (`Kind.BAG`, need.py, for a name bound to
a whole Bag rather than one object) - the shapes Need has an actual contract
for - and not to a plain value (a float, the backend module, ...), which
strict_needs still binds directly since there is nothing there for a Need to
check. It is opt-in per builder instance, off by default, so a builder that
has not been converted to declare its Needs keeps today's permissive
behaviour with zero change.

`bind(bag)` - a lone Bag, no name - is a different thing again, and unrelated
to strict_needs: it matches `bag`'s members against this builder's own
*declared* Needs (`.need()`) by name, binding whichever are still unbound
through each Need's own `.bind()`, and silently ignoring both a `bag` member
that names no declared Need and a declared Need with no matching member -
left unbound, reported later by unmet_needs() exactly like any other missing
Need. This is what replaces rebind()'s old all-or-nothing contract
(`bindings ⊆ bag, or raise`) with an incremental, Need-driven one; see
`bind()`'s own docstring for the full reasoning, including why no
check_handles-style guard is needed here.

Author: B.G (07/2026)
"""

import ast
import inspect
import textwrap
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from types import FunctionType
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

    A builder is used as a chain: any number of _bind_raw()/bind() calls, one
    ingest(), then compile(). Bound objects are not inspected as they arrive -
    what each one is (Parameter, Helper, Bag, handle, plain value) is worked
    out when the template is specialized, so binding accepts anything.

    _bind_raw() is the real binding implementation and the name to reach for
    in new code, for a factory's own internal, same-breath wiring. bind(name,
    obj) is the older alias to it, predating Need (need.py) and kept working
    unchanged for the many existing call sites that already use it - not the
    name to prefer first when writing something new. Not deprecated; both
    work identically.

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

    `strict_needs=True` (default False) makes every bind() call on this
    instance binding a Parameter, HelperBuilder or Bag require a
    pre-declared `.need()` for that name - see _bind_raw's docstring and the
    module docstring's paragraph on it.

    Author: B.G (07/2026)
    """

    def __init__(self, *needs: Need, strict_needs: bool = False):
        self._uid = new_uid()
        self._bindings: dict[str, Any] = {}
        self._template = None
        self._needs: dict[str, Need] = {}
        self._strict_needs = strict_needs
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
        through bind() to change them. This is a live view onto
        the builder's own dict, not a snapshot - the builder stays
        authoritative for its recipe (see the class docstring).

        Author: B.G (07/2026)
        """
        return self._bindings

    def _bind_raw(self, name: str, obj: Any) -> "CompileBuilder":
        """
        Register `obj` under `name` for injection into the template body.

        This is the name to reach for in new code for a factory's own
        internal, same-breath wiring - building a Parameter and binding it
        into a template two lines later, where no caller ever holds the
        object and there is no "where did this go" question to answer (see
        need.py's module docstring for the contrast with a caller-supplied
        Need). `bind()` is the older, still-working alias to this same
        method - see its own docstring.

        Binding a name a second time replaces what it pointed to - handy for
        editing a builder in place before recompiling.

        With `strict_needs=True` (see __init__), and only when `obj` is a
        Parameter, a HelperBuilder or a Bag - the three (four, counting
        nested Bags under kind=BAG) shapes Need actually has a contract for -
        `name` must already have a Need declared for it via `.need()`. Two
        shapes of declared Need are both legal here, and are told apart by
        whether the Need is already bound: an *unbound* Need is bound now,
        to `obj`, via its own `.bind()` (its kind/dtype/mode - or, for
        kind=BAG, its member contract - check runs there) - the "declare
        first, bind through the builder second" sequence. An *already-bound*
        Need - e.g. one built by backends.py's param_need()/helper_need(),
        which bind it at construction, meant to be handed straight to
        `.need()` and then `.bind(name, that_need.value)` in the same couple
        of lines (see make_helper) - is not bound again (a kind=PARAM/HELPER/
        BAG Need raises on a second bind, by design - see Need.bind); instead
        `obj` must be identically the Need's own already-bound `.value`, or
        this raises naming the mismatch. Either way `obj` is then recorded
        here exactly as the permissive path does.

        A plain value with no dtype/mode/identity to validate (a python
        float, the backend module, a dict of strings) is bound directly even
        under strict_needs=True - Need has no kind for "opaque value" and
        was never meant to gain one (see need.py's module docstring and the
        boundary settled for it: device call-time arguments are never Needs
        or bindings at all, and a bound *value* with no structure to check
        stays exactly that, a value - only Parameters/HelperBuilders/Bags,
        which Need can actually say something about, are required to go
        through one). This is the enforcement the Need-restructuring plan
        calls for; it is opt-in per builder instance so builders that have
        not been converted keep today's permissive behaviour unchanged (see
        the module docstring).

        Author: B.G (08/2026)
        """
        if self._strict_needs and isinstance(obj, (Parameter, HelperBuilder, Bag)):
            need = self._needs.get(name)
            if need is None:
                raise KeyError(
                    f"'{name}' has no Need declared on this builder (strict_needs=True) - "
                    f"call .need(Need({name!r}, kind=...)) before .bind()"
                )
            if need.is_bound:
                if need.value is not obj:
                    raise ValueError(
                        f"'{name}': bind() object does not match the already-bound Need's own "
                        f"value - a pre-bound Need (e.g. from param_need()/helper_need()) must be "
                        f"bound to exactly what is then passed to .bind(name, ...)"
                    )
            else:
                need.bind(obj)
        self._bindings[name] = obj
        return self

    def bind(self, name, obj: Any = None) -> "CompileBuilder":
        """
        Two forms.

        bind(name, obj) - the original, two-argument form: alias for
        _bind_raw() - see its docstring for what this actually does. Predates
        Need (need.py) and is kept working for the many existing call sites
        that already use this name across grid/, noise/, ops/ and flow/;
        still correct for internal, same-breath wiring, but not the name to
        prefer when writing something new - call _bind_raw() directly
        instead, and reach for a Need when what is being bound was
        constructed by a caller in a different module and consumed only
        later.

        bind(bag) - `name` given alone as a Bag, `obj` omitted: matches
        `bag`'s members against this builder's own *declared* Needs
        (`.need()`) by name, binding whichever are currently unbound through
        each Need's own `.bind()` (dtype/mode/contains checked there, same as
        any other Need bind - see need.py). A declared Need with no matching
        member in `bag` is simply left unbound - reported later by
        unmet_needs()/compile() exactly like any other still-missing Need,
        not an error here. A member of `bag` matching no declared Need is
        likewise ignored, not an error. kind=DATA needs are never touched by
        this form: a DATA Need is never a compile-time binding (need.py).

        Call this more than once, with different bags, to fill in a
        builder's Needs incrementally - a Need already bound (by an earlier
        bind(bag) call, or by an ordinary bind(name, obj)/`.need()`-then-
        `.bind()`) is simply skipped on a later call, never rebound. This is
        also why no check_handles (bag.py)-style cross-check runs here the
        way rebind() needs one at the RoutineBuilder/Sequence layer:
        check_handles exists to catch a *name* silently coming to mean two
        different objects across units once rebind's blind replacement is
        applied. A Need can never do that - kind=PARAM/HELPER/BAG needs are
        frozen after their one bind (need.py) and raise immediately on a
        second, conflicting bind - so the failure mode check_handles guards
        against cannot arise through this path; two Needs sharing a name are,
        by construction, either the same slot (one object, bound once) or two
        independent slots that never interact.

        Author: B.G (08/2026)
        """
        if isinstance(name, Bag) and obj is None:
            return self._bind_needs_from_bag(name)
        return self._bind_raw(name, obj)

    def _bind_needs_from_bag(self, bag: "Bag") -> "CompileBuilder":
        """
        The bind(bag) primitive - see bind()'s own docstring for the exact
        semantics. Not an alias of anything; this is the real implementation.

        Author: B.G (08/2026)
        """
        for need in self._needs.values():
            if need.kind is Kind.DATA or need.is_bound:
                continue
            if need.name in bag:
                need.bind(bag[need.name])
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

        This replaces bindings regardless of how they arrived - it does not
        go through bind() at all, so none of bind()'s own checks (e.g.
        strict_needs) run here; rebind swaps values, nothing else. Called by
        RoutineBuilder/SequenceBuilder (routine.py/sequence.py) to fold their
        own shared bag into each step/block's builder.

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


def resolve_binding_host(value):
    """
    Host-mode counterpart to resolve_binding: what a bound object becomes
    inside a host callback's globals, when there is no backend compilation
    step at all - only host python running the callback directly.

    Parameter      itself, unwrapped - host code reads/writes it through
                   parameter.py's own host-facing surface (.get()/.set()/
                   .read()), never through device_view(), which does not
                   exist for host use.
    HelperBuilder  raises - a device helper is compiled for and only
                   callable from device/kernel scope; there is nothing a
                   host callback could do with one.
    Bag            a _HostLazyBagView, so a dotted path resolves each member
                   the same way, recursively, on first access.
    anything else  passed through untouched, exactly as resolve_binding.

    Author: B.G (08/2026)
    """
    if isinstance(value, Parameter):
        return value
    if isinstance(value, HelperBuilder):
        name = getattr(value.template, "__name__", value.template)
        raise TypeError(
            f"cannot resolve device helper {name!r} for host use: a HelperBuilder is "
            f"specialized by a kernel's own compile() and only callable from device/kernel "
            f"scope, not host python - bind a Parameter or a plain value instead"
        )
    if isinstance(value, Bag):
        return _HostLazyBagView(value)
    return value


class _HostLazyBagView:
    """
    Host-mode counterpart to _LazyBagView: what a Bag looks like from inside
    a host callback once spliced into its globals by HostHelperBuilder.

    No _SpecializeCtx to thread through: resolving a member here never
    reaches a backend compile (a Parameter unwraps to itself, a HelperBuilder
    raises - see resolve_binding_host), so there is nothing across two
    resolutions of the same object to memoize the way device resolution does
    (see _SpecializeCtx). Member resolution is still deferred to first access
    and cached in the instance dict from then on, same as _LazyBagView.

    Author: B.G (08/2026)
    """

    def __init__(self, bag: "Bag"):
        object.__setattr__(self, "_bag", bag)

    def __getattr__(self, name: str) -> Any:
        # only called on a genuine miss (cached hits never reach here)
        bag = object.__getattribute__(self, "_bag")
        if name not in bag:
            raise AttributeError(name)
        resolved = resolve_binding_host(bag[name])
        self.__dict__[name] = resolved
        return resolved


class HostHelperBuilder(CompileBuilder):
    """
    Builds a host callback: a plain python function plus its bindings, held
    as a recipe, resolved for host python rather than for any device
    backend.

    A brother class to HelperBuilder, not a subclass of it - deliberately
    much lighter. HelperBuilder's three backend subclasses, its
    _SpecializeCtx memoization/cycle detection, and its Parameter ->
    device_view() resolution step all exist because *device* code needs
    backend-specific compilation; one HelperBuilder subclass per backend, one
    ClosureHelper/ti.func per specialize. A host callback is already a plain
    python function on every backend - there is nothing backend-specific left
    to dispatch on, so one class covers Taichi, Quadrants and cupy alike.
    "Compiling" it is exactly the closure-injection _closure_backend.py's
    specialize_closure already does for a device func - rebuild the function
    around a globals dict carrying the resolved bindings - minus handing the
    result to ti.func/qd.func; the rebuilt function is returned directly,
    already callable from host python.

    Reuses CompileBuilder's .need()/.bind()/strict_needs/.needs/
    .unmet_needs()/_resolve_needs() unchanged, by inheritance - nothing new
    is added to that surface here.

    Author: B.G (08/2026)
    """

    def compile(self) -> FunctionType:
        """
        Splice this builder's referenced bindings - resolved host-mode, see
        resolve_binding_host - into the ingested template's own globals, and
        return the rebuilt function. The result is ordinary host python,
        directly callable; no backend object is involved anywhere in this
        path.

        Author: B.G (08/2026)
        """
        self._resolve_needs()
        template = self._template
        filtered = filter_bindings(template, self._bindings)
        resolved = {name: resolve_binding_host(value) for name, value in filtered.items()}

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


