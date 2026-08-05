"""
FrozenKernel / FrozenHelper: what ingest() (builder.py) hands back - the
immutable, value-like result of a KernelBuilder/HelperBuilder's build phase.

Both are produced only by KernelBuilder.ingest() / HelperBuilder.ingest()
(builder.py), never constructed directly. Each holds:

  template   the ingested template, unchanged (a python def or CUDA text).
  slots      a SlotGroup snapshot (slot.py) - this builder's own wired
             PARAM/HELPER/DATA slots, frozen at the size they had when
             ingest() ran.
  composed   {name: FrozenKernel|FrozenHelper} - the already-frozen
             sub-structures compose()d in during build, by identity: the very
             object handed to compose() is what sits here, never a copy.
  contract   the Contract (contract.py) derived from `template` at ingest
             time.
  split      {composed_name: frozenset(relative Address)} - which of a
             composed FrozenGroup's own shared() paths (see FrozenGroup,
             below) this object's own compose(name, frozen, split=[...])
             call opted back out of that group's default collapse, keyed by
             the composed slot name they were declared under. Empty for a
             composed child that either is not a FrozenGroup or was composed
             with no `split=`. See bound.py's `_walk_group`/`_walk_group_
             subtree` for where this is actually consulted - build() time,
             the only point split is decided (see GroupBuilder.share()'s own
             docstring, builder.py).

Nothing here is a recipe any more - a frozen object is done being built.
Mutability alternates through the scheme this module is one step of: builder
mutable -> frozen builder immutable (here) -> bound object mutable (1b) ->
compiled callable immutable (1c). `__setattr__` raises FrozenBuilderError
unconditionally after construction, so any code path that tries to poke a new
value into a frozen object - rather than building a new one - fails loudly
and by name.

A frozen object is shared, not copied: compose() the same FrozenHelper into
two different builders and both results hold that one object, checked by
identity anywhere sameness matters (uid, `is`). This is what lets one grid
neighbour helper, built once, back eighty different kernels without eighty
copies of its recipe.

`.provides` is what a *further-out* compose() sees when this object is itself
composed one level up: the set of this object's own top-level PARAM/HELPER
slot names, plus its own composed root names. DATA slots are excluded - a
DATA slot is never reached through `ctx.*` (see slot.py), so it is not part
of what a chain like `outer.this.member` could ever ask this object to
provide.

`.build()` is the entry point into the bind phase (bound.py, 1b): it walks
this object's whole composition tree - recursing into every composed
FrozenHelper/FrozenKernel in turn - and mints one independently-bindable
slot per full dotted path it finds, returning a BoundKernel/BoundHelper. This
is where "one FrozenHelper composed into eighty kernels is one frozen object
but eighty independently-bindable slot sets" actually happens: `build()`
never mutates `self` (nothing here could - see `__setattr__` above) and
allocates a fresh bind-time table on every call. See bound.py's module
docstring for the walk itself, the address grammar, and why every wired
HELPER slot must already be composed by the time `build()` runs (deferred
here rather than in `ingest()` - see bound.py for the reasoning).

Author: B.G (08/2026)
"""

from typing import Any

from ..pool.base import new_uid
from .contract import Contract
from .slot import SlotGroup, SlotKind


class FrozenBuilderError(Exception):
    """
    Raised on any attempt to mutate a frozen object - a FrozenKernel/
    FrozenHelper directly, or a KernelBuilder/HelperBuilder that has already
    been ingest()-ed (see builder.py's `_check_mutable`).

    Author: B.G (08/2026)
    """


class _Frozen:
    """
    Shared base of FrozenKernel/FrozenHelper. Not instantiated directly - see
    the module docstring.

    Author: B.G (08/2026)
    """

    def __init__(
        self,
        template: Any,
        slots: SlotGroup,
        composed: dict[str, "_Frozen"],
        contract: Contract,
        split: "dict[str, frozenset] | None" = None,
    ):
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "composed", dict(composed))
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "split", {k: frozenset(v) for k, v in (split or {}).items()})
        object.__setattr__(self, "_uid", new_uid())

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction, from the same
        counter as Parameter/HelperBuilder/Bag (parameter.py, compile.py,
        bag.py). Two references to one FrozenKernel/FrozenHelper share a uid;
        composing "the same" frozen object into two builders never changes
        it.

        Author: B.G (08/2026)
        """
        return self._uid

    @property
    def provides(self) -> set[str]:
        """
        This object's own top-level PARAM/HELPER slot names, plus its own
        composed root names - what a compose() one level further out checks
        a chain's next segment against. See the module docstring.

        Author: B.G (08/2026)
        """
        return self.slots.names(SlotKind.PARAM) | self.slots.names(SlotKind.HELPER) | set(self.composed)

    def build(self) -> "Any":
        """
        Walk this object's whole composition tree and return a
        BoundKernel/BoundHelper minting one independently-bindable slot per
        full dotted path. See the module docstring and bound.py.

        Imported locally to avoid a module-level import cycle (bound.py
        itself imports FrozenKernel/FrozenHelper from here, to tell which of
        the two `build()` produces).

        Author: B.G (08/2026)
        """
        from .bound import build as _build

        return _build(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenBuilderError(
            f"{type(self).__name__}(uid={self._uid}) is frozen and cannot be mutated - "
            f"build a new {type(self).__name__} instead"
        )

    def __delattr__(self, name: str) -> None:
        raise FrozenBuilderError(
            f"{type(self).__name__}(uid={self._uid}) is frozen and cannot be mutated - "
            f"build a new {type(self).__name__} instead"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(uid={self._uid}, provides={sorted(self.provides)})"


class FrozenKernel(_Frozen):
    """
    The frozen result of a KernelBuilder's ingest(). See the module
    docstring.

    Author: B.G (08/2026)
    """


class FrozenHelper(_Frozen):
    """
    The frozen result of a HelperBuilder's ingest(). See the module
    docstring.

    Author: B.G (08/2026)
    """


class FrozenGroup(_Frozen):
    """
    The frozen result of a GroupBuilder's close() (builder.py): a
    non-callable, navigable composite - PARAM/HELPER slots and composed
    sub-structures only, `template` always None, never itself the target of
    a device call. `ctx.grid.NX.get(0)` (a PARAM leaf reached through it) and
    `ctx.grid.neighbour(i, k)` (a composed HELPER child called through it)
    both resolve by ordinary chain recursion through `.slots`/`.composed`
    exactly as they would through a FrozenHelper one level in - a
    FrozenGroup differs only in having no template of its own to compile,
    so `ctx.grid(...)` (calling it bare) is illegal: compile_closure.py's
    `_build_ctx_node` attaches its built ctx node directly, uncompiled and
    non-callable, instead of wrapping it in `backend.func`; compile_cupy.py's
    `_resolve_chain` raises CompileError if a chain ever tries to call it
    with no further segment.

    `.contract` is always empty (a group's own build phase derives nothing -
    see GroupBuilder.close()), which is exactly right for
    compile_shared.check_legal_accessors' walk: it recurses into a
    FrozenGroup's own composed children (where real contracts live) but
    finds no PARAM chain of the group's own to check.

    `.shared` is build-phase sharing (GroupBuilder.share()): {canonical PARAM
    slot name (wired directly on this group): frozenset(relative Address)},
    each Address a dotted path into this group's own composed subtree that
    reads the "same" quantity as `canonical` - the private per-axis blocks a
    public helper composes for its own use (e.g. `neighbour_raw`'s own `row`)
    read `NX` again independently of the group's own top-level `NX` slot,
    otherwise. bound.py's build() (`_walk_group`/`_walk_group_subtree`) is
    what actually acts on this: by default, every Address in `.shared`'s
    values is never independently minted at all - only `canonical` is - so
    `grid.NX` is the one PARAM address a caller sees and binds, not
    `grid.NX` plus every private occurrence. A composer may opt specific
    paths back out at compose() time (`split=` - builder.py's `_Builder.
    compose()`, recorded as the composing object's own `.split`), re-minting
    them as independent addresses again - see bound.py's module docstring
    for the full mechanism and why it needs no separate machinery beyond a
    build-time redirect table alongside the usual address table.

    Author: B.G (08/2026)
    """

    def __init__(
        self,
        template: Any,
        slots: SlotGroup,
        composed: dict[str, "_Frozen"],
        contract: Contract,
        split: "dict[str, frozenset] | None" = None,
        shared: "dict[str, list] | None" = None,
    ):
        super().__init__(template, slots, composed, contract, split=split)
        object.__setattr__(self, "shared", {k: frozenset(v) for k, v in (shared or {}).items()})
