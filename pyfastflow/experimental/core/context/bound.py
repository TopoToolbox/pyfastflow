"""
BoundKernel / BoundHelper: the bind phase - build() a frozen builder into one
of these, then bind()/wire() its slots freely, any number of times, in any
order. See parameter.py's module docstring for the overall build -> bind ->
compile scheme; builder.py/frozen.py are the build phase this continues from.

build(frozen) - also reachable as `frozen.build()`, see frozen.py - walks
`frozen`'s whole composition tree and mints one independently-bindable slot
per full dotted path it finds: `frozen`'s own top-level PARAM/DATA slots get
single-segment addresses, and every composed root recurses with its own name
prefixed on, all the way down through nested composed FrozenHelpers. This is
where instancing happens - one FrozenHelper composed into eighty different
KernelBuilders is one frozen object (frozen.py), but eighty separate calls to
`.build()` each mint their own, independently-bindable copy of its address
tree, because each call allocates a fresh table. Every wired HELPER slot
still reachable at this point (see builder.py: ingest() does not require one
to be composed, only that a template's actual usage of it be locally known)
MUST be composed by now, or this raises naming the exact address - see the
module docstring of frozen.py for why the check waits until here rather than
running at ingest(): a HELPER slot's content is never filled by bind() (it is
fixed structurally at build time, per the design this module implements), so
build() - which is about to fix the address tree for good - is the last and
only point left where "nothing was ever composed here" can still be caught.

Addressing is by qualified dotted path, always rooted at the explicit name a
slot or a compose() root was given - `flux.grad.z`, never a positional
`step0.*`. Every address a BoundKernel/BoundHelper accepts is represented
internally as a tuple of segments (`Address`), not a bare string, precisely
so a future pattern/glob layer over these paths (explicitly deferred, not
built here) can match per-segment without this module's own addressing
scheme standing in the way; `parse_address`/`format_address` are the only
places a dotted string and a segment tuple convert between each other. Only
PARAM/DATA leaves are ever minted an address - a prefix that names a
composed sub-structure rather than one of its leaves (`flux.grad.grid`, as
opposed to `flux.grad.grid.NX`) was never given a table entry at all, so
addressing it anywhere (bind, wire, inspect) raises the same "unknown
address" any other typo would.

bind(addr, obj) fills a slot. Rebinding is normal, not an error - there is no
freeze-once rule at this layer, since immutability is a property of the
*compiled* artifact (1c), not of a slot. What bind() checks depends on the
slot's kind: PARAM accepts any Parameter, of any mode (no mode constraint -
see slot.py's module docstring, genericity across modes is the entire point
of a PARAM slot); DATA checks the dtype declared at wire_data(..., dtype=...)
time, if one was declared, against whatever is bound.

wire(addr_a, addr_b) makes two slots the same thing, as an equivalence
relation resolved *before* any value is looked at: binding either address
afterwards is visible through the other, symmetrically, and a chain of
wire() calls merges transitively (an ordinary union-find over addresses).
Wiring two slots of different kinds raises; nothing else is guarded - the
caller is trusted about which slots ought to mean the same thing.

inspect() is the framework's primary debugging surface - the only place a
caller sees the whole binding contract as pasteable addresses, current state,
and wired equivalences at a glance. See its own docstring for the exact
output shape; cross-path leaf-name collisions are reported there as
informational text, never as an error - two unrelated slots happening to
share a last segment (two different `z`s at two different addresses) is
completely ordinary and not a caller's mistake to fix.

Build-phase sharing (FrozenGroup.shared)
-------------------------------------------
A composed FrozenGroup (frozen.py) may declare, via GroupBuilder.share()
(builder.py), that several of its own composed children's PARAM slots read
the "same" quantity as one of the group's own top-level PARAM slots - grid's
`neighbour_raw.row.NX` and `is_on_edge.row.NX` (among many others) both mean
the grid's own `NX`, structurally, because a device template can only call
what is composed directly onto its own scope (builder.py's module docstring)
and so grid's own public helpers each end up re-composing the same private
`row`/`col` blocks under their own local names. Left alone, build() would
mint one independent address per occurrence - correct, but a caller then has
to bind (or wire-then-bind) every one of them by hand for what is
conceptually one value.

`_walk_group`/`_walk_group_subtree` are `_walk`'s own group-aware variant:
walking into a composed FrozenGroup with any `.shared` entries switches to
these, which mint the group's own top-level PARAM/HELPER slots exactly as
`_walk` always has, but - for every relative path any canonical's `.shared`
set names - do NOT mint an independent table entry at all; instead they
record `full_address -> canonical_address` in a side `redirect` table
threaded alongside the usual one. The net effect: `build()` mints exactly
ONE address (the canonical) for the whole equivalence class, by default - not
`bind()`-time wire()-together-many, which still leaves every address
independently listed (see wire(), above) - this is coarser and happens
before a caller ever sees the address tree at all.

Nested groups: `_ShareScope` and outermost-wins
--------------------------------------------------
A composed FrozenGroup may itself compose another FrozenGroup that also
carries `.shared` entries (visu's hillshade group composing grid, itself
composed independently under each of two private gradient blocks - see
visu/__init__.py's own module docstring for the concrete case this was
designed against). Both layers' sharing must apply at once: grid's own
internal declarations still collapse its own private duplicates to its own
top-level canonical (`...GRID.dist_from_k.DX` -> `...GRID.DX`), and the
enclosing group's declarations may additionally redirect a path that reaches
INTO that nested group (`share("DX", "at.grad_x.GRID.DX", ...)`) to the
enclosing group's own canonical (`...GRID.DX` -> `hillshade.DX`).

`_ShareScope` is one enclosing group's own sharing declarations, tagged with
the full address (`start`) its own paths are expressed relative to. Walking
into a shared FrozenGroup pushes a new scope onto an ordered list threaded
through the recursion - outermost first, most-recently-entered last -
rather than resetting to a fresh scope per boundary the way the old,
single-scope implementation did (which is exactly what dropped an
enclosing group's own declarations the moment a nested group's own boundary
was crossed). `_resolve_shared` is the one place every PARAM leaf - a
group's own top-level name (`_walk_group`) or one reached descending its
composed subtree (`_walk_group_subtree`) - is checked against every active
scope at once.

Overlapping scopes: checked OUTERMOST first, and the first match wins
outright - an inner group's own declaration for the same leaf is never even
consulted once an outer one has already claimed it. This is deliberate, not
an artifact of iteration order, and there is exactly one reason for it:
an outer scope's own canonical is always a genuinely, unconditionally minted
address (a group's own top-level PARAM loop never itself redirects, by
construction - see `_walk_group` below), so resolving outer-first can never
produce a redirect that points at another redirect. Resolving inner-first
could: `...GRID.dist_from_k.DX` might be collapsed by grid's own inner scope
to `...GRID.DX`, which is *itself* one of the outer scope's declared paths -
inner-first would leave a `redirect` entry pointing at `...GRID.DX`, which is
not itself in `table` (having been redirected further, to `hillshade.DX`),
and `value_at()`'s single-hop lookup does not chase a redirect chain. Outer-
first sidesteps this rather than requiring one: `...GRID.dist_from_k.DX` is
checked against the outer scope first, matches directly (its own full
address, translated relative to the outer scope's `start`, is one of the
outer scope's own declared paths too - grid/noise/visu's own `_find_param_
paths` helpers do not stop at the shallowest match, so both the nested
canonical and its own private occurrences typically end up declared at the
outer scope as well), and redirects straight to `hillshade.DX` in one hop,
without ever consulting grid's own inner scope for that leaf at all.

`split_paths` stays scope-local: a leaf exempted from one scope's own
collapse (that scope's own compose-site `split=`) simply falls through to
the next scope in line (or, if none claim it, mints as its own independent
address) - splitting a leaf out of the outer scope's collapse does not
affect whether some inner scope still collapses it, and vice versa.

`redirect` is consulted only by `value_at()` - the read used internally by
compile_closure.py/compile_cupy.py/compile_shared.py's own structural walks,
which always compute the FULL address as they descend the frozen tree and
need SOME resolution at every PARAM leaf they reach, collapsed or not. It is
never consulted by `bind()`/`wire()`/`unmet()`/`addresses()`/`inspect()` - a
collapsed address is not independently bindable and does not appear in any
of those, which is the intended, visible consequence of collapsing it: `.
addresses()` after composing a D8 grid reports one `NX`, not seventeen.

compose(name, frozen, split=[...]) (builder.py) opts specific relative paths
back OUT of a composed FrozenGroup's collapse, at the point that group is
composed into some other builder: `_walk`/`_walk_group_subtree` mint those
paths as ordinary, independent addresses instead of adding them to
`redirect`, exactly as if they had never been declared shared at that
compose site. This is a build-time decision, recorded on the composing
object's own `.split` (frozen.py) and read back only while `build()` walks
that one composed occurrence - never something bind() or a caller after the
fact can change; splitting the same shared path back out at a second,
different compose() site of the same group is independent and unaffected.

Sharing across two separately-built composites (`kA.grid.NX` and
`kB.grid.NX`, two different KernelBuilders each composing their own
occurrence of the same FrozenGroup) is not this mechanism at all - those are
already two different address trees by construction (two separate `build()`
calls, per frozen.py's own instancing guarantee), and reconciling them, if
ever wanted, is ordinary bind-phase wire() between the two BoundKernels'
own addresses.

Author: B.G (08/2026)
"""

from typing import Any, NamedTuple

import numpy as np

from ..pool.base import new_uid
from .frozen import FrozenKernel, _Frozen
from .slot import SlotKind

Address = tuple[str, ...]


class BindError(Exception):
    """
    Raised by anything in the bind phase: an unknown address, a wire()
    between mismatched slot kinds, a bind() of the wrong kind of object or
    the wrong dtype, or build() finding a wired HELPER slot nothing was ever
    composed into. Every case names the exact address involved.

    Author: B.G (08/2026)
    """


def parse_address(addr: str) -> Address:
    """`"flux.grad.z"` -> `("flux", "grad", "z")`. Raises on an empty string."""
    if not addr:
        raise BindError("address must not be empty")
    return tuple(addr.split("."))


def format_address(addr: Address) -> str:
    """`("flux", "grad", "z")` -> `"flux.grad.z"`."""
    return ".".join(addr)


class _LeafInfo(NamedTuple):
    """
    The fixed, never-rebound metadata build() mints for one address: which
    kind of slot it is, and - DATA only - the dtype declared at wire_data()
    time (None if left open). Distinct from the *bound value* itself, which
    lives in `_Bound._values` and is free to change via bind()/rebind().

    Author: B.G (08/2026)
    """

    kind: SlotKind
    dtype: Any


class _ShareScope(NamedTuple):
    """
    One enclosing group's own build-phase-sharing declarations, active while
    walking anywhere inside that group's own composed subtree - see the
    module docstring's "Nested groups" section.

    `start` is the full address at which this scope's own group root sits -
    every path in `shared_paths`/`split_paths` is expressed relative to it,
    so a leaf's own relative path for this scope is always `full_addr[len(
    start):]`. `shared_paths` maps a relative PARAM path to this scope's own
    canonical full address; `split_paths` is the (also relative) set this
    scope's own compose-site `split=` opted back out of that collapse.

    Author: B.G (08/2026)
    """

    start: Address
    shared_paths: "dict[Address, Address]"
    split_paths: frozenset


def _resolve_shared(full_addr: Address, scopes: "list[_ShareScope]") -> "Address | None":
    """
    The full address this PARAM leaf should redirect to, per every currently
    active enclosing group's own sharing declarations, or None if none of
    them claim it - see the module docstring's "Overlapping scopes" rule for
    why `scopes` (ordered outermost-first by construction - see
    `_walk_group`) is checked in that order, with the first match winning
    outright.

    Author: B.G (08/2026)
    """
    for scope in scopes:
        rel = full_addr[len(scope.start) :]
        canonical = scope.shared_paths.get(rel)
        if canonical is not None and rel not in scope.split_paths:
            return canonical
    return None


def _walk(
    prefix: Address,
    frozen: _Frozen,
    table: dict[Address, _LeafInfo],
    redirect: "dict[Address, Address] | None" = None,
) -> None:
    """
    Populate `table` with one entry per PARAM/DATA leaf reachable from
    `frozen`, at its full dotted path under `prefix`, recursing into every
    composed root. Raises BindError, naming the address, for a wired HELPER
    slot with nothing composed into it - see the module docstring for why
    this is where that gets caught.

    A composed child with any `.shared` entries of its own (a FrozenGroup,
    typically, but a FrozenHelper composed as a child may also carry them -
    see `_Builder.share()`, builder.py) is walked by `_walk_group` instead -
    see the module docstring's "Build-phase sharing" section. `redirect`, optional, collects the
    collapsed-address -> canonical-address table that mechanism needs;
    every caller that does not care about it (routine.py/sequence.py/
    host_block.py's own direct `_walk()` calls, which only ever want a
    reduced `table`) may simply omit it.

    Author: B.G (08/2026)
    """
    if redirect is None:
        redirect = {}
    for name in frozen.slots.names(SlotKind.PARAM):
        table[prefix + (name,)] = _LeafInfo(SlotKind.PARAM, None)
    for name in frozen.slots.names(SlotKind.DATA):
        table[prefix + (name,)] = _LeafInfo(SlotKind.DATA, frozen.slots[name].dtype)

    helper_roots = frozen.slots.names(SlotKind.HELPER) | set(frozen.composed)
    for name in helper_roots:
        addr = prefix + (name,)
        if name not in frozen.composed:
            raise BindError(
                f"'{format_address(addr)}' is a wired HELPER slot with nothing composed into "
                f"it - compose() a frozen helper under that name before build()"
            )
        child = frozen.composed[name]
        if child.shared:
            _walk_group(addr, child, table, redirect, frozen.split.get(name, frozenset()))
        else:
            _walk(addr, child, table, redirect)


def _walk_group(
    prefix: Address,
    group: _Frozen,
    table: dict[Address, _LeafInfo],
    redirect: dict[Address, Address],
    split_paths: frozenset,
    scopes: "list[_ShareScope]" = (),
) -> None:
    """
    `_walk`'s group-aware entry point: mints `group`'s own top-level PARAM/
    DATA/HELPER slots exactly as `_walk` always has - except each PARAM name
    is first checked against `scopes`, the ENCLOSING scopes already active
    when this group was reached (empty at the outermost group in a tree),
    since an enclosing group's own sharing may claim this group's own
    top-level name for further redirection (see the module docstring's
    "Nested groups" section) - then descends into its composed subtree via
    `_walk_group_subtree`, pushing this group's own scope (built from
    `group.shared`/`split_paths`) onto `scopes` for that descent.

    `group` is not always a GroupBuilder's own FrozenGroup (which indeed
    never carries DATA - GroupBuilder.wire_data always raises): a
    KernelBuilder that calls `share()` on itself produces a FrozenKernel that
    also reaches this function, as build()'s own top-level dispatch (see
    build(), below) - and a FrozenKernel's own DATA slots are exactly as real
    as a plain `_walk` would mint them, so this mints them here too rather
    than silently dropping them.

    Author: B.G (08/2026)
    """
    own_shared_paths: dict[Address, Address] = {}
    for canonical, paths in group.shared.items():
        canonical_addr = prefix + (canonical,)
        for p in paths:
            own_shared_paths[p] = canonical_addr
    own_scope = _ShareScope(prefix, own_shared_paths, frozenset(split_paths))
    nested_scopes = list(scopes) + [own_scope]

    for name in group.slots.names(SlotKind.PARAM):
        full = prefix + (name,)
        canonical = _resolve_shared(full, scopes)
        if canonical is not None:
            redirect[full] = canonical
        else:
            table[full] = _LeafInfo(SlotKind.PARAM, None)
    for name in group.slots.names(SlotKind.DATA):
        table[prefix + (name,)] = _LeafInfo(SlotKind.DATA, group.slots[name].dtype)

    helper_roots = group.slots.names(SlotKind.HELPER) | set(group.composed)
    for name in helper_roots:
        addr = prefix + (name,)
        if name not in group.composed:
            raise BindError(
                f"'{format_address(addr)}' is a wired HELPER slot with nothing composed into "
                f"it - compose() a frozen helper under that name before build()"
            )
        _walk_group_subtree(addr, group.composed[name], table, redirect, nested_scopes)


def _walk_group_subtree(
    prefix: Address,
    frozen: _Frozen,
    table: dict[Address, _LeafInfo],
    redirect: dict[Address, Address],
    scopes: "list[_ShareScope]",
) -> None:
    """
    One level of `_walk_group`'s own descent into a group's composed
    subtree. `scopes` carries every enclosing group's own sharing
    declarations, outermost first (see the module docstring's "Nested
    groups" section) - a PARAM leaf whose full address resolves against any
    of them (`_resolve_shared`) is collapsed, a `redirect` entry rather than
    an independent `table` entry; every other PARAM/DATA leaf mints
    normally, exactly as plain `_walk` would. A nested composed FrozenGroup
    (with its own `.shared`) pushes its own scope onto `scopes` for its own
    descent (`_walk_group`) rather than starting a fresh, disconnected one -
    sharing is declared per group, but an enclosing group's own declarations
    stay active reaching through a group-within-a-group, which is exactly
    what this fixes relative to the single-scope implementation this module
    used to have.

    Author: B.G (08/2026)
    """
    for name in frozen.slots.names(SlotKind.PARAM):
        full = prefix + (name,)
        canonical = _resolve_shared(full, scopes)
        if canonical is not None:
            redirect[full] = canonical
        else:
            table[full] = _LeafInfo(SlotKind.PARAM, None)
    for name in frozen.slots.names(SlotKind.DATA):
        table[prefix + (name,)] = _LeafInfo(SlotKind.DATA, frozen.slots[name].dtype)

    helper_roots = frozen.slots.names(SlotKind.HELPER) | set(frozen.composed)
    for name in helper_roots:
        addr = prefix + (name,)
        if name not in frozen.composed:
            raise BindError(
                f"'{format_address(addr)}' is a wired HELPER slot with nothing composed into "
                f"it - compose() a frozen helper under that name before build()"
            )
        child = frozen.composed[name]
        child_split = frozen.split.get(name, frozenset())
        if child.shared:
            _walk_group(addr, child, table, redirect, child_split, scopes=scopes)
        else:
            _walk_group_subtree(addr, child, table, redirect, scopes)


def build(frozen: _Frozen) -> "BoundKernel | BoundHelper":
    """
    Walk `frozen`'s composition tree and return a fresh BoundKernel (if
    `frozen` is a FrozenKernel) or BoundHelper (FrozenHelper), with one
    independently-bindable slot minted per full address found - collapsed
    per any build-phase sharing reachable in the tree (module docstring,
    "Build-phase sharing"). Also reachable as `frozen.build()` (frozen.py).

    Author: B.G (08/2026)
    """
    table: dict[Address, _LeafInfo] = {}
    redirect: dict[Address, Address] = {}
    if frozen.shared:
        # `frozen` is itself the object being build()-ed directly (e.g. for
        # standalone inspection, or a KernelBuilder/HelperBuilder that
        # declared its own share() - see _Builder.share(), builder.py)
        # rather than reached as someone else's composed child - no
        # enclosing object exists to have declared a `split`, so there is
        # none.
        _walk_group((), frozen, table, redirect, frozenset())
    else:
        _walk((), frozen, table, redirect)
    cls = BoundKernel if isinstance(frozen, FrozenKernel) else BoundHelper
    return cls(frozen, table, redirect)


def _format_state(info: _LeafInfo, value: Any) -> str:
    """
    The state column of one inspect() line - see _Bound.inspect.

    Author: B.G (08/2026)
    """
    if value is None:
        return "UNBOUND"
    if info.kind is SlotKind.PARAM:
        mode = getattr(value, "mode", None)
        if mode == "const":
            return f"bound(const {value.get()})"
        if mode is not None:
            return f"bound({mode})"
    return "bound"


_DTYPE_SHORT = {
    "float32": "f32", "float64": "f64",
    "int32": "i32", "int64": "i64",
    "uint8": "u8", "uint32": "u32",
}


def _short_dtype(dtype: Any) -> str:
    """
    A dtype in the short spelling this package writes everywhere else
    ("f32", "i64", ...) rather than python's own repr - `<class
    'numpy.float32'>` is not a pasteable dtype, it is noise. Tries a numpy
    coercion first (covers numpy dtypes/dtype classes and the cupy backend's
    own dtype objects, which already are numpy dtypes); falls back to
    `str(dtype)` for anything numpy cannot make sense of (a Taichi/Quadrants
    dtype token, which already prints short - `ti.f32` reprs as `f32`).

    Author: B.G (08/2026)
    """
    try:
        name = np.dtype(dtype).name
    except TypeError:
        return str(dtype)
    return _DTYPE_SHORT.get(name, name)


class _Bound:
    """
    Shared machinery behind BoundKernel/BoundHelper. Not instantiated
    directly - see build().

    Author: B.G (08/2026)
    """

    def __init__(
        self,
        frozen: _Frozen,
        table: dict[Address, _LeafInfo],
        redirect: "dict[Address, Address] | None" = None,
    ):
        self._uid = new_uid()
        self._frozen = frozen
        self._table = table
        # union-find over addresses: wire() merges groups, bind()/inspect()
        # always resolve through _find() first, so a value lives once per
        # group regardless of which member address it was bound through.
        self._parent: dict[Address, Address] = {addr: addr for addr in table}
        self._values: dict[Address, Any] = {}
        # build-phase-collapsed addresses (module docstring, "Build-phase
        # sharing") -> their canonical table address. Consulted by value_at()
        # only - never by bind()/wire()/unmet()/addresses(), so a collapsed
        # address stays genuinely absent from every caller-facing listing.
        self._redirect: dict[Address, Address] = dict(redirect) if redirect else {}

    @property
    def uid(self) -> int:
        """Process-wide identity assigned at construction. See Parameter.uid (parameter.py)."""
        return self._uid

    @property
    def frozen(self) -> _Frozen:
        """The FrozenKernel/FrozenHelper this object was build()-ed from."""
        return self._frozen

    def addresses(self) -> set[Address]:
        """Every address this object has a slot for - the full, fixed address tree build() minted."""
        return set(self._table)

    def value_at(self, addr: "Address | str") -> Any:
        """
        The object currently bound at `addr` (following wire()-d equivalence
        to its group's representative), or None if that group is unbound.
        Read-only counterpart to bind() - for compile.py's use, and for
        anything else that wants to read a binding without going through
        inspect()'s formatted report.

        `addr` may be a build-phase-collapsed address (module docstring,
        "Build-phase sharing") even though it is not one of `.addresses()`'
        own members - compile_closure.py/compile_cupy.py/compile_shared.py's
        structural walks compute the full address at every PARAM leaf they
        reach regardless of whether build() minted it independently or
        redirected it, and this is the one read path required to resolve
        transparently either way. `bind()`/`wire()` do not get this
        treatment - a collapsed address is not independently bindable.

        Author: B.G (08/2026)
        """
        a = self._addr_or_redirect(addr)
        return self._values.get(self._find(a))

    def _addr_or_redirect(self, addr: "Address | str") -> Address:
        """
        `addr`, validated against `.addresses()` as `_addr()` always has, OR
        - if `addr` is not itself one of this object's minted addresses -
        its build-phase-collapsed canonical address, if one was recorded.
        Raises the same "unknown address" `_addr()` always has if neither
        applies. See value_at()'s own docstring for why only that method
        uses this instead of `_addr()` directly.

        Author: B.G (08/2026)
        """
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if a in self._table:
            return a
        redirected = self._redirect.get(a)
        if redirected is not None:
            return redirected
        raise BindError(
            f"unknown address {format_address(a)!r} - not one of this object's slots "
            f"(see .addresses() for the full set)"
        )

    def slot_info(self, addr: "Address | str") -> _LeafInfo:
        """This address's fixed kind/dtype, as minted by build() - never changes after that."""
        a = self._addr(addr)
        return self._table[self._find(a)]

    def unmet(self) -> list[Address]:
        """
        Every address whose equivalence group has no bound value yet, sorted.
        Empty means every slot build() minted is filled - the precondition
        compile() checks first (see compile_shared.py's check_unmet).

        Author: B.G (08/2026)
        """
        return sorted(addr for addr in self._table if self._values.get(self._find(addr)) is None)

    def _addr(self, addr: "Address | str") -> Address:
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if a not in self._table:
            raise BindError(
                f"unknown address {format_address(a)!r} - not one of this object's slots "
                f"(see .addresses() for the full set)"
            )
        return a

    def _find(self, addr: Address) -> Address:
        parent = self._parent
        root = addr
        while parent[root] != root:
            root = parent[root]
        while parent[addr] != root:
            parent[addr], addr = root, parent[addr]
        return root

    def bind(self, addr: "Address | str", obj: Any) -> "_Bound":
        """
        Fill the slot at `addr` (or the whole equivalence group it belongs
        to, if wire()-d) with `obj`. Rebinding is normal - see the module
        docstring - and simply overwrites what was there.

        A PARAM slot accepts any Parameter, of any mode. A DATA slot with a
        declared dtype (wire_data(..., dtype=...)) checks `obj.dtype`
        against it, when `obj` has one; an open (dtype=None) DATA slot
        accepts anything. There is no HELPER case to handle here: build()
        only ever mints table entries for PARAM/DATA leaves (see the module
        docstring and `_walk`) - a dotted prefix that names a composed
        sub-structure rather than one of its leaves (e.g. `flux.grad.grid`,
        as opposed to `flux.grad.grid.NX`) was never minted an address at
        all, so `_addr` above already raises "unknown address" for it before
        this method's own kind dispatch ever runs.

        Author: B.G (08/2026)
        """
        a = self._addr(addr)
        r = self._find(a)
        info = self._table[r]
        if info.kind is SlotKind.PARAM:
            from .parameter import Parameter

            if not isinstance(obj, Parameter):
                raise BindError(
                    f"{format_address(a)!r} is a PARAM slot; expected a Parameter, got "
                    f"{type(obj).__name__}"
                )
        else:
            assert info.kind is SlotKind.DATA
            if info.dtype is not None:
                obj_dtype = getattr(obj, "dtype", None)
                if obj_dtype is not None and obj_dtype != info.dtype:
                    raise BindError(
                        f"{format_address(a)!r}: dtype mismatch, slot declares {info.dtype}, "
                        f"got {obj_dtype}"
                    )
        self._values[r] = obj
        return self

    def bind_leaf(self, mapping: dict[str, Any], *, prefix: "Address | str" = (), strict: bool = False) -> "_Bound":
        """
        Bind every address under `prefix` whose last segment is a key of
        `mapping`, to that key's value - one bind() call per match, same
        checks (PARAM/DATA kind, dtype) as an ordinary bind().

        The bulk-bind counterpart to hand-writing one bind() per address: a
        caller that does not know in advance exactly how many addresses under
        `prefix` will need a given name bound - because that count depends on
        backend/method/config, not on anything the caller controls (a
        FrozenRoutine/FrozenSequence built differently per combination - see
        e.g. _cupy_depressions.py's module docstring on real-launch
        splitting) - matches against whatever `.addresses()` actually
        minted, instead of a hand-typed address list that would need to
        change with the combination.

        `prefix`, if given, restricts the match to addresses starting with
        it - this is what resolves a leaf name recurring under two different
        meanings at two different prefixes (make_accumulation's
        pointer_jump_push ping-pong, where "rec_curr" means one buffer under
        "step_a" and a different one under "step_b" - two bind_leaf() calls,
        one per prefix, rather than one call that would bind both to
        whichever value came last).

        `strict=True` raises if any `mapping` key matched no address under
        `prefix` at all - off by default, since an existing caller may
        deliberately pass one mapping wider than what a particular
        combination's own address tree contains (leaf-name binding across
        several method/reroute combinations relies on exactly this). A key
        that genuinely never matches anything under any combination is
        almost always a typo, though - pass `strict=True` wherever the
        address set is known fixed.

        Author: B.G (08/2026)
        """
        p = parse_address(prefix) if isinstance(prefix, str) else tuple(prefix)
        plen = len(p)
        matched: set[str] = set()
        for addr in self._table:
            if addr[:plen] == p and addr[-1] in mapping:
                self.bind(addr, mapping[addr[-1]])
                matched.add(addr[-1])
        if strict:
            unused = sorted(set(mapping) - matched)
            if unused:
                raise BindError(f"bind_leaf(prefix={format_address(p)!r}): {unused} matched no address")
        return self

    def bind_pattern(self, pattern: str, obj: Any) -> "_Bound":
        """
        Bind every address matching `pattern`, a dotted string the same
        length as the addresses it may match - a `*` segment matches any one
        segment, any other segment must match literally. `"step_a.*.rec_curr"`
        matches `("step_a", "get_src", "rec_curr")` but neither
        `("step_a", "rec_curr")` (too short) nor
        `("step_a", "a", "b", "rec_curr")` (too long) - there is deliberately
        no multi-segment wildcard (see the module docstring's note on Address
        being a tuple precisely so a pattern/glob layer over these paths
        could be added without disturbing the addressing scheme itself).

        Raises if `pattern` matches zero addresses: unlike bind_leaf's
        `strict` flag (off by default, since one mapping is often
        deliberately wider than one combination's own address tree), a
        single bind_pattern() call names one specific intended match, so a
        pattern that resolves to nothing is almost always a typo, not a
        legitimately-absent combination.

        Author: B.G (08/2026)
        """
        segs = tuple(pattern.split("."))
        matched = False
        for addr in self._table:
            if len(addr) == len(segs) and all(s == "*" or s == a for s, a in zip(segs, addr)):
                self.bind(addr, obj)
                matched = True
        if not matched:
            raise BindError(f"bind_pattern({pattern!r}): matched no address")
        return self

    def wire(self, addr_a: "Address | str", addr_b: "Address | str") -> "_Bound":
        """
        Make `addr_a` and `addr_b` the same slot: binding either afterwards
        fills both, and any address already wired to either joins the same
        group (transitively - an ordinary union-find). Raises if the two
        resolve to different slot kinds, or if both sides are already bound
        to different objects (which wiring them together could not resolve
        without silently discarding one). No other guard runs - see the
        module docstring.

        Author: B.G (08/2026)
        """
        a, b = self._addr(addr_a), self._addr(addr_b)
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return self
        if self._table[ra].kind is not self._table[rb].kind:
            raise BindError(
                f"wire({format_address(a)!r}, {format_address(b)!r}): kind mismatch "
                f"({self._table[ra].kind.value} vs {self._table[rb].kind.value})"
            )
        va, vb = self._values.get(ra), self._values.get(rb)
        if va is not None and vb is not None and va is not vb:
            raise BindError(
                f"wire({format_address(a)!r}, {format_address(b)!r}): both sides are already "
                f"bound to different objects - rebind one to match before wiring"
            )
        self._parent[ra] = rb
        if rb not in self._values and va is not None:
            self._values[rb] = va
        return self

    def inspect(self) -> str:
        """
        The full binding contract, one line per address, as exact pasteable
        addresses, columns aligned to whatever the actual addresses/types on
        this object need (never a fixed width - an address only ever gets
        longer as a composition tree grows deeper):

            flux.grad.dx       PARAM  -    bound(const 30.0)
            flux.grad.z        PARAM  -    UNBOUND
            flux.acc           DATA   f32  UNBOUND
            update.dt          PARAM  -    bound(scalar)

        PARAM and DATA rows share one column layout - address, kind, type,
        state - rather than DATA carrying an extra field: a PARAM slot's
        type column reads "-" (the slot itself declares no dtype - see
        slot.py), a DATA slot's reads its declared dtype in this package's
        short spelling ("f32", not "<class 'numpy.float32'>" - see
        _short_dtype) or "any" if wire_data() left it open.

        A wire()-d address carries a trailing `[wired: ...]` note listing
        every other address in its equivalence group; the state column is
        only padded when at least one row needs that trailing note, so a
        report with no wire()-d addresses at all has no dangling whitespace.
        After the per-address lines, any leaf name (an address's own last
        segment) shared by two or more addresses that are *not* in the same
        wire()-d group is listed once more, under an "Informational"
        heading - never as an error; see the module docstring for why this
        is deliberately not a conflict.

        Author: B.G (08/2026)
        """
        groups: dict[Address, list[Address]] = {}
        for addr in self._table:
            groups.setdefault(self._find(addr), []).append(addr)

        rows: list[tuple[str, str, str, str, str]] = []
        for addr in sorted(self._table):
            info = self._table[addr]
            root = self._find(addr)
            state = _format_state(info, self._values.get(root))
            peers = sorted(a for a in groups[root] if a != addr)
            wired = f"[wired: {', '.join(format_address(p) for p in peers)}]" if peers else ""
            if info.kind is SlotKind.DATA:
                type_col = _short_dtype(info.dtype) if info.dtype is not None else "any"
            else:
                type_col = "-"
            rows.append((format_address(addr), info.kind.value.upper(), type_col, state, wired))

        by_leaf: dict[str, list[Address]] = {}
        for addr in self._table:
            by_leaf.setdefault(addr[-1], []).append(addr)
        collisions = []
        for leaf, addrs in sorted(by_leaf.items()):
            if len(addrs) < 2 or len({self._find(a) for a in addrs}) < 2:
                continue
            collisions.append(f"  '{leaf}': {', '.join(format_address(a) for a in sorted(addrs))}")

        if not rows:
            report = "(no slots)"
        else:
            w_addr = max(len(r[0]) for r in rows)
            w_kind = max(len(r[1]) for r in rows)
            w_type = max(len(r[2]) for r in rows)
            w_state = max(len(r[3]) for r in rows)
            pad_state = any(r[4] for r in rows)
            lines = []
            for addr_s, kind_s, type_s, state_s, wired_s in rows:
                parts = [addr_s.ljust(w_addr), kind_s.ljust(w_kind), type_s.ljust(w_type)]
                parts.append(state_s.ljust(w_state) if pad_state else state_s)
                line = "  ".join(parts)
                if wired_s:
                    line += f"  {wired_s}"
                lines.append(line)
            report = "\n".join(lines)

        if collisions:
            report += "\n\nInformational - same leaf name at multiple, unwired addresses (not an error):\n"
            report += "\n".join(collisions)
        return report

    def __repr__(self) -> str:
        return f"{type(self).__name__}(uid={self._uid}, slots={len(self._table)})"


class BoundKernel(_Bound):
    """
    The bound result of build()-ing a FrozenKernel. See the module
    docstring.

    Author: B.G (08/2026)
    """

    def compile(self, backend: str, **kwargs) -> Any:
        """
        The compile phase (1c) - produce a frozen, immutable callable from
        this object's current bindings. A snapshot: this BoundKernel stays
        live and rebindable afterwards, and a later compile() (with or
        without edits in between) produces an independent callable - see
        compile_shared.py's module docstring for the full contract
        (CompiledKernel, swap(), the legal-PARAM-accessor and unmet-slot
        checks every backend runs first).

        `backend` is `"taichi"`, `"quadrants"` or `"cupy"` - the same three
        names `backends.py`'s `backend_classes()` uses elsewhere in this
        package. `**kwargs` is backend-specific: cupy's `compile_kernel`
        accepts `grid=`/`block=` launch-dimension defaults (see
        compile_cupy.py); the closure backends take none.

        Imported locally, per backend, to avoid importing taichi/quadrants/
        cupy at module load time for a caller that only uses one of them -
        the same reasoning `backends.py.backend_classes` follows.

        Author: B.G (08/2026)
        """
        if backend == "taichi":
            import taichi as ti

            from . import compile_closure

            return compile_closure.compile_kernel(self, ti, **kwargs)
        if backend == "quadrants":
            import quadrants as qd

            from . import compile_closure

            return compile_closure.compile_kernel(self, qd, **kwargs)
        if backend == "cupy":
            from . import compile_cupy

            return compile_cupy.compile_kernel(self, **kwargs)
        raise BindError(f"compile: unknown backend {backend!r}, expected 'taichi', 'quadrants' or 'cupy'")


class BoundHelper(_Bound):
    """
    The bound result of build()-ing a FrozenHelper. See the module
    docstring.

    Author: B.G (08/2026)
    """

    def compile(self, backend: str, **kwargs) -> Any:
        """
        Always raises: a device helper has no standalone compiled form, on
        any backend - it is compiled as part of the BoundKernel that
        composes it (see compile_shared.py/compile_closure.py/
        compile_cupy.py). Mirrors HelperBuilder.compile() (builder.py) at
        the build phase.

        Author: B.G (08/2026)
        """
        raise TypeError(
            "BoundHelper.compile() is not supported: a device helper is compiled as part of "
            "the BoundKernel that composes it, not on its own. Compose this helper's "
            "FrozenHelper into a KernelBuilder and call compile() on the resulting BoundKernel."
        )
