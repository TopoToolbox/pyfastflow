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

Author: B.G (08/2026)
"""

from typing import Any, NamedTuple

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


def _walk(prefix: Address, frozen: _Frozen, table: dict[Address, _LeafInfo]) -> None:
    """
    Populate `table` with one entry per PARAM/DATA leaf reachable from
    `frozen`, at its full dotted path under `prefix`, recursing into every
    composed root. Raises BindError, naming the address, for a wired HELPER
    slot with nothing composed into it - see the module docstring for why
    this is where that gets caught.

    Author: B.G (08/2026)
    """
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
        _walk(addr, frozen.composed[name], table)


def build(frozen: _Frozen) -> "BoundKernel | BoundHelper":
    """
    Walk `frozen`'s composition tree and return a fresh BoundKernel (if
    `frozen` is a FrozenKernel) or BoundHelper (FrozenHelper), with one
    independently-bindable slot minted per full address found. See the
    module docstring. Also reachable as `frozen.build()` (frozen.py).

    Author: B.G (08/2026)
    """
    table: dict[Address, _LeafInfo] = {}
    _walk((), frozen, table)
    cls = BoundKernel if isinstance(frozen, FrozenKernel) else BoundHelper
    return cls(frozen, table)


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


class _Bound:
    """
    Shared machinery behind BoundKernel/BoundHelper. Not instantiated
    directly - see build().

    Author: B.G (08/2026)
    """

    def __init__(self, frozen: _Frozen, table: dict[Address, _LeafInfo]):
        self._uid = new_uid()
        self._frozen = frozen
        self._table = table
        # union-find over addresses: wire() merges groups, bind()/inspect()
        # always resolve through _find() first, so a value lives once per
        # group regardless of which member address it was bound through.
        self._parent: dict[Address, Address] = {addr: addr for addr in table}
        self._values: dict[Address, Any] = {}

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
        addresses:

            flux.grad.z      PARAM  UNBOUND
            flux.grad.dx     PARAM  bound(const 30.0)
            flux.acc         DATA   f32  UNBOUND
            update.dt        PARAM  bound(scalar)

        A wire()-d address carries a trailing `[wired: ...]` note listing
        every other address in its equivalence group. After the per-address
        lines, any leaf name (an address's own last segment) shared by two
        or more addresses that are *not* in the same wire()-d group is
        listed once more, under an "Informational" heading - never as an
        error; see the module docstring for why this is deliberately not a
        conflict.

        Author: B.G (08/2026)
        """
        groups: dict[Address, list[Address]] = {}
        for addr in self._table:
            groups.setdefault(self._find(addr), []).append(addr)

        lines: list[str] = []
        for addr in sorted(self._table):
            info = self._table[addr]
            root = self._find(addr)
            state = _format_state(info, self._values.get(root))
            peers = sorted(a for a in groups[root] if a != addr)
            suffix = f"   [wired: {', '.join(format_address(p) for p in peers)}]" if peers else ""
            kind_col = info.kind.value.upper()
            if info.kind is SlotKind.DATA:
                dtype_col = str(info.dtype) if info.dtype is not None else "any"
                lines.append(f"{format_address(addr):<18} {kind_col:<7} {dtype_col:<8} {state}{suffix}")
            else:
                lines.append(f"{format_address(addr):<18} {kind_col:<7} {state}{suffix}")

        by_leaf: dict[str, list[Address]] = {}
        for addr in self._table:
            by_leaf.setdefault(addr[-1], []).append(addr)
        collisions = []
        for leaf, addrs in sorted(by_leaf.items()):
            if len(addrs) < 2 or len({self._find(a) for a in addrs}) < 2:
                continue
            collisions.append(f"  '{leaf}': {', '.join(format_address(a) for a in sorted(addrs))}")

        report = "\n".join(lines) if lines else "(no slots)"
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


class BoundHelper(_Bound):
    """
    The bound result of build()-ing a FrozenHelper. See the module
    docstring.

    Author: B.G (08/2026)
    """
