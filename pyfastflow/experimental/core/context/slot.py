"""
Slot: one named place declared on a builder during the build phase, plus
SlotGroup, the flat container that holds a builder's own set of them.

This is the vocabulary wire_param()/wire_helper()/wire_data() (builder.py)
speak in. A Slot carries a name and a kind - PARAM, HELPER or DATA - plus,
for a DATA slot only, an optional dtype (see DataSlot). Nothing else: no
mode, no backend concern. The kind says how the slot is reached once the
template is written:

  PARAM   read through ctx in device code (ctx.z.get(i)) - a Parameter is
          bound to this slot later, in the bind phase (1b), not here. Access
          is uniform across a Parameter's modes: ctx.z.get(i) reads z
          whether it is const, scalar or field, with `i` simply ignored for
          const/scalar - the same "read all three modes identically" design
          Parameter itself states (parameter.py's module docstring). This is
          a stated convention, not something enforced here or in the bind
          phase: which accessor spellings are legal on device code is a
          backend-emission question, settled where the backend actually
          knows what it can generate (1c).
  HELPER  called through ctx in device code (ctx.grad(...)) - a HelperBuilder
          is composed or bound to this slot, likewise later.
  DATA    a trusted device call argument of the compiled kernel/helper's own
          signature, never reached through ctx at all - see the module
          docstring of builder.py for why wire_data raises on a HelperBuilder.
          An optional dtype declared at wire_data(name, dtype=...) time is
          the one thing checked early: a wrong-dtype buffer is what corrupts
          a launch or segfaults silently, the same failure need.py's own
          kind=DATA discussion (need.py) already exists to catch. Where that
          check actually runs against a call-time value is bind/compile
          (1b/1c) work; the dtype is only declared and carried here.

A Slot is local to the one builder that declared it via wire_*(); nothing
here is process-wide or shared the way a Parameter's or a HelperBuilder's own
uid is (compile.py, parameter.py). Two builders wanting "the same" slot get
there through compose() (an already-frozen sub-structure, shared by
identity - see frozen.py) or, through bind()'s addressing (1b), never by two
Slots comparing equal.

SlotGroup is one level of depth on purpose - unlike Bag (bag.py), which
nests and is reached in-kernel by dotted path, a SlotGroup only ever
enumerates one builder's own local names. Depth comes from compose(), which
attaches an entire other frozen builder's own address tree under one slot
name; SlotGroup itself never nests.

Naming note, settled after 1a: "Handle" is reserved exclusively for
pool.base.DataHandle (core/pool/) - the device buffer handle behind a
Parameter's scalar/field storage. This module never uses that word for
anything of its own; every declared place at this layer is a Slot.

Author: B.G (08/2026)
"""

from enum import Enum
from typing import Iterator


class SlotKind(Enum):
    """
    What a Slot's place will eventually hold. See the module docstring for
    how each kind is reached from device code.

    Author: B.G (08/2026)
    """

    PARAM = "param"
    HELPER = "helper"
    DATA = "data"


class SlotGroupError(Exception):
    """
    Raised when a builder's local slot namespace is misused - wiring a name
    twice (as a slot or as a compose() root), or looking up a name that was
    never wired.

    Author: B.G (08/2026)
    """


class Slot:
    """
    One named place of a given SlotKind, local to the builder that declared
    it. See the module docstring.

    Author: B.G (08/2026)
    """

    __slots__ = ("name", "kind")

    def __init__(self, name: str, kind: SlotKind):
        self.name = name
        self.kind = kind

    def __repr__(self) -> str:
        return f"Slot({self.name!r}, kind={self.kind.value})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Slot) and self.name == other.name and self.kind is other.kind

    def __hash__(self) -> int:
        return hash((self.name, self.kind))


class ParamSlot(Slot):
    """A PARAM slot - wire_param()'s own slot type. See the module docstring."""

    __slots__ = ()

    def __init__(self, name: str):
        super().__init__(name, SlotKind.PARAM)


class HelperSlot(Slot):
    """A HELPER slot - wire_helper()'s own slot type. See the module docstring."""

    __slots__ = ()

    def __init__(self, name: str):
        super().__init__(name, SlotKind.HELPER)


class DataSlot(Slot):
    """
    A DATA slot - wire_data()'s own slot type. See the module docstring.

    `dtype` is optional: None means the slot stays open (any dtype accepted
    whenever it is eventually checked against a call-time value), anything
    else declares a contract wire_data(name, dtype=...) callers can rely on
    being validated downstream (1b/1c).

    Author: B.G (08/2026)
    """

    __slots__ = ("dtype",)

    def __init__(self, name: str, dtype=None):
        super().__init__(name, SlotKind.DATA)
        self.dtype = dtype

    def __repr__(self) -> str:
        if self.dtype is None:
            return f"Slot({self.name!r}, kind=data)"
        return f"Slot({self.name!r}, kind=data, dtype={self.dtype})"


class SlotGroup:
    """
    The flat {name: Slot} namespace behind one builder's wire_*() calls.

    Add-only during the build phase: a name may be wired once, checked here
    rather than left to collide silently later. copy() gives the snapshot a
    frozen builder (frozen.py) keeps for itself once ingest() has run, so a
    later mutation of the *builder's* group (which cannot happen anyway,
    since ingest() freezes the builder - see builder.py) could never reach
    back into an already-frozen result even if that guard were ever loosened.

    Author: B.G (08/2026)
    """

    def __init__(self):
        self._slots: dict[str, Slot] = {}

    def add(self, slot: Slot) -> None:
        """
        Register `slot` under its own name. Raises if that name is already
        wired on this group, as a Slot or otherwise.

        Author: B.G (08/2026)
        """
        if slot.name in self._slots:
            raise SlotGroupError(
                f"'{slot.name}' is already wired on this builder "
                f"(as {self._slots[slot.name]!r})"
            )
        self._slots[slot.name] = slot

    def __contains__(self, name: str) -> bool:
        return name in self._slots

    def __getitem__(self, name: str) -> Slot:
        return self._slots[name]

    def __iter__(self) -> Iterator[Slot]:
        return iter(self._slots.values())

    def __len__(self) -> int:
        return len(self._slots)

    def names(self, kind: SlotKind | None = None) -> set[str]:
        """
        Every wired name, or just those of one `kind` if given.

        Author: B.G (08/2026)
        """
        if kind is None:
            return set(self._slots)
        return {name for name, slot in self._slots.items() if slot.kind is kind}

    def copy(self) -> "SlotGroup":
        """
        A fresh SlotGroup holding the same Slot objects (Slot is itself
        immutable data, so nothing needs a deeper copy).

        Author: B.G (08/2026)
        """
        new = SlotGroup()
        new._slots = dict(self._slots)
        return new

    def __repr__(self) -> str:
        if not self._slots:
            return "SlotGroup()"
        body = ", ".join(repr(s) for s in self._slots.values())
        return f"SlotGroup({body})"
