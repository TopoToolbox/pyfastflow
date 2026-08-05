"""
KernelBuilder / HelperBuilder: the build phase - wire_param()/wire_helper()/
wire_data()/compose(), then ingest() to close it out. See parameter.py's
module docstring for what the overall scheme (build -> bind -> compile) is
for; this module is the first of those three phases only.

A builder is mutable exactly until ingest() runs. wire_param(name)/
wire_helper(name)/wire_data(name) each declare one local Slot (slot.py);
compose(name, frozen) attaches an already-frozen sub-structure (frozen.py)
under an explicit slot name - never positionally - so a template can reach
`flux.grad.z` once `flux` names what was composed. Every name - wired or
composed - lives in one flat namespace per builder; wiring the same name
twice, or composing over a name already used by a PARAM/DATA slot, raises.
compose() may target a name already wire_helper()'d, though: a HELPER slot
declares that a name *will* be reachable as `ctx.{name}(...)`, and compose()
is how that promise gets kept - see compose()'s own docstring for why this
is allowed while every other double-use of a name is not.

ingest(template) is where the local contract is checked and the structural
contract is derived (contract.py) - a python def by static AST walk, CUDA
source text by scanning its own `$ctx....$` spans, dispatched on
`isinstance(template, str)`. Every chain the contract requires must resolve:
its root is either a wired PARAM/HELPER slot (further segments trusted,
unchecked - see the class docstrings below) or a composed root, in which case
the *next* segment must be among what the composed candidate `.provides`
(frozen.py) - otherwise this raises naming exactly what is missing. A chain
rooted at a wired DATA slot is a contract violation of a different kind: a
DATA slot is a plain call argument of the template's own signature, never
reached through `ctx`, so this raises with a hint toward that instead of the
generic "no declared slot" message. ingest() does not itself require a wired
HELPER slot to already be composed - a template need not reference every
slot it declares, and an unreferenced, uncomposed HELPER slot is harmless to
ingest(). It becomes a hard requirement one phase later, at build() (see
frozen.py, bound.py): the address tree build() walks has no way to
represent "reachable, but nothing composed here yet".

ingest() returns a frozen, immutable FrozenKernel/FrozenHelper and freezes
the builder itself in the same call - every wire_*/compose/ingest afterwards
raises FrozenBuilderError. A builder is therefore used once, start to finish;
build a new one for a different template rather than trying to reuse an
ingested one. This is a deliberate choice this design left open (see the
Phase-1a report): nothing forces "one builder, one ingest" as opposed to
letting a builder be re-ingested with a different template against the same
wired slots, but a builder holding live, still-mutable slot state after
having already handed out one frozen, immutable result reads as exactly the
kind of aliasing hazard the frozen/mutable split exists to rule out, so this
implementation closes it off entirely instead.

Author: B.G (08/2026)
"""

from typing import Any

from ..pool.base import new_uid
from .contract import Contract, ContractError, extract_cupy_contract, extract_python_contract
from .frozen import FrozenBuilderError, FrozenHelper, FrozenKernel, _Frozen
from .slot import DataSlot, HelperSlot, ParamSlot, Slot, SlotGroup, SlotGroupError, SlotKind


class _Builder:
    """
    Shared build-phase machinery behind KernelBuilder/HelperBuilder. Not
    instantiated directly.

    Author: B.G (08/2026)
    """

    def __init__(self):
        self._uid = new_uid()
        self._slots = SlotGroup()
        self._composed: dict[str, _Frozen] = {}
        self._frozen = False

    @property
    def uid(self) -> int:
        """Process-wide identity assigned at construction. See Parameter.uid (parameter.py)."""
        return self._uid

    @property
    def slots(self) -> SlotGroup:
        """This builder's currently wired slots. Read-only - go through wire_*() to add more."""
        return self._slots

    @property
    def composed(self) -> dict[str, _Frozen]:
        """This builder's currently composed {name: frozen sub-structure}. Read-only - go through compose()."""
        return dict(self._composed)

    def _check_mutable(self) -> None:
        if self._frozen:
            raise FrozenBuilderError(
                f"{type(self).__name__}(uid={self._uid}) has already been ingest()-ed and is "
                f"frozen - build a new {type(self).__name__} instead of reusing this one"
            )

    def _wire(self, slot: Slot) -> "_Builder":
        self._check_mutable()
        if slot.name in self._composed:
            raise SlotGroupError(f"'{slot.name}' is already composed on this builder")
        self._slots.add(slot)
        return self

    def wire_param(self, name: str) -> "_Builder":
        """
        Declare a PARAM slot named `name`: reached in device code as
        `ctx.{name}.get(...)` / `ctx.{name}.set_node(...)`, uniformly across
        whatever mode the Parameter eventually bound to it has (see slot.py's
        module docstring). Deliberately generic - a slot declares a place to
        plug in a Parameter (parameter.py) later (1b), not a shape; nothing
        here constrains mode or dtype, which is the entire point of a
        Parameter being able to move between const/scalar/field without
        touching a template.

        Author: B.G (08/2026)
        """
        return self._wire(ParamSlot(name))

    def wire_helper(self, name: str) -> "_Builder":
        """
        Declare a HELPER slot named `name`: called in device code as
        `ctx.{name}(...)`. Filled at build time via compose() under this
        same name - see the module docstring and compose()'s own docstring
        for why compose() (and only compose()) may target an already-wired
        HELPER slot.

        Author: B.G (08/2026)
        """
        return self._wire(HelperSlot(name))

    def wire_data(self, name: str, *, dtype: Any = None) -> "_Builder":
        """
        Declare a DATA slot named `name`: a trusted call argument of the
        compiled kernel/helper's own signature, never reached through `ctx`.
        See slot.py's module docstring for the PARAM/HELPER/DATA distinction.
        Overridden on HelperBuilder to always raise - a helper is
        device-only and takes data only as its caller's own trusted
        argument, never as a declared slot of its own.

        `dtype`, optional, declares this slot's data-argument contract -
        checked later (1b/1c) against whatever value is actually bound or
        passed. Left as None, the slot stays open to any dtype.

        Author: B.G (08/2026)
        """
        return self._wire(DataSlot(name, dtype=dtype))

    def compose(self, name: str, frozen: _Frozen) -> "_Builder":
        """
        Attach an already-frozen sub-structure (a FrozenKernel or
        FrozenHelper - frozen.py) under slot `name`, giving a template
        reaching `ctx.{name}` access to whatever `frozen` itself provides -
        `{name}.{member}` for any PARAM/HELPER slot or composed root
        `frozen` carries at its own top level.

        `frozen` is stored by identity, not copied: compose the same object
        into any number of builders and every one of them shares it.

        `name` may be either fresh (nothing wired or composed under it yet)
        or a HELPER slot already declared via wire_helper() on this same
        builder - composing there is how that slot's "reachable, filled in
        later" promise is kept, and is the one case where a name may be used
        twice: once to wire the slot, once to compose its content. Composing
        under a name already composed, or already wired as PARAM/DATA
        (a kind compose() has no business filling), raises.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        if not isinstance(frozen, _Frozen):
            raise TypeError(f"compose({name!r}, ...): expected a FrozenKernel/FrozenHelper, got {type(frozen).__name__}")
        if name in self._composed:
            raise SlotGroupError(f"'{name}' is already composed on this builder")
        if name in self._slots and self._slots[name].kind is not SlotKind.HELPER:
            raise SlotGroupError(
                f"'{name}' is already wired on this builder as {self._slots[name]!r}; compose() "
                f"only fills a HELPER slot (or a fresh name), never a PARAM/DATA one"
            )
        self._composed[name] = frozen
        return self

    def _derive_and_check(self, template: Any) -> tuple[SlotGroup, dict[str, _Frozen], Contract]:
        """
        Derive `template`'s Contract and check every chain it requires
        against this builder's wired slots and composed sub-structures -
        see the module docstring for exactly what each chain shape needs.
        Returns the (slots, composed, contract) triple ingest() freezes
        into a FrozenKernel/FrozenHelper; raises nothing itself, letting
        ContractError/SlotGroupError from the checks below propagate.

        Author: B.G (08/2026)
        """
        contract = extract_cupy_contract(template) if isinstance(template, str) else extract_python_contract(template)

        param_and_helper_roots = self._slots.names(SlotKind.PARAM) | self._slots.names(SlotKind.HELPER)
        data_roots = self._slots.names(SlotKind.DATA)

        for chain in contract.chains:
            root = chain[0]
            if root in self._composed:
                contract.check_root(root, self._composed[root].provides)
            elif root in param_and_helper_roots:
                continue
            elif root in data_roots:
                raise ContractError(
                    f"ctx.{root} is not reachable: '{root}' is a wire_data slot, and data is "
                    f"a trusted call argument of the template's own signature, never reached "
                    f"through ctx - pass it as a plain parameter instead of wiring it as a slot"
                )
            else:
                raise ContractError(
                    f"ctx.{root} has no declared slot - call wire_param({root!r}) or "
                    f"wire_helper({root!r}) before ingest(), or compose({root!r}, ...) an "
                    f"already-frozen sub-structure"
                )

        return self._slots.copy(), dict(self._composed), contract


class HelperBuilder(_Builder):
    """
    Builds a device helper: PARAM/HELPER slots only, no data of its own. See
    the module docstring's local-contract rules and frozen.py for what
    ingest() returns.

    A helper takes data only as a trusted argument passed by whatever calls
    it - never a declared slot of its own - so wire_data always raises here.

    Author: B.G (08/2026)
    """

    def wire_data(self, name: str, *, dtype: Any = None) -> "HelperBuilder":
        """
        Always raises: a HelperBuilder is device-only and carries PARAM and
        HELPER slots only. Data reaches a helper as a trusted call argument
        supplied by whatever calls it, never as a slot declared on the
        helper itself. Declare the data slot on the enclosing KernelBuilder
        instead.

        Author: B.G (08/2026)
        """
        raise TypeError(
            "HelperBuilder.wire_data() is not allowed: a helper is device-only and takes data "
            "only as a trusted call argument of its caller. Declare wire_data on the enclosing "
            "KernelBuilder, and pass the value through as an ordinary template argument."
        )

    def ingest(self, template: Any) -> FrozenHelper:
        """
        Close out the build phase: derive and check `template`'s contract
        (see the module docstring), freeze this builder, and return the
        resulting FrozenHelper.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        slots, composed, contract = self._derive_and_check(template)
        self._frozen = True
        return FrozenHelper(template, slots, composed, contract)


class KernelBuilder(_Builder):
    """
    Builds a kernel: PARAM/HELPER/DATA slots all allowed. See the module
    docstring's local-contract rules and frozen.py for what ingest() returns.

    Author: B.G (08/2026)
    """

    def ingest(self, template: Any) -> FrozenKernel:
        """
        Close out the build phase: derive and check `template`'s contract
        (see the module docstring), freeze this builder, and return the
        resulting FrozenKernel.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        slots, composed, contract = self._derive_and_check(template)
        self._frozen = True
        return FrozenKernel(template, slots, composed, contract)
