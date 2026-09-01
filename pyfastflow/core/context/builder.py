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

compose(name, frozen, split=[...]) is the other half of build-phase sharing
(GroupBuilder.share(), below): when `frozen` is a FrozenGroup carrying
`.shared` paths, every one of them is collapsed into its own canonical
address by default (bound.py's build() - see that module's docstring for
the full mechanism), and `split` opts specific dotted relative paths back
out into their own, independently-bindable addresses again at THIS compose
site. Each path in `split` must already be one of `frozen.shared`'s own
declared relative paths, checked here, eagerly - naming the exact path if
not. `split` on anything that is not a FrozenGroup with `.shared` entries
raises: there is nothing to split.

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

`RESERVED_BK_NAME` ("bk", bk.py) may never be wired (wire_param/wire_helper/
wire_data, via `_wire`) or composed (`compose`) - `ctx.bk` is reserved,
backend-recognised grammar (the backend-intrinsics namespace: `ctx.bk.sqrt`,
`ctx.bk.atan2`, ...), not a name any factory's own template surface may
repurpose. See bk.py's module docstring for the full mechanism and
contract.py for the matching rule on the derivation side (a `ctx.bk.*` chain
is dropped before it ever becomes a contract requirement, so ingest() never
asks for a "bk" slot to be wired in the first place).

ingest() returns a frozen, immutable FrozenKernel/FrozenHelper and freezes
the builder itself in the same call - every wire_*/compose/ingest afterwards
raises FrozenBuilderError. A builder is therefore used once, start to finish;
build a new one for a different template rather than trying to reuse an
ingested one: a builder holding live, still-mutable slot state after having
already handed out one frozen, immutable result would be exactly the kind
of aliasing hazard the frozen/mutable split exists to rule out.

Author: B.G (08/2026)
"""

from typing import Any

from ..pool.base import new_uid
from .bk import RESERVED_BK_NAME
from .contract import Contract, ContractError, extract_cupy_contract, extract_python_contract
from .frozen import FrozenBuilderError, FrozenGroup, FrozenHelper, FrozenKernel, _Frozen
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
        self._split: dict[str, frozenset] = {}
        self._shared: dict[str, list[tuple]] = {}
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
                f"{type(self).__name__}(uid={self._uid}) has already closed its build phase "
                f"(ingest()/freeze()) and is frozen - build a new {type(self).__name__} "
                f"instead of reusing this one"
            )

    def _wire(self, slot: Slot) -> "_Builder":
        self._check_mutable()
        if slot.name == RESERVED_BK_NAME:
            raise SlotGroupError(
                f"'{RESERVED_BK_NAME}' is reserved - ctx.{RESERVED_BK_NAME} is the "
                f"backend-intrinsics namespace (bk.py) and can never be wired as a slot"
            )
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
        plug in a Parameter (parameter.py) later, not a shape; nothing
        here constrains mode or dtype, which is the entire point of a
        Parameter being able to move between const/scalar/field without
        touching a template.

        Parameters
        ----------
        name : str
            Slot name.

        Returns
        -------
        _Builder
            self, for chaining.

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

        Parameters
        ----------
        name : str
            Slot name.

        Returns
        -------
        _Builder
            self, for chaining.

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

        Parameters
        ----------
        name : str
        dtype : optional
            Declares this slot's data-argument contract, checked later at
            bind/compile time against whatever value is actually bound or
            passed. Left as None, the slot stays open to any dtype.

        Returns
        -------
        _Builder
            self, for chaining.

        Author: B.G (08/2026)
        """
        return self._wire(DataSlot(name, dtype=dtype))

    def compose(self, name: str, frozen: _Frozen, *, split: "list[str] | None" = None) -> "_Builder":
        """
        Attach an already-frozen sub-structure (a FrozenKernel, FrozenHelper
        or FrozenGroup - frozen.py) under slot `name`, giving a template
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
        (a kind compose() has no business filling), raises. `frozen` must be
        a FrozenHelper or a FrozenGroup - a FrozenKernel raises: a kernel is
        a host entry point, not something device code can call, and on a GPU
        backend a kernel cannot call another kernel.

        `split`, optional, is a list of dotted relative paths (e.g.
        `"neighbour_raw.row.NX"`) that opt back out of `frozen`'s own
        build-phase sharing (GroupBuilder.share(), FrozenGroup.shared) at
        THIS compose site, re-minting each as its own independently-bindable
        address instead of collapsing into its shared canonical - see the
        module docstring and bound.py's module docstring for the full
        mechanism.

        Parameters
        ----------
        name : str
            Slot name to compose `frozen` under.
        frozen : FrozenHelper or FrozenGroup
            Already-frozen sub-structure to attach.
        split : list[str], optional
            Dotted relative paths to opt back out of `frozen`'s build-phase
            sharing at this compose site.

        Returns
        -------
        _Builder
            self, for chaining.

        Raises
        ------
        TypeError
            If `frozen` is not a `_Frozen`, or is a FrozenKernel.
        SlotGroupError
            If `name` is reserved, already composed, already wired as a
            non-HELPER slot, or `split` names a path not in
            `frozen.shared`, or `split` is given for a `frozen` with no
            shared paths at all.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        if name == RESERVED_BK_NAME:
            raise SlotGroupError(
                f"'{RESERVED_BK_NAME}' is reserved - ctx.{RESERVED_BK_NAME} is the "
                f"backend-intrinsics namespace (bk.py) and can never be composed as a root"
            )
        if not isinstance(frozen, _Frozen):
            raise TypeError(f"compose({name!r}, ...): expected a FrozenKernel/FrozenHelper, got {type(frozen).__name__}")
        if isinstance(frozen, FrozenKernel):
            raise TypeError(
                f"compose({name!r}, ...): got a FrozenKernel, not a FrozenHelper - a kernel is a "
                f"host entry point, not a device-callable helper, and cannot be composed into "
                f"another builder (on a GPU backend a kernel cannot call another kernel). Build "
                f"the shared logic as a HelperBuilder instead."
            )
        if name in self._composed:
            raise SlotGroupError(f"'{name}' is already composed on this builder")
        if name in self._slots and self._slots[name].kind is not SlotKind.HELPER:
            raise SlotGroupError(
                f"'{name}' is already wired on this builder as {self._slots[name]!r}; compose() "
                f"only fills a HELPER slot (or a fresh name), never a PARAM/DATA one"
            )
        self._composed[name] = frozen
        if split:
            shared = getattr(frozen, "shared", None)
            if not shared:
                raise SlotGroupError(
                    f"compose({name!r}, ..., split={split!r}): {name!r}'s frozen object has no "
                    f"build-phase-shared PARAM paths to split - split only applies to a "
                    f"FrozenGroup composed with at least one share() declaration"
                )
            all_shared = {p for paths in shared.values() for p in paths}
            resolved = set()
            for path in split:
                segs = tuple(path.split("."))
                if segs not in all_shared:
                    raise SlotGroupError(
                        f"compose({name!r}, ..., split=...): {path!r} is not a shared path on "
                        f"the composed group (shared paths: "
                        f"{sorted('.'.join(p) for p in all_shared)})"
                    )
                resolved.add(segs)
            self._split[name] = frozenset(resolved)
        return self

    def share(self, canonical: str, *paths: str) -> "_Builder":
        """
        Declare that `canonical` - a PARAM slot already wire_param()'d on
        THIS builder - is the same value as each dotted `paths`, a relative
        address reaching a PARAM slot somewhere in this builder's own
        already-composed subtree (e.g. `"neighbour_raw.row.NX"`: the `row`
        helper composed inside the `neighbour_raw` helper composed on this
        builder, its own `NX` slot). bound.py's build() acts on this: by
        default, every declared path collapses into `canonical`'s own
        address - only `canonical` is independently minted, not every
        private occurrence - which is the whole point (see bound.py's
        module docstring for why this needed a build-phase mechanism rather
        than being left to bind-phase wire() or bulk/pattern binding).

        This is explicit and local to one builder's own authoring - never
        name-based matching across independently-authored composites (which
        is exactly the kind of accidental collision this architecture's
        addressing exists to prevent). A caller composing this builder's
        frozen result elsewhere opts specific paths back OUT of the collapse
        via compose()'s own `split=` (`_Builder.compose()`).

        Available on KernelBuilder and HelperBuilder as well as GroupBuilder:
        a kernel or helper that both reads a composed sub-structure's PARAM
        slot directly (via its own wire_param()) and also composes something
        that re-composes the same sub-structure may collapse those
        occurrences itself, exactly as a GroupBuilder does for its own
        composed children - no group wrapper needed purely to reach share().
        For a KernelBuilder, `canonical` is only actually usable as a
        collapse target - and this builder's own `.shared` only actually
        takes effect - when this object is later reached as build()'s own
        top-level frozen argument (a FrozenKernel is never itself composed
        as someone else's child, per compose()'s own FrozenKernel guard
        above); for a HelperBuilder composed as a child elsewhere, its
        `.shared` is consulted exactly as a FrozenGroup's is (see bound.py's
        module docstring).

        Parameters
        ----------
        canonical : str
            PARAM slot, already wire_param()'d on this builder, that the
            given `paths` collapse into.
        *paths : str
            Dotted relative addresses into this builder's own composed
            subtree, each naming a PARAM slot to share with `canonical`.

        Returns
        -------
        _Builder
            self, for chaining.

        Raises
        ------
        SlotGroupError
            If `canonical` is not a PARAM slot wired on this builder, if a
            path does not resolve (through this builder's already-composed
            children) to a real PARAM slot, or if a path is already declared
            shared under a different (or the same) canonical - each relative
            path may be shared at most once.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        if canonical not in self._slots or self._slots[canonical].kind is not SlotKind.PARAM:
            raise SlotGroupError(
                f"share({canonical!r}, ...): {canonical!r} is not a PARAM slot wired on this "
                f"builder - call wire_param({canonical!r}) before share()"
            )
        if not paths:
            raise SlotGroupError(f"share({canonical!r}): at least one path is required")

        already_shared = {p for ps in self._shared.values() for p in ps}
        resolved: list[tuple] = []
        for path in paths:
            segs = tuple(path.split("."))
            if len(segs) < 2:
                raise SlotGroupError(
                    f"share({canonical!r}, {path!r}): a shared path must reach into a composed "
                    f"child (at least 'child.PARAM'), got {path!r}"
                )
            root = segs[0]
            if root not in self._composed:
                raise SlotGroupError(f"share({canonical!r}, {path!r}): {root!r} is not composed on this builder")
            node: _Frozen = self._composed[root]
            walked = root
            for seg in segs[1:-1]:
                if seg not in node.composed:
                    raise SlotGroupError(f"share({canonical!r}, {path!r}): {seg!r} is not composed under {walked!r}")
                node = node.composed[seg]
                walked = f"{walked}.{seg}"
            leaf = segs[-1]
            if leaf not in node.slots.names(SlotKind.PARAM):
                raise SlotGroupError(f"share({canonical!r}, {path!r}): {leaf!r} is not a PARAM slot under {walked!r}")
            if segs in already_shared:
                raise SlotGroupError(f"share({canonical!r}, {path!r}): {path!r} is already shared")
            resolved.append(segs)
            already_shared.add(segs)

        self._shared.setdefault(canonical, [])
        self._shared[canonical].extend(resolved)
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

        Parameters
        ----------
        template : Any
            A python def (closure backends) or CUDA source text (cupy).

        Returns
        -------
        FrozenHelper

        Raises
        ------
        ContractError
            If a chain `template` requires has no matching slot/composed
            root.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        slots, composed, contract = self._derive_and_check(template)
        self._frozen = True
        return FrozenHelper(template, slots, composed, contract, split=self._split, shared=self._shared)


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

        Parameters
        ----------
        template : Any
            A python def (closure backends) or CUDA source text (cupy).

        Returns
        -------
        FrozenKernel

        Raises
        ------
        ContractError
            If a chain `template` requires has no matching slot/composed
            root.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        slots, composed, contract = self._derive_and_check(template)
        self._frozen = True
        return FrozenKernel(template, slots, composed, contract, split=self._split, shared=self._shared)


class GroupBuilder(_Builder):
    """
    Builds a non-callable, navigable composite: PARAM/HELPER slots and
    composed sub-structures only, no template of its own and never callable
    in device code - see frozen.py's FrozenGroup for what this closes into
    and why it exists (a caller needing both `ctx.grid.neighbour(i, k)`, a
    composed HELPER call, and `ctx.grid.NX.get(0)`, a PARAM leaf reached
    straight through the same composite, one level in).

    `wire_data` always raises, for the same reason it does on HelperBuilder:
    a group is device-structure-only, never a call argument's own signature.

    `share()` (inherited from `_Builder` - see its own docstring for the full
    mechanism) is build-phase sharing: a group PARAM slot the group's own
    author declares once, that stands in for the same value re-read by
    several of the group's own composed children - frozen.py's FrozenGroup
    is what it freezes into.

    Author: B.G (08/2026)
    """

    def wire_data(self, name: str, *, dtype: Any = None) -> "GroupBuilder":
        """
        Always raises: a GroupBuilder declares PARAM/HELPER slots only. See
        HelperBuilder.wire_data() (same reasoning) and frozen.py's
        FrozenGroup.

        Author: B.G (08/2026)
        """
        raise TypeError(
            "GroupBuilder.wire_data() is not allowed: a group is a passive, device-structure-"
            "only composite - it is never the template a call argument belongs to. Declare "
            "wire_data on whichever KernelBuilder eventually composes this group."
        )

    def freeze(self) -> FrozenGroup:
        """
        Close out the build phase and return the resulting FrozenGroup.
        Unlike KernelBuilder.ingest()/HelperBuilder.ingest(), there is no
        template to derive a Contract from - a group is never itself the
        target of a ctx.* chain resolution of its own body (see frozen.py),
        so its Contract is always empty. Every wired HELPER slot must still
        end up composed by build() time (frozen.py/bound.py), exactly as for
        a HelperBuilder/KernelBuilder - unreferenced here since there is no
        template to check it against at this phase, but still enforced one
        phase later.

        Returns
        -------
        FrozenGroup

        Author: B.G (08/2026)
        """
        self._check_mutable()
        self._frozen = True
        return FrozenGroup(
            None, self._slots.copy(), dict(self._composed), Contract(frozenset()),
            split=self._split, shared=self._shared,
        )


def find_param_paths(frozen: "_Frozen", leaf_name: str, prefix: tuple = ()) -> list:
    """
    Every relative dotted path, as a `"a.b.NAME"` string, under `frozen`'s own
    composed subtree whose PARAM slot is literally named `leaf_name` - the
    itemized list `share_leaf` hands to GroupBuilder.share(). Recurses through
    `.composed` only (a HELPER slot with nothing composed raises earlier, at
    that structure's own ingest()/build(), never reached here). Generic over
    whether a composed node is itself a FrozenHelper or a nested FrozenGroup.

    Shared by grid/noise/visu's own factories - see grid/__init__.py's module
    docstring ("Build-phase sharing collapses the duplicate addresses") for
    why this exists.

    Parameters
    ----------
    frozen : _Frozen
        Sub-structure to search.
    leaf_name : str
        PARAM slot name to find.
    prefix : tuple, optional
        Path segments prepended to every result; used internally for
        recursion.

    Returns
    -------
    list[str]
        Dotted relative paths to every occurrence of `leaf_name`.

    Author: B.G (08/2026)
    """
    paths = []
    if leaf_name in frozen.slots.names(SlotKind.PARAM):
        paths.append(".".join(prefix + (leaf_name,)))
    for name, child in frozen.composed.items():
        paths.extend(find_param_paths(child, leaf_name, prefix + (name,)))
    return paths


def share_leaf(group: "GroupBuilder", canonical: str) -> None:
    """
    Declare every occurrence of a PARAM slot named `canonical` anywhere in
    `group`'s already-composed subtree as build-phase-shared with `group`'s
    own top-level `canonical` slot. A no-op if `canonical` occurs nowhere in
    the subtree (e.g. OUTLET_MASK when no block happens to reference it under
    the current config) - share() itself requires at least one path, so this
    only calls it when there is something to share.

    Parameters
    ----------
    group : GroupBuilder
        Builder whose own `canonical` PARAM slot every found occurrence
        collapses into.
    canonical : str
        PARAM slot name to search for and share.

    Author: B.G (08/2026)
    """
    paths = []
    for name, child in group.composed.items():
        paths.extend(find_param_paths(child, canonical, (name,)))
    if paths:
        group.share(canonical, *paths)
