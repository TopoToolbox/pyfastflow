"""
HostBlockBuilder / FrozenHostBlock / BoundHostBlock: a leaf builder, in the
same build -> freeze(ingest) -> bind -> compile family as KernelBuilder/
HelperBuilder (builder.py/frozen.py/bound.py), for host-side python code that
needs to read/write Parameters between device launches - the layer
Sequence's (sequence_v2.py) loop control (`loop`'s `max_times`/`until`) and
inter-block bookkeeping (a depression solver zeroing a counter Parameter
before each pass) run on.

A host block is a leaf, not a composite: it declares PARAM slots only.
wire_helper()/wire_data() both raise here - a device helper cannot run on the
host (there is nothing to trace it against), and data is never what a host
block reads; it reads Parameters. compose() raises too, for the same
"leaf" reason - there is no sub-structure to attach.

`ctx` resolves unwrapped
--------------------------
Where a kernel/helper's `ctx.z` resolves to a Parameter's `device_view()`
(compile_closure.py/compile_cupy.py), a host block's `ctx.z` resolves to the
bound Parameter itself - parameter.py's host-facing surface, `.get()`/
`.set(value)`/`.read()`, not `.get(node)`/`.set_node(node, value)`. A device
view genuinely cannot run outside kernel-trace context, so there is nothing
else `ctx.z` could mean here. `check_legal_host_accessors` enforces the
matching legal-chain set - `(name, "get"|"set"|"read")`, two segments, one of
those three - the same role compile_shared.py's `check_legal_accessors` plays
for device code, just against a different legal set.

One class for every backend
-----------------------------
"Compiling" a host block is resolving names, not emitting device code: build
the ctx tree of raw Parameters and return `lambda: template(ctx)`. There is
no Taichi/Quadrants/cupy variant of that, so `BoundHostBlock.compile()` takes
no backend argument (it accepts and ignores `backend`, only to keep a call
site that already carries a `backend` variable from having to special-case
this block kind).

Author: B.G (08/2026)
"""

import inspect
from typing import Any

from .bound import _Bound, _walk
from .builder import _Builder
from .compile_shared import CompileError, check_unmet
from .ctx import CTX_PARAM_NAME
from .frozen import _Frozen
from .slot import SlotKind

_LEGAL_HOST_ACCESSORS = ("get", "set", "read")


class HostBlockBuilder(_Builder):
    """
    Builds a host block: PARAM slots only, no HELPER, no DATA, no compose().
    See the module docstring.

    Author: B.G (08/2026)
    """

    def wire_helper(self, name: str) -> "HostBlockBuilder":
        """
        Always raises: a host block runs on the host, and a device helper
        cannot run there - there is nothing to trace it against. Declare the
        Parameters this block needs with wire_param() instead.

        Author: B.G (08/2026)
        """
        raise TypeError(
            "HostBlockBuilder.wire_helper() is not allowed: a host block is a PARAM-only leaf "
            "that runs on the host, and a device helper has no host-side form to call."
        )

    def wire_data(self, name: str, *, dtype: Any = None) -> "HostBlockBuilder":
        """
        Always raises: a host block is a PARAM-only leaf. A host block reads
        state through bound Parameters (ctx.z.get()/.read()), never through a
        trusted call argument the way a kernel's DATA slot does.

        Author: B.G (08/2026)
        """
        raise TypeError(
            "HostBlockBuilder.wire_data() is not allowed: a host block is a PARAM-only leaf - "
            "it reads state through bound Parameters (ctx.z.get()/.read()), never as a call "
            "argument."
        )

    def compose(self, name: str, frozen: _Frozen) -> "HostBlockBuilder":
        """
        Always raises: a host block is a leaf. There is no sub-structure to
        attach - compose a HelperBuilder into a KernelBuilder instead if
        device-side composition is what is actually wanted.

        Author: B.G (08/2026)
        """
        raise TypeError(
            "HostBlockBuilder.compose() is not allowed: a host block is a leaf (PARAM slots "
            "only) - there is nothing here for a sub-structure to attach to."
        )

    def ingest(self, template: Any) -> "FrozenHostBlock":
        """
        Close out the build phase exactly as KernelBuilder/HelperBuilder.ingest()
        does (builder.py): derive and check `template`'s contract, freeze this
        builder, return the resulting FrozenHostBlock. `template` must be a
        plain python `def tmpl(ctx): ...` - no data arguments, since
        wire_data() can never have wired one - checked properly at compile()
        time (see BoundHostBlock.compile).

        Author: B.G (08/2026)
        """
        self._check_mutable()
        slots, composed, contract = self._derive_and_check(template)
        self._frozen = True
        return FrozenHostBlock(template, slots, composed, contract)


class FrozenHostBlock(_Frozen):
    """
    The frozen result of a HostBlockBuilder's ingest(). See the module
    docstring. `composed` is always empty (compose() raises during build), so
    `.provides` reports only this block's own wired PARAM names.

    Author: B.G (08/2026)
    """

    def build(self) -> "BoundHostBlock":
        """
        Mint one bindable address per wired PARAM slot and return a
        BoundHostBlock. Overrides `_Frozen.build()` (frozen.py), whose own
        dispatch (bound.py's `build()`) only knows FrozenKernel/FrozenHelper
        and would hand back a BoundHelper here, the wrong type. Reuses
        bound.py's `_walk` directly instead - it only needs `.slots`/
        `.composed`, both of which a FrozenHostBlock has, `.composed` always
        empty.

        Author: B.G (08/2026)
        """
        table: dict = {}
        _walk((), self, table)
        return BoundHostBlock(self, table)


def check_legal_host_accessors(bound: "_Bound") -> None:
    """
    Raise on the first PARAM chain in `bound`'s frozen contract that is not
    exactly `(name, "get"|"set"|"read")` - the host-facing accessor set
    (parameter.py), as opposed to compile_shared.py's device-facing
    `(name, "get"|"set_node")`. A host block never composes anything, so -
    unlike compile_shared.check_legal_accessors - there is no composition
    tree to walk, just this block's own contract.

    Author: B.G (08/2026)
    """
    frozen = bound.frozen
    param_names = frozen.slots.names(SlotKind.PARAM)
    for chain in frozen.contract.chains:
        root = chain[0]
        if root not in param_names:
            continue
        if len(chain) != 2 or chain[1] not in _LEGAL_HOST_ACCESSORS:
            raise CompileError(
                f"{root!r}: illegal host PARAM accessor 'ctx.{'.'.join(chain)}' - legal "
                f"accessors on a host block are .get(), .set(...) and .read()"
            )


class _HostCtxNode:
    """
    What `ctx` resolves to inside a compiled host block's template body - a
    plain attribute bag holding this block's Parameters unwrapped. See the
    module docstring's "ctx resolves unwrapped" section.

    Author: B.G (08/2026)
    """


class BoundHostBlock(_Bound):
    """
    The bound result of build()-ing a FrozenHostBlock. See the module
    docstring.

    Author: B.G (08/2026)
    """

    def compile(self, backend: "str | None" = None, **kwargs) -> Any:
        """
        Resolve this block's ctx (each wired PARAM slot's bound Parameter,
        unwrapped) and return `lambda: template(ctx)` - see the module
        docstring's "One class for every backend" section for why `backend`
        is accepted and ignored rather than dispatched on. Checks unmet slots
        and legal host accessors first, and that `template`'s own signature
        declares exactly one parameter (`ctx` - wire_data() can never have
        wired a data argument here, so a template declaring one more is
        always a mistake, not a valid combination this layer forgot to
        support).

        Author: B.G (08/2026)
        """
        check_unmet(self)
        check_legal_host_accessors(self)
        frozen = self._frozen
        template = frozen.template
        params = list(inspect.signature(template).parameters)
        if params != [CTX_PARAM_NAME]:
            label = getattr(template, "__name__", "?")
            raise CompileError(
                f"host block template {label!r} must declare exactly one parameter "
                f"({CTX_PARAM_NAME!r}) - got {params}; a host block has no DATA slots, so "
                f"there is nothing else for a second parameter to mean"
            )
        ctx = _HostCtxNode()
        for name in frozen.slots.names(SlotKind.PARAM):
            setattr(ctx, name, self.value_at((name,)))
        return lambda: template(ctx)
