"""
RoutineBuilder / FrozenRoutine / BoundRoutine / CompiledRoutine: the same
build -> freeze -> bind -> compile lifecycle a kernel goes through (builder.py
/frozen.py/bound.py), one level up - an ordered, device-only sequence of
already-built kernels, sharing one address space, launched back to back as
one unit.

Named `routine_v2` (not `routine`) only because `routine.py` already names
the pre-1d implementation this replaces, which stays in the tree untouched
until Phase 3 deletes it - see the Phase 1d report for this naming fork.

Composition, not registration
-------------------------------
`compose(name, frozen_kernel)` is this module's only way to add a step: it
both names the step (an explicit handle, never a positional `step0` - see
builder.py's compose() for the same rule one level down) and fixes its
position in launch order, which is insertion order. There is no separate
"declare a step" call followed by a separate "now order it" call - the two
are the same call, so a routine's execution order is always exactly its own
composition order, readable straight off `FrozenRoutine.order`.

Composing the *same* FrozenKernel object under two different step names
(`rb.compose("diffuse1", diffuse).compose("diffuse2", diffuse)`) is how a
kernel runs twice in one routine with independently-bound data at each
occurrence - the routine-level extension of the instancing property
frozen.py's module docstring describes for a helper composed into eighty
kernels: one FrozenKernel, two addresses (`diffuse1.*`, `diffuse2.*`), two
independently bindable slot sets, and - after compile() - two independent
CompiledKernel launches. This is also what replaces the old routine.py's
add_swap: instead of relabelling one shared data name between two launches of
one compiled kernel, two separate step names each hold their own DATA
address, bound to whichever buffer that occurrence needs.

compose() rejects a FrozenHelper (a helper has no standalone launch - compose
it into a KernelBuilder first, then compose that kernel's FrozenKernel here)
and anything that is not a FrozenKernel.

Addressing
-----------
`build()` walks every composed step's own composition tree via bound.py's
`_walk`, prefixed with that step's own name - `flux.grad.z` names the `z`
PARAM slot of the `grad` helper composed inside the kernel composed under
`flux`, exactly the nesting a kernel's own address tree already has, with one
more level of handle name in front of it. No separate bookkeeping: the
address table is derived fresh, every `build()` call, straight from what each
composed FrozenKernel's own `.build()` would mint on its own - see bound.py's
module docstring for why that is exactly the instancing guarantee this relies
on.

Compiling
----------
`BoundRoutine.compile(backend, **kwargs)` checks this routine's own unmet
slots first (routine-level addresses, not a per-step address a caller would
have to map back), then, per step in order: builds a *fresh* BoundKernel from
that step's own FrozenKernel, copies in whatever this BoundRoutine currently
holds at each of that step's local addresses (routine address `name.*` ->
step-local address `*`), and calls that BoundKernel's own `.compile(backend,
**step_kwargs)` - reusing compile_closure.py/compile_cupy.py entirely
unchanged. The result is one CompiledRoutine wrapping the steps' own
CompiledKernels, in order.

Per-step launch config
------------------------
`compose(name, frozen_kernel, launch=None)` accepts an optional dict of
compile()-kwargs (cupy's `grid=`/`block=`, in practice) that apply to this
step only - the obvious case being `ops`' scan kernels, which need a
different launch shape than whatever else shares a routine. `step_kwargs`
above is `{**kwargs, **launch}` - the routine-level `compile(backend,
**kwargs)` call is the default every step falls back to, `launch` overrides
it key by key for that one step. A step composed with no `launch` uses the
routine-level default outright.

`CompiledRoutine.swap(addr, buf)` routes `name.*` to the matching step's own
CompiledKernel.swap() - a dict write on that step alone, exactly as free as
CompiledKernel.swap() itself (compile_shared.py). Calling a CompiledRoutine
launches every step in order, each with whatever its own swap() state
currently holds.

Author: B.G (08/2026)
"""

from typing import Any

from ..pool.base import new_uid
from .bound import Address, BindError, _Bound, _walk, _walk_group, format_address, parse_address
from .compile_shared import CompileError, check_unmet
from .frozen import FrozenBuilderError, FrozenHelper, FrozenKernel


class RoutineBuilderError(Exception):
    """
    Raised by the RoutineBuilder build phase: a step name reused, an
    attempt to compose a non-FrozenKernel, or a mutation after freeze().

    Author: B.G (08/2026)
    """


class RoutineBuilder:
    """
    Collects an ordered set of named kernel steps and freeze()s them into a
    FrozenRoutine. See the module docstring.

    Author: B.G (08/2026)
    """

    def __init__(self):
        self._uid = new_uid()
        self._order: list[str] = []
        self._composed: dict[str, FrozenKernel] = {}
        self._launch: dict[str, dict] = {}
        self._frozen = False

    @property
    def uid(self) -> int:
        """Process-wide identity assigned at construction. See Parameter.uid (parameter.py)."""
        return self._uid

    def _check_mutable(self) -> None:
        if self._frozen:
            raise FrozenBuilderError(
                f"RoutineBuilder(uid={self._uid}) has already been freeze()-ed and is frozen - "
                f"build a new RoutineBuilder instead of reusing this one"
            )

    def compose(self, name: str, frozen_kernel: FrozenKernel, *, launch: "dict | None" = None) -> "RoutineBuilder":
        """
        Append a step named `name`, launching `frozen_kernel` at this
        position in the routine's launch order (= composition order). See
        the module docstring for the FrozenHelper rejection and the
        same-kernel-twice-under-two-names instancing pattern.

        `launch`, optional, is a dict of compile()-kwargs (cupy's `grid=`/
        `block=`) that override the routine-level default for this step
        only - see the module docstring's "Per-step launch config" section.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        if isinstance(frozen_kernel, FrozenHelper):
            raise TypeError(
                f"compose({name!r}, ...): got a FrozenHelper, not a FrozenKernel - a helper has "
                f"no standalone launch and cannot be a routine step. Compose it into a "
                f"KernelBuilder first, then compose that kernel's FrozenKernel here."
            )
        if not isinstance(frozen_kernel, FrozenKernel):
            raise TypeError(f"compose({name!r}, ...): expected a FrozenKernel, got {type(frozen_kernel).__name__}")
        if name in self._composed:
            raise RoutineBuilderError(f"'{name}' is already composed on this routine")
        self._composed[name] = frozen_kernel
        self._launch[name] = dict(launch) if launch else {}
        self._order.append(name)
        return self

    def freeze(self) -> "FrozenRoutine":
        """
        Close out the build phase: freeze this builder and return the
        resulting FrozenRoutine. Raises if no step was ever composed - an
        empty routine has nothing to launch.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        if not self._order:
            raise RoutineBuilderError("freeze: routine has no steps - compose() at least one kernel first")
        self._frozen = True
        return FrozenRoutine(self._order, self._composed, self._launch)


class FrozenRoutine:
    """
    The frozen result of a RoutineBuilder's freeze(): an ordered, immutable
    {name: FrozenKernel}, plus each step's own launch-kwargs override. See
    the module docstring.

    Author: B.G (08/2026)
    """

    def __init__(self, order: list, composed: dict, launch: "dict | None" = None):
        self._uid = new_uid()
        self._order = tuple(order)
        self._composed = dict(composed)
        self._launch = dict(launch) if launch else {}

    @property
    def uid(self) -> int:
        """Process-wide identity assigned at construction. See Parameter.uid (parameter.py)."""
        return self._uid

    @property
    def order(self) -> tuple:
        """Step names in launch order (= composition order)."""
        return self._order

    @property
    def composed(self) -> dict:
        """{step name: FrozenKernel}, read-only copy."""
        return dict(self._composed)

    @property
    def launch(self) -> dict:
        """{step name: launch-kwargs override dict}, read-only copy. See compose()'s `launch=`."""
        return dict(self._launch)

    def build(self) -> "BoundRoutine":
        """
        Walk every step's own composition tree (bound.py's `_walk`, prefixed
        with that step's own name) and return a fresh BoundRoutine. See the
        module docstring's "Addressing" section.

        A step's own `.shared` (`_Builder.share()`, builder.py) is honoured
        exactly as bound.py's own top-level `build()` honours it for a
        standalone FrozenKernel - dispatching to `_walk_group` rather than
        plain `_walk` - so a step composed with its own share() declarations
        collapses identically whether it is built standalone or as one step
        of a routine.

        Author: B.G (08/2026)
        """
        table: dict[Address, Any] = {}
        for name in self._order:
            step = self._composed[name]
            if step.shared:
                _walk_group((name,), step, table, {}, frozenset())
            else:
                _walk((name,), step, table)
        return BoundRoutine(self, table)

    def __repr__(self) -> str:
        return f"FrozenRoutine(uid={self._uid}, steps={list(self._order)})"


class BoundRoutine(_Bound):
    """
    The bound result of build()-ing a FrozenRoutine - bind()/wire()/
    inspect() work exactly as on a BoundKernel (_Bound, bound.py), over the
    routine's whole `name.*` address space. See the module docstring's
    "Compiling" section for compile().

    Author: B.G (08/2026)
    """

    def compile(self, backend: str, **kwargs) -> "CompiledRoutine":
        """
        See the module docstring's "Compiling" section.

        Author: B.G (08/2026)
        """
        check_unmet(self)
        frozen: FrozenRoutine = self._frozen
        steps: list[tuple[str, Any]] = []
        for name in frozen.order:
            step_frozen = frozen.composed[name]
            step_bound = step_frozen.build()
            for local_addr in step_bound.addresses():
                val = self.value_at((name,) + local_addr)
                if val is not None:
                    step_bound.bind(local_addr, val)
            step_kwargs = {**kwargs, **frozen.launch.get(name, {})}
            compiled = step_bound.compile(backend, **step_kwargs)
            steps.append((name, compiled))
        return CompiledRoutine(steps)


class CompiledRoutine:
    """
    An immutable, ordered sequence of already-compiled kernels, ready to
    launch as one unit. See the module docstring.

    Author: B.G (08/2026)
    """

    def __init__(self, steps: list):
        self._steps = list(steps)
        self._by_name = dict(steps)

    @property
    def step_names(self) -> list:
        """Step names in launch order."""
        return [name for name, _ in self._steps]

    def swap(self, addr: "Address | str", buf: Any) -> "CompiledRoutine":
        """
        Re-point one step's DATA address at `buf` - routes `name.*` to that
        step's own CompiledKernel.swap(), a dict write on that step alone.
        See the module docstring.

        Author: B.G (08/2026)
        """
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if not a:
            raise BindError("swap: address must not be empty")
        name, local = a[0], a[1:]
        if name not in self._by_name:
            raise BindError(
                f"swap: {format_address(a)!r} - no such routine step {name!r} "
                f"(steps: {sorted(self._by_name)})"
            )
        self._by_name[name].swap(local, buf)
        return self

    def __call__(self) -> None:
        """Launch every step in order, each with whatever its own swap() state currently holds."""
        for _, compiled in self._steps:
            compiled()

    def __repr__(self) -> str:
        return f"CompiledRoutine(steps={[n for n, _ in self._steps]})"
