"""
An ordered, device-only sequence of already-built kernels that share one
address space and launch back to back as a single unit.

`RoutineBuilder` / `FrozenRoutine` / `BoundRoutine` / `CompiledRoutine` follow
the same build -> freeze -> bind -> compile lifecycle as a single kernel
(see builder.py, frozen.py, bound.py), one level up.

Composing steps
----------------
`compose(name, frozen_kernel)` appends a step: `name` is its address prefix
and its position in launch order is its position in composition order - there
is no separate ordering call. `FrozenRoutine.order` reads back exactly the
sequence `compose()` was called in.

Composing the same `FrozenKernel` under two step names runs it twice with
independently bound data at each occurrence:

    rb.compose("diffuse1", diffuse).compose("diffuse2", diffuse)

gives two addresses (`diffuse1.*`, `diffuse2.*`), two independently bindable
slot sets, and two independent `CompiledKernel` launches after `compile()`.

`compose()` rejects a `FrozenHelper` - a helper has no standalone launch.
Compose it into a `KernelBuilder` first, then compose that kernel here.

Addressing
-----------
`build()` walks each step's own composition tree, prefixed with that step's
name: `flux.grad.z` names the `z` PARAM slot of the `grad` helper composed
inside the kernel composed under `flux`.

Compiling
----------
`BoundRoutine.compile(backend, **kwargs)` checks this routine's own unmet
slots, then per step: builds a fresh `BoundKernel` from that step's
`FrozenKernel`, copies over whatever is bound at that step's addresses
(`name.*` -> the step's own local addresses), and compiles it. The result is
a `CompiledRoutine` wrapping each step's `CompiledKernel`, in order.

Per-step launch config
------------------------
`compose(name, frozen_kernel, launch=None)` takes an optional dict of
compile()-kwargs (cupy's `grid=`/`block=`) applied to that step only,
overriding the routine-level `compile(backend, **kwargs)` defaults key by
key - e.g. `ops`' scan kernels, which need a different launch shape than
the rest of a routine they share with.

`CompiledRoutine.swap(addr, buf)` routes `name.*` to that step's own
`CompiledKernel.swap()`. Calling a `CompiledRoutine` launches every step in
order, each with whatever its own `swap()` state currently holds.

Author: B.G (08/2026)
"""

from typing import Any

from ..pool.base import new_uid
from .bound import Address, BindError, _Bound, _walk, _walk_group, format_address, parse_address
from .compile_shared import CompileError, check_unmet
from .frozen import FrozenBuilderError, FrozenHelper, FrozenKernel
from .slot import BuildError


class RoutineBuilderError(BuildError):
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
        position in the routine's launch order.

        Parameters
        ----------
        name : str
            Address prefix for this step. Must be unique within the routine.
        frozen_kernel : FrozenKernel
        launch : dict, optional
            compile()-kwargs (cupy's `grid=`/`block=`) applied to this step
            only, overriding the routine-level default. See the module
            docstring's "Per-step launch config" section.

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
        Close out the build phase and return the resulting FrozenRoutine.

        Raises
        ------
        RoutineBuilderError
            No step was ever composed - an empty routine has nothing to
            launch.

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
        Return a fresh BoundRoutine with every step's addresses walked and
        prefixed by that step's name. See the module docstring's
        "Addressing" section.

        A step's own `share()` declarations (builder.py) are honoured the
        same way whether the step is built standalone or as part of a
        routine.

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
        Compile every step and return the resulting CompiledRoutine. See
        the module docstring's "Compiling" section.

        Parameters
        ----------
        backend : str
            "taichi", "quadrants" or "cupy".
        **kwargs
            Routine-level compile()-kwargs, the default every step falls
            back to unless overridden by its own `launch=`.

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
        Re-point one step's DATA address at `buf`.

        Parameters
        ----------
        addr : Address or str
            `name.*`, routed to step `name`'s own CompiledKernel.swap().
        buf : Any
            Replacement buffer.

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
