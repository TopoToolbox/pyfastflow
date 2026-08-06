"""
SequenceBuilder / FrozenSequence / BoundSequence / CompiledSequence: the
host-driven layer above Routine (routine.py) - an ordered list of blocks
(kernel, whole routine, host block) plus one loop whose trip count and break
are evaluated on the host. Same build -> freeze -> bind -> compile lifecycle
as everything else in this package. Named `sequence` for the same reason
routine.py is - `sequence.py` already names the pre-1d implementation this
replaces, kept untouched until Phase 3.

What this is for
------------------
A Routine is device-only and linear - fixed steps, no python between them,
nothing about repeat count decided at run time. That is the wrong shape for
an outer pass whose trip count is not known until the device has been asked -
depression routing reads a pass count back from the device and either goes
round again or stops. A Sequence runs blocks in order, calls host code
between them, and loops with a host-evaluated predicate; `Parameter.read()`
(parameter.py) underpins every such predicate, and it synchronizes - the
layer's whole cost model, paid at block boundaries, never inside a block.

Composition vs. order
-----------------------
Unlike RoutineBuilder, composing a block here (`compose(name, frozen)`) does
not by itself place it in execution order - a name may be composed once and
then referenced from `step(name)` and/or from inside `loop(...)`'s body more
than once (the old sequence.py's `zero_ndep` callback, called once before a
loop and again every iteration, is exactly this shape). `step(name)` appends
`name` to the top-level order; `loop(body, max_times, until=None)` appends one
loop entry whose `body` is a sequence of already-composed names, run in order,
`max_times` times, stopping early when `until` returns True.

`max_times`/`until` are each either a plain value (an int for `max_times`,
`None` for `until`, meaning "run to completion") or the *name* of an
already-composed host block (host_block.py) - a FrozenHostBlock, never a bare
python callable, since a name is the only handle this layer's addressing
scheme has to bind that block's own Parameters through. That host block's
compiled, zero-argument callable is invoked once per check; its return value
is coerced with `int()` for `max_times`, `bool()` for `until`.

compose() accepts a FrozenKernel, a FrozenRoutine (routine.py) or a
FrozenHostBlock (host_block.py); a FrozenHelper raises, matching
RoutineBuilder.compose() - a device helper has no standalone host-callable
form on its own.

Addressing
-----------
`build()` walks every composed block's own tree under that block's compose()
name: bound.py's `_walk` directly for a FrozenKernel or FrozenHostBlock (both
real `_Frozen` objects), and, for a FrozenRoutine, one more level of
recursion through its own `.composed` steps - so a routine composed under
`saddlesort` and internally stepping `label`/`sort` reaches
`saddlesort.label.*`/`saddlesort.sort.*`, exactly the address a standalone
Routine's own `build()` would have minted, with the sequence-level compose()
name prefixed on top.

Compiling
----------
`BoundSequence.compile(backend, **kwargs)` checks this sequence's own unmet
slots first, then compiles each composed name at most once (cached by name,
since one composed block may be referenced from several places in order),
decomposing exactly as BoundRoutine.compile() does: a fresh bound object from
that block's own `.build()`, filled from this BoundSequence's current values
at that block's addresses, then that object's own `.compile()` - a
FrozenHostBlock's ignoring `backend` (host_block.py), everything else taking
it. The result, CompiledSequence, is an ordered list of zero-argument
callables (blocks already resolved to their own compiled form) plus loop
entries carrying their own body/max_times/until, evaluated on the host at
call time exactly as sequence.py's own `Sequence.__call__`/`_run_loop` did.

Per-block launch config
--------------------------
`compose(name, frozen, launch=None)` accepts the same optional launch-kwargs
override `RoutineBuilder.compose()` does (routine.py) - a dict merged over
this sequence's own `compile(backend, **kwargs)` call, `{**kwargs, **launch}`,
for that one composed block only. Composing a whole FrozenRoutine under `name`
with a `launch` override hands that merged dict to the routine's own
`compile()` as *its* default - which the routine's own per-step `launch`
overrides then apply on top of, exactly as they would against any other
default.

`CompiledSequence.swap(addr, buf)` routes `name.*` to the matching compiled
block's own `.swap()` (CompiledKernel.swap / CompiledRoutine.swap); raises if
that block has nothing to swap (a host block has no DATA of its own - see
host_block.py).

Author: B.G (08/2026)
"""

from typing import Any

from ..pool.base import new_uid
from .bound import Address, BindError, _Bound, _walk, _walk_group, format_address, parse_address
from .compile_shared import CompileError, check_unmet
from .frozen import FrozenBuilderError, FrozenHelper, FrozenKernel
from .host_block import BoundHostBlock, FrozenHostBlock
from .routine import BoundRoutine, FrozenRoutine


class SequenceBuilderError(Exception):
    """
    Raised by the SequenceBuilder build phase: a name reused or unknown, an
    attempt to compose an unsupported frozen type, a malformed loop, or a
    mutation after freeze().

    Author: B.G (08/2026)
    """


def _walk_leaf(prefix: Address, frozen: Any, table: dict) -> None:
    """
    `_walk` a single frozen object (FrozenKernel or FrozenHostBlock),
    honouring its own top-level `.shared` (`_Builder.share()`, builder.py)
    exactly as bound.py's own top-level `build()` does for a standalone
    object - dispatching to `_walk_group` rather than plain `_walk` when it
    has any - so a block composed with its own share() declarations
    collapses identically whether built standalone or as part of a
    sequence/routine.

    Author: B.G (08/2026)
    """
    if frozen.shared:
        _walk_group(prefix, frozen, table, {}, frozenset())
    else:
        _walk(prefix, frozen, table)


def _walk_block(prefix: Address, frozen: Any, table: dict) -> None:
    """
    Populate `table` with every PARAM/DATA leaf reachable from `frozen`, at
    its full path under `prefix` - dispatching on which of the three
    supported block kinds `frozen` is. See the module docstring's
    "Addressing" section.

    Author: B.G (08/2026)
    """
    if isinstance(frozen, FrozenRoutine):
        for name, step_frozen in frozen.composed.items():
            _walk_leaf(prefix + (name,), step_frozen, table)
    else:
        _walk_leaf(prefix, frozen, table)


class SequenceBuilder:
    """
    Collects a set of named blocks and an ordered list of steps/loops over
    them, and freeze()s them into a FrozenSequence. See the module docstring.

    Author: B.G (08/2026)
    """

    def __init__(self):
        self._uid = new_uid()
        self._composed: dict[str, Any] = {}
        self._launch: dict[str, dict] = {}
        self._order: list[tuple] = []
        self._frozen = False

    @property
    def uid(self) -> int:
        """Process-wide identity assigned at construction. See Parameter.uid (parameter.py)."""
        return self._uid

    def _check_mutable(self) -> None:
        if self._frozen:
            raise FrozenBuilderError(
                f"SequenceBuilder(uid={self._uid}) has already been freeze()-ed and is frozen - "
                f"build a new SequenceBuilder instead of reusing this one"
            )

    def _require_composed(self, name: str) -> Any:
        if name not in self._composed:
            raise SequenceBuilderError(f"{name!r} is not composed on this sequence - call compose({name!r}, ...) first")
        return self._composed[name]

    def compose(self, name: str, frozen: Any, *, launch: "dict | None" = None) -> "SequenceBuilder":
        """
        Register `frozen` (a FrozenKernel, FrozenRoutine or FrozenHostBlock)
        under `name`, without placing it in execution order - see step()/
        loop() for that, and the module docstring for why the two are
        separate calls here (unlike RoutineBuilder.compose()).

        `launch`, optional, is a dict of compile()-kwargs overriding this
        sequence's own compile()-level default for this block only - see the
        module docstring's "Per-block launch config" section. Ignored for a
        FrozenHostBlock (BoundHostBlock.compile() takes no backend-specific
        kwargs), accepted here regardless so a caller need not special-case
        which kind of block it is composing.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        if isinstance(frozen, FrozenHelper):
            raise TypeError(
                f"compose({name!r}, ...): got a FrozenHelper, not a FrozenKernel/FrozenRoutine/"
                f"FrozenHostBlock - a helper has no standalone host-callable form. Compose it "
                f"into a KernelBuilder first."
            )
        if not isinstance(frozen, (FrozenKernel, FrozenRoutine, FrozenHostBlock)):
            raise TypeError(
                f"compose({name!r}, ...): expected a FrozenKernel, FrozenRoutine or "
                f"FrozenHostBlock, got {type(frozen).__name__}"
            )
        if name in self._composed:
            raise SequenceBuilderError(f"'{name}' is already composed on this sequence")
        self._composed[name] = frozen
        self._launch[name] = dict(launch) if launch else {}
        return self

    def step(self, name: str) -> "SequenceBuilder":
        """
        Append a top-level step launching the block composed under `name`.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        self._require_composed(name)
        self._order.append(("step", name))
        return self

    def loop(self, body, max_times, until: "str | None" = None) -> "SequenceBuilder":
        """
        Append a loop running the composed blocks named in `body`, in order,
        `max_times` times, stopping early once `until` reports True. See the
        module docstring for the accepted shapes of `max_times`/`until`.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        body = tuple(body)
        if not body:
            raise SequenceBuilderError("loop: body is empty")
        for name in body:
            self._require_composed(name)
        if isinstance(max_times, str):
            frozen = self._require_composed(max_times)
            if not isinstance(frozen, FrozenHostBlock):
                raise TypeError(f"loop: max_times={max_times!r} must name a host block, got {type(frozen).__name__}")
        elif not isinstance(max_times, int):
            raise TypeError("loop: max_times must be an int or the name of a composed host block")
        if until is not None:
            if not isinstance(until, str):
                raise TypeError("loop: until must be None or the name of a composed host block")
            frozen = self._require_composed(until)
            if not isinstance(frozen, FrozenHostBlock):
                raise TypeError(f"loop: until={until!r} must name a host block, got {type(frozen).__name__}")
        self._order.append(("loop", body, max_times, until))
        return self

    def freeze(self) -> "FrozenSequence":
        """
        Close out the build phase: freeze this builder and return the
        resulting FrozenSequence. Raises if no step()/loop() was ever
        recorded.

        Author: B.G (08/2026)
        """
        self._check_mutable()
        if not self._order:
            raise SequenceBuilderError("freeze: sequence has no steps - call step()/loop() at least once")
        self._frozen = True
        return FrozenSequence(self._composed, self._order, self._launch)


class FrozenSequence:
    """
    The frozen result of a SequenceBuilder's freeze(): an immutable
    {name: block} composition, each block's own launch-kwargs override, plus
    the ordered step/loop list. See the module docstring.

    Author: B.G (08/2026)
    """

    def __init__(self, composed: dict, order: list, launch: "dict | None" = None):
        self._uid = new_uid()
        self._composed = dict(composed)
        self._launch = dict(launch) if launch else {}
        self._order = list(order)

    @property
    def uid(self) -> int:
        """Process-wide identity assigned at construction. See Parameter.uid (parameter.py)."""
        return self._uid

    @property
    def composed(self) -> dict:
        """{name: FrozenKernel|FrozenRoutine|FrozenHostBlock}, read-only copy."""
        return dict(self._composed)

    @property
    def launch(self) -> dict:
        """{name: launch-kwargs override dict}, read-only copy. See compose()'s `launch=`."""
        return dict(self._launch)

    @property
    def order(self) -> list:
        """The ordered step/loop list, read-only copy."""
        return list(self._order)

    def build(self) -> "BoundSequence":
        """
        Walk every composed block's own tree (_walk_block, prefixed with its
        compose() name) and return a fresh BoundSequence. See the module
        docstring's "Addressing" section.

        Author: B.G (08/2026)
        """
        table: dict[Address, Any] = {}
        for name, frozen in self._composed.items():
            _walk_block((name,), frozen, table)
        return BoundSequence(self, table)

    def __repr__(self) -> str:
        return f"FrozenSequence(uid={self._uid}, blocks={sorted(self._composed)})"


class BoundSequence(_Bound):
    """
    The bound result of build()-ing a FrozenSequence - bind()/wire()/
    inspect() work exactly as on a BoundKernel (_Bound, bound.py), over the
    sequence's whole `name.*` address space. See the module docstring's
    "Compiling" section for compile().

    Author: B.G (08/2026)
    """

    def compile(self, backend: str, **kwargs) -> "CompiledSequence":
        """
        See the module docstring's "Compiling" section.

        Author: B.G (08/2026)
        """
        check_unmet(self)
        frozen: FrozenSequence = self._frozen
        compiled_blocks: dict[str, Any] = {}

        def _compile_name(name: str) -> Any:
            if name in compiled_blocks:
                return compiled_blocks[name]
            child = frozen.composed[name]
            child_bound = child.build()
            for local_addr in child_bound.addresses():
                val = self.value_at((name,) + local_addr)
                if val is not None:
                    child_bound.bind(local_addr, val)
            block_kwargs = {**kwargs, **frozen.launch.get(name, {})}
            compiled = child_bound.compile() if isinstance(child, FrozenHostBlock) else child_bound.compile(backend, **block_kwargs)
            compiled_blocks[name] = compiled
            return compiled

        entries: list[_SeqEntry] = []
        for item in frozen.order:
            if item[0] == "step":
                _, name = item
                entries.append(_SeqEntry("run", run=_compile_name(name)))
            else:
                _, body, max_times, until = item
                body_compiled = tuple(_compile_name(n) for n in body)
                mt = max_times if isinstance(max_times, int) else _compile_name(max_times)
                un = None if until is None else _compile_name(until)
                entries.append(_SeqEntry("loop", body=body_compiled, max_times=mt, until=un))

        return CompiledSequence(entries, compiled_blocks)


class _SeqEntry:
    """
    One entry of a CompiledSequence: a "run" entry wraps a single already-
    resolved zero-argument callable; a "loop" entry carries its own compiled
    body, max_times and until (each itself a plain value or a zero-argument
    callable). Not constructed directly outside BoundSequence.compile().

    Author: B.G (08/2026)
    """

    __slots__ = ("kind", "run", "body", "max_times", "until")

    def __init__(self, kind: str, run: Any = None, body: tuple = (), max_times: Any = None, until: Any = None):
        self.kind = kind
        self.run = run
        self.body = body
        self.max_times = max_times
        self.until = until


class CompiledSequence:
    """
    An immutable, ordered list of resolved blocks and host-evaluated loops,
    ready to run. See the module docstring.

    `last_trip_counts` reports how many body iterations each loop entry took
    on the most recent call, in the order the loop entries appear.

    Author: B.G (08/2026)
    """

    def __init__(self, entries: list, compiled_blocks: dict):
        self._entries = entries
        self._compiled_blocks = compiled_blocks
        self._last_trip_counts: tuple = ()

    @property
    def last_trip_counts(self) -> tuple:
        """Body iterations taken by each loop entry on the most recent call, in entry order."""
        return self._last_trip_counts

    def swap(self, addr: "Address | str", buf: Any) -> "CompiledSequence":
        """
        Re-point one composed block's DATA address at `buf` - routes
        `name.*` to that block's own compiled `.swap()`. Raises if that
        block has nothing to swap (a compiled host block is a plain
        callable with no data addresses of its own).

        Author: B.G (08/2026)
        """
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if not a:
            raise BindError("swap: address must not be empty")
        name, local = a[0], a[1:]
        if name not in self._compiled_blocks:
            raise BindError(
                f"swap: {format_address(a)!r} - no such composed block {name!r} "
                f"(blocks: {sorted(self._compiled_blocks)})"
            )
        target = self._compiled_blocks[name]
        if not hasattr(target, "swap"):
            raise BindError(f"swap: {format_address(a)!r} - block {name!r} has no data to swap (it is a host block)")
        target.swap(local, buf)
        return self

    def __call__(self) -> None:
        """Run every entry in order - see the module docstring's cost-model paragraph."""
        trips: list[int] = []
        for entry in self._entries:
            if entry.kind == "loop":
                trips.append(self._run_loop(entry))
            else:
                entry.run()
        self._last_trip_counts = tuple(trips)

    def _run_loop(self, entry: _SeqEntry) -> int:
        """
        Evaluate `max_times` once on entry, run the body that many times,
        evaluating `until` after each iteration and stopping when it returns
        True. Returns the number of iterations actually run.

        Author: B.G (08/2026)
        """
        max_times = entry.max_times
        times = int(max_times()) if callable(max_times) else int(max_times)
        taken = 0
        for _ in range(max(0, times)):
            for inner in entry.body:
                inner()
            taken += 1
            if entry.until is not None and bool(entry.until()):
                break
        return taken

    def __repr__(self) -> str:
        return f"CompiledSequence(entries={len(self._entries)})"
