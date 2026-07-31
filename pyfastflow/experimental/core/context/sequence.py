"""
A Sequence is a host-driven ordering of blocks - kernels, whole Routines,
and plain python callbacks - sharing one bag, with a loop whose trip count
and stopping condition are decided on the host while it runs.

What this is for
-----------------
A Routine (routine.py) is device-only and linear: a fixed list of steps, no
python between them, nothing about how many times anything runs decided at
run time. That is what lets it fuse into one generated kernel on the closure
backends and replay as one captured graph on cupy, and it is the right shape
for the inner passes of an algorithm.

It is the wrong shape for the outer pass of one. Depression routing runs
label/saddlesort/reroute, reads a depression count back from the device,
and either goes round again or stops - a trip count that is not known until
the device has been asked, and a break that depends on a value only the host
can branch on. A Sequence is that layer and nothing more: it runs blocks in
order, calls host code between them, and loops with a host-evaluated
predicate.

    sb = TaichiSequenceBuilder()
    sb.add_data("rec", rec_h.data)
    sb.bind_bag(shared_bag)
    sb.add_routine(label_builder, data_handle_ref=("rec",))
    sb.add_loop(
        body=[
            routine_step(pass_builder, data_handle_ref=("rec",)),
            host_step(lambda bag: bag.stats.ndep.read()),
        ],
        max_times=lambda bag: ceil(log2(max(2, bag.stats.ndep.read()))) + 2,
        until=lambda bag: bag.stats.ndep.read() == 0,
    )
    seq = sb.compile()
    seq()

Blocks
-------
Four kinds, all recorded in the order added:

- a kernel (add_kernel / kernel_step): one KernelBuilder, compiled and
  launched with the data names given, exactly as a Routine step is;
- a Routine (add_routine / routine_step): a whole RoutineBuilder, compiled
  independently and called as one block. This is the one that matters. An
  inner Routine keeps whatever its own backend gives it - a fused generated
  kernel on Taichi/Quadrants, its own captured CUDA graph on cupy - so a
  loop over it replays that graph per iteration and pays the host sync only
  at block boundaries, not per kernel;
- host code (add_host / host_step): any callable, called with the bag. It
  may read Parameters with read() and write them with set();
- a loop (add_loop): a body of the three above, run under a host-evaluated
  trip count and predicate.

`max_times` is evaluated once, on entry to the loop: an int, or a callable
taking the bag and returning one. The body then runs that many times, and
`until` - a callable taking the bag, returning a bool, optional - is
evaluated after each iteration, stopping the loop when it returns True.
`max_times <= 0` runs the body zero times; that is a correct answer, not an
error, and it is what "the device reports nothing to do" looks like on
entry.

Deliberately absent: conditionals, nested loops, sub-Sequence reuse, and a
`check_every` stride on the predicate. Nested add_loop raises. None of them
have a caller yet, and each of them is a semantics decision better made
against real code than guessed at here.

Cost model
-----------
Every host callback and every predicate evaluation is a point where the host
must have the device's answer, so `Parameter.read()` synchronizes (see
parameter.py, Parameter.read). One `until` per iteration is one sync per
iteration. That is the price of the layer, and the reason the inner passes
belong in a Routine where no such point exists: the sync is paid at block
boundaries, never inside a block.

Contract
---------
One bag for the whole Sequence, given to bind_bag(); every block, including
every block in a loop body, is rebound against it at compile time.
check_handles (bag.py) runs first, across every block's bindings as
authored, so two blocks disagreeing about what a name means is caught before
rebinding would silently make them agree. Each inner RoutineBuilder is bound
to that same bag and then does its own validation when it compiles, net-swap
-identity included - which matters more here than in a standalone Routine,
since a Sequence may run it an unknown number of times.

What host code may and may not do, between blocks:

- set() on a scalar or field Parameter is legal, and is the point of this
  layer: the write lands in the storage every compiled block already reads,
  including a captured graph's, so the next block sees it.
- set() on a const Parameter raises, as everywhere else. A Sequence holds
  already-compiled Routines and kernels with that literal baked in, so there
  is no in-place remedy: rebuild the Sequence.
- destroy(), or anything else returning a handle's buffer to the pool,
  invalidates storage - and, on cupy, pointers already baked into a captured
  graph - that compiled blocks still point at. Forbidden mid-Sequence. This
  is documented and not detected, the same way routine.py documents it.
- a host callback or a predicate must not add or remove blocks. A compiled
  Sequence is inert, like every other compiled object here; its block list
  is fixed at compile().

There is no whole-Sequence graph capture and there will not be one: a loop
whose trip count is read back from the device is not expressible as a single
graph. Capture lives inside the blocks, where it belongs.

Author: B.G (07/2026)
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from .bag import Bag, check_handles
from .compile import CompileBuilder
from .routine import RoutineBuilder, _flatten_bindings, _template_label


class _Block:
    """
    One recorded block of a Sequence, as authored: its kind ("kernel",
    "routine", "host" or "loop") and the arguments that kind carries.

    Built by the module-level kernel_step/routine_step/host_step helpers and
    by SequenceBuilder's add_* methods, which are the same thing with an
    append. Inert - compile() reads it and builds a callable from it.

    Author: B.G (07/2026)
    """

    __slots__ = ("kind", "builder", "data_handle_ref", "grid", "block", "fn", "body", "max_times", "until")

    def __init__(
        self,
        kind: str,
        *,
        builder: Any = None,
        data_handle_ref: tuple = (),
        grid: Any = None,
        block: Any = None,
        fn: Any = None,
        body: tuple = (),
        max_times: Any = None,
        until: Any = None,
    ):
        self.kind = kind
        self.builder = builder
        self.data_handle_ref = data_handle_ref
        self.grid = grid
        self.block = block
        self.fn = fn
        self.body = body
        self.max_times = max_times
        self.until = until

    def label(self) -> str:
        """
        A short human-readable name for this block, for error messages.

        Author: B.G (07/2026)
        """
        if self.kind == "kernel":
            return f"kernel:{_template_label(self.builder.template)}"
        if self.kind == "routine":
            return f"routine:{type(self.builder).__name__}"
        if self.kind == "host":
            return f"host:{getattr(self.fn, '__name__', repr(self.fn)[:40])}"
        return "loop"


def kernel_step(kernel_builder: CompileBuilder, data_handle_ref: tuple = (), *, grid=None, block=None) -> _Block:
    """
    A kernel block, for use in an add_loop() body. The same block
    add_kernel() appends, as a value rather than an append.

    Author: B.G (07/2026)
    """
    return _Block("kernel", builder=kernel_builder, data_handle_ref=tuple(data_handle_ref), grid=grid, block=block)


def routine_step(routine_builder: RoutineBuilder, data_handle_ref: tuple = ()) -> _Block:
    """
    A Routine block, for use in an add_loop() body. The same block
    add_routine() appends, as a value rather than an append.

    Author: B.G (07/2026)
    """
    return _Block("routine", builder=routine_builder, data_handle_ref=tuple(data_handle_ref))


def host_step(fn: Callable[[Bag], Any]) -> _Block:
    """
    A host-code block, for use in an add_loop() body. The same block
    add_host() appends, as a value rather than an append.

    Author: B.G (07/2026)
    """
    return _Block("host", fn=fn)


class _CompiledBlock:
    """
    One block of a compiled Sequence, reduced to a callable of no arguments
    plus the loop control a "loop" block carries.

    A kernel or Routine block's callable already has its data handles bound
    in; a host block's is the callback with the bag applied; a loop block
    holds its body's compiled blocks and the (max_times, until) it was built
    with.

    Author: B.G (07/2026)
    """

    __slots__ = ("kind", "run", "body", "max_times", "until")

    def __init__(self, kind: str, run=None, body: tuple = (), max_times=None, until=None):
        self.kind = kind
        self.run = run
        self.body = body
        self.max_times = max_times
        self.until = until


class Sequence:
    """
    A compiled Sequence: an ordered list of blocks, ready to run.

    Inert, like every other compiled object in this package. Calling it runs
    every block in order; a loop block evaluates its trip count once on
    entry, runs its body that many times, and stops early when its `until`
    returns True. Nothing about the Sequence's own state changes across
    calls, so calling it twice runs the same blocks against whatever the bag
    and its storage hold at the time of each call.

    `last_trip_counts` reports how many body iterations each loop block
    actually took on the most recent call, in the order the loop blocks
    appear - a Sequence exists precisely because that number is not known
    until it runs, so it is worth reporting rather than reconstructing.

    Author: B.G (07/2026)
    """

    def __init__(self, blocks: list[_CompiledBlock], bag: Bag):
        self._blocks = blocks
        self._bag = bag
        self._last_trip_counts: tuple = ()

    @property
    def bag(self) -> Bag:
        """
        The one bag every block of this Sequence was rebound against.

        Author: B.G (07/2026)
        """
        return self._bag

    @property
    def last_trip_counts(self) -> tuple:
        """
        Body iterations taken by each loop block on the most recent call, in
        block order. Empty before the first call.

        Author: B.G (07/2026)
        """
        return self._last_trip_counts

    def __call__(self) -> None:
        """
        Run every block in order. Takes no arguments: a Sequence's data
        handles are fixed at compile time, unlike a Routine's, because a
        loop body and a host callback both hold references a call-time
        override could not reach.

        Author: B.G (07/2026)
        """
        trips: list[int] = []
        for block in self._blocks:
            if block.kind == "loop":
                trips.append(self._run_loop(block))
            else:
                block.run()
        self._last_trip_counts = tuple(trips)

    def _run_loop(self, block: _CompiledBlock) -> int:
        """
        Run one loop block: evaluate `max_times` once on entry, run the body
        that many times, evaluating `until` after each iteration and stopping
        when it returns True. Returns the number of iterations actually run.

        Author: B.G (07/2026)
        """
        max_times = block.max_times
        times = int(max_times(self._bag)) if callable(max_times) else int(max_times)
        taken = 0
        for _ in range(max(0, times)):
            for inner in block.body:
                inner.run()
            taken += 1
            if block.until is not None and block.until(self._bag):
                break
        return taken


class SequenceBuilder(ABC):
    """
    Collects data names, a shared bag, and an ordered list of blocks, and
    compiles them into a Sequence.

    add_data(name, handle) registers a sequence-local name for a data handle,
    exactly as RoutineBuilder.add_data does. add_kernel/add_routine/add_host
    append one block each; add_loop appends a loop over a body built from the
    module-level kernel_step/routine_step/host_step helpers. bind_bag(bag)
    sets the one bag every block - and every inner RoutineBuilder - is bound
    against at compile time.

    There is no add_swap here. A Routine needs one because it never returns
    to python and so cannot swap buffers itself; a Sequence returns to python
    between every block, where a host callback can do whatever relabeling is
    wanted directly.

    Author: B.G (07/2026)
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._blocks: list[_Block] = []
        self._bag: "Bag | None" = None

    def add_data(self, name: str, handle: Any) -> "SequenceBuilder":
        """
        Register `handle` under sequence-local `name`, for a kernel or
        Routine block's data_handle_ref to refer to.

        Author: B.G (07/2026)
        """
        if name in self._data:
            raise KeyError(f"add_data: '{name}' is already registered")
        self._data[name] = handle
        return self

    def bind_bag(self, bag: "Bag") -> "SequenceBuilder":
        """
        Set the one bag every block is bound against at compile time.

        Author: B.G (07/2026)
        """
        self._bag = bag
        return self

    def add_kernel(
        self,
        kernel_builder: CompileBuilder,
        data_handle_ref: tuple = (),
        *,
        grid=None,
        block=None,
    ) -> "SequenceBuilder":
        """
        Append a block launching `kernel_builder`'s compiled kernel with the
        data handles named in `data_handle_ref`, mapped positionally onto the
        template's own declared data arguments. `grid`/`block` are accepted
        on every backend but only meaningful on cupy.

        Author: B.G (07/2026)
        """
        self._blocks.append(kernel_step(kernel_builder, data_handle_ref, grid=grid, block=block))
        return self

    def add_routine(self, routine_builder: RoutineBuilder, data_handle_ref: tuple = ()) -> "SequenceBuilder":
        """
        Append a block running a whole Routine, compiled independently from
        `routine_builder` at this Sequence's compile() time.

        `data_handle_ref` names the data handles to call it with, mapped
        positionally onto the compiled Routine's own `data_names`; leave it
        empty to let the Routine use the defaults its own add_data() calls
        gave it, which is the usual case. A non-empty ref is an override at
        every call, and a captured cupy Routine rejects overrides outright
        (see cupy_backend.py, _CapturedRoutine) - compile that routine with
        captured=False if it needs them, or leave the ref empty.

        Author: B.G (07/2026)
        """
        self._blocks.append(routine_step(routine_builder, data_handle_ref))
        return self

    def add_host(self, fn: Callable[[Bag], Any]) -> "SequenceBuilder":
        """
        Append a block calling `fn(bag)` on the host, with the Sequence's
        bag. Its return value is discarded; it is there to read Parameters
        with read() and write them with set() between blocks.

        Author: B.G (07/2026)
        """
        self._blocks.append(host_step(fn))
        return self

    def add_loop(self, body, max_times, until: Callable[[Bag], bool] | None = None) -> "SequenceBuilder":
        """
        Append a loop over `body`, a non-empty sequence of blocks built with
        kernel_step/routine_step/host_step.

        `max_times` is evaluated once on entry to the loop: an int, or a
        callable taking the bag and returning one. The body runs that many
        times; `until`, if given, is a callable taking the bag and returning
        a bool, evaluated after each iteration, and stops the loop when it
        returns True. A `max_times` of zero or less runs the body zero times.

        A loop block inside `body` raises: nested loops are out of scope.

        Author: B.G (07/2026)
        """
        body = tuple(body)
        if not body:
            raise ValueError("add_loop: body is empty")
        for i, entry in enumerate(body):
            if not isinstance(entry, _Block):
                raise TypeError(
                    f"add_loop: body[{i}] is {type(entry).__name__}, expected a block from "
                    f"kernel_step()/routine_step()/host_step()"
                )
            if entry.kind == "loop":
                raise ValueError("add_loop: nested loops are not supported")
        if until is not None and not callable(until):
            raise TypeError("add_loop: until must be a callable taking the bag and returning a bool")
        if not callable(max_times) and not isinstance(max_times, int):
            raise TypeError("add_loop: max_times must be an int or a callable taking the bag")
        self._blocks.append(_Block("loop", body=body, max_times=max_times, until=until))
        return self

    @abstractmethod
    def _data_arity(self, kernel_builder: CompileBuilder) -> int:
        """
        The number of data arguments `kernel_builder`'s ingested template
        declares, for a kernel block's arity check.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def _make_caller(self, compiled_kernel, grid, block):
        """
        A callable(*data_args) that launches `compiled_kernel` the way this
        backend requires.

        Author: B.G (07/2026)
        """
        ...

    def _routine_compile_kwargs(self) -> dict:
        """
        Extra kwargs for a Routine block's own builder.compile() call, one
        per distinct RoutineBuilder identity (see _compile_block's
        routine_cache). Empty on Taichi/Quadrants - fused compile() has no
        such knob, and no need for one: fusion generates one kernel per
        Routine, no per-step real launch happens at compile time at all, so
        there is nothing here for a later block to see, correctly or not.

        CupySequenceBuilder overrides this to pass restore=False - see
        CupyRoutineBuilder.compile()'s `restore` parameter and
        _snapshot_data/_restore_data below for why.

        Author: B.G (07/2026)
        """
        return {}

    def _snapshot_data(self):
        """
        An opaque, backend-defined snapshot of every add_data() buffer this
        Sequence reaches, taken once before any block compiles - paired with
        _restore_data, taken once after every block (loop bodies included)
        has compiled. No-op (returns None) except on cupy.

        Why this exists: a captured cupy Routine's compile() warms up with a
        real launch, computing real values into its buffers - by design,
        see CupyRoutineBuilder.compile()'s docstring point 1. Two Routine
        blocks in the same Sequence with a real data dependency between them
        (block B reads what block A's real output is meant to be, e.g.
        depression routing's saddlesort reading label_basins' `bid`) need
        that real value from A still in the buffer when B's own warmup runs
        - which means restoring after every individual block's compile(),
        independently, is wrong: it erases A's real output before B's
        warmup ever sees it, leaving B's warmup to run on whatever the
        buffer held before A ran at all (uninitialised/pool-recycled
        garbage on a fresh build - proven to reach illegal array indices and
        crash, not just compute a wrong answer). One snapshot before
        anything compiles, one restore after everything has, lets each
        block's warmup see the previous block's genuine output while still
        leaving the Sequence's compile() side-effect-free overall, matching
        every other compile() in this package.

        Author: B.G (07/2026)
        """
        return None

    def _restore_data(self, snapshot) -> None:
        """
        Undo _snapshot_data's effect. No-op when `snapshot` is None (every
        backend but cupy, or a Sequence that registered no data at all).

        Author: B.G (07/2026)
        """

    def _flat_blocks(self) -> list[tuple[str, _Block]]:
        """
        Every block of this Sequence paired with a path naming where it sits,
        loop bodies flattened in place - so validation and compilation walk
        one list and a loop's body is checked exactly like a top-level block.

        Author: B.G (07/2026)
        """
        flat: list[tuple[str, _Block]] = []
        for i, block in enumerate(self._blocks):
            if block.kind == "loop":
                for j, inner in enumerate(block.body):
                    flat.append((f"block{i}.body{j}", inner))
            else:
                flat.append((f"block{i}", block))
        return flat

    def _validate(self) -> None:
        """
        Everything checked before any block is compiled.

        check_handles (bag.py) runs first, across every block's bindings as
        authored - a kernel block contributes its own, a Routine block one
        entry per step - so two blocks disagreeing about what one name means
        is caught before rebinding makes them agree. Kernel blocks are then
        rebound against the bag given to bind_bag(), and every inner
        RoutineBuilder is bound to that same bag, leaving each Routine's own
        validation (net swap identity included) to run when it compiles.
        Data names are checked against add_data(), and kernel arities against
        the template's declared data arguments.

        Author: B.G (07/2026)
        """
        if self._bag is None:
            raise ValueError("compile: no bag bound - call bind_bag() first")
        if not self._blocks:
            raise ValueError("compile: sequence has no blocks")

        flat = self._flat_blocks()

        units: dict[str, dict[str, Any]] = {}
        for path, block in flat:
            if block.kind == "kernel":
                units[f"{path}:{block.label()}"] = _flatten_bindings(block.builder.bindings)
            elif block.kind == "routine":
                for k, step in enumerate(block.builder._steps):
                    label = f"{path}:step{k}:{_template_label(step.kernel_builder.template)}"
                    units[label] = _flatten_bindings(step.kernel_builder.bindings)
        check_handles(units)

        for path, block in flat:
            if block.kind == "kernel":
                unknown = [n for n in block.data_handle_ref if n not in self._data]
                if unknown:
                    raise KeyError(f"compile: {path} refers to data not registered via add_data: {sorted(unknown)}")
                arity = self._data_arity(block.builder)
                if arity != len(block.data_handle_ref):
                    raise ValueError(
                        f"compile: {path} template {_template_label(block.builder.template)!r} declares "
                        f"{arity} data argument(s), data_handle_ref gives {len(block.data_handle_ref)}"
                    )
                try:
                    block.builder.rebind(self._bag)
                except KeyError as exc:
                    raise KeyError(f"compile: {path} cannot be satisfied by the sequence's bag: {exc}") from exc
            elif block.kind == "routine":
                unknown = [n for n in block.data_handle_ref if n not in self._data]
                if unknown:
                    raise KeyError(f"compile: {path} refers to data not registered via add_data: {sorted(unknown)}")
                block.builder.bind_bag(self._bag)
            elif block.kind == "host":
                if not callable(block.fn):
                    raise TypeError(f"compile: {path} host block is not callable")

    def _compile_block(self, path: str, block: _Block, routine_cache: dict) -> _CompiledBlock:
        """
        Compile one non-loop block into a `_CompiledBlock` whose `run` takes
        no arguments, its data handles already resolved.

        A RoutineBuilder appearing in more than one place compiles once,
        keyed on its identity, so a Routine used both before a loop and
        inside it is not compiled twice - and, on cupy, not captured twice.

        Author: B.G (07/2026)
        """
        if block.kind == "host":
            fn = block.fn
            bag = self._bag
            return _CompiledBlock("host", run=lambda: fn(bag))

        if block.kind == "kernel":
            compiled = block.builder.compile()
            caller = self._make_caller(compiled, block.grid, block.block)
            args = tuple(self._data[name] for name in block.data_handle_ref)
            return _CompiledBlock("kernel", run=lambda: caller(*args))

        key = id(block.builder)
        routine = routine_cache.get(key)
        if routine is None:
            routine = block.builder.compile(**self._routine_compile_kwargs())
            routine_cache[key] = routine
        if block.data_handle_ref:
            if len(block.data_handle_ref) != len(routine.data_names):
                raise ValueError(
                    f"compile: {path} gives {len(block.data_handle_ref)} data name(s), the compiled "
                    f"routine takes {len(routine.data_names)} matching data_names={routine.data_names}"
                )
            args = tuple(self._data[name] for name in block.data_handle_ref)
            return _CompiledBlock("routine", run=lambda: routine(*args))
        return _CompiledBlock("routine", run=routine)

    def compile(self) -> Sequence:
        """
        Validate (see _validate) and compile every block, loop bodies
        included, into a Sequence.

        Each kernel block compiles to one kernel and each Routine block to a
        whole Routine, compiled by its own builder with that backend's
        defaults - fused on Taichi/Quadrants, graph-captured on cupy. A loop
        body's blocks are compiled once, here, and re-run per iteration; the
        loop's `max_times` and `until` are kept as given and evaluated on the
        host while the Sequence runs.

        _snapshot_data/_restore_data (cupy only - see their docstrings)
        bracket the whole compile loop, not each block: every Routine
        block's own compile() warms up for real and (via
        _routine_compile_kwargs' restore=False) leaves that real output in
        place for the next block, so data genuinely flows block to block
        exactly as it would at runtime, and only the Sequence's own
        snapshot, taken before any of this starts, is restored - once - at
        the end.

        Author: B.G (07/2026)
        """
        self._validate()

        snapshot = self._snapshot_data()
        routine_cache: dict[int, Any] = {}
        compiled: list[_CompiledBlock] = []
        for i, block in enumerate(self._blocks):
            if block.kind == "loop":
                body = tuple(
                    self._compile_block(f"block{i}.body{j}", inner, routine_cache)
                    for j, inner in enumerate(block.body)
                )
                compiled.append(_CompiledBlock("loop", body=body, max_times=block.max_times, until=block.until))
            else:
                compiled.append(self._compile_block(f"block{i}", block, routine_cache))
        self._restore_data(snapshot)
        return Sequence(compiled, self._bag)
