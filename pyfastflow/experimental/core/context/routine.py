"""
A Routine is an ordered, device-only, linear sequence of kernels compiled and
launched as one unit, sharing a single bag of Parameters and Helpers across
every step.

What this is for
-----------------
A single kernel is one pass over the grid. Most models need several passes
run back to back, over the same underlying parameters, every substep - heat
diffusion needs a diffuse pass then a source pass, repeated. Wiring that by
hand means keeping several already-compiled Kernels around plus a python loop
that launches them in order and juggles which buffer currently holds what.
A Routine collects that sequence once, as a builder, and compiles it into one
callable that launches every step in order.

A Routine is built from KernelBuilders that are otherwise ordinary - built
exactly as they would be for a standalone compile, bind()ed against whatever
Parameters and Helpers the step needs. What is different is that a Routine's
steps do not each keep their own bindings forever: at compile() time, every
step is rebound (see compile.py, CompileBuilder.rebind) against the one bag the
whole Routine shares, so a Parameter reached under a given name means the same
object in every step that reaches it. check_handles (bag.py) is run across
every step's own bindings, as authored, before that rebind happens - so a
step built against one object under a name another step expects to mean
something else is caught at compile time, even though rebind would otherwise
silently make both agree with whatever the routine's bag holds.

Bulk data - the buffers each step reads and writes - is handled separately
from the bag. add_data(name, handle) gives a buffer a routine-local name;
add_kernel's data_handle_ref maps positional template arguments onto those
names. add_swap(a, b) relabels which buffer a name currently means, for every
step added after it - it is bookkeeping only, resolved once at compile time,
and costs nothing at launch. Because a Routine is called and re-called without
returning to Python between steps, there is no place to do the python-level
`T0, T1 = T1, T0` a hand-written ping-pong loop uses; add_swap is what stands
in for it, and the compiled Routine keeps re-running correctly precisely
because the net effect of every swap in it is required to be the identity -
see RoutineBuilder.compile.

Executing a routine
--------------------
    routine = (TaichiRoutineBuilder()
               .add_data("T0", T0.data)
               .add_data("T1", T1.data)
               .bind_bag(shared_bag)
               .add_kernel(diffuse_builder, data_handle_ref=("T1", "T0"))
               .add_kernel(source_builder, data_handle_ref=("T1",))
               .add_swap("T0", "T1")
               .add_kernel(diffuse_builder, data_handle_ref=("T1", "T0"))
               .add_kernel(source_builder, data_handle_ref=("T1",))
               .add_swap("T0", "T1")
               .compile())
    routine()                 # uses the add_data() defaults
    routine(T0.data, T1.data) # or override them positionally, per data_names

compile() rebinds and compiles each step, in the order added, into a Routine:
an inert sequence of already-compiled kernels plus, per step, which of the
routine's data names it launches with. Calling a Routine simply launches its
steps in that order. This base implementation keeps every step a fully
separate kernel launch; both backend subclasses build on it rather than
replacing it outright. The closure-backend subclass (Taichi/Quadrants)
overrides compile() to splice consecutive steps' loop bodies into one
generated kernel by default; split() marks where that splicing should stop
and a new generated kernel should start. CupyRoutineBuilder, having no
source fusion available, instead defaults to capturing the compiled steps'
launches into a CUDA graph and replaying that graph on call - see
cupy_backend.py, CupyRoutineBuilder.compile and _CapturedRoutine. See
RoutineBuilder.compile, ClosureRoutineBuilder.compile and
CupyRoutineBuilder.compile for what a repeated call composes to and why the
swaps must balance either way.

Contract: no set()/destroy() mid-routine
------------------------------------------
A compiled Routine holds the same kind of frozen snapshot a Kernel does (see
base.py, "Lifetime of a compiled object"): scalar/field Parameters are baked
in by their storage, const Parameters by their literal value. Calling set() or
destroy() on any Parameter the routine's bag reaches, between two calls to the
routine (or between two of its steps, if that were possible), is undefined:
a scalar/field write is visible immediately since the routine holds the same
storage, a const write is not since the literal was already baked into every
step at compile time, and destroy() invalidates the storage a step still
points at. None of this is enforced at runtime - recompile the routine after
such a change, the same way a Kernel is recompiled.

Contract: a captured graph bakes in pointers, not just storage
-----------------------------------------------------------------
A CupyRoutineBuilder-compiled Routine (captured=True, its default - see
cupy_backend.py) has everything above still true, plus one more thing a
CUDA graph adds on top of what a plain kernel launch already has: every
step's launch arguments, pointers included, are baked into the graph at
capture time, not re-read at replay time.

- A write to a scalar or field Parameter's storage - the ordinary set() -
  still lands where the graph's launches already point, so it is seen on the
  very next replay with no recompile needed; this is the intended way to
  feed a captured routine changing data, exactly as for an uncaptured one.
- set() on a const Parameter still only changes generated source the graph
  never re-reads, so it goes stale exactly as an uncaptured Routine's kernels
  would - recompile.
- destroy(), or anything else that returns a data handle's buffer to the
  pool, invalidates a pointer the graph's launches were captured with.
  Recompile - there is no cheaper fix, since the graph does not know which
  of its baked-in pointers came from which handle.
None of this is enforced at runtime; see cupy_backend.py, _CapturedRoutine
for the exact rule set and why.

Author: B.G (07/2026)
"""

import re
from abc import ABC, abstractmethod
from typing import Any

from .bag import Bag, check_handles
from .compile import CompileBuilder

_C_FUNC_NAME_RE = re.compile(r"(?:__global__|__device__)\s+[\w:\*&]*\s*(\w+)\s*\(")


def _template_label(template) -> str:
    """
    A short, human-readable name for a template, for error messages: a
    python def's own __name__, or the __global__/__device__ entry point read
    out of CUDA source text. Falls back to a truncated repr if neither
    applies.

    Author: B.G (07/2026)
    """
    name = getattr(template, "__name__", None)
    if name is not None:
        return name
    if isinstance(template, str):
        match = _C_FUNC_NAME_RE.search(template)
        if match:
            return match.group(1)
    return repr(template)[:60]


def _flatten_bindings(bindings: dict[str, Any]) -> dict[str, Any]:
    """
    A step's bound names, expanded with a dotted entry for every member of
    any bound Bag - the same shape Bag.walk() produces for a single bag,
    covering a step's whole binding set instead of one bag's contents.

    Author: B.G (07/2026)
    """
    flat: dict[str, Any] = {}
    for name, obj in bindings.items():
        flat[name] = obj
        if isinstance(obj, Bag):
            flat.update(dict(obj.walk(name)))
    return flat


class _Step:
    """
    One entry recorded by add_kernel: the kernel builder as given, and the
    data names it launches with, resolved through the swap table as it stood
    at the moment this step was added - see RoutineBuilder.add_kernel.

    Author: B.G (07/2026)
    """

    __slots__ = ("kernel_builder", "canonical_refs", "grid", "block")

    def __init__(self, kernel_builder, canonical_refs: tuple, grid, block):
        self.kernel_builder = kernel_builder
        self.canonical_refs = canonical_refs
        self.grid = grid
        self.block = block


class _CompiledStep:
    """
    One step of a compiled Routine: a callable that launches the already
    compiled kernel (backend launch convention baked in by
    RoutineBuilder._make_caller), plus the data names it is called with.

    Author: B.G (07/2026)
    """

    __slots__ = ("caller", "canonical_refs")

    def __init__(self, caller, canonical_refs: tuple):
        self.caller = caller
        self.canonical_refs = canonical_refs


class Routine:
    """
    A compiled, ordered sequence of kernels sharing one bag, ready to launch.

    Inert, like every other compiled object in this package (see base.py,
    Specializable): it retains the steps it was built from and nothing reads
    back from it to drive a later compile. Go through the RoutineBuilder that
    produced it to change anything and compile again.

    `data_names` lists the distinct names add_kernel's steps referenced, in
    the order they first appear across the steps as they were added. Calling
    the routine with no arguments launches every step using the handles given
    to add_data(); calling it with `len(data_names)` positional arguments
    overrides all of them for that call, matched up by position against
    `data_names`. A repeated call, with or without override arguments, is
    exactly what add_swap's net-identity requirement (see
    RoutineBuilder.compile) makes safe: nothing about a Routine's own state
    changes between calls, so calling it twice launches its steps against the
    same starting arrangement of buffers twice.

    Author: B.G (07/2026)
    """

    def __init__(self, steps: list[_CompiledStep], data_names: tuple, defaults: dict[str, Any]):
        self._steps = steps
        self._data_names = data_names
        self._defaults = defaults

    @property
    def data_names(self) -> tuple:
        """
        The distinct data names this routine's steps reference, in
        first-appearance order across the steps as they were added.

        Author: B.G (07/2026)
        """
        return self._data_names

    def __call__(self, *args) -> None:
        """
        Launch every step in order. With no arguments, each data name
        resolves to the handle given to add_data(); with
        `len(data_names)` positional arguments, those override the defaults
        for this call, matched by position against `data_names`. Any other
        argument count raises.

        Author: B.G (07/2026)
        """
        if args and len(args) != len(self._data_names):
            raise ValueError(
                f"Routine: expected 0 or {len(self._data_names)} argument(s) "
                f"matching data_names={self._data_names}, got {len(args)}"
            )
        table = dict(self._defaults)
        if args:
            table.update(zip(self._data_names, args))
        for step in self._steps:
            step.caller(*(table[name] for name in step.canonical_refs))


class RoutineBuilder(ABC):
    """
    Collects data names, a shared bag, and an ordered list of kernel steps,
    and compiles them into a Routine.

    add_data(name, handle) registers a routine-local name for a pooled data
    handle. add_kernel(kernel_builder, data_handle_ref=(...)) appends a step:
    `data_handle_ref` is a tuple of names, previously registered with
    add_data, mapped positionally onto the template's own declared data
    arguments. add_swap(a, b) is a build-time relabeling of which handle `a`
    and `b` currently mean - it emits no code, costs nothing at launch, and
    only affects steps added after it; a step added earlier keeps whatever
    the table resolved to at the time it was added. split() records a fusion
    boundary and otherwise does nothing; see its own docstring. bind_bag(bag)
    sets the one bag every step is rebound against at compile time.

    Author: B.G (07/2026)
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._perm: dict[str, str] = {}
        self._steps: list[_Step] = []
        self._splits: set[int] = set()
        self._bag: "Bag | None" = None

    def add_data(self, name: str, handle: Any) -> "RoutineBuilder":
        """
        Register `handle` under routine-local `name`, and the default it
        resolves to unless a call to the compiled Routine overrides it.

        Author: B.G (07/2026)
        """
        if name in self._data:
            raise KeyError(f"add_data: '{name}' is already registered")
        self._data[name] = handle
        self._perm[name] = name
        return self

    def add_kernel(
        self,
        kernel_builder: CompileBuilder,
        data_handle_ref: tuple = (),
        *,
        grid=None,
        block=None,
    ) -> "RoutineBuilder":
        """
        Append a step launching `kernel_builder`'s compiled kernel with the
        data handles named in `data_handle_ref`, mapped positionally onto
        the template's own declared data arguments.

        Each name in `data_handle_ref` must already be registered via
        add_data(); which underlying handle it currently means is resolved
        against the swap table as it stands right now; a later add_swap does
        not reach back and change this step. `grid`/`block` are accepted on
        every backend but only meaningful on cupy - see CupyRoutineBuilder.

        Author: B.G (07/2026)
        """
        data_handle_ref = tuple(data_handle_ref)
        arity = self._data_arity(kernel_builder)
        if arity != len(data_handle_ref):
            name = _template_label(kernel_builder.template)
            raise ValueError(
                f"add_kernel: template {name!r} declares {arity} data argument(s), "
                f"data_handle_ref gives {len(data_handle_ref)}"
            )
        unknown = [n for n in data_handle_ref if n not in self._perm]
        if unknown:
            raise KeyError(f"add_kernel: not registered via add_data: {sorted(unknown)}")
        canonical = tuple(self._perm[n] for n in data_handle_ref)
        self._steps.append(_Step(kernel_builder, canonical, grid, block))
        return self

    def add_swap(self, a: str, b: str) -> "RoutineBuilder":
        """
        Relabel the table so `a` and `b` swap which handle they currently
        mean. Emits no code and has no runtime cost - it only changes what a
        later add_kernel resolves its data_handle_ref against; steps added
        before this call are unaffected. See compile() for the requirement
        that every swap in a routine nets out to the identity.

        Author: B.G (07/2026)
        """
        if a not in self._perm or b not in self._perm:
            missing = sorted(n for n in (a, b) if n not in self._perm)
            raise KeyError(f"add_swap: not registered via add_data: {missing}")
        self._perm[a], self._perm[b] = self._perm[b], self._perm[a]
        return self

    def split(self) -> "RoutineBuilder":
        """
        Mark the boundary between this step and the next as a fusion
        boundary: a fused compile() launches everything before this call and
        everything after it as separate generated kernels, back to back,
        rather than splicing them into one. It has no effect on an unfused
        compile() - every step there is already its own kernel, in the order
        added, whether or not split() was called between them.

        Author: B.G (07/2026)
        """
        self._splits.add(len(self._steps))
        return self

    def bind_bag(self, bag: "Bag") -> "RoutineBuilder":
        """
        Set the one bag every step is rebound against at compile time.

        Author: B.G (07/2026)
        """
        self._bag = bag
        return self

    @abstractmethod
    def _data_arity(self, kernel_builder: CompileBuilder) -> int:
        """
        The number of data arguments `kernel_builder`'s ingested template
        declares, for add_kernel's arity check.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def _make_caller(self, compiled_kernel, grid, block):
        """
        A callable(*data_args) that launches `compiled_kernel` the way this
        backend requires - straight through for Taichi/Quadrants, with
        grid/block supplied for cupy.

        Author: B.G (07/2026)
        """
        ...

    def _validate(self) -> None:
        """
        Everything a compile needs checked before any step is actually
        compiled, fused or not.

        In order: check_handles (bag.py) runs across every step's own
        bindings, as authored - so two steps disagreeing about what one
        handle means is caught here, before rebind would otherwise silently
        make both agree with the routine's bag. Each step is then rebound
        against the bag set by bind_bag(); a step whose current bindings the
        bag cannot satisfy raises naming that step and everything missing.
        The swap table's net permutation is then checked: every add_swap in
        this routine must compose to the identity, since a Routine is called
        and re-called without ever returning to Python to swap buffers
        itself - an unbalanced set of swaps would compute into the wrong
        buffer on the second call.

        Author: B.G (07/2026)
        """
        if self._bag is None:
            raise ValueError("compile: no bag bound - call bind_bag() first")
        if not self._steps:
            raise ValueError("compile: routine has no steps")

        units = {}
        for i, step in enumerate(self._steps):
            label = f"step{i}:{_template_label(step.kernel_builder.template)}"
            units[label] = _flatten_bindings(step.kernel_builder.bindings)
        check_handles(units)

        for i, step in enumerate(self._steps):
            try:
                step.kernel_builder.rebind(self._bag)
            except KeyError as exc:
                label = _template_label(step.kernel_builder.template)
                raise KeyError(f"compile: step {i} ({label!r}) cannot be satisfied by the routine's bag: {exc}") from exc

        drift = {name: target for name, target in self._perm.items() if target != name}
        if drift:
            raise ValueError(f"compile: net swap permutation is not the identity, still swapped: {drift}")

    def _grouped_steps(self) -> list[list[_Step]]:
        """
        `self._steps` partitioned at every split() boundary, in order. With
        no split() calls this is a single group holding every step.

        Author: B.G (07/2026)
        """
        groups: list[list[_Step]] = []
        current: list[_Step] = []
        for i, step in enumerate(self._steps):
            if i in self._splits and current:
                groups.append(current)
                current = []
            current.append(step)
        if current:
            groups.append(current)
        return groups

    def compile(self, fused: bool = False, dump_source: str | None = None) -> Routine:
        """
        Validate (see _validate) and compile every step into a Routine.

        With fused=False (the default here - see the closure-backend
        subclass for a backend where fusion is actually available and
        defaults on), each step compiles to its own kernel and the Routine
        launches them in order, exactly one host-side call per step.
        `dump_source` is accepted for signature parity with a fusing
        subclass and ignored, since there is no generated source to dump.
        fused=True raises here: this base implementation backs cupy, which
        has no source fusion.

        Author: B.G (07/2026)
        """
        if fused:
            raise NotImplementedError(
                f"{type(self).__name__}: source fusion is not supported on this backend; "
                "call compile(fused=False) (the default) instead"
            )
        self._validate()

        compiled_steps: list[_CompiledStep] = []
        data_names: list[str] = []
        for step in self._steps:
            compiled = step.kernel_builder.compile()
            caller = self._make_caller(compiled, step.grid, step.block)
            compiled_steps.append(_CompiledStep(caller, step.canonical_refs))
            for name in step.canonical_refs:
                if name not in data_names:
                    data_names.append(name)

        defaults = {name: self._data[name] for name in data_names}
        return Routine(compiled_steps, tuple(data_names), defaults)
