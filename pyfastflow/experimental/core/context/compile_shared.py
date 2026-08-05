"""
Backend-agnostic pieces of the compile phase (1c): the two checks every
backend's `BoundKernel.compile()` runs before emitting anything, and
CompiledKernel - the callable every backend's compile() returns.

Legal PARAM accessors
----------------------
1a/1b deliberately left "does this chain spell a real accessor" unenforced -
only the emission layer knows what a PARAM slot may legally do in device
code, per slot.py's own module docstring. Settled here, identically on every
backend (Taichi, Quadrants, cupy all resolve a PARAM chain the same way, so
there is one answer, not three): a PARAM-rooted chain must be exactly
`(name, "get")` or `(name, "set_node")` - two segments, nothing else. A bare
`ctx.z` with no accessor, a chain with a third segment, or any other method
name is illegal. `set_node` is further illegal against a slot currently bound
to a const-mode Parameter (const is baked into generated code as a literal;
there is nothing to write). check_legal_accessors walks the whole composition
tree - the kernel's own contract plus every composed FrozenHelper's own,
recursively - and raises naming the exact address and the exact chain, before
any backend touches source generation. This is a compile()-time convenience
on top of what the backends already refuse structurally on their own -
Taichi/Quadrants' ClosureParamDeviceView simply has no `set_node` attribute
for a const Parameter (AttributeError at trace time), cupy's span expander
only implements `get`/`set_node` - check_legal_accessors exists so the error
arrives before any tracing/emission starts, naming the address plainly
instead of surfacing as a trace-time AttributeError or a malformed-span
ValueError deep in generated text.

Unmet slots
-----------
check_unmet raises listing every address BoundKernel.unmet() reports,
formatted exactly as inspect() would show them - pasteable, not paraphrased.

Data argument signature
------------------------
check_data_signature/the cupy-specific text equivalent in compile_cupy.py
validate that a template's own declared data arguments (its python
parameters after `ctx`, or a cupy `__global__`'s own C parameter names)
match this kernel's wire_data() slots by name exactly - this is what lets
CompiledKernel resolve DATA addresses to launch-argument *positions* without
either side (template author, wire_data caller) tracking an implicit order
by hand.

CompiledKernel
--------------
What every backend's compile() returns: an immutable snapshot around a
resolved data-address order and a `launch` callable. Data is bound by
address, never passed positionally at call time - `swap(addr, buf)` re-points
one DATA address's current buffer with a plain dict write, no re-trace, no
recompile, exactly the ping-pong cost `z`/`z_prime` needs. `__call__` reads
whatever `swap()` currently holds for every address, in the fixed order
data_order was built with, and passes those positionally to `launch` - this
positional pass-through is what makes swap() free: the *compiled* kernel
itself is never touched, only a python dict entry.

Author: B.G (08/2026)
"""

import inspect
from typing import Any, Callable

from .bound import Address, BindError, _Bound, format_address, parse_address
from .ctx import CTX_PARAM_NAME
from .slot import SlotKind

_LEGAL_PARAM_ACCESSORS = ("get", "set_node")


class CompileError(Exception):
    """
    Raised by the compile phase (1c): unmet slots, an illegal PARAM accessor,
    a data-argument signature mismatch between a template and its wire_data()
    slots, or any other structural problem caught before/while emitting
    device code. Every case names the exact address (and, for accessors, the
    exact chain) involved.

    Author: B.G (08/2026)
    """


def check_unmet(bound: _Bound) -> None:
    """
    Raise, listing every still-unbound address exactly as inspect() would
    print it, if `bound` has any. Every concrete `compile()` calls this
    first.

    Author: B.G (08/2026)
    """
    missing = bound.unmet()
    if missing:
        listing = ", ".join(format_address(a) for a in missing)
        raise CompileError(f"compile: unbound slot(s): {listing}")


def check_legal_accessors(bound: _Bound) -> None:
    """
    Walk `bound`'s whole composition tree (the kernel's own frozen object,
    then every composed FrozenHelper, recursively) and raise on the first
    illegal PARAM accessor found - see the module docstring for the exact
    legal set and why it is identical on every backend.

    Author: B.G (08/2026)
    """
    _walk_accessors((), bound.frozen, bound)


def _walk_accessors(prefix: Address, frozen, bound: _Bound) -> None:
    param_names = frozen.slots.names(SlotKind.PARAM)
    for chain in frozen.contract.chains:
        root = chain[0]
        if root not in param_names:
            continue
        addr = prefix + (root,)
        if len(chain) != 2 or chain[1] not in _LEGAL_PARAM_ACCESSORS:
            raise CompileError(
                f"{format_address(addr)!r}: illegal PARAM accessor 'ctx.{'.'.join(chain)}' - "
                f"legal accessors are .get(...) and .set_node(...)"
            )
        if chain[1] == "set_node":
            value = bound.value_at(addr)
            if value is not None and getattr(value, "mode", None) == "const":
                raise CompileError(
                    f"{format_address(addr)!r}: 'ctx.{'.'.join(chain)}' - set_node against a "
                    f"const-mode PARAM slot is illegal (const is a baked-in literal, nothing "
                    f"to write)"
                )

    for name in frozen.slots.names(SlotKind.HELPER) | set(frozen.composed):
        _walk_accessors(prefix + (name,), frozen.composed[name], bound)


def check_data_signature(template, declared_names: set[str]) -> list[str]:
    """
    A python template's own data-argument names, in declaration order (its
    parameters after `ctx`), validated to be exactly `declared_names` as a
    set - not a subset, not a superset. Raises CompileError naming the
    mismatch otherwise. The order returned is what CompiledKernel resolves
    DATA addresses against for positional launch.

    Author: B.G (08/2026)
    """
    label = getattr(template, "__name__", "?")
    params = list(inspect.signature(template).parameters)
    if not params or params[0] != CTX_PARAM_NAME:
        raise CompileError(f"template {label!r}: first parameter must be {CTX_PARAM_NAME!r}")
    data_params = params[1:]
    if set(data_params) != declared_names:
        raise CompileError(
            f"template {label!r} declares data argument(s) {data_params}, wire_data() "
            f"declared {sorted(declared_names)} - these must match exactly"
        )
    return data_params


class CompiledKernel:
    """
    The immutable callable every backend's `BoundKernel.compile()` returns.
    See the module docstring.

    Author: B.G (08/2026)
    """

    def __init__(
        self,
        bound: _Bound,
        launch: Callable,
        data_order: list[Address],
        *,
        needs_launch_dims: bool = False,
        grid: Any = None,
        block: Any = None,
    ):
        self._bound = bound
        self._launch = launch
        self._data_order = list(data_order)
        self._data: dict[Address, Any] = {addr: bound.value_at(addr) for addr in self._data_order}
        self._needs_launch_dims = needs_launch_dims
        self._grid = grid
        self._block = block

    @property
    def data_order(self) -> list[Address]:
        """This kernel's DATA addresses, in the fixed positional order `launch` is called with."""
        return list(self._data_order)

    def data_at(self, addr: "Address | str") -> Any:
        """The buffer `swap()` currently has parked at `addr` (or the value it was compiled with)."""
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if a not in self._data:
            raise BindError(f"data_at: {format_address(a)!r} is not one of this compiled kernel's data addresses")
        return self._data[a]

    def swap(self, addr: "Address | str", buf: Any) -> "CompiledKernel":
        """
        Re-point DATA address `addr` at `buf` - a dict write, nothing else:
        no re-trace, no recompile, the compiled kernel itself is untouched.
        Validates `buf`'s dtype against the slot's declared dtype
        (wire_data(..., dtype=...)), if one was declared, the same check
        bind() runs.

        Author: B.G (08/2026)
        """
        a = parse_address(addr) if isinstance(addr, str) else tuple(addr)
        if a not in self._data:
            raise BindError(
                f"swap: {format_address(a)!r} is not one of this compiled kernel's data "
                f"addresses ({', '.join(format_address(x) for x in self._data_order)})"
            )
        info = self._bound.slot_info(a)
        if info.dtype is not None:
            obj_dtype = getattr(buf, "dtype", None)
            if obj_dtype is not None and obj_dtype != info.dtype:
                raise BindError(
                    f"swap({format_address(a)!r}, ...): dtype mismatch, slot declares "
                    f"{info.dtype}, got {obj_dtype}"
                )
        self._data[a] = buf
        return self

    def __call__(self, *, grid: Any = None, block: Any = None):
        """
        Launch with whatever `swap()` currently holds for every DATA
        address, in `data_order`. `grid`/`block` matter only on a backend
        that needs explicit launch dimensions (cupy); ignored otherwise.

        Author: B.G (08/2026)
        """
        args = [self._data[addr] for addr in self._data_order]
        if not self._needs_launch_dims:
            return self._launch(*args)
        g = grid if grid is not None else self._grid
        b = block if block is not None else self._block
        if g is None or b is None:
            raise CompileError(
                "this compiled kernel needs explicit launch dimensions - pass grid=/block= "
                "to compile() or to this call"
            )
        return self._launch(*args, grid=g, block=b)

    def __repr__(self) -> str:
        return f"CompiledKernel(data={[format_address(a) for a in self._data_order]})"
