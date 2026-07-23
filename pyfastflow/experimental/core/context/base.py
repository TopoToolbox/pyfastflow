"""
Backend-agnostic building blocks for describing GPU work once and compiling it
against Taichi, Quadrants or cupy (or any future one).

Jargon
----------
Parameter       One named, typed value. Its `mode` says where the value lives:
                "const" (compile-time constant, non modifiable mid-run),
                "scalar" (single value modifiable) or "field" (a device array).
DeviceFunction  A compiled device-side helper: a small routine callable only
                from other device code (ti.func, qd.func, CUDA __device__).
Kernel          A compiled entry point - what the host launches (ti.kernel,
                qd.kernel, CUDA __global__).
Bag             A named collection of Parameters (ParamBag) or DeviceFunctions
                (HelperBag), so a group travels as one object and is read
                in-kernel by dotted path: phys.dx.get(i).

A "context" is any concrete class - GridContext, FlowContext, ... - that groups
Parameters and registers DeviceFunctions. There is deliberately no base Context
class: a context needing another context's parameters binds them explicitly,
rather than reaching through a registry of stored connections.

Compiling something
-------------------
Templates are written once, generically, and specialized by a builder:

    kernel = (TaichiKernelBuilder()
              .bind("phys", phys)        # a ParamBag
              .bind("ops", ops)          # a HelperBag
              .ingest(update_height)     # the template
              .compile())
    kernel(h_new, h_old)                 # bulk data passed at call time

bind(name, obj) makes `obj` visible inside the template body under `name`.
ingest() takes the template - a python def for Taichi/Quadrants, a CUDA source
string for cupy. compile() returns a Kernel or DeviceFunction. Only the
abstract DeviceFunctionBuilder / KernelBuilder live here; the concrete
Taichi*, Quadrants* and Cupy* builders sit alongside this module.

Data at call time, configuration at compile time
------------------------------------------------
Bound objects are injected into the template body and never appear in the call
signature. A compiled Kernel takes exactly the arguments its template declares,
and that is where bulk data travels - the buffers read and written each step.
Everything that *describes* the problem rather than *being* it - grid spacing,
timestep, gravity, which helper implements the neighbour lookup - is bound.

Reading a Parameter in device code is uniform across modes: p.get(node) to
read, p.set_node(node, value) to write. A const Parameter declared solo=True
is the exception: it resolves to a bare compile-time literal, read as `p` with
no call.

Device helpers bind const parameters only
-----------------------------------------
A DeviceFunction may only bind const-mode Parameters; any data it needs is
passed to it as an explicit argument by the calling kernel. A helper is
spliced into its caller and has no way to acquire a pointer argument of its
own, so this holds on all three backends alike. It is checked at compile time.
Kernels carry no such restriction and may bind any mode.

Lifetime of a compiled object
-----------------------------
compile() freezes what it was given: const Parameters are baked in as literals,
scalar and field Parameters as the storage behind their DataHandle. So:

  - Writing to a scalar or field Parameter *is* visible to already-compiled
    kernels, which hold that same storage. This is the normal way to feed
    changing data.
  - set() on a const Parameter is not. It drops the parameter's cached device
    view so the next compile() picks the new value up, but kernels compiled
    before it keep the old literal. Recompile them.
  - destroy() returns storage to the pool, which may hand the same buffer out
    again. Never destroy a Parameter that a live kernel still binds.

None of this is enforced at runtime.

Author: B.G (07/2026)
"""

import ast
import inspect
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, ClassVar

from ..pool.base import DataHandle


class Parameter(ABC):
    """
    One named, typed value owned by a context.

    `mode` decides where the value lives - "const" in the generated code,
    "scalar" in a single device cell, "field" in a device array - and every
    backend must offer all three (REQUIRED_MODES). A backend may widen
    SUPPORTED_MODES with further storage kinds; the check runs when the
    subclass is defined, so an incomplete backend fails at import.

    Two surfaces. From the host: get(), set(value), set_node(node, value).
    From device code: device_view(), which returns a backend object whose
    .get(node) / .set_node(node, val) let a kernel read and write the
    parameter identically whatever its mode.

    Author: B.G (07/2026)
    """

    REQUIRED_MODES: ClassVar[frozenset[str]] = frozenset({"const", "scalar", "field"})
    SUPPORTED_MODES: ClassVar[frozenset[str]] = REQUIRED_MODES

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = Parameter.REQUIRED_MODES - cls.SUPPORTED_MODES
        if missing:
            raise TypeError(f"{cls.__name__} must support modes {sorted(missing)}")

    name: str
    dtype: Any
    mode: str
    solo: bool = False

    @abstractmethod
    def get(self):
        """
        Host-side value: a python scalar for const mode, a DataHandle for scalar/field.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def set(self, value) -> None:
        """
        Update the whole parameter value in place, according to its mode. On
        const mode this does not reach already-compiled kernels - see the
        module docstring, "Lifetime of a compiled object".

        Author: B.G (07/2026)
        """
        ...

    def set_node(self, node, value) -> None:
        """
        Host-side single-cell write. scalar ignores node; const is read-only.
        Overridden by concrete backends; device-side writes go through
        device_view().set_node instead.

        Author: B.G (07/2026)
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement host set_node")

    def device_view(self):
        """
        An object whose .get(node) / .set_node(node, val) work inside device
        code. Taichi and Quadrants compile one out of ti/qd funcs. cupy leaves
        this unimplemented, having no use for it: its parser substitutes
        parameters into the source directly.

        Author: B.G (07/2026)
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement device_view")

    @abstractmethod
    def destroy(self) -> None:
        """
        Release any backing storage owned by this parameter. Unsafe while a
        compiled kernel still binds it - see the module docstring, "Lifetime
        of a compiled object".

        Author: B.G (07/2026)
        """
        ...


class Specializable(ABC):
    """
    Anything a builder can compile: DeviceFunction or Kernel.

    Beyond the call contract, each instance keeps the raw material it was made
    from - source text, AST, and a manifest of what it bound. Nothing in this
    module reads those back; they are here so a higher layer (kernel fusion, a
    graph compiler) can work from the originals instead of re-deriving them.

    Author: B.G (07/2026)
    """

    name: str
    _source: str | None = None
    _ast: ast.AST | None = None
    _dependencies: dict[str, str] | None = None

    @property
    @abstractmethod
    def compiled(self):
        """
        Raw backend callable/source (ti.func / qd.func object, CUDA text),
        for injection into another template's bindings.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs): ...


class DeviceFunction(Specializable):
    """
    Compiled device-side helper (e.g. a ti.func specialization).

    Not necessarily callable from host Python - backends where device
    functions can only run inside kernel/func scope raise on __call__.

    Author: B.G (07/2026)
    """


class Kernel(Specializable):
    """
    Compiled entry point (e.g. a ti.kernel specialization).

    Unlike DeviceFunction, __call__ works from host Python - that is how
    compute gets launched. Its arguments are the template's own declared data
    arguments; see the module docstring on data at call time.

    Author: B.G (07/2026)
    """


class _LazyBagView:
    """
    What a Bag looks like from inside a template body.

    `phys.g` resolves member `g` on first access and caches the result in the
    instance dict, so later lookups skip __getattr__ altogether. Resolving a
    member is not free - for a Parameter it compiles a device view - and a
    template usually touches only a few members of the bag it binds, so
    resolution is deferred to the members actually named. Members that are
    themselves Bags resolve to another _LazyBagView, keeping nested bags lazy
    all the way down.

    Author: B.G (07/2026)
    """

    def __init__(self, bag: "Bag"):
        object.__setattr__(self, "_bag", bag)

    def __getattr__(self, name: str) -> Any:
        # only called on a genuine miss (cached hits never reach here)
        bag = object.__getattribute__(self, "_bag")
        if name not in bag:
            raise AttributeError(name)
        resolved = resolve_binding(bag[name])
        self.__dict__[name] = resolved
        return resolved


def resolve_binding(value):
    """
    Turn a bound object into what a template body should see in its place.

    Parameter      a bare literal when solo, otherwise device_view() - the
                   carrier of .get / .set_node for in-kernel access.
    Specializable  its .compiled backend callable or source.
    Bag            a _LazyBagView, so a dotted path like grid.nx.get(i) traces
                   as plain attribute lookups.
    anything else  passed through untouched.

    Author: B.G (07/2026)
    """
    if isinstance(value, Parameter):
        if getattr(value, "solo", False):
            resolved = value.get()
            return resolved.data if isinstance(resolved, DataHandle) else resolved
        return value.device_view()
    if isinstance(value, Specializable):
        return value.compiled
    if isinstance(value, Bag):
        return _LazyBagView(value)
    return value


@lru_cache(maxsize=256)
def capture_template_meta(template) -> tuple[str | None, ast.AST | None]:
    """
    Return (source_text, ast) for a template. A python def is introspected; a
    raw string (CUDA source) is kept verbatim and has no AST.

    Cached because every compile() asks twice - once to filter bindings, once
    for attach_meta - and a miss costs an inspect.getsource plus a parse. The
    tree handed back is therefore shared by every Specializable built from that
    template: treat it as read-only.

    The cache key is the template object itself, so the bound size matters -
    unbounded, it would pin every dynamically generated template and every CUDA
    source string for the life of the process. An eviction only costs one
    re-parse.

    Author: B.G (07/2026)
    """
    if isinstance(template, str):
        return template, None
    try:
        source = inspect.getsource(template)
    except (OSError, TypeError):
        return None, None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    return source, tree


def _used_bindings(template, bindings: dict[str, Any]) -> dict[str, Any]:
    """
    The subset of `bindings` whose name appears in the template body.

    Collecting every ast.Name id in the tree is enough: the root of an
    attribute chain - `phys` in `phys.g.get(0)` - is itself an ast.Name.

    With no AST to consult (a CUDA source string, or a def whose source cannot
    be recovered) this returns `bindings` unchanged, rather than dropping a
    binding it cannot prove is unused.

    Author: B.G (07/2026)
    """
    _, tree = capture_template_meta(template)
    if tree is None:
        return bindings
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return {name: value for name, value in bindings.items() if name in used}


def filter_bindings(template, bindings: dict[str, Any]) -> dict[str, Any]:
    """
    The bindings a template actually references, ready to inject.

    Anything bound but never referenced is reported in a single warning per
    compile, which is what catches a misspelled bind() name in a context with
    many parameters. Nothing is reported when there is no AST to check
    against, since _used_bindings then treats every binding as used.

    Author: B.G (07/2026)
    """
    filtered = _used_bindings(template, bindings)
    unused = sorted(set(bindings) - set(filtered))
    if unused:
        warnings.warn(
            f"template '{getattr(template, '__name__', '?')}': bound but unused: {unused}",
            UserWarning,
            stacklevel=3,
        )
    return filtered


def attach_meta(obj: Specializable, template, bindings: dict[str, Any]) -> None:
    """
    Record on a freshly built Specializable what it was made from: its source,
    its AST, and the type name of each thing it bound. See Specializable.

    Author: B.G (07/2026)
    """
    obj._source, obj._ast = capture_template_meta(template)
    obj._dependencies = {name: type(value).__name__ for name, value in bindings.items()}


class CompileBuilder(ABC):
    """
    Collects dependencies and a template, and compiles them into one
    Specializable.

    A builder is used once, as a chain: any number of bind() calls, one
    ingest(), then compile(). Bound objects are not inspected as they arrive -
    what each one is (Parameter, DeviceFunction, Bag, handle, plain value) is
    worked out when the template is specialized, so bind() accepts anything.

    Everything here is backend-independent. A backend supplies compile(), and
    may override ingest() if its templates need different handling.

    Author: B.G (07/2026)
    """

    def __init__(self):
        self._bindings: dict[str, Any] = {}
        self._template = None

    def bind(self, name: str, obj: Any) -> "CompileBuilder":
        """
        Register `obj` under `name` for injection into the template body.
        Raises if `name` is already bound.

        Author: B.G (07/2026)
        """
        if name in self._bindings:
            raise KeyError(f"'{name}' is already bound")
        self._bindings[name] = obj
        return self

    def bind_bag(self, bag: "Bag") -> "CompileBuilder":
        """
        Bind every member of `bag` at top level under its own name (flat), for
        when a kernel refers to members directly rather than via a bag path.

        Author: B.G (07/2026)
        """
        for name, item in bag.items():
            self.bind(name, item)
        return self

    def ingest(self, template) -> "CompileBuilder":
        """
        Take the generic template (a python def for closure backends, a CUDA
        source string for cupy). Backends may override with their own handling.

        Author: B.G (07/2026)
        """
        self._template = template
        return self

    @abstractmethod
    def compile(self) -> Specializable:
        """
        Produce the compiled DeviceFunction/Kernel.

        Author: B.G (07/2026)
        """
        ...


class DeviceFunctionBuilder(CompileBuilder):
    """
    Builds a DeviceFunction. compile() -> DeviceFunction.

    Author: B.G (07/2026)
    """


class KernelBuilder(CompileBuilder):
    """
    Builds a Kernel. compile() -> host-callable Kernel.

    Author: B.G (07/2026)
    """


class Bag:
    """
    A named collection that can be handed to a builder in one go.

    Two ways to use one: bind it whole - bind("grid", bag) - and reach its
    members by dotted path in the template body, or bind_bag(bag) to merge
    every member in at top level under its own name.

    A subclass sets `_member_type` to restrict what it will hold. Bags are
    always accepted whatever that restriction says, since bags nest: a ParamBag
    may legitimately group its parameters into sub-bags.

    Author: B.G (07/2026)
    """

    _member_type: ClassVar[type | None] = None

    def __init__(self, items: dict[str, Any] | None = None):
        self._items: dict[str, Any] = {}
        for name, item in (items or {}).items():
            self.add(name, item)

    def add(self, name: str, item: Any) -> None:
        """
        Register `item` under `name`. Raises if `name` is already registered
        or if `item` doesn't match this bag's `_member_type` (a nested Bag is
        always accepted).

        Author: B.G (07/2026)
        """
        if name in self._items:
            raise KeyError(f"'{name}' is already registered in this bag")
        if self._member_type is not None and not isinstance(item, (self._member_type, Bag)):
            raise TypeError(
                f"{type(self).__name__}: member '{name}' must be a {self._member_type.__name__} "
                f"(or a Bag), got {type(item).__name__}"
            )
        self._items[name] = item

    def __getattr__(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name: str) -> Any:
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items)

    def items(self):
        return self._items.items()


class ParamBag(Bag):
    """
    Named collection of Parameter objects.

    Author: B.G (07/2026)
    """

    _member_type = Parameter


class HelperBag(Bag):
    """
    Named collection of DeviceFunction objects.

    Author: B.G (07/2026)
    """

    _member_type = DeviceFunction
