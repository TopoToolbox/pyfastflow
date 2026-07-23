"""
Backend-agnostic context building blocks.

No "Context" class here on purpose: a context is just whatever concrete class
(GridContext, FlowContext, ...) groups a set of Parameters and registers
DeviceFunctions. Cross-context references are plain explicit bindings passed
at bind() time, not a stored connection registry.

Load-bearing rule: data flows at call time (a template's own explicit
arguments); configuration flows at compile time (bind()ed Parameters/helpers/
bags, resolved into the template body).

A device helper (DeviceFunction) may only bind const-mode Parameters. Any
data it needs (scalar/field buffers) is passed to it as an explicit argument
by the calling kernel, not bound in - a device helper has no way to receive
an extra pointer argument once spliced into a caller, so this holds across
all three backends alike. Kernels are unrestricted: they may bind params of
any mode.

The compile surface is a two-layer builder: an abstract DeviceFunctionBuilder /
KernelBuilder here, backend variants (Taichi*/Quadrants*/Cupy*) elsewhere. A
builder collects bind()ed dependencies + one ingest()ed template and produces a
compiled DeviceFunction/Kernel. Bound Parameters/helpers/bags are injected into
the template body (never passed at call time); only a template's own explicit
data-field arguments are passed to the front callable.

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

    REQUIRED_MODES is the baseline every backend must support; a backend
    widens SUPPORTED_MODES to add more storage kinds. Enforced at subclass
    definition time via __init_subclass__, not at instantiation.

    Host surface: get() / set(value) / set_node(node, value). Device surface:
    device_view(), returning a backend object whose .get(node) / .set_node(
    node, val) are usable from inside device code - the uniform accessor that
    lets one kernel read/write a Parameter identically no matter its mode.

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
        Update the whole parameter value in place, according to its mode.

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
        Return a backend device-view object (get/set_node usable in device
        code). Implemented per backend; closure backends compile ti/qd funcs,
        cupy returns parser metadata.

        Author: B.G (07/2026)
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement device_view")

    @abstractmethod
    def destroy(self) -> None:
        """
        Release any backing storage owned by this parameter.

        Author: B.G (07/2026)
        """
        ...


class Specializable(ABC):
    """
    Shared call contract for DeviceFunction and Kernel, plus the hidden raw
    material (source text, AST, dependency manifest) a later fusion/graph
    compiler can reuse. Built by a *Builder, not by a classmethod.

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

    Unlike DeviceFunction, __call__ works from host Python - that's how the
    compute gets launched. Its template's own parameters are data fields only;
    params/helpers arrive via bindings and are resolved into the kernel body.

    Author: B.G (07/2026)
    """


class _LazyBagView:
    """
    Lazy stand-in for a Bag in a template body: `phys.g` resolves `g` only on
    first attribute access (via resolve_binding), then caches it in the
    instance dict so later lookups bypass __getattr__ entirely. Avoids
    filter_bindings' top-level-only reach: binding one large bag no longer
    eagerly resolves (and device_view()-compiles) every member, only the ones
    a template actually touches. Nested Bag members resolve to another
    _LazyBagView, so nesting stays lazy all the way down.

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
    Unwrap a bound object to what a closure-backend template body should see.

    Parameter -> literal if solo (const), else device_view() (a .get/.set_node
        carrier for uniform in-kernel access).
    Specializable (DeviceFunction/Kernel) -> .compiled.
    Bag -> a _LazyBagView resolving each member on first attribute access, so
        dotted paths (grid.nx.get(i)) trace as plain attribute lookups without
        forcing every other member in the bag to resolve too.
    Anything else passes through unchanged.

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


@lru_cache(maxsize=None)
def capture_template_meta(template) -> tuple[str | None, ast.AST | None]:
    """
    Return (source_text, ast) for a template. A python def is introspected;
    a raw string (CUDA source) is kept verbatim with no AST (not python).

    Cached per template: every compile() asks twice (once to filter bindings,
    once for attach_meta), and each miss costs an inspect.getsource + a parse.
    The returned tree is therefore SHARED by every Specializable built from
    that template - treat it as read-only.

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


def used_bindings(template, bindings: dict[str, Any]) -> dict[str, Any]:
    """
    Subset of `bindings` whose key is actually referenced by `template`'s
    body (by name - an attribute chain's root, e.g. `phys` in
    `phys.g.get(0)`, is itself an `ast.Name`, so collecting every `ast.Name`
    id in the tree is sufficient). Falls back to returning `bindings`
    unchanged whenever the AST isn't available (non-python template, source
    unavailable) - never silently drop a binding we cannot prove is unused.

    Author: B.G (07/2026)
    """
    _, tree = capture_template_meta(template)
    if tree is None:
        return bindings
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return {name: value for name, value in bindings.items() if name in used}


def filter_bindings(template, bindings: dict[str, Any]) -> dict[str, Any]:
    """
    used_bindings(), plus one warning per compile listing everything bound but
    never referenced - catches typos in a large context. Backend-agnostic: the
    closure builders feed the result to their specializer. No warning when the
    AST isn't available (used_bindings then returns `bindings` unchanged, so
    nothing looks unused).

    Author: B.G (07/2026)
    """
    filtered = used_bindings(template, bindings)
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
    Stash hidden raw material on a freshly built Specializable for later reuse
    by a higher-level compiler (kernel fusion / graph building).

    Author: B.G (07/2026)
    """
    obj._source, obj._ast = capture_template_meta(template)
    obj._dependencies = {name: type(value).__name__ for name, value in bindings.items()}


class CompileBuilder(ABC):
    """
    Two-layer compile surface: collect bind()ed dependencies + one ingest()ed
    template, then compile() to a backend Specializable. bind() detects the
    kind of each object at resolution time (Parameter / DeviceFunction / Bag /
    handle / plain value); backends implement compile() (and may override
    ingest()) their own way.

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
    Simple named collection, mergeable into a builder via bind_bag() or bound
    whole (bind('grid', bag)) for dotted-path access in the template body.

    `_member_type` (None on the base class) restricts what a subclass's
    members may be; a nested Bag is always accepted regardless, since bags
    nest (a ParamBag may legitimately contain another bag of params).

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

    def as_bindings(self) -> dict[str, Any]:
        """
        Return {name: item} for merging into a builder's bindings.

        Author: B.G (07/2026)
        """
        return dict(self._items)


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
