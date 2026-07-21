"""
Backend-agnostic context building blocks.

No "Context" class here on purpose: a context is just whatever concrete class
(GridContext, FlowContext, ...) groups a set of Parameters and registers
DeviceFunctions. Cross-context references are plain explicit bindings passed
at compile() time, not a stored connection registry.

Author: B.G (07/2026)
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..pool.base import DataHandle


class Parameter(ABC):
    """
    One named, typed value owned by a context.

    REQUIRED_MODES is the baseline every backend must support; a backend
    widens SUPPORTED_MODES to add more storage kinds. Enforced at subclass
    definition time via __init_subclass__, not at instantiation.

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
        Update the parameter's value in place, according to its mode.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def destroy(self) -> None:
        """
        Release any backing storage owned by this parameter.

        Author: B.G (07/2026)
        """
        ...


class Specializable(ABC):
    """
    Shared compile/call contract for DeviceFunction and (later) Kernel.

    compile() specializes a template with explicit bindings injected as
    globals - the mechanism that replaces a stored connection registry.

    Author: B.G (07/2026)
    """

    name: str

    @classmethod
    @abstractmethod
    def compile(cls, template, *, bindings: dict[str, Any]) -> "Specializable":
        """
        Specialize `template` with `bindings` injected as globals.

        Author: B.G (07/2026)
        """
        ...

    @property
    @abstractmethod
    def compiled(self):
        """
        Raw backend callable (e.g. the ti.func/ti.kernel object), for
        injection as a global into another template's bindings.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs): ...


class DeviceFunction(Specializable):
    """
    Compiled device-side helper (e.g. a ti.func specialization).

    Not necessarily callable from host Python - backends where device
    functions can only run inside kernel/func scope may raise on __call__.

    Author: B.G (07/2026)
    """


class Kernel(Specializable):
    """
    Compiled entry point (e.g. a ti.kernel specialization).

    Unlike DeviceFunction, __call__ is expected to work from host Python -
    that's how the compute actually gets launched. Its template's own
    parameters are data fields only; params/helpers arrive via bindings
    and are resolved into the kernel body, not passed at call time.

    Author: B.G (07/2026)
    """


def resolve_binding(value):
    """
    Unwrap a Parameter/Specializable to its backend-workable object.

    Parameter -> get() -> .data if that's a DataHandle, else the raw value.
    Specializable (DeviceFunction/Kernel) -> .compiled.
    Anything else passes through unchanged.

    Author: B.G (07/2026)
    """
    if isinstance(value, Parameter):
        resolved = value.get()
        return resolved.data if isinstance(resolved, DataHandle) else resolved
    if isinstance(value, Specializable):
        return value.compiled
    return value


class Bag:
    """
    Simple named collection, mergeable into a Specializable.compile() bindings dict via as_bindings().

    Author: B.G (07/2026)
    """

    def __init__(self, items: dict[str, Any] | None = None):
        self._items: dict[str, Any] = dict(items or {})

    def add(self, name: str, item: Any) -> None:
        """
        Register `item` under `name`. Raises if `name` is already taken.

        Author: B.G (07/2026)
        """
        if name in self._items:
            raise KeyError(f"'{name}' is already registered in this bag")
        self._items[name] = item

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
        Return {name: item} for merging into a compile() bindings dict.

        Author: B.G (07/2026)
        """
        return dict(self._items)


class ParamBag(Bag):
    """
    Named collection of Parameter objects.

    Author: B.G (07/2026)
    """


class HelperBag(Bag):
    """
    Named collection of DeviceFunction objects.

    Author: B.G (07/2026)
    """
