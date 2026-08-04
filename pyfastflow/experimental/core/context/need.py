"""
Need: one declared, typed, identity-bearing slot a builder must have filled
before it can be used - the single binding contract PARAM, DATA and HELPER
bindings now share, replacing the two different mechanisms this module used
to have (Parameters bound eagerly and invisibly at construction via
CompileBuilder.bind(), data left as unlabeled positional call arguments).

A Need is created standalone, independent of any builder:

    source_need = Need("source", kind=Kind.PARAM, dtype=f32)
    kb = KernelCls().need(source_need).ingest(template_fn)
    source_need.bind(source_p)
    kb.compile()

Two builders share a slot because they were handed the same Need object, not
because they used the same name string - see the module docstring in
compile.py for how CompileBuilder consumes a builder's declared needs.

kind=PARAM / kind=HELPER are frozen forever once bound: rebind raises. This
matches the existing Parameter invariant ("a Parameter's build-time identity
is fixed at construction" - parameter.py) and extends it to helpers, which
are themselves recipes closing over their own Parameters. To change one,
construct a new Need (or a new Parameter/HelperBuilder) and rebuild whatever
bound the old one.

kind=DATA needs stay call-time arguments (see make_accumulation's
"persistent_mfd" discussion for why this was chosen over binding data the
same way as everything else): a DATA Need documents and validates a
template's data-argument contract (name, dtype), but the compiled Kernel/
Routine still takes the actual buffer positionally at every call - `.bind()`
on a DATA Need is only ever used to validate a *declared* dtype against
whatever gets passed at call time, never to freeze which buffer a compiled
object reads.

Author: B.G (08/2026)
"""

from enum import Enum
from typing import Any


class Kind(Enum):
    """
    What a Need's slot holds. PARAM: a Parameter (parameter.py). DATA: a raw
    buffer, checked at call time, never frozen into a compiled object. HELPER:
    a HelperBuilder (compile.py), itself possibly carrying its own unmet
    Needs - see Need.unmet_needs.

    Author: B.G (08/2026)
    """

    PARAM = "param"
    DATA = "data"
    HELPER = "helper"


class Need:
    """
    One named slot, declared once, bound once (PARAM/HELPER) or validated
    per-call (DATA). See the module docstring for the full picture.

    `dtype`, if given, is checked against the bound object's own dtype at
    bind() time (PARAM/HELPER) - fail fast, not at compile() or worse, at a
    CUDA launch. `modes`, only meaningful for kind=PARAM, is the set of
    Parameter.mode values this slot accepts (e.g. rake_compress's
    iteration_p needing exactly {"scalar"}) - checked the same way.

    Author: B.G (08/2026)
    """

    def __init__(self, name: str, kind: Kind, *, dtype: Any = None, modes=None):
        if kind != Kind.PARAM and modes is not None:
            raise ValueError(f"Need({name!r}): modes= is only meaningful for kind=Kind.PARAM")
        self.name = name
        self.kind = kind
        self.dtype = dtype
        self.modes = frozenset(modes) if modes is not None else None
        self._bound: Any = None
        self._is_bound = False

    def __repr__(self) -> str:
        state = "bound" if self._is_bound else "UNBOUND"
        return f"Need({self.name!r}, kind={self.kind.value}, {state})"

    @property
    def is_bound(self) -> bool:
        """
        Whether bind() has been called - for kind=DATA this only reflects
        whether a dtype/shape contract check has ever been run, not whether a
        buffer is "currently" attached (there is none between calls - see the
        module docstring).

        Author: B.G (08/2026)
        """
        return self._is_bound

    @property
    def value(self) -> Any:
        """
        The bound object. Raises if unbound - callers must check is_bound (or
        let this raise) rather than silently proceeding with None, which is
        exactly the failure mode Need replaces.

        Author: B.G (08/2026)
        """
        if not self._is_bound:
            raise ValueError(f"Need({self.name!r}, kind={self.kind.value}) is not bound yet")
        return self._bound

    def bind(self, obj: Any) -> "Need":
        """
        Attach `obj` to this slot, validating its kind/dtype/mode immediately
        (bind-time checking, not compile-time or runtime).

        kind=PARAM/HELPER: raises if already bound - these are frozen
        forever (see the module docstring); construct a new Need instead of
        rebinding one already in use.
        kind=DATA: never raises for "already bound" - re-binding (i.e.
        re-validating a new call-time buffer against this slot's contract)
        is exactly its intended repeated use.

        Author: B.G (08/2026)
        """
        if self.kind in (Kind.PARAM, Kind.HELPER) and self._is_bound:
            raise ValueError(
                f"Need({self.name!r}, kind={self.kind.value}) is already bound and frozen - "
                "construct a new Need (and a new Parameter/HelperBuilder) instead of rebinding"
            )
        self._check(obj)
        self._bound = obj
        self._is_bound = True
        return self

    def _check(self, obj: Any) -> None:
        """
        kind/dtype/mode validation for bind(). Import-cycles against
        parameter.py/compile.py are avoided with local imports (this module
        must not be imported by either of those at module scope).

        Author: B.G (08/2026)
        """
        from .compile import HelperBuilder
        from .parameter import Parameter

        if self.kind is Kind.PARAM:
            if not isinstance(obj, Parameter):
                raise TypeError(f"Need({self.name!r}, kind=param): expected a Parameter, got {type(obj).__name__}")
            if self.dtype is not None and obj.dtype != self.dtype:
                raise TypeError(f"Need({self.name!r}): dtype mismatch, need {self.dtype}, got {obj.dtype}")
            if self.modes is not None and obj.mode not in self.modes:
                raise ValueError(f"Need({self.name!r}): mode {obj.mode!r} not in allowed {sorted(self.modes)}")
        elif self.kind is Kind.HELPER:
            if not isinstance(obj, HelperBuilder):
                raise TypeError(f"Need({self.name!r}, kind=helper): expected a HelperBuilder, got {type(obj).__name__}")
        elif self.kind is Kind.DATA:
            if self.dtype is not None and getattr(obj, "dtype", None) is not None and obj.dtype != self.dtype:
                raise TypeError(f"Need({self.name!r}, kind=data): dtype mismatch, need {self.dtype}, got {obj.dtype}")
        else:
            raise ValueError(f"Need({self.name!r}): unknown kind {self.kind!r}")

    def unmet_needs(self) -> list["Need"]:
        """
        This need's own unmet-ness, flattened with whatever a bound
        kind=HELPER need's HelperBuilder itself still needs (automatic
        flattening, all the way down). Empty list if this need is fully
        satisfied.

        Author: B.G (08/2026)
        """
        if not self._is_bound:
            return [] if self.kind is Kind.DATA else [self]
        if self.kind is Kind.HELPER:
            return self._bound.unmet_needs()
        return []
