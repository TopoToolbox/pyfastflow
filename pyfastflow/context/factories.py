"""
Minimal shared factories for the cleaned context system.

Author: B.G (03/2026)
"""

from dataclasses import dataclass
from types import FunctionType, SimpleNamespace

import numpy as np
import taichi as ti

from .. import pool as ppool


def unwrap_field(field_like):
    """
    Return the raw Taichi field from a TPField-like wrapper or field handle.

    Author: B.G (03/2026)
    """
    return field_like.field if hasattr(field_like, "field") else field_like


def require_flat_field(field_like, label):
    """
    Ensure one field-like input is a flat 1D Taichi field.

    Author: B.G (03/2026)
    """
    field = unwrap_field(field_like)
    if len(tuple(field.shape)) != 1:
        raise ValueError(f"{label} must be a flat field")
    return field


@dataclass(frozen=True)
class ContextRef:
    """
    Explicit reference to an attribute already bound on one context object.

    Author: B.G (03/2026)
    """

    path: str


class ParameterFactory:
    """
    Minimal manager for context parameter storage and setters.

    Author: B.G (03/2026)
    """

    def __init__(self, owner, n_flat=None):
        self.owner = owner
        self.n_flat = None if n_flat is None else int(n_flat)
        self._specs = {}
        self._owned_tpfields = []

    def declare(
        self,
        name,
        *,
        dtype,
        mode,
        value,
        extra_modes=None,
        mode_validator=None,
    ):
        """
        Declare one named parameter and allocate its backing storage.

        Author: B.G (03/2026)
        """
        mode_value = self._normalize_mode(name, mode, extra_modes=extra_modes)
        if mode_validator is not None:
            mode_value = mode_validator(mode_value)

        self._specs[name] = {
            "dtype": dtype,
            "mode": mode_value,
        }

        setattr(self.owner, f"{name}_mode", mode_value)
        setattr(self.owner, f"{name}_const", self._cast_scalar(dtype, value) if mode_value == "const" else 0)
        setattr(self.owner, f"{name}_scalar", None)
        setattr(self.owner, f"{name}_field", None)
        setattr(self.owner, f"_{name}_scalar_tpfield", None)
        setattr(self.owner, f"_{name}_field_tpfield", None)

        if mode_value == "scalar":
            tpfield = ppool.taipool.get_tpfield(dtype=dtype, shape=())
            setattr(self.owner, f"_{name}_scalar_tpfield", tpfield)
            setattr(self.owner, f"{name}_scalar", tpfield.field)
            self._owned_tpfields.append(tpfield)
        elif mode_value == "field":
            if self.n_flat is None:
                raise ValueError(f"{name}_mode='field' requires a flat field size")
            tpfield = ppool.taipool.get_tpfield(dtype=dtype, shape=(self.n_flat))
            setattr(self.owner, f"_{name}_field_tpfield", tpfield)
            setattr(self.owner, f"{name}_field", tpfield.field)
            self._owned_tpfields.append(tpfield)

        self.set_value(name, value)

    def set_value(self, name, value):
        """
        Update one declared parameter according to its configured mode.

        Author: B.G (03/2026)
        """
        spec = self._specs[name]
        mode = spec["mode"]
        dtype = spec["dtype"]
        if mode == "const":
            setattr(self.owner, f"{name}_const", self._cast_scalar(dtype, value))
        elif mode == "scalar":
            getattr(self.owner, f"{name}_scalar")[None] = self._cast_scalar(dtype, value)
        elif mode == "field":
            self._copy_flat_values(
                value,
                getattr(self.owner, f"{name}_field"),
                dtype=dtype,
                label=name,
            )

    def bind_setter(self, name, method_name=None):
        """
        Attach one explicit setter method to the owning context.

        Author: B.G (03/2026)
        """
        target_name = method_name if method_name is not None else f"set_{name}"

        def setter(value, _name=name, _factory=self):
            _factory.set_value(_name, value)

        setattr(self.owner, target_name, setter)

    def destroy(self):
        """
        Release all pooled storage owned by this parameter factory.

        Author: B.G (03/2026)
        """
        while self._owned_tpfields:
            tpfield = self._owned_tpfields.pop()
            tpfield.release()
        for name in self._specs:
            setattr(self.owner, f"{name}_scalar", None)
            setattr(self.owner, f"{name}_field", None)
            setattr(self.owner, f"_{name}_scalar_tpfield", None)
            setattr(self.owner, f"_{name}_field_tpfield", None)

    def _normalize_mode(self, name, value, extra_modes=None):
        allowed = {"const", "scalar", "field"}
        if extra_modes is not None:
            allowed |= set(extra_modes)
        mode = str(value).lower()
        if mode not in allowed:
            raise ValueError(f"{name}_mode must be one of {sorted(allowed)}")
        return mode

    def _cast_scalar(self, dtype, value):
        if dtype == ti.u8:
            return int(value)
        return float(value)

    def _copy_flat_values(self, values, dst, *, dtype, label):
        src = values.field if hasattr(values, "field") else values
        if hasattr(src, "shape") and hasattr(dst, "copy_from"):
            try:
                if tuple(src.shape) == tuple(dst.shape):
                    dst.copy_from(src)
                    return
            except (AttributeError, TypeError):
                pass

        if hasattr(values, "to_numpy"):
            arr = np.asarray(values.to_numpy(), dtype=self._numpy_dtype(dtype))
        else:
            arr = np.asarray(values, dtype=self._numpy_dtype(dtype))
        arr = arr.reshape(-1)
        if self.n_flat is None:
            raise ValueError(f"{label} field mode requires a declared flat size")
        if arr.size != self.n_flat:
            raise ValueError(f"{label} expects {self.n_flat} values, got {arr.size}")
        dst.from_numpy(arr)

    def _numpy_dtype(self, dtype):
        if dtype == ti.u8:
            return np.uint8
        return np.float32


class CallableFactory:
    """
    Shared callable specialization layer for helpers and kernels.

    Author: B.G (03/2026)
    """

    def __init__(self, bindings=None):
        self.bindings = dict(bindings or {})

    def bind(self, **bindings):
        """
        Return one extended callable factory with additional explicit bindings.

        Author: B.G (03/2026)
        """
        merged = dict(self.bindings)
        merged.update(bindings)
        return CallableFactory(merged)

    def compile(self, template, *, kind, bindings=None):
        """
        Specialize one generic helper or kernel with explicit globals.

        Author: B.G (03/2026)
        """
        decorator = ti.func if kind == "func" else ti.kernel
        source = getattr(template, "__wrapped__", template)
        func_globals = dict(source.__globals__)
        func_globals.update(self.bindings)
        if bindings is not None:
            func_globals.update(bindings)

        specialised = FunctionType(
            source.__code__,
            func_globals,
            source.__name__,
            source.__defaults__,
            source.__closure__,
        )
        specialised.__kwdefaults__ = source.__kwdefaults__
        specialised.__annotations__ = dict(source.__annotations__)
        specialised.__doc__ = source.__doc__
        specialised.__qualname__ = source.__qualname__
        return decorator(specialised)


class ContextFactory:
    """
    Declarative assembler for cleaned specialization-only contexts.

    Author: B.G (03/2026)
    """

    def __init__(self, owner, *, bindings=None, n_flat=None):
        self.owner = owner
        self.params = ParameterFactory(owner, n_flat=n_flat)
        self.callables = CallableFactory(bindings)

    def ensure_namespace(self, path):
        """
        Create and return one nested namespace path on the owning context.

        Author: B.G (03/2026)
        """
        current = self.owner
        for part in path.split("."):
            if not hasattr(current, part):
                setattr(current, part, SimpleNamespace())
            current = getattr(current, part)
        return current

    def compile_block(self, specs):
        """
        Compile and bind one explicit block of helpers or kernels.

        Author: B.G (03/2026)
        """
        for spec in specs:
            target = self.ensure_namespace(spec.get("target", "kernels"))
            compiled = self.callables.compile(
                spec["template"],
                kind=spec["kind"],
                bindings=self._resolve_bindings(spec.get("bindings")),
            )
            setattr(target, spec["name"], compiled)

    def export(self, exports):
        """
        Expose selected nested callables directly on the context root.

        Author: B.G (03/2026)
        """
        for root_name, path in exports.items():
            setattr(self.owner, root_name, self._resolve_path(path))

    def _resolve_bindings(self, bindings):
        if bindings is None:
            return None
        resolved = {}
        for name, value in bindings.items():
            if isinstance(value, ContextRef):
                resolved[name] = self._resolve_path(value.path)
            else:
                resolved[name] = value
        return resolved

    def _resolve_path(self, path):
        current = self.owner
        for part in path.split("."):
            current = getattr(current, part)
        return current
