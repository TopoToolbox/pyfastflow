from types import FunctionType, SimpleNamespace

import numpy as np
import taichi as ti

from .. import pool as ppool
from ._gridapi_helpers_flat import build_flat_helpers
from ._gridapi_helpers_2d import build_2d_helpers


class GridContext:
    """
    Lightweight kernel-facing grid context with pre-bound Taichi helpers.

    This object stores only the compile-time constants and the optional internal
    boundary-condition field required by the neighbouring helpers. It
    exposes a single ``tfunc`` namespace containing both flat and 2D helper
    variants so user kernels can stay generic and compact.

    The context does not own general simulation fields. It is intentionally kept
    small so several contexts can coexist and specialize the same generic kernel
    body independently through :meth:`make_kernel`.

    Author: B.G (02/2026)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        boundary_mode: str = "normal",
        topology: str = "D4",
        has_bcs: bool = False,
        bcs=None,
    ):
        # Small immutable kernel-facing state; grid data stays outside this context.
        self.nx = int(nx)
        self.ny = int(ny)
        self.rshp = (self.ny,self.nx)
        self.dx = float(dx)
        self.boundary_mode = boundary_mode
        self.topology = str(topology).upper()
        self.has_bcs = bool(has_bcs)

        if self.boundary_mode not in {"normal", "periodic_EW", "periodic_NS"}:
            raise ValueError(f"Unsupported boundary_mode: {self.boundary_mode}")
        if self.topology not in {"D4", "D8"}:
            raise ValueError(f"Unsupported topology: {self.topology}")

        self.n_neighbours = 8 if self.topology == "D8" else 4
        self._bcs_tpfield = None
        self.bcs = None

        # _^^_
        # (.. )  Optional internal boundary mask, injected statically in helpers when enabled.
        # /||\\
        if self.has_bcs:
            self._bcs_tpfield = ppool.taipool.get_tpfield(dtype=ti.u8, shape=(self.nx * self.ny))
            self.bcs = self._bcs_tpfield.field
            if bcs is None:
                self.bcs.fill(1)
            else:
                self.set_bcs(bcs)
        elif bcs is not None:
            raise ValueError("bcs data was provided but has_bcs is False")

        self.tfunc = SimpleNamespace()
        self._compile_helpers()

    def _compile_helpers(self):
        """
        Compile and bind the full helper surface for this context.

        The flat and 2D helper builders both capture the current constants and
        the optional internal boundary field. Their outputs are merged into the
        unique ``tfunc`` namespace exposed by the context.

        Author: B.G (02/2026)
        """
        # Build both flat and 2D helper namespaces once, then expose them through one surface.
        flat_helpers = build_flat_helpers(self)
        two_d_helpers = build_2d_helpers(self)
        for name, value in flat_helpers.__dict__.items():
            setattr(self.tfunc, name, value)
        for name, value in two_d_helpers.__dict__.items():
            setattr(self.tfunc, name, value)

    def set_bcs(self, values):
        """
        Copy boundary codes into the internal ``bcs`` field.

        Parameters
        ----------
        values:
            Flat or 2D numpy-like array, or a Taichi field exposing ``to_numpy``,
            containing exactly ``nx * ny`` unsigned byte boundary codes.

        Notes
        -----
        This method is only available when ``has_bcs=True``.

        Author: B.G (02/2026)
        """
        if self._bcs_tpfield is None:
            raise ValueError("This GridContext has no internal bcs field")

        # Accept numpy arrays or Taichi fields; normalize once before copying to device.
        if hasattr(values, "to_numpy"):
            arr = np.asarray(values.to_numpy(), dtype=np.uint8)
        else:
            arr = np.asarray(values, dtype=np.uint8)

        arr = arr.reshape(-1)
        expected = self.nx * self.ny
        if arr.size != expected:
            raise ValueError(f"Expected {expected} boundary codes, got {arr.size}")
        self.bcs.from_numpy(arr)

    def make_kernel(self, kernel_template, **extra_globals):
        """
        Specialize a generic Taichi kernel against this context.

        The input kernel is expected to refer to ``gridctx`` as a global name
        inside its body. This method clones the original Python function,
        injects the current context under that name, optionally overrides other
        globals with already-specialized helper funcs, and re-wraps the cloned
        function with ``ti.kernel`` so the compiled kernel is specialized for
        this context.

        Parameters
        ----------
        kernel_template:
            A generic ``@ti.kernel`` function whose body uses ``gridctx`` to
            access constants and helpers.

        Returns
        -------
        function
            A context-specialized Taichi kernel ready to be called directly.

        Author: B.G (02/2026)
        """
        return self._specialize_callable(kernel_template, ti.kernel, **extra_globals)

    def make_func(self, func_template, **extra_globals):
        """
        Specialize a generic Taichi function against this context.

        The input function is expected to refer to ``gridctx`` as a global name
        inside its body. This method clones the original Python function,
        injects the current context under that name, optionally overrides other
        globals with already-specialized helper funcs, and re-wraps the cloned
        function with ``ti.func``.

        Author: B.G (02/2026)
        """
        return self._specialize_callable(func_template, ti.func, **extra_globals)

    def _specialize_callable(self, func_template, decorator, **extra_globals):
        """
        Clone a Python function and bind this context plus optional helper overrides.

        This is the shared implementation behind ``make_kernel`` and ``make_func``.

        Author: B.G (02/2026)
        """
        # `@ti.kernel` and `@ti.func` both keep the undecorated Python body in
        # `__wrapped__` when available. That is the version we clone.
        source = getattr(func_template, "__wrapped__", func_template)

        # Clone the original globals dictionary, then inject the bound grid
        # context and any already-specialized helper funcs/kernels requested by
        # the caller.
        func_globals = dict(source.__globals__)
        func_globals["gridctx"] = self
        func_globals.update(extra_globals)

        # Rebuild a plain Python function object from the original code object.
        specialised = FunctionType(
            source.__code__,
            func_globals,
            source.__name__,
            source.__defaults__,
            source.__closure__,
        )

        # Preserve usual Python metadata so the specialized callable still reads
        # like the original one under inspection.
        specialised.__kwdefaults__ = source.__kwdefaults__
        specialised.__annotations__ = dict(source.__annotations__)
        specialised.__doc__ = source.__doc__
        specialised.__qualname__ = source.__qualname__

        # Re-apply the requested Taichi decorator to obtain the final specialized
        # callable.
        return decorator(specialised)

    def destroy(self):
        """
        Release pooled internal fields held by this context.

        This should be called when a context with internal pooled resources is no
        longer needed. After destruction, the context should not be reused.

        Author: B.G (02/2026)
        """
        if hasattr(self, "_bcs_tpfield") and self._bcs_tpfield is not None:
            self._bcs_tpfield.release()
            self._bcs_tpfield = None
            self.bcs = None

    def __del__(self):
        """
        Destructor - automatically release pooled resources when possible.

        Author: B.G (02/2026)
        """
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass
