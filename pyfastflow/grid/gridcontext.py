from types import SimpleNamespace

import numpy as np
import taichi as ti

from .. import pool as ppool
from ..context import ContextFactory
from ._gridapi_helpers_flat import build_flat_helpers


class GridContext:
    """
    Flat-index grid specialization context.

    This context only stores compile-time grid constants, optional boundary
    codes, and the flat helper/callable surface specialized against that grid.

    Author: B.G (03/2026)
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
        self.nx = int(nx)
        self.ny = int(ny)
        self.n_flat = self.nx * self.ny
        self.rshp = (self.ny, self.nx)
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

        if self.has_bcs:
            self._bcs_tpfield = ppool.taipool.get_tpfield(dtype=ti.u8, shape=(self.n_flat))
            self.bcs = self._bcs_tpfield.field
            if bcs is None:
                self.bcs.fill(1)
            else:
                self.set_bcs(bcs)
        elif bcs is not None:
            raise ValueError("bcs data was provided but has_bcs is False")

        self._factory = ContextFactory(self, bindings={"gridctx": self}, n_flat=self.n_flat)
        self.tfunc = SimpleNamespace()
        self._compile_helpers()

    def _compile_helpers(self):
        """
        Compile and bind the flat helper surface for this grid.

        Author: B.G (03/2026)
        """
        helpers = build_flat_helpers(self)
        for name, value in helpers.__dict__.items():
            setattr(self.tfunc, name, value)

        canonical = {
            "is_active": self.tfunc.is_active_flat,
            "nodata": self.tfunc.nodata_flat,
            "neighbour": self.tfunc.neighbour_flat,
            "neighbour_raw": self.tfunc.neighbour_raw_flat,
            "neighbours": self.tfunc.neighbours_flat,
            "neighbours_raw": self.tfunc.neighbours_raw_flat,
            "is_on_edge": self.tfunc.is_on_edge_flat,
            "which_edge": self.tfunc.which_edge_flat,
            "can_out": self.tfunc.can_out_flat,
            "dist_from_k": self.tfunc.dist_from_k_flat,
            "dist_between_nodes": self.tfunc.dist_between_nodes_flat,
        }
        for name, value in canonical.items():
            setattr(self.tfunc, name, value)

    def set_bcs(self, values):
        """
        Copy boundary codes into the internal flat boundary field.

        Author: B.G (03/2026)
        """
        if self._bcs_tpfield is None:
            raise ValueError("This GridContext has no internal bcs field")

        if hasattr(values, "to_numpy"):
            arr = np.asarray(values.to_numpy(), dtype=np.uint8)
        else:
            arr = np.asarray(values, dtype=np.uint8)

        arr = arr.reshape(-1)
        if arr.size != self.n_flat:
            raise ValueError(f"Expected {self.n_flat} boundary codes, got {arr.size}")
        self.bcs.from_numpy(arr)

    def make_kernel(self, kernel_template, **extra_globals):
        """
        Specialize one generic Taichi kernel against this grid context.

        Author: B.G (03/2026)
        """
        return self._factory.callables.compile(
            kernel_template,
            kind="kernel",
            bindings=extra_globals,
        )

    def make_func(self, func_template, **extra_globals):
        """
        Specialize one generic Taichi helper against this grid context.

        Author: B.G (03/2026)
        """
        return self._factory.callables.compile(
            func_template,
            kind="func",
            bindings=extra_globals,
        )

    def destroy(self):
        """
        Release pooled internal fields owned by this context.

        Author: B.G (03/2026)
        """
        if self._bcs_tpfield is not None:
            self._bcs_tpfield.release()
            self._bcs_tpfield = None
            self.bcs = None

    def __del__(self):
        """
        Best-effort pooled resource cleanup.

        Author: B.G (03/2026)
        """
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass
