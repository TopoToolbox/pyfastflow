"""
Clean GraphFlood object exposing a simplified API around the flooder.

GGF_Object mirrors Flooder's public getters/properties while providing a
tidier GraphFlood control loop and explicit LisFlood handling.

Author: B.G.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import taichi as ti

import pyfastflow as pf

from .. import constants as cte
from . import gf_ls as ls


class GGF_Object:
    """
    Standalone flood model with the same external API as the original Flooder
    but a focused GraphFlood routine based on the shared reference script.
    """

    def __init__(
        self,
        router,
        precipitation_rates: float = 10e-3 / 3600,
        manning: float = 0.033,
        edge_slope: float = 1e-2,
        dt_hydro: float = 1e-3,
        dt_hydro_ls: Optional[float] = None,
    ):
        self.router = router
        self.grid = router.grid

        self.og_z = self.grid.z.field.to_numpy()

        self.h = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny))
        self.dh = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny))
        self.nQ = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny))
        self.qx_LS = None
        self.qy_LS = None
        self.qx = None
        self.qy = None

        self.h.field.fill(0.0)
        self.dh.field.fill(0.0)
        self.nQ.field.fill(0.0)
        self.nQ_init = False

        self.precipitation_rates = precipitation_rates
        self.manning = manning
        self.edge_slope = edge_slope

        self.verbose = False

        cte.PREC = precipitation_rates
        cte.MANNING = manning
        cte.EDGESW = edge_slope
        cte.DT_HYDRO = dt_hydro
        cte.DT_HYDRO_LS = dt_hydro if dt_hydro_ls is None else dt_hydro_ls

    # ------------------------------------------------------------------
    # Grid metadata
    # ------------------------------------------------------------------
    @property
    def nx(self):
        return self.grid.nx

    @property
    def ny(self):
        return self.grid.ny

    @property
    def dx(self):
        return self.grid.dx

    @property
    def rshp(self):
        return self.grid.rshp

    # ------------------------------------------------------------------
    # GraphFlood control loop
    # ------------------------------------------------------------------
    def run_graphflood(
        self,
        iterations: int = 1000,
        diffusion_cycles: int = 1,
        weight_recomputations: int = 1,
        dt: float = 5e-3,
        temporal_dumping: float = 0.5,
        prec2D: Optional[Any] = None,
    ):
        """
        Execute the GraphFlood loop defined by the reference script.

        Args:
            iterations: Number of outer GraphFlood iterations.
            diffusion_cycles: Number of diffusion passes per iteration.
            weight_recomputations: Number of weight recomputations before each
                diffusion pass.
            dt: Timestep used inside ``run_graphflood_diffuse_nopropag``.
            temporal_dumping: Temporal damping parameter passed to
                ``diffuse_Q_with_weights``.
            prec2D: Optional 2D precipitation map replacing the scalar rate.
        """

        if iterations <= 0:
            raise ValueError("iterations must be >= 1")
        if diffusion_cycles <= 0:
            raise ValueError("diffusion_cycles must be >= 1")
        if weight_recomputations <= 0:
            raise ValueError("weight_recomputations must be >= 1")

        grid_shape = self.grid.z.field.shape

        wx = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=grid_shape)
        wy = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=grid_shape)
        Q_ = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=grid_shape)
        S = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=grid_shape)

        wx.field.fill(0.0)
        wy.field.fill(0.0)
        Q_.field.fill(0.0)

        try:
            self._fill_precipitation_field(S.field, prec2D)

            for _ in range(iterations):
                for _ in range(diffusion_cycles):
                    wx.field.fill(0.0)
                    wy.field.fill(0.0)

                    for _ in range(weight_recomputations):
                        pf.flood.gf_hydrodynamics.compute_weights_wxy(
                            wx.field,
                            wy.field,
                            self.grid.z.field,
                            self.h.field,
                        )

                    pf.flood.gf_hydrodynamics.diffuse_Q_with_weights(
                        self.router.Q.field,
                        Q_.field,
                        wx.field,
                        wy.field,
                        S.field,
                        temporal_dumping,
                    )

                self.run_graphflood_diffuse_nopropag(N=1, dt=dt)
        finally:
            wx.release()
            wy.release()
            Q_.release()
            S.release()

    def run_graphflood_diffuse_nopropag(self, N: int = 10, dt: Optional[float] = None, mask=None):
        """
        Run the hydrodynamic diffusion kernel without propagation.

        Args:
            N: Number of solver steps.
            dt: Optional explicit timestep overriding ``cte.DT_HYDRO``.
            mask: Optional mask restricting updates to a subset of nodes.
        """
        dh = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny))

        if mask is not None:
            tmask = pf.pool.taipool.get_tpfield(dtype=ti.u1, shape=(self.nx * self.ny))
            if isinstance(mask, np.ndarray):
                tmask.field.from_numpy(mask)
            else:
                tmask.copy_from(mask)
        else:
            tmask = None

        for _ in range(N):
            if tmask is None:
                pf.flood.gf_hydrodynamics.graphflood_cte_man_dt_nopropag(
                    self.grid.z.field,
                    self.h.field,
                    self.router.Q.field,
                    dh.field,
                    cte.DT_HYDRO if dt is None else dt,
                )
            else:
                pf.flood.gf_hydrodynamics.graphflood_cte_man_dt_nopropag_mask(
                    self.grid.z.field,
                    self.h.field,
                    self.router.Q.field,
                    dh.field,
                    tmask.field,
                    cte.DT_HYDRO if dt is None else dt,
                )

        dh.release()
        if tmask is not None:
            tmask.release()

    # ------------------------------------------------------------------
    # LisFlood explicit solver
    # ------------------------------------------------------------------
    def run_LS(self, N: int = 1000, input_mode: str = "constant_prec", mode=None):
        """
        Run the LisFlood explicit scheme while keeping dedicated discharge fields.

        Args:
            N: Number of LisFlood timesteps.
            input_mode: Either ``constant_prec`` or ``custom_func``.
            mode: Callable executed each step when ``input_mode`` equals
                ``custom_func``.
        """

        if self.qx_LS is None:
            self.qx_LS = pf.pool.taipool.get_tpfield(
                dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny)
            )
            self.qy_LS = pf.pool.taipool.get_tpfield(
                dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny)
            )
            self.qx_LS.field.fill(0.0)
            self.qy_LS.field.fill(0.0)
            self.qx = self.qx_LS
            self.qy = self.qy_LS

        for _ in range(N):
            if input_mode == "constant_prec":
                ls.init_LS_on_hw_from_constant_effective_prec(
                    self.h.field, self.grid.z.field
                )
            elif input_mode == "custom_func":
                if mode is None:
                    raise ValueError("mode callable must be provided when input_mode='custom_func'")
                mode()
            else:
                raise ValueError(f"Unsupported input_mode '{input_mode}'")

            ls.flow_route(
                self.h.field,
                self.grid.z.field,
                self.qx_LS.field,
                self.qy_LS.field,
            )
            ls.depth_update(
                self.h.field,
                self.grid.z.field,
                self.qx_LS.field,
                self.qy_LS.field,
            )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def fill_lakes_full(self, compute_Qsfd: bool = False, epsilon: float = 2e-3):
        """
        Fill depressions in the combined surface and optionally recompute
        discharge using constant precipitation.

        Args:
            compute_Qsfd: Whether to rebuild router discharges after filling.
            epsilon: Minimum increment applied during the fill.
        """
        z_ = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny))
        receivers_ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))
        receivers__ = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(self.nx * self.ny))

        pf.general_algorithms.util_taichi.add_B_to_A(self.grid.z.field, self.h.field)

        self.router.compute_receivers()
        self.router.reroute_flow()

        pf.flow.fill_z_add_delta(
            self.grid.z.field,
            self.h.field,
            z_.field,
            self.router.receivers.field,
            receivers_.field,
            receivers__.field,
            epsilon=epsilon,
        )

        if compute_Qsfd:
            self.router.accumulate_constant_Q(cte.PREC, area=True)

        pf.general_algorithms.util_taichi.add_B_to_weighted_A(
            self.grid.z.field, self.h.field, -1.0
        )

        z_.release()
        receivers_.release()
        receivers__.release()

    def _fill_precipitation_field(self, target_field: ti.Field, source: Optional[Any]):
        """
        Fill the temporary precipitation field with the default rate or a custom
        2D distribution supplied by the caller.
        """
        if source is None:
            target_field.fill(self.precipitation_rates)
            return

        if isinstance(source, float):
            target_field.fill(source)
            return

        if isinstance(source, np.ndarray):
            target_field.from_numpy(np.asarray(source, dtype=np.float32).ravel())
            return

        if hasattr(source, "field"):
            target_field.copy_from(source.field)
            return

        if hasattr(source, "copy_from"):
            target_field.copy_from(source)
            return

        raise TypeError(
            "prec2D must be None, a numpy array, a Taichi field, "
            "or a pool field with a `.field` attribute.",
        )

    # ------------------------------------------------------------------
    # Getters / setters (API parity with Flooder)
    # ------------------------------------------------------------------
    def set_h(self, val):
        """Replace the depth field with ``val`` (numpy array expected)."""
        self.h.field.from_numpy(val.ravel())

    def get_h(self):
        """Return water depth as a 2D numpy array."""
        return self.h.field.to_numpy().reshape(self.rshp)

    def get_qx(self):
        """Return the LisFlood x-direction discharge."""
        if self.qx_LS is None:
            raise RuntimeError("LisFlood fields not initialized yet. Call run_LS first.")
        return self.qx_LS.field.to_numpy().reshape(self.rshp)

    def get_qy(self):
        """Return the LisFlood y-direction discharge."""
        if self.qy_LS is None:
            raise RuntimeError("LisFlood fields not initialized yet. Call run_LS first.")
        return self.qy_LS.field.to_numpy().reshape(self.rshp)

    def get_dh(self):
        """Return the cumulative depth change field."""
        return self.dh.field.to_numpy().reshape(self.rshp)

    def get_Q(self):
        """Return the router discharge as a numpy array."""
        return self.router.Q.field.to_numpy().reshape(self.rshp)

    def get_Qo(self):
        """Return the GraphFlood discharge field."""
        Q_ = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny))
        pf.flood.gf_hydrodynamics.graphflood_get_Qo(
            self.grid.z.field, self.h.field, Q_.field
        )
        Qo = Q_.field.to_numpy().reshape(self.rshp)
        Q_.release()
        return Qo

    def get_sw(self):
        """Return the water-surface slope field."""
        sw = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny))
        pf.flood.gf_hydrodynamics.compute_sw(self.grid.z.field, self.h.field, sw.field)
        out = sw.field.to_numpy().reshape(self.rshp)
        sw.release()
        return out

    def get_tau(self, rho: float = 1000.0):
        """Return the bed shear stress for the provided density."""
        tau = pf.pool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.nx * self.ny))
        pf.flood.gf_hydrodynamics.compute_tau(
            self.grid.z.field, self.h.field, tau.field, rho
        )
        out = tau.field.to_numpy().reshape(self.rshp)
        tau.release()
        return out

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def destroy(self):
        """Release all pooled fields owned by this object."""
        if getattr(self, "h", None) is not None:
            self.h.release()
            self.h = None

        if getattr(self, "dh", None) is not None:
            self.dh.release()
            self.dh = None

        if getattr(self, "nQ", None) is not None:
            self.nQ.release()
            self.nQ = None

        if getattr(self, "qx_LS", None) is not None:
            self.qx_LS.release()
            self.qx_LS = None
            self.qx = None

        if getattr(self, "qy_LS", None) is not None:
            self.qy_LS.release()
            self.qy_LS = None
            self.qy = None

    def __del__(self):
        """Best-effort guard to release GPU memory when the object is GC'd."""
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass
