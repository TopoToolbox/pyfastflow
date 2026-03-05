from types import SimpleNamespace

import numpy as np
import taichi as ti

from .. import constants as cte
from .. import pool as ppool
from ..flow import FlowContext
from ._flood_param_helpers import (
    compute_qo_from_h_slope,
    compute_q_from_h_slope,
    compute_u_from_h_slope,
    get_boundary_h,
    get_dth,
    get_dt_morpho,
    get_dt_morpho_coeff,
    get_friction_coeff,
    get_friction_exponent,
    get_gf_min_increment,
    get_gravity,
    get_rho_s,
    get_rho_w,
    get_source_w,
    source_to_h,
    source_to_Q,
)
from .flood_graphflood_kernels import (
    add_source_to_h_kernel,
    add_source_to_Q_kernel,
    compute_Qo_kernel,
    compute_tau_kernel,
    compute_u_kernel,
    distribute_flow_kernel,
    graphflood_core_kernel,
    make_surface_kernel,
)
from .flood_ls_kernels import (
    ls_add_source_to_h_kernel,
    ls_depth_update_kernel,
    ls_flow_route_kernel,
)


class FloodContext:
    """
    Grid-bound context for cleaned flood hydrodynamics.

    The context exposes a unified parameter-helper surface and binds GraphFlood
    and LisFlood kernels with compile-time static behavior switches.

    Author: B.G (02/2026)
    """

    def __init__(
        self,
        gridctx,
        flowctx=None,
        dth_mode: str = "const",
        dth: float = 1e-3,
        source_w_mode: str = "const",
        source_w: float = 0.0,
        source_w_kind: str = "precip",
        friction_coeff_mode: str = "const",
        friction_coeff: float = 0.033,
        friction_exponent_mode: str = "const",
        friction_exponent: float = 2.0 / 3.0,
        friction_law: str = "manning",
        dt_morpho_mode: str = "n_dthydro",
        dt_morpho: float = 1.0,
        dt_morpho_coeff_mode: str = "const",
        dt_morpho_coeff: float = 1.0,
        boundary_h_mode: str = "const",
        boundary_h: float = 0.0,
        gf_min_increment_mode: str = "const",
        gf_min_increment: float = 0.0,
        gravity_mode: str = "const",
        gravity: float = 9.8,
        rho_w_mode: str = "const",
        rho_w: float = 1000.0,
        rho_s_mode: str = "const",
        rho_s: float = 2600.0,
    ):
        """
        Initialize FloodContext and bind helper/kernels.

        Author: B.G (02/2026)
        """
        if hasattr(gridctx, "layout") and str(gridctx.layout).lower() == "2d":
            raise ValueError("FloodContext only supports flat grid logic")
        if gridctx.topology not in {"D4", "D8"}:
            raise ValueError("FloodContext only supports D4 or D8 grid contexts")

        self.gridctx = gridctx
        self.flowctx = flowctx if flowctx is not None else FlowContext(gridctx)
        self.gactx = self.flowctx.gactx

        self.n_flat = self.gridctx.nx * self.gridctx.ny

        self._owned_param_fields = []
        self._tpfield_attrs = []

        self.source_w_kind = self._normalize_source_kind(source_w_kind)
        self.friction_law = self._normalize_friction_law(friction_law)

        self._init_param("dth", dth_mode, dth)
        self._init_param("source_w", source_w_mode, source_w)
        self._init_param("friction_coeff", friction_coeff_mode, friction_coeff)
        self._init_param("friction_exponent", friction_exponent_mode, friction_exponent)
        self._init_param("boundary_h", boundary_h_mode, boundary_h)
        self._init_param("gf_min_increment", gf_min_increment_mode, gf_min_increment)
        self._init_param("gravity", gravity_mode, gravity)
        self._init_param("rho_w", rho_w_mode, rho_w)
        self._init_param("rho_s", rho_s_mode, rho_s)

        dtm_mode = self._normalize_mode(
            dt_morpho_mode, "dt_morpho_mode", extra={"n_dthydro"}
        )
        self._init_param("dt_morpho_coeff", dt_morpho_coeff_mode, dt_morpho_coeff)
        if dtm_mode == "n_dthydro":
            self._init_param("dt_morpho", "const", dt_morpho)
            self.dt_morpho_mode = "n_dthydro"
        else:
            self._init_param("dt_morpho", dtm_mode, dt_morpho)
            self.dt_morpho_mode = dtm_mode

        self.tfunc = SimpleNamespace()
        self.kernels = SimpleNamespace()
        self.kernels.graphflood = SimpleNamespace()
        self.kernels.ls = SimpleNamespace()
        self._compile_helpers()
        self._compile_kernels()

        # Dedicated accumulation context avoids mutating caller FlowContext
        # parameter state when flood source weights are refreshed.
        self._owns_accum_flowctx = True
        self._accum_flowctx = FlowContext(
            self.gridctx,
            gactx=self.gactx,
            weight_mode="field",
            weight=np.zeros(self.n_flat, dtype=np.float32),
            min_slope_mode="const",
            min_slope=0.0,
            diagonal_partition_correction=self.flowctx.diagonal_partition_correction,
        )

        self.gridctx.flood = self

    def _normalize_mode(self, value, label, extra=None):
        """
        Normalize one mode string.

        Author: B.G (02/2026)
        """
        allowed = {"const", "scalar", "field"}
        if extra is not None:
            allowed |= set(extra)
        mode = str(value).lower()
        if mode not in allowed:
            raise ValueError(f"{label} must be one of {sorted(allowed)}")
        return mode

    def _normalize_source_kind(self, value):
        """
        Normalize source unit kind.

        Author: B.G (02/2026)
        """
        text = str(value)
        kind = text.lower()
        if text == "Q":
            return "Q"
        if kind not in {"q", "precip"}:
            raise ValueError("source_w_kind must be one of: 'Q', 'q', 'precip'")
        if kind == "q":
            return "q"
        return "precip"

    def _normalize_friction_law(self, value):
        """
        Normalize friction law selector.

        Author: B.G (02/2026)
        """
        law = str(value).lower()
        if law != "manning":
            raise ValueError("Only friction_law='manning' is currently supported")
        return law

    def _register_tpfield(self, attr_name, tpfield):
        """
        Track one internal pooled field for cleanup.

        Author: B.G (02/2026)
        """
        self._owned_param_fields.append(tpfield)
        self._tpfield_attrs.append(attr_name)

    def _init_param(self, name, mode, value):
        """
        Initialize one flood parameter storage.

        Author: B.G (02/2026)
        """
        mode_norm = self._normalize_mode(mode, f"{name}_mode")
        setattr(self, f"{name}_mode", mode_norm)
        setattr(self, f"{name}_const", float(value) if mode_norm == "const" else 0.0)
        setattr(self, f"{name}_scalar", None)
        setattr(self, f"{name}_field", None)
        setattr(self, f"_{name}_scalar_tpfield", None)
        setattr(self, f"_{name}_field_tpfield", None)

        cur_mode = getattr(self, f"{name}_mode")
        if cur_mode == "scalar":
            tpfield = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=())
            setattr(self, f"_{name}_scalar_tpfield", tpfield)
            setattr(self, f"{name}_scalar", tpfield.field)
            self._register_tpfield(f"_{name}_scalar_tpfield", tpfield)
        elif cur_mode == "field":
            tpfield = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
            setattr(self, f"_{name}_field_tpfield", tpfield)
            setattr(self, f"{name}_field", tpfield.field)
            self._register_tpfield(f"_{name}_field_tpfield", tpfield)

        self._set_param(name, value)

    def _copy_flat_values(self, values, dst, label):
        """
        Copy flat values into an internal field parameter.

        Author: B.G (02/2026)
        """
        src = values.field if hasattr(values, "field") else values
        if hasattr(src, "shape") and hasattr(dst, "copy_from"):
            try:
                if tuple(src.shape) == tuple(dst.shape):
                    dst.copy_from(src)
                    return
            except (AttributeError, TypeError):
                pass

        if hasattr(values, "to_numpy"):
            arr = np.asarray(values.to_numpy(), dtype=np.float32)
        else:
            arr = np.asarray(values, dtype=np.float32)
        arr = arr.reshape(-1)
        if arr.size != self.n_flat:
            raise ValueError(f"{label} expects {self.n_flat} values, got {arr.size}")
        dst.from_numpy(arr)

    def _set_param(self, name, value):
        """
        Set one parameter value according to its mode.

        Author: B.G (02/2026)
        """
        mode = getattr(self, f"{name}_mode")
        if mode == "const":
            setattr(self, f"{name}_const", float(value))
        elif mode == "scalar":
            getattr(self, f"{name}_scalar")[None] = float(value)
        else:
            self._copy_flat_values(value, getattr(self, f"{name}_field"), name)

    def set_dth(self, value):
        """Set dth parameter. Author: B.G (02/2026)"""
        self._set_param("dth", value)

    def set_source_w(self, value):
        """Set source_w parameter. Author: B.G (02/2026)"""
        self._set_param("source_w", value)

    def set_friction_coeff(self, value):
        """Set friction coefficient parameter. Author: B.G (02/2026)"""
        self._set_param("friction_coeff", value)

    def set_friction_exponent(self, value):
        """Set friction exponent parameter. Author: B.G (02/2026)"""
        self._set_param("friction_exponent", value)

    def set_boundary_h(self, value):
        """Set boundary_h parameter. Author: B.G (02/2026)"""
        self._set_param("boundary_h", value)

    def set_gf_min_increment(self, value):
        """Set minimum GraphFlood increment parameter. Author: B.G (02/2026)"""
        self._set_param("gf_min_increment", value)

    def set_gravity(self, value):
        """Set gravity parameter. Author: B.G (02/2026)"""
        self._set_param("gravity", value)

    def set_rho_w(self, value):
        """Set rho_w parameter. Author: B.G (02/2026)"""
        self._set_param("rho_w", value)

    def set_rho_s(self, value):
        """Set rho_s parameter. Author: B.G (02/2026)"""
        self._set_param("rho_s", value)

    def set_dt_morpho(self, value):
        """Set dt_morpho parameter. Author: B.G (02/2026)"""
        if self.dt_morpho_mode == "n_dthydro":
            raise ValueError("dt_morpho mode is n_dthydro; set dt_morpho_coeff instead")
        self._set_param("dt_morpho", value)

    def set_dt_morpho_coeff(self, value):
        """Set dt_morpho coefficient parameter. Author: B.G (02/2026)"""
        self._set_param("dt_morpho_coeff", value)

    def make_kernel(self, kernel_template, **extra_globals):
        """
        Specialize one generic Taichi kernel against this flood context.

        Author: B.G (02/2026)
        """
        return self.gridctx._specialize_callable(
            kernel_template, ti.kernel, floodctx=self, flowctx=self.flowctx, **extra_globals
        )

    def make_func(self, func_template, **extra_globals):
        """
        Specialize one generic Taichi function against this flood context.

        Author: B.G (02/2026)
        """
        return self.gridctx._specialize_callable(
            func_template, ti.func, floodctx=self, flowctx=self.flowctx, **extra_globals
        )

    def _compile_helpers(self):
        """
        Bind all flood helper functions.

        Author: B.G (02/2026)
        """
        self.tfunc.get_dth = self.make_func(get_dth)
        self.tfunc.get_source_w = self.make_func(get_source_w)
        self.tfunc.source_to_Q = self.make_func(
            source_to_Q, get_source_w=self.tfunc.get_source_w
        )
        self.tfunc.source_to_h = self.make_func(
            source_to_h,
            get_source_w=self.tfunc.get_source_w,
            get_dth=self.tfunc.get_dth,
        )
        self.tfunc.get_friction_coeff = self.make_func(get_friction_coeff)
        self.tfunc.get_friction_exponent = self.make_func(get_friction_exponent)
        self.tfunc.get_boundary_h = self.make_func(get_boundary_h)
        self.tfunc.get_gf_min_increment = self.make_func(get_gf_min_increment)
        self.tfunc.get_gravity = self.make_func(get_gravity)
        self.tfunc.get_rho_w = self.make_func(get_rho_w)
        self.tfunc.get_rho_s = self.make_func(get_rho_s)
        self.tfunc.get_dt_morpho_coeff = self.make_func(get_dt_morpho_coeff)
        self.tfunc.get_dt_morpho = self.make_func(
            get_dt_morpho,
            get_dth=self.tfunc.get_dth,
            get_dt_morpho_coeff=self.tfunc.get_dt_morpho_coeff,
        )
        self.tfunc.compute_u_from_h_slope = self.make_func(
            compute_u_from_h_slope,
            get_friction_coeff=self.tfunc.get_friction_coeff,
            get_friction_exponent=self.tfunc.get_friction_exponent,
        )
        self.tfunc.compute_q_from_h_slope = self.make_func(
            compute_q_from_h_slope, compute_u_from_h_slope=self.tfunc.compute_u_from_h_slope
        )
        self.tfunc.compute_qo_from_h_slope = self.make_func(
            compute_qo_from_h_slope, compute_q_from_h_slope=self.tfunc.compute_q_from_h_slope
        )
        # Stable helper aliases used directly in kernels via floodctx.tfunc.*.
        self.tfunc.dth = self.tfunc.get_dth
        self.tfunc.boundary_h = self.tfunc.get_boundary_h
        self.tfunc.gf_minimum_increment = self.tfunc.get_gf_min_increment
        self.tfunc.gravity = self.tfunc.get_gravity
        self.tfunc.rho_w = self.tfunc.get_rho_w
        self.tfunc.rho_s = self.tfunc.get_rho_s
        self.tfunc.friction_coeff = self.tfunc.get_friction_coeff
        self.tfunc.friction_exponent = self.tfunc.get_friction_exponent
        self.tfunc.dt_morpho = self.tfunc.get_dt_morpho
        self.tfunc.dt_morpho_coeff = self.tfunc.get_dt_morpho_coeff
        self.tfunc.u_from_h_slope = self.tfunc.compute_u_from_h_slope
        self.tfunc.q_from_h_slope = self.tfunc.compute_q_from_h_slope
        self.tfunc.qo_from_h_slope = self.tfunc.compute_qo_from_h_slope

    def _compile_kernels(self):
        """
        Bind GraphFlood and LS kernel families.

        Author: B.G (02/2026)
        """
        self.kernels.graphflood.add_source_to_Q = self.make_kernel(add_source_to_Q_kernel)
        self.kernels.graphflood.add_source_to_h = self.make_kernel(add_source_to_h_kernel)
        self.kernels.graphflood.make_surface = self.make_kernel(make_surface_kernel)
        self.kernels.graphflood.distribute_flow = self.make_kernel(distribute_flow_kernel)
        self.kernels.graphflood.core = self.make_kernel(graphflood_core_kernel)
        self.kernels.graphflood.compute_Qo = self.make_kernel(compute_Qo_kernel)
        self.kernels.graphflood.compute_u = self.make_kernel(compute_u_kernel)
        self.kernels.graphflood.compute_tau = self.make_kernel(compute_tau_kernel)

        self.kernels.ls.add_source_to_h = self.make_kernel(ls_add_source_to_h_kernel)
        self.kernels.ls.flow_route = self.make_kernel(ls_flow_route_kernel)
        self.kernels.ls.depth_update = self.make_kernel(ls_depth_update_kernel)

    def _unwrap_field(self, field_like):
        """
        Return raw Taichi field handle from TPField or field.

        Author: B.G (02/2026)
        """
        return field_like.field if hasattr(field_like, "field") else field_like

    def _require_flat(self, field_like, label):
        """
        Ensure one field is flat and return its raw handle.

        Author: B.G (02/2026)
        """
        field = self._unwrap_field(field_like)
        if len(tuple(field.shape)) != 1:
            raise ValueError(f"{label} must be a flat field")
        return field

    def _alloc_tpfield(self, dtype, shape):
        """
        Allocate one pooled temporary field.

        Author: B.G (02/2026)
        """
        return ppool.taipool.get_tpfield(dtype=dtype, shape=shape)

    def gf_distribute_with_fields(self, z, h, Q_in, Q_next):
        """
        Distribute discharge once with caller-provided output buffer.

        Author: B.G (02/2026)
        """
        self.kernels.graphflood.distribute_flow(
            self._require_flat(z, "z"),
            self._require_flat(h, "h"),
            self._require_flat(Q_in, "Q_in"),
            self._require_flat(Q_next, "Q_next"),
        )

    def gf_distribute(self, z, h, Q):
        """
        Distribute discharge once (in-place on Q) using pooled temporary memory.

        Author: B.G (02/2026)
        """
        q_next = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        try:
            q_field = self._require_flat(Q, "Q")
            self.gf_distribute_with_fields(z, h, Q, q_next)
            q_field.copy_from(q_next.field)
        finally:
            q_next.release()

    def gf_core_with_fields(self, z, h, receivers, Q_in, h_next):
        """
        Run GraphFlood core update with caller-provided temporary field.

        Author: B.G (02/2026)
        """
        self.kernels.graphflood.core(
            self._require_flat(z, "z"),
            self._require_flat(h, "h"),
            self._require_flat(receivers, "receivers"),
            self._require_flat(Q_in, "Q_in"),
            self._require_flat(h_next, "h_next"),
        )
        self._require_flat(h, "h").copy_from(self._require_flat(h_next, "h_next"))

    def gf_core(self, z, h, receivers, Q_in):
        """
        Run GraphFlood core update with pooled temporary field.

        Author: B.G (02/2026)
        """
        h_next = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        try:
            self.gf_core_with_fields(z, h, receivers, Q_in, h_next)
        finally:
            h_next.release()

    def gf_localpass_with_fields(
        self, z, h, receivers, Q, Q_next, h_next, n_iterations: int
    ):
        """
        Execute N local GraphFlood distribute/core passes with caller buffers.

        Author: B.G (02/2026)
        """
        q_field = self._require_flat(Q, "Q")
        q_next_field = self._require_flat(Q_next, "Q_next")
        for _ in range(int(n_iterations)):
            self.gf_distribute_with_fields(z, h, q_field, q_next_field)
            q_field.copy_from(q_next_field)
            self.gf_core_with_fields(z, h, receivers, q_field, h_next)

    def gf_localpass(self, z, h, receivers, Q, n_iterations: int):
        """
        Execute N local GraphFlood distribute/core passes with pooled buffers.

        Author: B.G (02/2026)
        """
        q_next = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        h_next = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        try:
            self.gf_localpass_with_fields(
                z,
                h,
                receivers,
                Q,
                q_next,
                h_next,
                n_iterations=n_iterations,
            )
        finally:
            q_next.release()
            h_next.release()

    def gf_propagate_with_fields(
        self,
        z,
        h,
        receivers,
        Q_out,
        source_q,
        mode: str = "sfd",
        reroute: bool = False,
        fill: bool = False,
        carve: bool = True,
        sfd_buffers: dict | None = None,
        mfd_buffers: dict | None = None,
        reroute_buffers: dict | None = None,
        fill_buffers: dict | None = None,
    ):
        """
        Propagate source-driven Qin field through SFD or MFD routing.

        Author: B.G (02/2026)
        """
        z_field = self._require_flat(z, "z")
        h_field = self._require_flat(h, "h")
        rec_field = self._require_flat(receivers, "receivers")
        q_out_field = self._require_flat(Q_out, "Q_out")
        source_q_field = self._require_flat(source_q, "source_q")

        mode_l = str(mode).lower()
        if mode_l not in {"sfd", "mfd"}:
            raise ValueError("mode must be 'sfd' or 'mfd'")

        source_q_field.fill(0.0)
        self.kernels.graphflood.add_source_to_Q(source_q_field)
        self._accum_flowctx.set_weight(source_q_field)

        if mode_l == "mfd":
            fill = True

        surface_owned = False
        if mfd_buffers is not None and "surface" in mfd_buffers:
            surface_field = self._require_flat(mfd_buffers["surface"], "surface")
        else:
            surface = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
            surface_field = surface.field
            surface_owned = True

        try:
            # All routing-related operations are computed on the water surface.
            self.kernels.graphflood.make_surface(z_field, h_field, surface_field)
            # Fill and reroute require a valid receiver map on the current surface.
            self.flowctx.compute_receivers(surface_field, rec_field)

            # Optional fill applies to the surface and is written back into h.
            if fill:
                if fill_buffers is None:
                    self.flowctx.fill_topography_delta(surface_field, h_field, rec_field)
                else:
                    self._require_flat(fill_buffers["z_ref"], "z_ref").copy_from(surface_field)
                    self.flowctx.fill_topography_delta_with_temps(
                        fill_buffers["z_ref"],
                        h_field,
                        rec_field,
                        fill_buffers["z_work"],
                        fill_buffers["receivers_work"],
                        fill_buffers["receivers_next"],
                    )
                # Surface changed after fill-through-h update.
                self.kernels.graphflood.make_surface(z_field, h_field, surface_field)
                self.flowctx.compute_receivers(surface_field, rec_field)

            if reroute:
                if reroute_buffers is None:
                    rerouted = self.flowctx.reroute_flow(surface_field, rec_field, carve=carve)
                    rerouted.release()
                else:
                    self.flowctx.reroute_flow_with_temps(
                        surface_field,
                        rec_field,
                        reroute_buffers["bid"],
                        reroute_buffers["receivers_work"],
                        reroute_buffers["receivers_jump"],
                        reroute_buffers["z_prime"],
                        reroute_buffers["is_border"],
                        reroute_buffers["outlet"],
                        reroute_buffers["basin_saddle"],
                        reroute_buffers["basin_saddlenode"],
                        reroute_buffers["tag"],
                        reroute_buffers["tag_alt"],
                        reroute_buffers["change"],
                        reroute_buffers["rerouted"],
                        carve=carve,
                    )

            if mode_l == "sfd":
                if sfd_buffers is None:
                    self._accum_flowctx.accumulate_sfd(rec_field, q_out_field)
                else:
                    self._accum_flowctx.accumulate_sfd_with_temps(
                        rec_field,
                        q_out_field,
                        sfd_buffers["donors"],
                        sfd_buffers["ndonors"],
                        sfd_buffers["donors_alt"],
                        sfd_buffers["ndonors_alt"],
                        sfd_buffers["q_alt"],
                        sfd_buffers["src"],
                    )
            else:
                if mfd_buffers is None:
                    self._accum_flowctx.accumulate_mfd(surface_field, q_out_field)
                else:
                    self._accum_flowctx.accumulate_mfd_with_temps(
                        surface_field,
                        q_out_field,
                        mfd_buffers["routing_weights"],
                        mfd_buffers["routing_sum"],
                        mfd_buffers["source"],
                        mfd_buffers["q_tmp"],
                        mfd_buffers["eps"],
                    )
        finally:
            if surface_owned:
                surface.release()

    def gf_propagate(
        self,
        z,
        h,
        receivers,
        Q_out,
        mode: str = "sfd",
        reroute: bool = False,
        fill: bool = False,
        carve: bool = True,
    ):
        """
        Propagate source-driven Qin with pooled helper fields.

        Author: B.G (02/2026)
        """
        source_q = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        try:
            self.gf_propagate_with_fields(
                z,
                h,
                receivers,
                Q_out,
                source_q.field,
                mode=mode,
                reroute=reroute,
                fill=fill,
                carve=carve,
            )
        finally:
            source_q.release()

    def compute_Qo(self, z, h, receivers, Qo=None):
        """
        Compute Qo field from current flood state.

        Author: B.G (02/2026)
        """
        own = Qo is None
        if own:
            Qo = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        self.kernels.graphflood.compute_Qo(
            self._require_flat(z, "z"),
            self._require_flat(h, "h"),
            self._require_flat(receivers, "receivers"),
            self._require_flat(Qo, "Qo"),
        )
        return Qo

    def compute_u(self, z, h, receivers, u=None):
        """
        Compute velocity field from current flood state.

        Author: B.G (02/2026)
        """
        own = u is None
        if own:
            u = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        self.kernels.graphflood.compute_u(
            self._require_flat(z, "z"),
            self._require_flat(h, "h"),
            self._require_flat(receivers, "receivers"),
            self._require_flat(u, "u"),
        )
        return u

    def compute_tau(self, z, h, tau=None):
        """
        Compute shear-stress proxy field from current flood state.

        Author: B.G (02/2026)
        """
        own = tau is None
        if own:
            tau = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        self.kernels.graphflood.compute_tau(
            self._require_flat(z, "z"),
            self._require_flat(h, "h"),
            self._require_flat(tau, "tau"),
        )
        return tau

    def ls_run_with_fields(self, z, h, qx, qy, n_iterations: int):
        """
        Run LisFlood iterations using caller-provided discharge buffers.

        Only available for D4 topology and Manning friction.

        Author: B.G (02/2026)
        """
        if self.gridctx.topology != "D4":
            raise RuntimeError("LS mode is only available when topology is D4")
        if self.friction_law != "manning":
            raise RuntimeError("LS mode currently requires friction_law='manning'")

        z_field = self._require_flat(z, "z")
        h_field = self._require_flat(h, "h")
        qx_field = self._require_flat(qx, "qx")
        qy_field = self._require_flat(qy, "qy")

        for _ in range(int(n_iterations)):
            self.kernels.ls.add_source_to_h(h_field)
            self.kernels.ls.flow_route(h_field, z_field, qx_field, qy_field)
            self.kernels.ls.depth_update(h_field, z_field, qx_field, qy_field)

    def ls_run(self, z, h, n_iterations: int):
        """
        Run LisFlood iterations with pooled discharge buffers.

        Author: B.G (02/2026)
        """
        qx = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        qy = self._alloc_tpfield(cte.FLOAT_TYPE_TI, (self.n_flat))
        qx.field.fill(0.0)
        qy.field.fill(0.0)
        try:
            self.ls_run_with_fields(z, h, qx.field, qy.field, n_iterations)
        finally:
            qx.release()
            qy.release()

    def destroy(self):
        """
        Release all pooled resources owned by this context.

        Author: B.G (02/2026)
        """
        for attr in self._tpfield_attrs:
            tpfield = getattr(self, attr, None)
            if tpfield is not None:
                tpfield.release()
                setattr(self, attr, None)

        for name in (
            "dth",
            "source_w",
            "friction_coeff",
            "friction_exponent",
            "boundary_h",
            "gf_min_increment",
            "gravity",
            "rho_w",
            "rho_s",
            "dt_morpho",
            "dt_morpho_coeff",
        ):
            setattr(self, f"{name}_scalar", None)
            setattr(self, f"{name}_field", None)

        if self._owns_accum_flowctx and self._accum_flowctx is not None:
            self._accum_flowctx.destroy()
            self._accum_flowctx = None

    def __del__(self):
        """
        Destructor - best-effort pooled resource cleanup.

        Author: B.G (02/2026)
        """
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass
