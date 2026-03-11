from math import isclose
from types import SimpleNamespace

import numpy as np
import taichi as ti

from .. import constants as cte
from .. import pool as ppool
from ..flow import FlowContext
from ._lem_param_helpers import (
    get_K_bedrock,
    get_K_sed,
    get_domain,
    get_dt,
    get_kappa_bedrock,
    get_kappa_sed,
    get_m_exp,
    get_n_exp,
    get_uplift_rate,
)
from .lem_kernels import (
    assemble_hillslope_adi_row_system_kernel,
    build_hillslope_fixed_mask_kernel,
    copy_flat_kernel,
    flat_to_grid_2d_kernel,
    grid_2d_to_flat_kernel,
    init_erode_spl_kernel,
    iteration_erode_spl_kernel,
    solve_cyclic_rows_kernel,
    solve_tridiagonal_rows_kernel,
    tectonic_uplift_kernel,
    transpose_grid_2d_kernel,
    uplift_baselevel_kernel,
)


class LEMContext:
    """
    Grid-bound landscape-evolution context.

    This first cleaned version only binds the uplift helpers and the implicit
    SPL incision path. Future sediment and hillslope pieces can reuse the same
    parameter-helper surface without changing external kernel signatures.

    Author: B.G (02/2026)
    """

    def __init__(
        self,
        gridctx,
        flowctx=None,
        dt_mode: str = "const",
        dt: float = 1.0,
        uplift_rate_mode: str = "const",
        uplift_rate: float = 0.0,
        K_bedrock_mode: str = "const",
        K_bedrock: float = 0.0,
        m_exp_mode: str = "const",
        m_exp: float = 0.5,
        n_exp_mode: str = "const",
        n_exp: float = 1.0,
        K_sed_mode: str = "const",
        K_sed: float = 0.0,
        kappa_bedrock_mode: str = "const",
        kappa_bedrock: float = 0.0,
        kappa_sed_mode: str = "const",
        kappa_sed: float = 0.0,
        domain_mode: str = "const",
        domain: int = 1,
    ):
        """
        Initialize one landscape-evolution context bound to one GridContext.

        Author: B.G (02/2026)
        """
        if hasattr(gridctx, "layout") and str(gridctx.layout).lower() == "2d":
            raise ValueError("LEMContext only supports flat grid logic")

        self.gridctx = gridctx
        self.flowctx = flowctx if flowctx is not None else FlowContext(gridctx)
        self.gactx = self.flowctx.gactx
        self.n_flat = self.gridctx.nx * self.gridctx.ny
        self.logn = self.flowctx.logn

        self._owned_param_fields = []
        self._tpfield_attrs = []

        self._init_float_param("dt", dt_mode, dt)
        self._init_float_param("uplift_rate", uplift_rate_mode, uplift_rate)
        self._init_float_param("K_bedrock", K_bedrock_mode, K_bedrock)
        self._init_float_param("m_exp", m_exp_mode, m_exp)
        self._init_float_param("n_exp", n_exp_mode, n_exp)
        self._init_float_param("K_sed", K_sed_mode, K_sed)
        self._init_float_param("kappa_bedrock", kappa_bedrock_mode, kappa_bedrock)
        self._init_float_param("kappa_sed", kappa_sed_mode, kappa_sed)
        self._init_u8_param("domain", domain_mode, domain)

        self.tfunc = SimpleNamespace()
        self.kernels = SimpleNamespace()
        self._compile_helpers()
        self._compile_kernels()

        self.gridctx.lem = self

    def _normalize_mode(self, value, label):
        """
        Normalize one classic parameter-storage mode.

        Author: B.G (02/2026)
        """
        mode = str(value).lower()
        if mode not in {"const", "scalar", "field"}:
            raise ValueError(f"{label} must be one of: const, scalar, field")
        return mode

    def _register_tpfield(self, attr_name, tpfield):
        """
        Track one pooled parameter field for later cleanup.

        Author: B.G (02/2026)
        """
        self._owned_param_fields.append(tpfield)
        self._tpfield_attrs.append(attr_name)

    def _init_float_param(self, name, mode, value):
        """
        Initialize one floating-point parameter and its storage.

        Author: B.G (02/2026)
        """
        mode_norm = self._normalize_mode(mode, f"{name}_mode")
        setattr(self, f"{name}_mode", mode_norm)
        setattr(self, f"{name}_const", float(value) if mode_norm == "const" else 0.0)
        setattr(self, f"{name}_scalar", None)
        setattr(self, f"{name}_field", None)
        setattr(self, f"_{name}_scalar_tpfield", None)
        setattr(self, f"_{name}_field_tpfield", None)

        if mode_norm == "scalar":
            tpfield = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=())
            setattr(self, f"_{name}_scalar_tpfield", tpfield)
            setattr(self, f"{name}_scalar", tpfield.field)
            self._register_tpfield(f"_{name}_scalar_tpfield", tpfield)
        elif mode_norm == "field":
            tpfield = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
            setattr(self, f"_{name}_field_tpfield", tpfield)
            setattr(self, f"{name}_field", tpfield.field)
            self._register_tpfield(f"_{name}_field_tpfield", tpfield)

        self._set_float_param(name, value)

    def _init_u8_param(self, name, mode, value):
        """
        Initialize one unsigned-byte parameter and its storage.

        Author: B.G (02/2026)
        """
        mode_norm = self._normalize_mode(mode, f"{name}_mode")
        setattr(self, f"{name}_mode", mode_norm)
        setattr(self, f"{name}_const", int(value) if mode_norm == "const" else 0)
        setattr(self, f"{name}_scalar", None)
        setattr(self, f"{name}_field", None)
        setattr(self, f"_{name}_scalar_tpfield", None)
        setattr(self, f"_{name}_field_tpfield", None)

        if mode_norm == "scalar":
            tpfield = ppool.taipool.get_tpfield(dtype=ti.u8, shape=())
            setattr(self, f"_{name}_scalar_tpfield", tpfield)
            setattr(self, f"{name}_scalar", tpfield.field)
            self._register_tpfield(f"_{name}_scalar_tpfield", tpfield)
        elif mode_norm == "field":
            tpfield = ppool.taipool.get_tpfield(dtype=ti.u8, shape=(self.n_flat))
            setattr(self, f"_{name}_field_tpfield", tpfield)
            setattr(self, f"{name}_field", tpfield.field)
            self._register_tpfield(f"_{name}_field_tpfield", tpfield)

        self._set_u8_param(name, value)

    def _copy_flat_float_values(self, values, dst, label):
        """
        Copy flat float values into an internal parameter field.

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

    def _copy_flat_u8_values(self, values, dst, label):
        """
        Copy flat ``u8`` values into an internal parameter field.

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
            arr = np.asarray(values.to_numpy(), dtype=np.uint8)
        else:
            arr = np.asarray(values, dtype=np.uint8)
        arr = arr.reshape(-1)
        if arr.size != self.n_flat:
            raise ValueError(f"{label} expects {self.n_flat} values, got {arr.size}")
        dst.from_numpy(arr)

    def _set_float_param(self, name, value):
        """
        Update one floating-point parameter according to its mode.

        Author: B.G (02/2026)
        """
        mode = getattr(self, f"{name}_mode")
        if mode == "const":
            setattr(self, f"{name}_const", float(value))
        elif mode == "scalar":
            getattr(self, f"{name}_scalar")[None] = float(value)
        else:
            self._copy_flat_float_values(value, getattr(self, f"{name}_field"), name)

    def _set_u8_param(self, name, value):
        """
        Update one unsigned-byte parameter according to its mode.

        Author: B.G (02/2026)
        """
        mode = getattr(self, f"{name}_mode")
        if mode == "const":
            setattr(self, f"{name}_const", int(value))
        elif mode == "scalar":
            getattr(self, f"{name}_scalar")[None] = int(value)
        else:
            self._copy_flat_u8_values(value, getattr(self, f"{name}_field"), name)

    def set_dt(self, value):
        """Update ``dt`` according to its storage mode.

        Author: B.G (02/2026)
        """
        self._set_float_param("dt", value)

    def set_uplift_rate(self, value):
        """Update ``uplift_rate`` according to its storage mode.

        Author: B.G (02/2026)
        """
        self._set_float_param("uplift_rate", value)

    def set_K_bedrock(self, value):
        """Update ``K_bedrock`` according to its storage mode.

        Author: B.G (02/2026)
        """
        self._set_float_param("K_bedrock", value)

    def set_m_exp(self, value):
        """Update ``m_exp`` according to its storage mode.

        Author: B.G (02/2026)
        """
        self._set_float_param("m_exp", value)

    def set_n_exp(self, value):
        """Update ``n_exp`` according to its storage mode.

        Author: B.G (02/2026)
        """
        self._set_float_param("n_exp", value)

    def set_K_sed(self, value):
        """Update ``K_sed`` according to its storage mode.

        Author: B.G (02/2026)
        """
        self._set_float_param("K_sed", value)

    def set_kappa_bedrock(self, value):
        """Update ``kappa_bedrock`` according to its storage mode.

        Author: B.G (02/2026)
        """
        self._set_float_param("kappa_bedrock", value)

    def set_kappa_sed(self, value):
        """Update ``kappa_sed`` according to its storage mode.

        Author: B.G (02/2026)
        """
        self._set_float_param("kappa_sed", value)

    def set_domain(self, value):
        """Update ``domain`` according to its storage mode.

        Author: B.G (02/2026)
        """
        self._set_u8_param("domain", value)

    def make_kernel(self, kernel_template, **extra_globals):
        """
        Specialize one generic Taichi kernel against this context.

        Author: B.G (02/2026)
        """
        return self.gridctx._specialize_callable(
            kernel_template,
            ti.kernel,
            lemctx=self,
            flowctx=self.flowctx,
            **extra_globals,
        )

    def make_func(self, func_template, **extra_globals):
        """
        Specialize one generic Taichi helper against this context.

        Author: B.G (02/2026)
        """
        return self.gridctx._specialize_callable(
            func_template,
            ti.func,
            lemctx=self,
            flowctx=self.flowctx,
            **extra_globals,
        )

    def _compile_helpers(self):
        """
        Bind the unified LEM parameter-helper surface.

        Author: B.G (02/2026)
        """
        self.tfunc.get_dt = self.make_func(get_dt)
        self.tfunc.get_uplift_rate = self.make_func(get_uplift_rate)
        self.tfunc.get_K_bedrock = self.make_func(get_K_bedrock)
        self.tfunc.get_m_exp = self.make_func(get_m_exp)
        self.tfunc.get_n_exp = self.make_func(get_n_exp)
        self.tfunc.get_K_sed = self.make_func(get_K_sed)
        self.tfunc.get_kappa_bedrock = self.make_func(get_kappa_bedrock)
        self.tfunc.get_kappa_sed = self.make_func(get_kappa_sed)
        self.tfunc.get_domain = self.make_func(get_domain)

    def _compile_kernels(self):
        """
        Bind the first landscape-evolution kernel family.

        Author: B.G (02/2026)
        """
        self.kernels.tectonic_uplift = self.make_kernel(
            tectonic_uplift_kernel,
            get_dt=self.tfunc.get_dt,
            get_uplift_rate=self.tfunc.get_uplift_rate,
        )
        self.kernels.uplift_baselevel = self.make_kernel(
            uplift_baselevel_kernel,
            get_dt=self.tfunc.get_dt,
            get_uplift_rate=self.tfunc.get_uplift_rate,
        )
        self.kernels.init_erode_spl = self.make_kernel(
            init_erode_spl_kernel,
            get_dt=self.tfunc.get_dt,
            get_K_bedrock=self.tfunc.get_K_bedrock,
            get_m_exp=self.tfunc.get_m_exp,
        )
        self.kernels.iteration_erode_spl = self.make_kernel(iteration_erode_spl_kernel)
        self.kernels.copy_flat = self.make_kernel(copy_flat_kernel)
        self.kernels.flat_to_grid_2d = self.make_kernel(flat_to_grid_2d_kernel)
        self.kernels.grid_2d_to_flat = self.make_kernel(grid_2d_to_flat_kernel)
        self.kernels.transpose_grid_2d = self.make_kernel(transpose_grid_2d_kernel)
        self.kernels.build_hillslope_fixed_mask = self.make_kernel(
            build_hillslope_fixed_mask_kernel,
            transposed=False,
        )
        self.kernels.build_hillslope_fixed_mask_transposed = self.make_kernel(
            build_hillslope_fixed_mask_kernel,
            transposed=True,
        )
        self.kernels.assemble_hillslope_rows = self.make_kernel(
            assemble_hillslope_adi_row_system_kernel,
            get_dt=self.tfunc.get_dt,
            get_kappa_bedrock=self.tfunc.get_kappa_bedrock,
            transposed=False,
        )
        self.kernels.assemble_hillslope_rows_transposed = self.make_kernel(
            assemble_hillslope_adi_row_system_kernel,
            get_dt=self.tfunc.get_dt,
            get_kappa_bedrock=self.tfunc.get_kappa_bedrock,
            transposed=True,
        )
        self.kernels.solve_hillslope_rows = self.make_kernel(
            solve_tridiagonal_rows_kernel,
            transposed=False,
        )
        self.kernels.solve_hillslope_rows_transposed = self.make_kernel(
            solve_tridiagonal_rows_kernel,
            transposed=True,
        )
        self.kernels.solve_hillslope_rows_cyclic = self.make_kernel(
            solve_cyclic_rows_kernel,
            transposed=False,
        )
        self.kernels.solve_hillslope_rows_cyclic_transposed = self.make_kernel(
            solve_cyclic_rows_kernel,
            transposed=True,
        )

    def _unwrap_field(self, field_like):
        """
        Return the raw Taichi field from a TPField or field handle.

        Author: B.G (02/2026)
        """
        return field_like.field if hasattr(field_like, "field") else field_like

    def _require_flat(self, field_like, label):
        """
        Ensure one input field is flat.

        Author: B.G (02/2026)
        """
        field = self._unwrap_field(field_like)
        if len(tuple(field.shape)) != 1:
            raise ValueError(f"{label} must be a flat field")
        return field

    def _current_scalar_or_const(self, name):
        """
        Return the host-visible value of one const/scalar parameter.

        Author: B.G (02/2026)
        """
        mode = getattr(self, f"{name}_mode")
        if mode == "const":
            return float(getattr(self, f"{name}_const"))
        if mode == "scalar":
            return float(getattr(self, f"{name}_scalar")[None])
        raise ValueError(f"{name} is field-varying and has no single scalar value")

    def _validate_implicit_spl_support(self):
        """
        Validate the subset currently supported by the implicit SPL kernels.

        The present formulation is the cleaned linear legacy path. It supports
        spatially varying ``m_exp`` and ``K_bedrock`` but not a general
        nonlinear slope exponent yet.

        Author: B.G (02/2026)
        """
        if self.n_exp_mode == "field":
            raise ValueError("Implicit SPL does not support field-varying n_exp yet")
        if not isclose(self._current_scalar_or_const("n_exp"), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Implicit SPL currently requires n_exp == 1")

    def run_SPL_with_fields(
        self,
        z,
        area,
        receivers,
        z_work,
        z_aux,
        alpha,
        alpha_aux,
        rec_work,
        rec_aux,
        n_iterations: int = 1,
        reroute: bool = False,
        fill: bool = False,
        carve: bool = True,
        bid=None,
        receivers_jump=None,
        z_prime=None,
        is_border=None,
        outlet=None,
        basin_saddle=None,
        basin_saddlenode=None,
        tag=None,
        tag_alt=None,
        change=None,
        rerouted=None,
        fill_z_work=None,
        fill_receivers_work=None,
        fill_receivers_next=None,
    ):
        """
        Run ``n_iterations`` of uplift plus implicit SPL using caller temps.

        Workflow per iteration:
        1. compute receivers on the current topography,
        2. optionally reroute and/or fill the landscape before routing,
        3. accumulate drainage area with the active FlowContext weighting mode,
        4. apply tectonic uplift away from outlets,
        5. solve one implicit SPL sweep with pointer jumping.

        Author: B.G (02/2026)
        """
        self._validate_implicit_spl_support()

        z_field = self._require_flat(z, "z")
        area_field = self._require_flat(area, "area")
        rec_field = self._require_flat(receivers, "receivers")
        z_work_field = self._require_flat(z_work, "z_work")
        z_aux_field = self._require_flat(z_aux, "z_aux")
        alpha_field = self._require_flat(alpha, "alpha")
        alpha_aux_field = self._require_flat(alpha_aux, "alpha_aux")
        rec_work_field = self._require_flat(rec_work, "rec_work")
        rec_aux_field = self._require_flat(rec_aux, "rec_aux")

        if reroute:
            if any(
                value is None
                for value in (
                    bid,
                    receivers_jump,
                    z_prime,
                    is_border,
                    outlet,
                    basin_saddle,
                    basin_saddlenode,
                    tag,
                    tag_alt,
                    change,
                    rerouted,
                )
            ):
                raise ValueError("reroute=True requires all reroute temp fields")
            bid_field = self._require_flat(bid, "bid")
            receivers_jump_field = self._require_flat(receivers_jump, "receivers_jump")
            z_prime_field = self._require_flat(z_prime, "z_prime")
            is_border_field = self._require_flat(is_border, "is_border")
            outlet_field = self._require_flat(outlet, "outlet")
            basin_saddle_field = self._require_flat(basin_saddle, "basin_saddle")
            basin_saddlenode_field = self._require_flat(
                basin_saddlenode, "basin_saddlenode"
            )
            tag_field = self._require_flat(tag, "tag")
            tag_alt_field = self._require_flat(tag_alt, "tag_alt")
            change_field = self._unwrap_field(change)
            rerouted_field = self._require_flat(rerouted, "rerouted")

        if fill:
            if any(value is None for value in (fill_z_work, fill_receivers_work, fill_receivers_next)):
                raise ValueError("fill=True requires all fill temp fields")
            fill_z_work_field = self._require_flat(fill_z_work, "fill_z_work")
            fill_receivers_work_field = self._require_flat(
                fill_receivers_work, "fill_receivers_work"
            )
            fill_receivers_next_field = self._require_flat(
                fill_receivers_next, "fill_receivers_next"
            )

        for _ in range(int(n_iterations)):
            self.flowctx.compute_receivers(z_field, rec_field)
            if reroute:
                self.flowctx.reroute_flow_with_temps(
                    z_field,
                    rec_field,
                    bid_field,
                    rec_work_field,
                    receivers_jump_field,
                    z_prime_field,
                    is_border_field,
                    outlet_field,
                    basin_saddle_field,
                    basin_saddlenode_field,
                    tag_field,
                    tag_alt_field,
                    change_field,
                    rerouted_field,
                    carve=carve,
                )
            if fill:
                self.flowctx.fill_topography_inplace_with_temps(
                    z_field,
                    rec_field,
                    fill_z_work_field,
                    fill_receivers_work_field,
                    fill_receivers_next_field,
                )
            self.flowctx.accumulate_sfd(rec_field, area_field)
            self.kernels.tectonic_uplift(z_field)
            self.kernels.init_erode_spl(
                z_field,
                z_work_field,
                z_aux_field,
                alpha_field,
                alpha_aux_field,
                area_field,
                rec_field,
            )
            rec_work_field.copy_from(rec_field)
            rec_aux_field.copy_from(rec_field)
            for _ in range(self.logn):
                self.kernels.iteration_erode_spl(
                    z_work_field,
                    z_aux_field,
                    rec_work_field,
                    rec_aux_field,
                    alpha_field,
                    alpha_aux_field,
                )
            z_field.copy_from(z_work_field)

    def run_SPL(
        self,
        z,
        n_iterations: int = 1,
        reroute: bool = False,
        fill: bool = False,
        carve: bool = True,
    ):
        """
        Run the cleaned uplift + implicit SPL loop with pooled temporaries.

        Author: B.G (02/2026)
        """
        area = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        receivers = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
        z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        z_aux = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        alpha = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        alpha_aux = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        rec_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
        rec_aux = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))

        bid = None
        receivers_jump = None
        z_prime = None
        is_border = None
        outlet = None
        basin_saddle = None
        basin_saddlenode = None
        tag = None
        tag_alt = None
        change = None
        rerouted = None

        fill_z_work = None
        fill_receivers_work = None
        fill_receivers_next = None

        if reroute:
            bid = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
            receivers_jump = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
            z_prime = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
            is_border = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(self.n_flat))
            outlet = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(self.n_flat))
            basin_saddle = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(self.n_flat))
            basin_saddlenode = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
            tag = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(self.n_flat))
            tag_alt = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(self.n_flat))
            change = ppool.taipool.get_tpfield(dtype=ti.i32, shape=())
            rerouted = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(self.n_flat))

        if fill:
            fill_z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
            fill_receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
            fill_receivers_next = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))

        try:
            self.run_SPL_with_fields(
                z,
                area,
                receivers,
                z_work,
                z_aux,
                alpha,
                alpha_aux,
                rec_work,
                rec_aux,
                n_iterations=n_iterations,
                reroute=reroute,
                fill=fill,
                carve=carve,
                bid=bid,
                receivers_jump=receivers_jump,
                z_prime=z_prime,
                is_border=is_border,
                outlet=outlet,
                basin_saddle=basin_saddle,
                basin_saddlenode=basin_saddlenode,
                tag=tag,
                tag_alt=tag_alt,
                change=change,
                rerouted=rerouted,
                fill_z_work=fill_z_work,
                fill_receivers_work=fill_receivers_work,
                fill_receivers_next=fill_receivers_next,
            )
        finally:
            area.release()
            receivers.release()
            z_work.release()
            z_aux.release()
            alpha.release()
            alpha_aux.release()
            rec_work.release()
            rec_aux.release()
            if reroute:
                bid.release()
                receivers_jump.release()
                z_prime.release()
                is_border.release()
                outlet.release()
                basin_saddle.release()
                basin_saddlenode.release()
                tag.release()
                tag_alt.release()
                change.release()
                rerouted.release()
            if fill:
                fill_z_work.release()
                fill_receivers_work.release()
                fill_receivers_next.release()

    def _run_linear_hillslope_diffusion_with_fields(
        self,
        z,
        z_grid,
        z_half,
        z_transposed,
        z_transposed_out,
        fixed_mask,
        fixed_mask_t,
        row_a,
        row_b,
        row_c,
        row_rhs,
        row_cp,
        row_dp,
        row_y,
        row_z,
        col_a,
        col_b,
        col_c,
        col_rhs,
        col_cp,
        col_dp,
        col_y,
        col_z,
        n_iterations: int = 1,
    ):
        """
        Execute linear hillslope diffusion with caller-provided ADI buffers.

        The bedrock diffusivity comes from ``kappa_bedrock`` and the timestep
        from ``dt``. Fixed cells are rebuilt from the active grid geometry,
        outlet mask, and optional nodata mask.

        Author: B.G (02/2026)
        """
        z_field = self._require_flat(z, "z")
        self.kernels.flat_to_grid_2d(z_field, self._unwrap_field(z_grid))
        self.kernels.build_hillslope_fixed_mask(self._unwrap_field(fixed_mask))
        self.kernels.transpose_grid_2d(
            self._unwrap_field(fixed_mask), self._unwrap_field(fixed_mask_t)
        )

        for _ in range(int(n_iterations)):
            self.kernels.assemble_hillslope_rows(
                self._unwrap_field(z_grid),
                self._unwrap_field(fixed_mask),
                self._unwrap_field(row_a),
                self._unwrap_field(row_b),
                self._unwrap_field(row_c),
                self._unwrap_field(row_rhs),
            )
            if self.gridctx.boundary_mode == "periodic_EW":
                self.kernels.solve_hillslope_rows_cyclic(
                    self._unwrap_field(row_a),
                    self._unwrap_field(row_b),
                    self._unwrap_field(row_c),
                    self._unwrap_field(row_rhs),
                    self._unwrap_field(row_cp),
                    self._unwrap_field(row_dp),
                    self._unwrap_field(row_y),
                    self._unwrap_field(row_z),
                    self._unwrap_field(z_half),
                )
            else:
                self.kernels.solve_hillslope_rows(
                    self._unwrap_field(row_a),
                    self._unwrap_field(row_b),
                    self._unwrap_field(row_c),
                    self._unwrap_field(row_rhs),
                    self._unwrap_field(row_cp),
                    self._unwrap_field(row_dp),
                    self._unwrap_field(z_half),
                )

            self.kernels.transpose_grid_2d(
                self._unwrap_field(z_half), self._unwrap_field(z_transposed)
            )
            self.kernels.assemble_hillslope_rows_transposed(
                self._unwrap_field(z_transposed),
                self._unwrap_field(fixed_mask_t),
                self._unwrap_field(col_a),
                self._unwrap_field(col_b),
                self._unwrap_field(col_c),
                self._unwrap_field(col_rhs),
            )
            if self.gridctx.boundary_mode == "periodic_NS":
                self.kernels.solve_hillslope_rows_cyclic_transposed(
                    self._unwrap_field(col_a),
                    self._unwrap_field(col_b),
                    self._unwrap_field(col_c),
                    self._unwrap_field(col_rhs),
                    self._unwrap_field(col_cp),
                    self._unwrap_field(col_dp),
                    self._unwrap_field(col_y),
                    self._unwrap_field(col_z),
                    self._unwrap_field(z_transposed_out),
                )
            else:
                self.kernels.solve_hillslope_rows_transposed(
                    self._unwrap_field(col_a),
                    self._unwrap_field(col_b),
                    self._unwrap_field(col_c),
                    self._unwrap_field(col_rhs),
                    self._unwrap_field(col_cp),
                    self._unwrap_field(col_dp),
                    self._unwrap_field(z_transposed_out),
                )

            self.kernels.transpose_grid_2d(
                self._unwrap_field(z_transposed_out), self._unwrap_field(z_grid)
            )

        self.kernels.grid_2d_to_flat(self._unwrap_field(z_grid), z_field)

    def run_SPL_hillslope(
        self,
        z,
        n_iterations: int = 1,
        hillslope_substeps: int = 1,
        reroute: bool = False,
        fill: bool = False,
        carve: bool = True,
    ):
        """
        Run coupled linear hillslope diffusion and implicit SPL.

        Each outer iteration applies ``hillslope_substeps`` ADI diffusion steps
        followed by one uplift + implicit SPL step. All temporary buffers are
        allocated once for the full run.

        Author: B.G (02/2026)
        """
        area = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        receivers = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
        z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        z_aux = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        alpha = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        alpha_aux = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        rec_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
        rec_aux = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))

        bid = None
        receivers_jump = None
        z_prime = None
        is_border = None
        outlet = None
        basin_saddle = None
        basin_saddlenode = None
        tag = None
        tag_alt = None
        change = None
        rerouted = None

        fill_z_work = None
        fill_receivers_work = None
        fill_receivers_next = None

        if reroute:
            bid = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
            receivers_jump = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
            z_prime = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
            is_border = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(self.n_flat))
            outlet = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(self.n_flat))
            basin_saddle = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(self.n_flat))
            basin_saddlenode = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
            tag = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(self.n_flat))
            tag_alt = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(self.n_flat))
            change = ppool.taipool.get_tpfield(dtype=ti.i32, shape=())
            rerouted = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(self.n_flat))

        if fill:
            fill_z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
            fill_receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
            fill_receivers_next = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))

        z_grid = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        z_half = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        z_transposed = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        z_transposed_out = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        fixed_mask = ppool.taipool.get_tpfield(
            dtype=ti.u8, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        fixed_mask_t = ppool.taipool.get_tpfield(
            dtype=ti.u8, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        row_a = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        row_b = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        row_c = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        row_rhs = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        row_cp = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        row_dp = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        row_y = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        row_z = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx)
        )
        col_a = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        col_b = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        col_c = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        col_rhs = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        col_cp = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        col_dp = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        col_y = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )
        col_z = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx, self.gridctx.ny)
        )

        try:
            for _ in range(int(n_iterations)):
                self._run_linear_hillslope_diffusion_with_fields(
                    z,
                    z_grid,
                    z_half,
                    z_transposed,
                    z_transposed_out,
                    fixed_mask,
                    fixed_mask_t,
                    row_a,
                    row_b,
                    row_c,
                    row_rhs,
                    row_cp,
                    row_dp,
                    row_y,
                    row_z,
                    col_a,
                    col_b,
                    col_c,
                    col_rhs,
                    col_cp,
                    col_dp,
                    col_y,
                    col_z,
                    n_iterations=hillslope_substeps,
                )
                self.run_SPL_with_fields(
                    z,
                    area,
                    receivers,
                    z_work,
                    z_aux,
                    alpha,
                    alpha_aux,
                    rec_work,
                    rec_aux,
                    n_iterations=1,
                    reroute=reroute,
                    fill=fill,
                    carve=carve,
                    bid=bid,
                    receivers_jump=receivers_jump,
                    z_prime=z_prime,
                    is_border=is_border,
                    outlet=outlet,
                    basin_saddle=basin_saddle,
                    basin_saddlenode=basin_saddlenode,
                    tag=tag,
                    tag_alt=tag_alt,
                    change=change,
                    rerouted=rerouted,
                    fill_z_work=fill_z_work,
                    fill_receivers_work=fill_receivers_work,
                    fill_receivers_next=fill_receivers_next,
                )
        finally:
            area.release()
            receivers.release()
            z_work.release()
            z_aux.release()
            alpha.release()
            alpha_aux.release()
            rec_work.release()
            rec_aux.release()
            if reroute:
                bid.release()
                receivers_jump.release()
                z_prime.release()
                is_border.release()
                outlet.release()
                basin_saddle.release()
                basin_saddlenode.release()
                tag.release()
                tag_alt.release()
                change.release()
                rerouted.release()
            if fill:
                fill_z_work.release()
                fill_receivers_work.release()
                fill_receivers_next.release()
            z_grid.release()
            z_half.release()
            z_transposed.release()
            z_transposed_out.release()
            fixed_mask.release()
            fixed_mask_t.release()
            row_a.release()
            row_b.release()
            row_c.release()
            row_rhs.release()
            row_cp.release()
            row_dp.release()
            row_y.release()
            row_z.release()
            col_a.release()
            col_b.release()
            col_c.release()
            col_rhs.release()
            col_cp.release()
            col_dp.release()
            col_y.release()
            col_z.release()

    def destroy(self):
        """
        Release pooled parameter storages owned by this context.

        Author: B.G (02/2026)
        """
        while self._owned_param_fields:
            tpfield = self._owned_param_fields.pop()
            tpfield.release()
        while self._tpfield_attrs:
            setattr(self, self._tpfield_attrs.pop(), None)

    def __del__(self):
        """
        Destructor - release pooled parameter storage when possible.

        Author: B.G (02/2026)
        """
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass
