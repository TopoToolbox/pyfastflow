import numpy as np

from .. import constants as cte
from ..context import ContextFactory, ContextRef
from ..flow import FlowContext
from ._flood_param_helpers import (
    compute_q_from_h_slope,
    compute_qo_from_h_slope,
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
    source_to_Q,
    source_to_h,
)
from .flood_graphflood_kernels import (
    add_source_to_Q_kernel,
    add_source_to_h_kernel,
    compute_q_direction_kernel,
    compute_Q_direction_kernel,
    compute_q_kernel,
    compute_Qo_kernel,
    compute_Sw_direction_kernel,
    compute_Sw_kernel,
    compute_tau_direction_kernel,
    compute_tau_kernel,
    compute_u_direction_kernel,
    compute_u_kernel,
    distribute_flow_local_kernel,
    graphflood_core_kernel,
    graphflood_core_unsafe_kernel,
    localpass_kernel,
    make_surface_kernel,
)
from .flood_ls_kernels import (
    ls_add_source_to_h_kernel,
    ls_depth_update_kernel,
    ls_flow_route_kernel,
)


class FloodContext:
    """
    Flat flood specialization context.

    This context only owns parameter state plus the compiled helper/kernel API
    specialized against explicit grid and flow dependencies.

    Author: B.G (03/2026)
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
        if gridctx.topology not in {"D4", "D8"}:
            raise ValueError("FloodContext only supports D4 or D8 grid contexts")

        self.gridctx = gridctx
        self.flowctx = flowctx if flowctx is not None else FlowContext(gridctx)
        self.gactx = self.flowctx.gactx
        self.n_flat = self.gridctx.n_flat
        self.source_w_kind = self._normalize_source_kind(source_w_kind)
        self.friction_law = self._normalize_friction_law(friction_law)

        self._factory = ContextFactory(
            self,
            bindings={
                "gridctx": self.gridctx,
                "flowctx": self.flowctx,
                "floodctx": self,
            },
            n_flat=self.n_flat,
        )

        float_params = [
            ("dth", dth_mode, dth),
            ("source_w", source_w_mode, source_w),
            ("friction_coeff", friction_coeff_mode, friction_coeff),
            ("friction_exponent", friction_exponent_mode, friction_exponent),
            ("boundary_h", boundary_h_mode, boundary_h),
            ("gf_min_increment", gf_min_increment_mode, gf_min_increment),
            ("gravity", gravity_mode, gravity),
            ("rho_w", rho_w_mode, rho_w),
            ("rho_s", rho_s_mode, rho_s),
            ("dt_morpho_coeff", dt_morpho_coeff_mode, dt_morpho_coeff),
        ]
        for name, mode, value in float_params:
            self._factory.params.declare(
                name,
                dtype=cte.FLOAT_TYPE_TI,
                mode=mode,
                value=value,
            )
            self._factory.params.bind_setter(name)

        def validate_dt_mode(mode_value):
            return mode_value

        self._factory.params.declare(
            "dt_morpho",
            dtype=cte.FLOAT_TYPE_TI,
            mode=dt_morpho_mode,
            value=dt_morpho,
            extra_modes={"n_dthydro"},
            mode_validator=validate_dt_mode,
        )
        if self.dt_morpho_mode != "n_dthydro":
            self._factory.params.bind_setter("dt_morpho")
        else:
            self.set_dt_morpho = self._set_dt_morpho_blocked
        self._factory.params.bind_setter("dt_morpho_coeff")

        self._factory.compile_block(
            [
                {"target": "tfunc", "name": "get_dth", "template": get_dth, "kind": "func"},
                {
                    "target": "tfunc",
                    "name": "get_source_w",
                    "template": get_source_w,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "source_to_Q",
                    "template": source_to_Q,
                    "kind": "func",
                    "bindings": {"get_source_w": ContextRef("tfunc.get_source_w")},
                },
                {
                    "target": "tfunc",
                    "name": "source_to_h",
                    "template": source_to_h,
                    "kind": "func",
                    "bindings": {
                        "get_source_w": ContextRef("tfunc.get_source_w"),
                        "get_dth": ContextRef("tfunc.get_dth"),
                    },
                },
                {
                    "target": "tfunc",
                    "name": "get_friction_coeff",
                    "template": get_friction_coeff,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "get_friction_exponent",
                    "template": get_friction_exponent,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "get_boundary_h",
                    "template": get_boundary_h,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "get_gf_min_increment",
                    "template": get_gf_min_increment,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "get_gravity",
                    "template": get_gravity,
                    "kind": "func",
                },
                {"target": "tfunc", "name": "get_rho_w", "template": get_rho_w, "kind": "func"},
                {"target": "tfunc", "name": "get_rho_s", "template": get_rho_s, "kind": "func"},
                {
                    "target": "tfunc",
                    "name": "get_dt_morpho_coeff",
                    "template": get_dt_morpho_coeff,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "get_dt_morpho",
                    "template": get_dt_morpho,
                    "kind": "func",
                    "bindings": {
                        "get_dth": ContextRef("tfunc.get_dth"),
                        "get_dt_morpho_coeff": ContextRef("tfunc.get_dt_morpho_coeff"),
                    },
                },
                {
                    "target": "tfunc",
                    "name": "compute_u_from_h_slope",
                    "template": compute_u_from_h_slope,
                    "kind": "func",
                    "bindings": {
                        "get_friction_coeff": ContextRef("tfunc.get_friction_coeff"),
                        "get_friction_exponent": ContextRef("tfunc.get_friction_exponent"),
                    },
                },
                {
                    "target": "tfunc",
                    "name": "compute_q_from_h_slope",
                    "template": compute_q_from_h_slope,
                    "kind": "func",
                    "bindings": {
                        "compute_u_from_h_slope": ContextRef("tfunc.compute_u_from_h_slope"),
                    },
                },
                {
                    "target": "tfunc",
                    "name": "compute_qo_from_h_slope",
                    "template": compute_qo_from_h_slope,
                    "kind": "func",
                    "bindings": {
                        "compute_q_from_h_slope": ContextRef("tfunc.compute_q_from_h_slope"),
                    },
                },
            ]
        )

        self._factory.compile_block(
            [
                {
                    "target": "kernels.graphflood",
                    "name": "add_source_to_Q",
                    "template": add_source_to_Q_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "add_source_to_h",
                    "template": add_source_to_h_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "make_surface",
                    "template": make_surface_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "distribute_flow_local",
                    "template": distribute_flow_local_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "core",
                    "template": graphflood_core_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "core_unsafe",
                    "template": graphflood_core_unsafe_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "localpass",
                    "template": localpass_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_Qo",
                    "template": compute_Qo_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_u",
                    "template": compute_u_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_tau",
                    "template": compute_tau_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_Sw",
                    "template": compute_Sw_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_q",
                    "template": compute_q_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_Sw_direction",
                    "template": compute_Sw_direction_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_u_direction",
                    "template": compute_u_direction_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_tau_direction",
                    "template": compute_tau_direction_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_q_direction",
                    "template": compute_q_direction_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.graphflood",
                    "name": "compute_Q_direction",
                    "template": compute_Q_direction_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.ls",
                    "name": "add_source_to_h",
                    "template": ls_add_source_to_h_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.ls",
                    "name": "flow_route",
                    "template": ls_flow_route_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.ls",
                    "name": "depth_update",
                    "template": ls_depth_update_kernel,
                    "kind": "kernel",
                },
            ]
        )

        self._factory.export(
            {
                "add_source_to_Q": "kernels.graphflood.add_source_to_Q",
                "add_source_to_h": "kernels.graphflood.add_source_to_h",
                "make_surface": "kernels.graphflood.make_surface",
                "distribute_flow_local": "kernels.graphflood.distribute_flow_local",
                "graphflood_core": "kernels.graphflood.core",
                "graphflood_core_unsafe": "kernels.graphflood.core_unsafe",
                "localpass": "kernels.graphflood.localpass",
                "compute_Qo": "kernels.graphflood.compute_Qo",
                "compute_u": "kernels.graphflood.compute_u",
                "compute_tau": "kernels.graphflood.compute_tau",
                "compute_Sw": "kernels.graphflood.compute_Sw",
                "compute_q": "kernels.graphflood.compute_q",
                "compute_Sw_direction": "kernels.graphflood.compute_Sw_direction",
                "compute_u_direction": "kernels.graphflood.compute_u_direction",
                "compute_tau_direction": "kernels.graphflood.compute_tau_direction",
                "compute_q_direction": "kernels.graphflood.compute_q_direction",
                "compute_Q_direction": "kernels.graphflood.compute_Q_direction",
                "ls_add_source_to_h": "kernels.ls.add_source_to_h",
                "ls_flow_route": "kernels.ls.flow_route",
                "ls_depth_update": "kernels.ls.depth_update",
            }
        )

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

    def _normalize_source_kind(self, value):
        text = str(value)
        kind = text.lower()
        if text == "Q":
            return "Q"
        if kind not in {"q", "precip"}:
            raise ValueError("source_w_kind must be one of: 'Q', 'q', 'precip'")
        return "q" if kind == "q" else "precip"

    def _normalize_friction_law(self, value):
        law = str(value).lower()
        if law != "manning":
            raise ValueError("Only friction_law='manning' is currently supported")
        return law

    def _set_dt_morpho_blocked(self, value):
        raise ValueError("dt_morpho mode is n_dthydro; set dt_morpho_coeff instead")

    def destroy(self):
        """
        Release pooled parameter storage owned by this context.

        Author: B.G (03/2026)
        """
        self._factory.params.destroy()
        if self._owns_accum_flowctx and self._accum_flowctx is not None:
            self._accum_flowctx.destroy()
            self._accum_flowctx = None

    def __del__(self):
        """
        Best-effort pooled resource cleanup.

        Author: B.G (03/2026)
        """
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass
