from math import ceil, log2

import taichi as ti

from .. import constants as cte
from ..context import ContextFactory, ContextRef
from ..general_algorithms import GAContext
from ._flow_param_helpers import (
    dist_between_nodes_corrected,
    dist_from_k_corrected,
    get_min_slope,
    get_weight,
    slope_between_nodes,
    slope_from_values_k,
)
from .flow_analysis_kernels import (
    monitor_lm_z_kernel,
    monitor_lm_zh_kernel,
    sum_at_can_out_kernel,
)
from .flow_fill_kernels import (
    apply_fill_delta_kernel,
    fill_h_epsilon_kernel,
    fill_topography_step_kernel,
    solve_lm_z_kernel,
    solve_lm_zh_kernel,
)
from .flow_mfd_kernels import (
    check_mfd_convergence_kernel,
    compute_mfd_routing_weights_kernel,
    init_mfd_source_kernel,
    mfd_power_iteration_step_kernel,
)
from .flow_receivers_kernels import (
    compute_sfd_receivers_kernel,
    compute_sfd_receivers_stochastic_kernel,
)
from .flow_reroute_kernels import (
    basin_id_init_kernel,
    depression_counter_kernel,
    finalise_reroute_carve_kernel,
    init_reroute_carve_kernel,
    iteration_reroute_carve_kernel,
    propagate_basin_final_kernel,
    propagate_basin_iter_kernel,
    reroute_jump_kernel,
    saddlesort_kernel,
)
from .flow_sfd_accum_kernels import (
    fuse_accum_buffers_kernel,
    init_weighted_source_kernel,
    rake_compress_accum_kernel,
    receivers_to_donors_kernel,
)


class FlowContext:
    """
    Flat flow specialization context.

    This context only stores parameter state and compiled helpers/kernels
    specialized against one grid and one general-algorithm context.

    Author: B.G (03/2026)
    """

    def __init__(
        self,
        gridctx,
        gactx=None,
        weight_mode: str = "const",
        weight: float = 1.0,
        min_slope_mode: str = "const",
        min_slope: float = 0.0,
        diagonal_partition_correction: bool = False,
    ):
        if gridctx.topology not in {"D4", "D8"}:
            raise ValueError("FlowContext only supports D4 or D8")

        self.gridctx = gridctx
        self.gactx = gactx if gactx is not None else getattr(gridctx, "ga", None)
        if self.gactx is None:
            self.gactx = GAContext(gridctx)

        self.n_flat = self.gridctx.n_flat
        self.logn = ceil(log2(self.n_flat)) + 1
        self.diagonal_partition_correction = bool(diagonal_partition_correction)

        self._factory = ContextFactory(
            self,
            bindings={
                "gridctx": self.gridctx,
                "flowctx": self,
            },
            n_flat=self.n_flat,
        )

        self._factory.params.declare(
            "weight",
            dtype=cte.FLOAT_TYPE_TI,
            mode=weight_mode,
            value=weight,
        )
        self._factory.params.declare(
            "min_slope",
            dtype=cte.FLOAT_TYPE_TI,
            mode=min_slope_mode,
            value=min_slope,
        )
        self._factory.params.bind_setter("weight")
        self._factory.params.bind_setter("min_slope")

        self._factory.compile_block(
            [
                {"target": "tfunc", "name": "get_weight", "template": get_weight, "kind": "func"},
                {"target": "tfunc", "name": "get_min_slope", "template": get_min_slope, "kind": "func"},
                {
                    "target": "tfunc",
                    "name": "dist_from_k_corrected",
                    "template": dist_from_k_corrected,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "dist_between_nodes_corrected",
                    "template": dist_between_nodes_corrected,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "slope_from_values_k",
                    "template": slope_from_values_k,
                    "kind": "func",
                    "bindings": {
                        "dist_from_k_corrected": ContextRef("tfunc.dist_from_k_corrected"),
                    },
                },
                {
                    "target": "tfunc",
                    "name": "slope_between_nodes",
                    "template": slope_between_nodes,
                    "kind": "func",
                    "bindings": {
                        "dist_between_nodes_corrected": ContextRef(
                            "tfunc.dist_between_nodes_corrected"
                        ),
                    },
                },
            ]
        )

        self._factory.compile_block(
            [
                {
                    "target": "kernels",
                    "name": "compute_sfd_receivers",
                    "template": compute_sfd_receivers_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "compute_sfd_receivers_stochastic",
                    "template": compute_sfd_receivers_stochastic_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "init_weighted_source",
                    "template": init_weighted_source_kernel,
                    "kind": "kernel",
                    "bindings": {"get_weight": ContextRef("tfunc.get_weight")},
                },
                {
                    "target": "kernels",
                    "name": "receivers_to_donors",
                    "template": receivers_to_donors_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "rake_compress_accum",
                    "template": rake_compress_accum_kernel,
                    "kind": "kernel",
                    "bindings": {
                        "get_weight": ContextRef("tfunc.get_weight"),
                        "get_src": self.gactx.tfunc.get_src,
                        "update_src": self.gactx.tfunc.update_src,
                    },
                },
                {
                    "target": "kernels",
                    "name": "fuse_accum_buffers",
                    "template": fuse_accum_buffers_kernel,
                    "kind": "kernel",
                    "bindings": {"get_src": self.gactx.tfunc.get_src},
                },
                {
                    "target": "kernels",
                    "name": "init_mfd_source",
                    "template": init_mfd_source_kernel,
                    "kind": "kernel",
                    "bindings": {"get_weight": ContextRef("tfunc.get_weight")},
                },
                {
                    "target": "kernels",
                    "name": "compute_mfd_routing_weights",
                    "template": compute_mfd_routing_weights_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "mfd_power_iteration_step",
                    "template": mfd_power_iteration_step_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "check_mfd_convergence",
                    "template": check_mfd_convergence_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "fill_topography_step",
                    "template": fill_topography_step_kernel,
                    "kind": "kernel",
                    "bindings": {"get_min_slope": ContextRef("tfunc.get_min_slope")},
                },
                {
                    "target": "kernels",
                    "name": "apply_fill_delta",
                    "template": apply_fill_delta_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "depression_counter",
                    "template": depression_counter_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "basin_id_init",
                    "template": basin_id_init_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "propagate_basin_iter",
                    "template": propagate_basin_iter_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "propagate_basin_final",
                    "template": propagate_basin_final_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "saddlesort",
                    "template": saddlesort_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "reroute_jump",
                    "template": reroute_jump_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "init_reroute_carve",
                    "template": init_reroute_carve_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "iteration_reroute_carve",
                    "template": iteration_reroute_carve_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "finalise_reroute_carve",
                    "template": finalise_reroute_carve_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "sum_at_can_out",
                    "template": sum_at_can_out_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "fill_h_epsilon",
                    "template": fill_h_epsilon_kernel,
                    "kind": "kernel",
                    "bindings": {"get_min_slope": ContextRef("tfunc.get_min_slope")},
                },
                {
                    "target": "kernels",
                    "name": "solve_lm_z",
                    "template": solve_lm_z_kernel,
                    "kind": "kernel",
                    "bindings": {"nextafter": ContextRef("gactx.tfunc.nextafter")},
                },
                {
                    "target": "kernels",
                    "name": "solve_lm_zh",
                    "template": solve_lm_zh_kernel,
                    "kind": "kernel",
                    "bindings": {"nextafter": ContextRef("gactx.tfunc.nextafter")},
                },
                {
                    "target": "kernels",
                    "name": "monitor_lm_z",
                    "template": monitor_lm_z_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "monitor_lm_zh",
                    "template": monitor_lm_zh_kernel,
                    "kind": "kernel",
                },
            ]
        )

        self._factory.export(
            {
                "compute_receivers": "kernels.compute_sfd_receivers",
                "compute_receivers_stochastic": "kernels.compute_sfd_receivers_stochastic",
                "init_weighted_source": "kernels.init_weighted_source",
                "receivers_to_donors": "kernels.receivers_to_donors",
                "rake_compress_accum": "kernels.rake_compress_accum",
                "fuse_accum_buffers": "kernels.fuse_accum_buffers",
                "init_mfd_source": "kernels.init_mfd_source",
                "compute_mfd_routing_weights": "kernels.compute_mfd_routing_weights",
                "mfd_power_iteration_step": "kernels.mfd_power_iteration_step",
                "check_mfd_convergence": "kernels.check_mfd_convergence",
                "fill_topography_step": "kernels.fill_topography_step",
                "apply_fill_delta": "kernels.apply_fill_delta",
                "depression_counter": "kernels.depression_counter",
                "basin_id_init": "kernels.basin_id_init",
                "propagate_basin_iter": "kernels.propagate_basin_iter",
                "propagate_basin_final": "kernels.propagate_basin_final",
                "saddlesort": "kernels.saddlesort",
                "reroute_jump": "kernels.reroute_jump",
                "init_reroute_carve": "kernels.init_reroute_carve",
                "iteration_reroute_carve": "kernels.iteration_reroute_carve",
                "finalise_reroute_carve": "kernels.finalise_reroute_carve",
                "sum_at_can_out": "kernels.sum_at_can_out",
                "fill_h_epsilon": "kernels.fill_h_epsilon",
                "solve_lm_z": "kernels.solve_lm_z",
                "solve_lm_zh": "kernels.solve_lm_zh",
                "monitor_lm_z": "kernels.monitor_lm_z",
                "monitor_lm_zh": "kernels.monitor_lm_zh",
            }
        )

        self.gridctx.flow = self

    def destroy(self):
        """
        Release pooled parameter storage owned by this context.

        Author: B.G (03/2026)
        """
        self._factory.params.destroy()

    def __del__(self):
        """
        Best-effort pooled resource cleanup.

        Author: B.G (03/2026)
        """
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass
