from math import ceil, log2
from types import SimpleNamespace

import numpy as np
import taichi as ti

from .. import constants as cte
from .. import pool as ppool
from ..general_algorithms import GAContext
from ._flow_param_helpers import (
    dist_between_nodes_corrected,
    dist_from_k_corrected,
    get_min_slope,
    get_weight,
    slope_between_nodes,
    slope_from_values_k,
)
from .flow_analysis_kernels import sum_at_can_out_kernel
from .flow_fill_kernels import apply_fill_delta_kernel, fill_topography_step_kernel
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
    Grid-bound context for cleaned flow-routing algorithms.

    This first pass targets the flat D4/D8 path and introduces the unified
    parameter-getter system used to specialize kernels at compile time while
    keeping the external kernel signatures compact.

    Author: B.G (02/2026)
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
        """
        Initialize the flow context and bind the cleaned kernel families.

        Author: B.G (02/2026)
        """
        if hasattr(gridctx, "layout") and str(gridctx.layout).lower() == "2d":
            raise ValueError("FlowContext only supports flat grid logic")
        if gridctx.topology not in {"D4", "D8"}:
            raise ValueError("FlowContext only supports D4 or D8")

        self.gridctx = gridctx
        self.gactx = gactx if gactx is not None else getattr(gridctx, "ga", None)
        if self.gactx is None:
            self.gactx = GAContext(gridctx)

        self.n_flat = self.gridctx.nx * self.gridctx.ny
        self.logn = ceil(log2(self.n_flat)) + 1

        self.weight_mode = self._normalize_mode(weight_mode, "weight_mode")
        self.min_slope_mode = self._normalize_mode(min_slope_mode, "min_slope_mode")
        self.diagonal_partition_correction = bool(diagonal_partition_correction)

        self.weight_const = float(weight) if self.weight_mode == "const" else 0.0
        self.min_slope_const = float(min_slope) if self.min_slope_mode == "const" else 0.0

        self._weight_scalar_tpfield = None
        self.weight_scalar = None
        self._weight_field_tpfield = None
        self.weight_field = None

        self._min_slope_scalar_tpfield = None
        self.min_slope_scalar = None
        self._min_slope_field_tpfield = None
        self.min_slope_field = None

        self._allocate_param_storage()
        self.set_weight(weight)
        self.set_min_slope(min_slope)

        self.tfunc = SimpleNamespace()
        self.kernels = SimpleNamespace()
        self._compile_helpers()
        self._compile_kernels()

        # Make the context reachable from the bound grid context for kernels
        # that need to see it indirectly.
        self.gridctx.flow = self

    def _normalize_mode(self, value, label):
        """
        Validate and normalize one parameter storage mode.

        Author: B.G (02/2026)
        """
        mode = str(value).lower()
        if mode not in {"const", "scalar", "field"}:
            raise ValueError(f"{label} must be one of: const, scalar, field")
        return mode

    def _allocate_param_storage(self):
        """
        Allocate the internal parameter storages required by the active modes.

        Author: B.G (02/2026)
        """
        if self.weight_mode == "scalar":
            self._weight_scalar_tpfield = ppool.taipool.get_tpfield(
                dtype=cte.FLOAT_TYPE_TI, shape=()
            )
            self.weight_scalar = self._weight_scalar_tpfield.field
        elif self.weight_mode == "field":
            self._weight_field_tpfield = ppool.taipool.get_tpfield(
                dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat)
            )
            self.weight_field = self._weight_field_tpfield.field

        if self.min_slope_mode == "scalar":
            self._min_slope_scalar_tpfield = ppool.taipool.get_tpfield(
                dtype=cte.FLOAT_TYPE_TI, shape=()
            )
            self.min_slope_scalar = self._min_slope_scalar_tpfield.field
        elif self.min_slope_mode == "field":
            self._min_slope_field_tpfield = ppool.taipool.get_tpfield(
                dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat)
            )
            self.min_slope_field = self._min_slope_field_tpfield.field

    def _copy_flat_values(self, values, dst, label):
        """
        Copy flat numeric values into one internal flat field.

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

    def set_weight(self, value):
        """
        Update the configured weight storage.

        In ``const`` mode this updates the Python-side constant. In ``scalar``
        and ``field`` modes it updates the internal GPU storages.

        Author: B.G (02/2026)
        """
        if self.weight_mode == "const":
            self.weight_const = float(value)
        elif self.weight_mode == "scalar":
            self.weight_scalar[None] = float(value)
        else:
            self._copy_flat_values(value, self.weight_field, "weight field")

    def set_min_slope(self, value):
        """
        Update the configured minimum-slope storage.

        Author: B.G (02/2026)
        """
        if self.min_slope_mode == "const":
            self.min_slope_const = float(value)
        elif self.min_slope_mode == "scalar":
            self.min_slope_scalar[None] = float(value)
        else:
            self._copy_flat_values(value, self.min_slope_field, "min_slope field")

    def make_kernel(self, kernel_template, **extra_globals):
        """
        Specialize a generic Taichi kernel against this flow context.

        Author: B.G (02/2026)
        """
        return self.gridctx._specialize_callable(
            kernel_template,
            ti.kernel,
            flowctx=self,
            **extra_globals,
        )

    def make_func(self, func_template, **extra_globals):
        """
        Specialize a generic Taichi function against this flow context.

        Author: B.G (02/2026)
        """
        return self.gridctx._specialize_callable(
            func_template,
            ti.func,
            flowctx=self,
            **extra_globals,
        )

    def _compile_helpers(self):
        """
        Bind the unified parameter getter helpers.

        Author: B.G (02/2026)
        """
        self.tfunc.get_weight = self.make_func(get_weight)
        self.tfunc.get_min_slope = self.make_func(get_min_slope)
        self.tfunc.dist_from_k_corrected = self.make_func(dist_from_k_corrected)
        self.tfunc.dist_between_nodes_corrected = self.make_func(
            dist_between_nodes_corrected
        )
        self.tfunc.slope_from_values_k = self.make_func(
            slope_from_values_k,
            dist_from_k_corrected=self.tfunc.dist_from_k_corrected,
        )
        self.tfunc.slope_between_nodes = self.make_func(
            slope_between_nodes,
            dist_between_nodes_corrected=self.tfunc.dist_between_nodes_corrected,
        )

    def _compile_kernels(self):
        """
        Bind the cleaned flow kernel families.

        Author: B.G (02/2026)
        """
        self.kernels.compute_sfd_receivers = self.make_kernel(compute_sfd_receivers_kernel)
        self.kernels.compute_sfd_receivers_stochastic = self.make_kernel(
            compute_sfd_receivers_stochastic_kernel
        )

        self.kernels.init_weighted_source = self.make_kernel(
            init_weighted_source_kernel,
            get_weight=self.tfunc.get_weight,
        )
        self.kernels.receivers_to_donors = self.make_kernel(receivers_to_donors_kernel)
        self.kernels.rake_compress_accum = self.make_kernel(
            rake_compress_accum_kernel,
            get_weight=self.tfunc.get_weight,
            get_src=self.gactx.tfunc.get_src,
            update_src=self.gactx.tfunc.update_src,
        )
        self.kernels.fuse_accum_buffers = self.make_kernel(
            fuse_accum_buffers_kernel,
            get_src=self.gactx.tfunc.get_src,
        )

        self.kernels.init_mfd_source = self.make_kernel(
            init_mfd_source_kernel,
            get_weight=self.tfunc.get_weight,
        )
        self.kernels.compute_mfd_routing_weights = self.make_kernel(
            compute_mfd_routing_weights_kernel
        )
        self.kernels.mfd_power_iteration_step = self.make_kernel(
            mfd_power_iteration_step_kernel
        )
        self.kernels.check_mfd_convergence = self.make_kernel(check_mfd_convergence_kernel)

        self.kernels.fill_topography_step = self.make_kernel(
            fill_topography_step_kernel,
            get_min_slope=self.tfunc.get_min_slope,
        )
        self.kernels.apply_fill_delta = self.make_kernel(apply_fill_delta_kernel)

        self.kernels.depression_counter = self.make_kernel(depression_counter_kernel)
        self.kernels.basin_id_init = self.make_kernel(basin_id_init_kernel)
        self.kernels.propagate_basin_iter = self.make_kernel(propagate_basin_iter_kernel)
        self.kernels.propagate_basin_final = self.make_kernel(propagate_basin_final_kernel)
        self.kernels.saddlesort = self.make_kernel(saddlesort_kernel)
        self.kernels.reroute_jump = self.make_kernel(reroute_jump_kernel)
        self.kernels.init_reroute_carve = self.make_kernel(init_reroute_carve_kernel)
        self.kernels.iteration_reroute_carve = self.make_kernel(iteration_reroute_carve_kernel)
        self.kernels.finalise_reroute_carve = self.make_kernel(finalise_reroute_carve_kernel)
        self.kernels.sum_at_can_out = self.make_kernel(sum_at_can_out_kernel)

    def _unwrap_field(self, field_like):
        """
        Return the raw Taichi field from a TPField or field handle.

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

    def compute_receivers(self, z, receivers):
        """
        Compute deterministic SFD receivers into ``receivers``.

        Author: B.G (02/2026)
        """
        z_field = self._require_flat(z, "z")
        rec_field = self._require_flat(receivers, "receivers")
        self.kernels.compute_sfd_receivers(z_field, rec_field)

    def compute_receivers_stochastic(self, z, receivers):
        """
        Compute stochastic SFD receivers into ``receivers``.

        Author: B.G (02/2026)
        """
        z_field = self._require_flat(z, "z")
        rec_field = self._require_flat(receivers, "receivers")
        self.kernels.compute_sfd_receivers_stochastic(z_field, rec_field)

    def sum_at_can_out_with_accumulator(self, field, out_sum):
        """
        Sum one field over outlet nodes using a caller-provided scalar field.

        ``out_sum`` must be a 0D scalar Taichi field (or TPField wrapper).

        Author: B.G (02/2026)
        """
        field_handle = self._require_flat(field, "field")
        out_handle = self._unwrap_field(out_sum)
        if tuple(out_handle.shape) != ():
            raise ValueError("out_sum must be a 0D scalar field")
        self.kernels.sum_at_can_out(field_handle, out_handle)

    def sum_at_can_out(self, field) -> float:
        """
        Sum one field over outlet nodes and return the scalar value.

        Author: B.G (02/2026)
        """
        out_sum = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=())
        try:
            self.sum_at_can_out_with_accumulator(field, out_sum.field)
            return float(out_sum.field[None])
        finally:
            out_sum.release()

    def accumulate_sfd_with_temps(
        self,
        receivers,
        q,
        donors,
        ndonors,
        donors_alt,
        ndonors_alt,
        q_alt,
        src,
    ):
        """
        Execute D4 SFD downstream accumulation using caller-provided temps.

        Author: B.G (02/2026)
        """
        rec_field = self._require_flat(receivers, "receivers")
        q_field = self._require_flat(q, "q")
        donors_field = self._require_flat(donors, "donors")
        ndonors_field = self._require_flat(ndonors, "ndonors")
        donors_alt_field = self._require_flat(donors_alt, "donors_alt")
        ndonors_alt_field = self._require_flat(ndonors_alt, "ndonors_alt")
        q_alt_field = self._require_flat(q_alt, "q_alt")
        src_field = self._require_flat(src, "src")

        ndonors_field.fill(0)
        ndonors_alt_field.fill(0)
        src_field.fill(0)

        self.kernels.init_weighted_source(q_field)
        self.kernels.receivers_to_donors(rec_field, donors_field, ndonors_field)

        for iteration in range(self.logn + 1):
            self.kernels.rake_compress_accum(
                donors_field,
                ndonors_field,
                q_field,
                src_field,
                donors_alt_field,
                ndonors_alt_field,
                q_alt_field,
                iteration,
            )

        self.kernels.fuse_accum_buffers(q_field, src_field, q_alt_field, self.logn)

    def accumulate_sfd(self, receivers, q):
        """
        Execute D4 SFD downstream accumulation with pooled temps.

        Author: B.G (02/2026)
        """
        donors = ppool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.n_flat * self.gridctx.n_neighbours)
        )
        ndonors = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
        donors_alt = ppool.taipool.get_tpfield(
            dtype=ti.i32, shape=(self.n_flat * self.gridctx.n_neighbours)
        )
        ndonors_alt = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
        q_alt = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        src = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))

        try:
            self.accumulate_sfd_with_temps(
                receivers,
                q,
                donors,
                ndonors,
                donors_alt,
                ndonors_alt,
                q_alt,
                src,
            )
        finally:
            donors.release()
            ndonors.release()
            donors_alt.release()
            ndonors_alt.release()
            q_alt.release()
            src.release()

    def accumulate_mfd_with_temps(
        self,
        z,
        q_out,
        routing_weights,
        routing_sum,
        source,
        q_tmp,
        eps,
        max_iterations: int = 2000,
        tol: float = 1e-6,
        check_interval: int = 20,
    ):
        """
        Execute D4 MFD power-iteration accumulation with caller-provided temps.

        Author: B.G (02/2026)
        """
        z_field = self._require_flat(z, "z")
        q_out_field = self._require_flat(q_out, "q_out")
        routing_weights_field = self._unwrap_field(routing_weights)
        routing_sum_field = self._require_flat(routing_sum, "routing_sum")
        source_field = self._require_flat(source, "source")
        q_tmp_field = self._require_flat(q_tmp, "q_tmp")
        eps_field = self._unwrap_field(eps)

        self.kernels.init_mfd_source(source_field)
        q_out_field.copy_from(source_field)
        q_tmp_field.copy_from(source_field)
        self.kernels.compute_mfd_routing_weights(z_field, routing_weights_field, routing_sum_field)

        for iteration in range(int(max_iterations)):
            self.kernels.mfd_power_iteration_step(
                source_field,
                q_tmp_field,
                routing_weights_field,
                q_out_field,
            )
            if iteration > 0 and iteration % int(check_interval) == 0:
                self.kernels.check_mfd_convergence(q_out_field, q_tmp_field, eps_field)
                if eps_field[None] < tol:
                    break
            q_tmp_field.copy_from(q_out_field)

    def accumulate_mfd(
        self,
        z,
        q_out,
        max_iterations: int = 2000,
        tol: float = 1e-6,
        check_interval: int = 20,
    ):
        """
        Execute D4 MFD power-iteration accumulation with pooled temps.

        Author: B.G (02/2026)
        """
        routing_weights = ppool.taipool.get_tpfield(
            dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat, self.gridctx.n_neighbours)
        )
        routing_sum = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        source = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        q_tmp = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        eps = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=())

        try:
            self.accumulate_mfd_with_temps(
                z,
                q_out,
                routing_weights,
                routing_sum,
                source,
                q_tmp,
                eps,
                max_iterations=max_iterations,
                tol=tol,
                check_interval=check_interval,
            )
        finally:
            routing_weights.release()
            routing_sum.release()
            source.release()
            q_tmp.release()
            eps.release()

    def fill_topography_inplace_with_temps(
        self,
        z,
        receivers,
        z_work,
        receivers_work,
        receivers_next,
    ):
        """
        Fill ``z`` in place using caller-provided temporary arrays.

        Author: B.G (02/2026)
        """
        z_field = self._require_flat(z, "z")
        rec_field = self._require_flat(receivers, "receivers")
        z_work_field = self._require_flat(z_work, "z_work")
        receivers_work_field = self._require_flat(receivers_work, "receivers_work")
        receivers_next_field = self._require_flat(receivers_next, "receivers_next")

        z_work_field.copy_from(z_field)
        receivers_work_field.copy_from(rec_field)
        receivers_next_field.copy_from(rec_field)

        for iteration in range(self.logn):
            self.kernels.fill_topography_step(
                z_field,
                z_work_field,
                receivers_work_field,
                receivers_next_field,
                iteration + 1,
            )

        z_field.copy_from(z_work_field)

    def fill_topography_inplace(self, z, receivers):
        """
        Fill ``z`` in place with pooled temporary arrays.

        Author: B.G (02/2026)
        """
        z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
        receivers_next = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))

        try:
            self.fill_topography_inplace_with_temps(
                z,
                receivers,
                z_work,
                receivers_work,
                receivers_next,
            )
        finally:
            z_work.release()
            receivers_work.release()
            receivers_next.release()

    def fill_topography_delta_with_temps(
        self,
        z,
        surplus,
        receivers,
        z_work,
        receivers_work,
        receivers_next,
    ):
        """
        Fill ``z`` and accumulate the surplus into ``surplus`` with caller temps.

        Author: B.G (02/2026)
        """
        z_field = self._require_flat(z, "z")
        rec_field = self._require_flat(receivers, "receivers")
        surplus_field = self._require_flat(surplus, "surplus")
        z_work_field = self._require_flat(z_work, "z_work")
        receivers_work_field = self._require_flat(receivers_work, "receivers_work")
        receivers_next_field = self._require_flat(receivers_next, "receivers_next")

        z_work_field.copy_from(z_field)
        receivers_work_field.copy_from(rec_field)
        receivers_next_field.copy_from(rec_field)

        for iteration in range(self.logn):
            self.kernels.fill_topography_step(
                z_field,
                z_work_field,
                receivers_work_field,
                receivers_next_field,
                iteration + 1,
            )

        self.kernels.apply_fill_delta(z_field, surplus_field, z_work_field)

    def fill_topography_delta(self, z, surplus, receivers):
        """
        Fill ``z`` and accumulate the fill surplus into ``surplus``.

        Author: B.G (02/2026)
        """
        z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.n_flat))
        receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
        receivers_next = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))

        try:
            self.fill_topography_delta_with_temps(
                z,
                surplus,
                receivers,
                z_work,
                receivers_work,
                receivers_next,
            )
        finally:
            z_work.release()
            receivers_work.release()
            receivers_next.release()

    def reroute_flow_with_temps(
        self,
        z,
        receivers,
        bid,
        receivers_work,
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
        carve: bool = True,
    ):
        """
        Execute the cleaned lake rerouting path with caller-provided temps.

        ``receivers`` is updated in place. ``rerouted`` is filled with the cells
        whose receiver changed during the reroute.

        Author: B.G (02/2026)
        """
        z_field = self._require_flat(z, "z")
        rec_field = self._require_flat(receivers, "receivers")
        bid_field = self._require_flat(bid, "bid")
        rec_work_field = self._require_flat(receivers_work, "receivers_work")
        rec_jump_field = self._require_flat(receivers_jump, "receivers_jump")
        z_prime_field = self._require_flat(z_prime, "z_prime")
        is_border_field = self._require_flat(is_border, "is_border")
        outlet_field = self._require_flat(outlet, "outlet")
        basin_saddle_field = self._require_flat(basin_saddle, "basin_saddle")
        basin_saddlenode_field = self._require_flat(basin_saddlenode, "basin_saddlenode")
        tag_field = self._require_flat(tag, "tag")
        tag_alt_field = self._require_flat(tag_alt, "tag_alt")
        change_field = self._unwrap_field(change)
        rerouted_field = self._require_flat(rerouted, "rerouted")

        del change_field  # reserved for the future convergence-check variant

        rec_work_field.copy_from(rec_field)
        rerouted_field.fill(False)

        ndep = self.kernels.depression_counter(rec_field)
        if ndep == 0:
            return

        ndep_iters = ceil(log2(max(1, int(ndep)))) + 1

        for _ in range(ndep_iters):
            ndep_bis = self.kernels.depression_counter(rec_work_field)
            # print('ndep_bis',ndep_bis)

            self.kernels.basin_id_init(bid_field)
            rec_jump_field.copy_from(rec_work_field)

            for _ in range(self.logn + 1):
                self.kernels.propagate_basin_iter(rec_jump_field)
            self.kernels.propagate_basin_final(bid_field, rec_jump_field)

            if ndep_bis == 0:
                break

            self.kernels.saddlesort(
                bid_field,
                is_border_field,
                z_prime_field,
                basin_saddle_field,
                basin_saddlenode_field,
                outlet_field,
                z_field,
            )

            if carve:
                self.kernels.init_reroute_carve(tag_field, tag_alt_field, basin_saddlenode_field)
                rec_field.copy_from(rec_work_field)
                rec_jump_field.copy_from(rec_work_field)

                for _ in range(self.logn + 1):
                    self.kernels.iteration_reroute_carve(
                        tag_field,
                        tag_alt_field,
                        rec_field,
                        rec_work_field,
                        bid_field,
                    )

                self.kernels.finalise_reroute_carve(
                    rec_field,
                    rec_jump_field,
                    tag_field,
                    basin_saddlenode_field,
                    outlet_field,
                    rerouted_field,
                )
                rec_work_field.copy_from(rec_field)
            else:
                self.kernels.reroute_jump(rec_work_field, outlet_field, rerouted_field)

        rec_field.copy_from(rec_work_field)

    def reroute_flow(self, z, receivers, carve: bool = True):
        """
        Execute the cleaned lake rerouting path with pooled temporary arrays.

        The returned pooled ``rerouted`` mask is owned by the caller and must be
        released by the caller when no longer needed.

        Author: B.G (02/2026)
        """
        bid = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
        receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(self.n_flat))
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

        try:
            self.reroute_flow_with_temps(
                z,
                receivers,
                bid,
                receivers_work,
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
                carve=carve,
            )
        finally:
            bid.release()
            receivers_work.release()
            receivers_jump.release()
            z_prime.release()
            is_border.release()
            outlet.release()
            basin_saddle.release()
            basin_saddlenode.release()
            tag.release()
            tag_alt.release()
            change.release()

        return rerouted

    def destroy(self):
        """
        Release the internal pooled parameter storages owned by this context.

        Author: B.G (02/2026)
        """
        for attr in (
            "_weight_scalar_tpfield",
            "_weight_field_tpfield",
            "_min_slope_scalar_tpfield",
            "_min_slope_field_tpfield",
        ):
            tpfield = getattr(self, attr, None)
            if tpfield is not None:
                tpfield.release()
                setattr(self, attr, None)

        self.weight_scalar = None
        self.weight_field = None
        self.min_slope_scalar = None
        self.min_slope_field = None

    def __del__(self):
        """
        Destructor - best-effort release of pooled resources.

        Author: B.G (02/2026)
        """
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass
