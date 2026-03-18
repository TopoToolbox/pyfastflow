"""
Runtime services for LEM workflows built from ``LEMContext``.

Author: B.G (03/2026)
"""

from math import isclose

import taichi as ti

from .. import constants as cte
from .. import pool as ppool
from ..context import require_flat_field
from ..flow import runtime as flow_runtime


def current_scalar_or_const(lemctx, name):
    """
    Return the host scalar value of one const/scalar parameter.

    Author: B.G (03/2026)
    """
    mode = getattr(lemctx, f"{name}_mode")
    if mode == "const":
        return float(getattr(lemctx, f"{name}_const"))
    if mode == "scalar":
        return float(getattr(lemctx, f"{name}_scalar")[None])
    raise ValueError(f"{name} is field-varying and has no single scalar value")


def validate_implicit_spl_support(lemctx):
    """
    Validate the subset currently supported by the implicit SPL kernels.

    Author: B.G (03/2026)
    """
    if lemctx.n_exp_mode == "field":
        raise ValueError("Implicit SPL does not support field-varying n_exp yet")
    if not isclose(current_scalar_or_const(lemctx, "n_exp"), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Implicit SPL currently requires n_exp == 1")


def run_spl_with_fields(
    lemctx,
    z,
    area,
    receivers,
    z_work,
    z_aux,
    alpha,
    alpha_aux,
    rec_work,
    rec_aux,
    *,
    n_iterations=1,
    reroute=False,
    fill=False,
    carve=True,
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
    Run uplift plus implicit SPL using caller-provided flat buffers.

    Author: B.G (03/2026)
    """
    validate_implicit_spl_support(lemctx)

    z_field = require_flat_field(z, "z")
    area_field = require_flat_field(area, "area")
    rec_field = require_flat_field(receivers, "receivers")
    z_work_field = require_flat_field(z_work, "z_work")
    z_aux_field = require_flat_field(z_aux, "z_aux")
    alpha_field = require_flat_field(alpha, "alpha")
    alpha_aux_field = require_flat_field(alpha_aux, "alpha_aux")
    rec_work_field = require_flat_field(rec_work, "rec_work")
    rec_aux_field = require_flat_field(rec_aux, "rec_aux")

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

    if fill and any(value is None for value in (fill_z_work, fill_receivers_work, fill_receivers_next)):
        raise ValueError("fill=True requires all fill temp fields")

    for _ in range(int(n_iterations)):
        lemctx.flowctx.compute_receivers(z_field, rec_field)
        if reroute:
            flow_runtime.reroute_flow_with_temps(
                lemctx.flowctx,
                z_field,
                rec_field,
                bid,
                rec_work_field,
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
        if fill:
            flow_runtime.fill_topography_inplace_with_temps(
                lemctx.flowctx,
                z_field,
                rec_field,
                fill_z_work,
                fill_receivers_work,
                fill_receivers_next,
            )
        flow_runtime.accumulate_sfd(lemctx.flowctx, rec_field, area_field)
        lemctx.tectonic_uplift(z_field)
        lemctx.init_erode_spl(
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
        for _ in range(lemctx.logn):
            lemctx.iteration_erode_spl(
                z_work_field,
                z_aux_field,
                rec_work_field,
                rec_aux_field,
                alpha_field,
                alpha_aux_field,
            )
        z_field.copy_from(z_work_field)


def run_spl(lemctx, z, *, n_iterations=1, reroute=False, fill=False, carve=True):
    """
    Run uplift plus implicit SPL with pooled flat buffers.

    Author: B.G (03/2026)
    """
    area = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    receivers = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
    z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    z_aux = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    alpha = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    alpha_aux = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    rec_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
    rec_aux = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))

    bid = receivers_jump = z_prime = is_border = None
    outlet = basin_saddle = basin_saddlenode = tag = tag_alt = change = rerouted = None
    fill_z_work = fill_receivers_work = fill_receivers_next = None

    if reroute:
        bid = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
        receivers_jump = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
        z_prime = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
        is_border = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(lemctx.n_flat))
        outlet = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(lemctx.n_flat))
        basin_saddle = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(lemctx.n_flat))
        basin_saddlenode = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
        tag = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(lemctx.n_flat))
        tag_alt = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(lemctx.n_flat))
        change = ppool.taipool.get_tpfield(dtype=ti.i32, shape=())
        rerouted = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(lemctx.n_flat))

    if fill:
        fill_z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
        fill_receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
        fill_receivers_next = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))

    try:
        run_spl_with_fields(
            lemctx,
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


def run_linear_hillslope_diffusion_with_fields(
    lemctx,
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
    *,
    n_iterations=1,
):
    """
    Execute linear hillslope diffusion with explicit 2D runtime buffers.

    Author: B.G (03/2026)
    """
    z_field = require_flat_field(z, "z")
    lemctx.kernels.hillslope.flat_to_grid(z_field, z_grid)
    lemctx.kernels.hillslope.build_fixed_mask(fixed_mask)
    lemctx.kernels.hillslope.transpose(fixed_mask, fixed_mask_t)

    for _ in range(int(n_iterations)):
        lemctx.kernels.hillslope.assemble_rows(
            z_grid,
            fixed_mask,
            row_a,
            row_b,
            row_c,
            row_rhs,
        )
        if lemctx.gridctx.boundary_mode == "periodic_EW":
            lemctx.kernels.hillslope.solve_rows_cyclic(
                row_a,
                row_b,
                row_c,
                row_rhs,
                row_cp,
                row_dp,
                row_y,
                row_z,
                z_half,
            )
        else:
            lemctx.kernels.hillslope.solve_rows(
                row_a,
                row_b,
                row_c,
                row_rhs,
                row_cp,
                row_dp,
                z_half,
            )

        lemctx.kernels.hillslope.transpose(z_half, z_transposed)
        lemctx.kernels.hillslope.assemble_rows_transposed(
            z_transposed,
            fixed_mask_t,
            col_a,
            col_b,
            col_c,
            col_rhs,
        )
        if lemctx.gridctx.boundary_mode == "periodic_NS":
            lemctx.kernels.hillslope.solve_rows_cyclic_transposed(
                col_a,
                col_b,
                col_c,
                col_rhs,
                col_cp,
                col_dp,
                col_y,
                col_z,
                z_transposed_out,
            )
        else:
            lemctx.kernels.hillslope.solve_rows_transposed(
                col_a,
                col_b,
                col_c,
                col_rhs,
                col_cp,
                col_dp,
                z_transposed_out,
            )

        lemctx.kernels.hillslope.transpose(z_transposed_out, z_grid)

    lemctx.kernels.hillslope.grid_to_flat(z_grid, z_field)


def run_spl_hillslope(
    lemctx,
    z,
    *,
    n_iterations=1,
    hillslope_substeps=1,
    reroute=False,
    fill=False,
    carve=True,
):
    """
    Run coupled hillslope diffusion and SPL with explicit runtime-managed 2D buffers.

    Author: B.G (03/2026)
    """
    area = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    receivers = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
    z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    z_aux = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    alpha = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    alpha_aux = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
    rec_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
    rec_aux = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))

    bid = receivers_jump = z_prime = is_border = None
    outlet = basin_saddle = basin_saddlenode = tag = tag_alt = change = rerouted = None
    fill_z_work = fill_receivers_work = fill_receivers_next = None

    if reroute:
        bid = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
        receivers_jump = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
        z_prime = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
        is_border = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(lemctx.n_flat))
        outlet = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(lemctx.n_flat))
        basin_saddle = ppool.taipool.get_tpfield(dtype=ti.i64, shape=(lemctx.n_flat))
        basin_saddlenode = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
        tag = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(lemctx.n_flat))
        tag_alt = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(lemctx.n_flat))
        change = ppool.taipool.get_tpfield(dtype=ti.i32, shape=())
        rerouted = ppool.taipool.get_tpfield(dtype=ti.u1, shape=(lemctx.n_flat))

    if fill:
        fill_z_work = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(lemctx.n_flat))
        fill_receivers_work = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))
        fill_receivers_next = ppool.taipool.get_tpfield(dtype=ti.i32, shape=(lemctx.n_flat))

    two_d_shape = (lemctx.gridctx.ny, lemctx.gridctx.nx)
    transposed_shape = (lemctx.gridctx.nx, lemctx.gridctx.ny)
    z_grid = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    z_half = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    z_transposed = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)
    z_transposed_out = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)
    fixed_mask = ppool.taipool.get_tpfield(dtype=ti.u8, shape=two_d_shape)
    fixed_mask_t = ppool.taipool.get_tpfield(dtype=ti.u8, shape=transposed_shape)
    row_a = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    row_b = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    row_c = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    row_rhs = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    row_cp = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    row_dp = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    row_y = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    row_z = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=two_d_shape)
    col_a = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)
    col_b = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)
    col_c = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)
    col_rhs = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)
    col_cp = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)
    col_dp = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)
    col_y = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)
    col_z = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=transposed_shape)

    try:
        for _ in range(int(n_iterations)):
            run_linear_hillslope_diffusion_with_fields(
                lemctx,
                z,
                z_grid.field,
                z_half.field,
                z_transposed.field,
                z_transposed_out.field,
                fixed_mask.field,
                fixed_mask_t.field,
                row_a.field,
                row_b.field,
                row_c.field,
                row_rhs.field,
                row_cp.field,
                row_dp.field,
                row_y.field,
                row_z.field,
                col_a.field,
                col_b.field,
                col_c.field,
                col_rhs.field,
                col_cp.field,
                col_dp.field,
                col_y.field,
                col_z.field,
                n_iterations=hillslope_substeps,
            )
            run_spl_with_fields(
                lemctx,
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
