"""
Generic parameter and law helpers for ``FloodContext``.

All getters are specialized at compile time through mode flags captured by the
bound flood context.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None
floodctx = None


@ti.func
def get_dth(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return local hydrodynamic time step. Author: B.G (02/2026)"""
    if ti.static(floodctx.dth_mode == "const"):
        return ti.cast(ti.static(floodctx.dth_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.dth_mode == "scalar"):
        return floodctx.dth_scalar[None]
    return floodctx.dth_field[i]


@ti.func
def get_source_w(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return local source intensity in its configured unit. Author: B.G (02/2026)"""
    if ti.static(floodctx.source_w_mode == "const"):
        return ti.cast(ti.static(floodctx.source_w_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.source_w_mode == "scalar"):
        return floodctx.source_w_scalar[None]
    return floodctx.source_w_field[i]


@ti.func
def source_to_Q(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Convert source intensity to volumetric discharge increment. Author: B.G (02/2026)"""
    s = get_source_w(i)
    dx = ti.cast(ti.static(gridctx.dx), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.source_w_kind == "precip"):
        return s * dx * dx
    if ti.static(floodctx.source_w_kind == "q"):
        return s * dx
    return s


@ti.func
def source_to_h(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Convert source intensity to water-depth increment. Author: B.G (02/2026)"""
    s = get_source_w(i)
    dx = ti.cast(ti.static(gridctx.dx), cte.FLOAT_TYPE_TI)
    dth = get_dth(i)
    if ti.static(floodctx.source_w_kind == "precip"):
        return s * dth
    if ti.static(floodctx.source_w_kind == "q"):
        return (s / dx) * dth
    return (s / (dx * dx)) * dth


@ti.func
def get_friction_coeff(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return local friction coefficient. Author: B.G (02/2026)"""
    if ti.static(floodctx.friction_coeff_mode == "const"):
        return ti.cast(ti.static(floodctx.friction_coeff_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.friction_coeff_mode == "scalar"):
        return floodctx.friction_coeff_scalar[None]
    return floodctx.friction_coeff_field[i]


@ti.func
def get_friction_exponent(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return local friction flow exponent. Author: B.G (02/2026)"""
    if ti.static(floodctx.friction_exponent_mode == "const"):
        return ti.cast(ti.static(floodctx.friction_exponent_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.friction_exponent_mode == "scalar"):
        return floodctx.friction_exponent_scalar[None]
    return floodctx.friction_exponent_field[i]


@ti.func
def get_boundary_h(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return target outlet depth for nodes that can out. Author: B.G (02/2026)"""
    if ti.static(floodctx.boundary_h_mode == "const"):
        return ti.cast(ti.static(floodctx.boundary_h_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.boundary_h_mode == "scalar"):
        return floodctx.boundary_h_scalar[None]
    return floodctx.boundary_h_field[i]


@ti.func
def get_gf_min_increment(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return minimum absolute GraphFlood depth increment. Author: B.G (02/2026)"""
    if ti.static(floodctx.gf_min_increment_mode == "const"):
        return ti.cast(ti.static(floodctx.gf_min_increment_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.gf_min_increment_mode == "scalar"):
        return floodctx.gf_min_increment_scalar[None]
    return floodctx.gf_min_increment_field[i]


@ti.func
def get_gravity(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return local gravity value. Author: B.G (02/2026)"""
    if ti.static(floodctx.gravity_mode == "const"):
        return ti.cast(ti.static(floodctx.gravity_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.gravity_mode == "scalar"):
        return floodctx.gravity_scalar[None]
    return floodctx.gravity_field[i]


@ti.func
def get_rho_w(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return local fluid density. Author: B.G (02/2026)"""
    if ti.static(floodctx.rho_w_mode == "const"):
        return ti.cast(ti.static(floodctx.rho_w_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.rho_w_mode == "scalar"):
        return floodctx.rho_w_scalar[None]
    return floodctx.rho_w_field[i]


@ti.func
def get_rho_s(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return local sediment density placeholder. Author: B.G (02/2026)"""
    if ti.static(floodctx.rho_s_mode == "const"):
        return ti.cast(ti.static(floodctx.rho_s_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.rho_s_mode == "scalar"):
        return floodctx.rho_s_scalar[None]
    return floodctx.rho_s_field[i]


@ti.func
def get_dt_morpho_coeff(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return the coefficient used by dt_morpho mode n_dthydro. Author: B.G (02/2026)"""
    if ti.static(floodctx.dt_morpho_coeff_mode == "const"):
        return ti.cast(ti.static(floodctx.dt_morpho_coeff_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.dt_morpho_coeff_mode == "scalar"):
        return floodctx.dt_morpho_coeff_scalar[None]
    return floodctx.dt_morpho_coeff_field[i]


@ti.func
def get_dt_morpho(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Return local morphodynamic time step. Author: B.G (02/2026)"""
    if ti.static(floodctx.dt_morpho_mode == "n_dthydro"):
        return get_dth(i) * get_dt_morpho_coeff(i)
    if ti.static(floodctx.dt_morpho_mode == "const"):
        return ti.cast(ti.static(floodctx.dt_morpho_const), cte.FLOAT_TYPE_TI)
    if ti.static(floodctx.dt_morpho_mode == "scalar"):
        return floodctx.dt_morpho_scalar[None]
    return floodctx.dt_morpho_field[i]


@ti.func
def compute_u_from_h_slope(h: cte.FLOAT_TYPE_TI, slope: cte.FLOAT_TYPE_TI, i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Compute velocity from local depth/slope and friction law. Author: B.G (02/2026)"""
    if ti.static(floodctx.friction_law == "manning"):
        coeff = ti.max(get_friction_coeff(i), ti.cast(1e-9, cte.FLOAT_TYPE_TI))
        expo = get_friction_exponent(i)
        return ti.pow(ti.max(h, ti.cast(0.0, cte.FLOAT_TYPE_TI)), expo) / coeff * ti.sqrt(
            ti.max(slope, ti.cast(1e-9, cte.FLOAT_TYPE_TI))
        )
    return ti.cast(0.0, cte.FLOAT_TYPE_TI)


@ti.func
def compute_q_from_h_slope(h: cte.FLOAT_TYPE_TI, slope: cte.FLOAT_TYPE_TI, i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Compute unit discharge from local depth/slope and friction law. Author: B.G (02/2026)"""
    return h * compute_u_from_h_slope(h, slope, i)


@ti.func
def compute_qo_from_h_slope(h: cte.FLOAT_TYPE_TI, slope: cte.FLOAT_TYPE_TI, i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """Compute volumetric outflow from local depth/slope and friction law. Author: B.G (02/2026)"""
    dx = ti.cast(ti.static(gridctx.dx), cte.FLOAT_TYPE_TI)
    return compute_q_from_h_slope(h, slope, i) * dx

