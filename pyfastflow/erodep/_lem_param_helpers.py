import taichi as ti

from .. import constants as cte


@ti.func
def get_dt(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the landscape-evolution timestep at one node.

    Author: B.G (02/2026)
    """
    if ti.static(lemctx.dt_mode == "const"):
        return ti.cast(ti.static(lemctx.dt_const), cte.FLOAT_TYPE_TI)
    if ti.static(lemctx.dt_mode == "scalar"):
        return lemctx.dt_scalar[None]
    return lemctx.dt_field[i]


@ti.func
def get_uplift_rate(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the tectonic uplift rate at one node.

    Author: B.G (02/2026)
    """
    if ti.static(lemctx.uplift_rate_mode == "const"):
        return ti.cast(ti.static(lemctx.uplift_rate_const), cte.FLOAT_TYPE_TI)
    if ti.static(lemctx.uplift_rate_mode == "scalar"):
        return lemctx.uplift_rate_scalar[None]
    return lemctx.uplift_rate_field[i]


@ti.func
def get_K_bedrock(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the bedrock erodibility coefficient at one node.

    Author: B.G (02/2026)
    """
    if ti.static(lemctx.K_bedrock_mode == "const"):
        return ti.cast(ti.static(lemctx.K_bedrock_const), cte.FLOAT_TYPE_TI)
    if ti.static(lemctx.K_bedrock_mode == "scalar"):
        return lemctx.K_bedrock_scalar[None]
    return lemctx.K_bedrock_field[i]


@ti.func
def get_m_exp(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the drainage-area exponent at one node.

    Author: B.G (02/2026)
    """
    if ti.static(lemctx.m_exp_mode == "const"):
        return ti.cast(ti.static(lemctx.m_exp_const), cte.FLOAT_TYPE_TI)
    if ti.static(lemctx.m_exp_mode == "scalar"):
        return lemctx.m_exp_scalar[None]
    return lemctx.m_exp_field[i]


@ti.func
def get_n_exp(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the slope exponent at one node.

    Author: B.G (02/2026)
    """
    if ti.static(lemctx.n_exp_mode == "const"):
        return ti.cast(ti.static(lemctx.n_exp_const), cte.FLOAT_TYPE_TI)
    if ti.static(lemctx.n_exp_mode == "scalar"):
        return lemctx.n_exp_scalar[None]
    return lemctx.n_exp_field[i]


@ti.func
def get_K_sed(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the future sediment erodibility coefficient at one node.

    Author: B.G (02/2026)
    """
    if ti.static(lemctx.K_sed_mode == "const"):
        return ti.cast(ti.static(lemctx.K_sed_const), cte.FLOAT_TYPE_TI)
    if ti.static(lemctx.K_sed_mode == "scalar"):
        return lemctx.K_sed_scalar[None]
    return lemctx.K_sed_field[i]


@ti.func
def get_kappa_bedrock(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the future bedrock hillslope diffusivity at one node.

    Author: B.G (02/2026)
    """
    if ti.static(lemctx.kappa_bedrock_mode == "const"):
        return ti.cast(ti.static(lemctx.kappa_bedrock_const), cte.FLOAT_TYPE_TI)
    if ti.static(lemctx.kappa_bedrock_mode == "scalar"):
        return lemctx.kappa_bedrock_scalar[None]
    return lemctx.kappa_bedrock_field[i]


@ti.func
def get_kappa_sed(i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    Return the future sediment hillslope diffusivity at one node.

    Author: B.G (02/2026)
    """
    if ti.static(lemctx.kappa_sed_mode == "const"):
        return ti.cast(ti.static(lemctx.kappa_sed_const), cte.FLOAT_TYPE_TI)
    if ti.static(lemctx.kappa_sed_mode == "scalar"):
        return lemctx.kappa_sed_scalar[None]
    return lemctx.kappa_sed_field[i]


@ti.func
def get_domain(i: ti.i32) -> ti.u8:
    """
    Return the future domain code at one node.

    Author: B.G (02/2026)
    """
    if ti.static(lemctx.domain_mode == "const"):
        return ti.cast(ti.static(lemctx.domain_const), ti.u8)
    if ti.static(lemctx.domain_mode == "scalar"):
        return lemctx.domain_scalar[None]
    return lemctx.domain_field[i]
