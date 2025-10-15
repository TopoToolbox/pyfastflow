import taichi as ti

import pyfastflow as pf
from .. import constants as cte


@ti.func
def sumslope_downstream_node(z: ti.template(), i: ti.i32) -> cte.FLOAT_TYPE_TI:
    """
    sum the slope of all downtream
    """
    sumslope: cte.FLOAT_TYPE_TI = 0.0
    for k in ti.static(range(4)):
        j = pf.flow.neighbourer_flat.neighbour(i, k)
        if j > -1:
            if z[j] < z[i]:
                sumslope += (z[i] - z[j]) / pf.constants.DX
    return sumslope


@ti.func
def slope_dir(z: ti.template(), i: ti.i32, k: ti.template()) -> cte.FLOAT_TYPE_TI:
    """
    sum the slope of all downtream
    """
    j = pf.flow.neighbourer_flat.neighbour(i, k)

    slope: cte.FLOAT_TYPE_TI = 0.0
    if j > -1:
        slope = (z[i] - z[j]) / pf.constants.DX

    return slope
