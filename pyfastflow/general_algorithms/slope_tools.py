import taichi as ti

import pyfastflow as pf


@ti.func
def sumslope_downstream_node(z: ti.template(), i: ti.i32) -> ti.f32:
    """
    sum the slope of all downtream
    """
    sumslope: ti.f32 = 0.0
    for k in ti.static(range(4)):
        j = pf.flow.neighbourer_flat.neighbour(i, k)
        if j > -1:
            if z[j] < z[i]:
                sumslope += (z[i] - z[j]) / pf.constants.DX
    return sumslope


@ti.func
def slope_dir(z: ti.template(), i: ti.i32, k: ti.template()) -> ti.f32:
    """
    sum the slope of all downtream
    """
    j = pf.flow.neighbourer_flat.neighbour(i, k)

    slope: ti.f32 = 0.0
    if j > -1:
        slope = (z[i] - z[j]) / pf.constants.DX

    return slope
