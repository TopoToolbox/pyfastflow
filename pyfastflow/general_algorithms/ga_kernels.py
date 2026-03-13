"""
Generic Taichi kernels bound by ``GAContext`` through ``gridctx.make_kernel``.

The kernels in this module intentionally stay generic. They only rely on the
universal ``gridctx`` global injected by :class:`pyfastflow.grid.GridContext`
at specialization time.

Author: B.G (02/2026)
"""

import taichi as ti

from .. import constants as cte


gridctx = None


@ti.kernel
def swap_flat_kernel(array1: ti.template(), array2: ti.template()):
    """
    Swap two flat arrays over the bound grid size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.nx * gridctx.ny):
        temp = array1[i]
        array1[i] = array2[i]
        array2[i] = temp


@ti.kernel
def add_to_flat_kernel(array1: ti.template(), array2: ti.template()):
    """
    Add array2 into array1 over the bound grid size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.nx * gridctx.ny):
        array1[i] += array2[i]


@ti.kernel
def add_weighted_to_flat_kernel(
    array1: ti.template(), array2: ti.template(), weight: cte.FLOAT_TYPE_TI
):
    """
    Add a weighted version of array2 into array1 over the bound grid size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.nx * gridctx.ny):
        array1[i] += array2[i] * weight


@ti.kernel
def weighted_mean_into_flat_kernel(
    array1: ti.template(), array2: ti.template(), weight: cte.FLOAT_TYPE_TI
):
    """
    Blend array2 into array1 with a weighted mean over the bound grid size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.nx * gridctx.ny):
        array1[i] = array2[i] * weight + array1[i] * (1.0 - weight)


@ti.kernel
def init_flat_arange_kernel(array: ti.template()):
    """
    Fill a flat array with its row-major indices over the bound grid size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.nx * gridctx.ny):
        array[i] = i


@ti.kernel
def multiply_flat_by_scalar_kernel(array: ti.template(), scalar: ti.template()):
    """
    Multiply a flat array by a scalar field over the bound grid size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.nx * gridctx.ny):
        array[i] *= scalar[None]


@ti.kernel
def scan_copy_input_to_work_kernel(src: ti.template(), dst: ti.template()):
    """
    Copy flat input data into the scan work buffer and zero-pad the tail.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.ga.scan_work_size):
        if i < gridctx.nx * gridctx.ny:
            dst[i] = src[i]
        else:
            dst[i] = 0


@ti.kernel
def scan_upsweep_step_kernel(data: ti.template(), stride: ti.i32):
    """
    Execute one upsweep step over the precomputed scan work size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.ga.scan_work_size):
        if (i + 1) % (stride * 2) == 0:
            data[i] += data[i - stride]


@ti.kernel
def scan_downsweep_step_kernel(data: ti.template(), stride: ti.i32):
    """
    Execute one downsweep step over the precomputed scan work size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.ga.scan_work_size):
        if (i + 1) % (stride * 2) == 0:
            temp = data[i - stride]
            data[i - stride] = data[i]
            data[i] += temp


@ti.kernel
def scan_set_root_zero_kernel(data: ti.template()):
    """
    Zero the scan-tree root element in the work buffer.

    Author: B.G (02/2026)
    """
    data[gridctx.ga.scan_work_size - 1] = 0


@ti.kernel
def scan_make_inclusive_and_copy_kernel(
    input_arr: ti.template(), work_arr: ti.template(), output_arr: ti.template()
):
    """
    Convert exclusive scan work data to inclusive output over the bound grid size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.nx * gridctx.ny):
        if i == 0:
            output_arr[i] = input_arr[i]
        else:
            output_arr[i] = work_arr[i] + input_arr[i]
