from types import SimpleNamespace

import taichi as ti

from .. import constants as cte
from .math_utils import atan
from .pingpong import getSrc, updateSrc


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
def _scan_copy_input_to_work_kernel(src: ti.template(), dst: ti.template()):
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
def _scan_upsweep_step_kernel(data: ti.template(), stride: ti.i32):
    """
    Execute one upsweep step over the precomputed scan work size.

    Author: B.G (02/2026)
    """
    for i in range(gridctx.ga.scan_work_size):
        if (i + 1) % (stride * 2) == 0:
            data[i] += data[i - stride]


@ti.kernel
def _scan_downsweep_step_kernel(data: ti.template(), stride: ti.i32):
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
def _scan_set_root_zero_kernel(data: ti.template()):
    """
    Zero the scan-tree root element in the work buffer.

    Author: B.G (02/2026)
    """
    data[gridctx.ga.scan_work_size - 1] = 0


@ti.kernel
def _scan_make_inclusive_and_copy_kernel(
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


class GAContext:
    """
    Grid-bound context for general Taichi algorithms.

    This context provides a small set of flat kernels specialized against one
    GridContext, plus direct access to reusable Taichi funcs that do not need
    per-context recompilation.

    Author: B.G (02/2026)
    """

    def __init__(self, gridctx):
        """
        Initialize the general-algorithm context for one GridContext.

        Author: B.G (02/2026)
        """
        self.gridctx = gridctx
        self.n_flat = self.gridctx.nx * self.gridctx.ny
        self.scan_work_size = 1
        while self.scan_work_size < self.n_flat:
            self.scan_work_size *= 2

        # Make the scan size visible as a static constant through the bound grid context.
        self.gridctx.ga = self

        self.tfunc = SimpleNamespace()
        self.kernels = SimpleNamespace()
        self._bind_tfunc()
        self._compile_kernels()

    def _bind_tfunc(self):
        """
        Bind reusable Taichi helper funcs.

        Author: B.G (02/2026)
        """
        self.tfunc.atan = atan
        self.tfunc.get_src = getSrc
        self.tfunc.update_src = updateSrc

    def _compile_kernels(self):
        """
        Compile the flat kernel family for this context.

        Author: B.G (02/2026)
        """
        self.kernels.swap_flat = self.gridctx.make_kernel(swap_flat_kernel)
        self.kernels.add_to_flat = self.gridctx.make_kernel(add_to_flat_kernel)
        self.kernels.add_weighted_to_flat = self.gridctx.make_kernel(add_weighted_to_flat_kernel)
        self.kernels.weighted_mean_into_flat = self.gridctx.make_kernel(
            weighted_mean_into_flat_kernel
        )
        self.kernels.init_flat_arange = self.gridctx.make_kernel(init_flat_arange_kernel)
        self.kernels.multiply_flat_by_scalar = self.gridctx.make_kernel(
            multiply_flat_by_scalar_kernel
        )

        self.kernels.scan_copy_input_to_work = self.gridctx.make_kernel(
            _scan_copy_input_to_work_kernel
        )
        self.kernels.scan_upsweep_step = self.gridctx.make_kernel(_scan_upsweep_step_kernel)
        self.kernels.scan_downsweep_step = self.gridctx.make_kernel(
            _scan_downsweep_step_kernel
        )
        self.kernels.scan_set_root_zero = self.gridctx.make_kernel(_scan_set_root_zero_kernel)
        self.kernels.scan_make_inclusive_and_copy = self.gridctx.make_kernel(
            _scan_make_inclusive_and_copy_kernel
        )

    def inclusive_scan_flat(self, input_arr, output_arr, work_arr):
        """
        Compute an inclusive scan over the bound flat grid size.

        ``work_arr`` must have at least ``scan_work_size`` elements.

        Author: B.G (02/2026)
        """
        self.kernels.scan_copy_input_to_work(input_arr, work_arr)

        stride = 1
        while stride < self.scan_work_size:
            self.kernels.scan_upsweep_step(work_arr, stride)
            stride *= 2

        self.kernels.scan_set_root_zero(work_arr)

        stride = self.scan_work_size // 2
        while stride > 0:
            self.kernels.scan_downsweep_step(work_arr, stride)
            stride //= 2

        self.kernels.scan_make_inclusive_and_copy(input_arr, work_arr, output_arr)
