from types import SimpleNamespace

from .ga_kernels import (
    add_to_flat_kernel,
    add_weighted_to_flat_kernel,
    init_flat_arange_kernel,
    multiply_flat_by_scalar_kernel,
    scan_copy_input_to_work_kernel,
    scan_downsweep_step_kernel,
    scan_make_inclusive_and_copy_kernel,
    scan_set_root_zero_kernel,
    scan_upsweep_step_kernel,
    swap_flat_kernel,
    weighted_mean_into_flat_kernel,
)
from .math_utils import atan
from .pingpong import getSrc, updateSrc


class GAContext:
    """
    Grid-bound context for general Taichi algorithms.

    This context only exposes the final specialized API. The raw Taichi helper
    funcs and kernels live in dedicated modules and are rebound here through
    ``gridctx.make_func`` and ``gridctx.make_kernel``.

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

        # Make the scan-size metadata visible from the universal ``gridctx``
        # entry point used by the generic scan kernels.
        self.gridctx.ga = self

        self.tfunc = SimpleNamespace()
        self.kernels = SimpleNamespace()
        self._compile_helpers()
        self._compile_kernels()

    def _compile_helpers(self):
        """
        Bind reusable Taichi helper funcs to this context.

        Author: B.G (02/2026)
        """
        self.tfunc.atan = self.gridctx.make_func(atan)
        self.tfunc.get_src = self.gridctx.make_func(getSrc)
        self.tfunc.update_src = self.gridctx.make_func(updateSrc)

    def _compile_kernels(self):
        """
        Bind the flat kernel family to this context.

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
            scan_copy_input_to_work_kernel
        )
        self.kernels.scan_upsweep_step = self.gridctx.make_kernel(scan_upsweep_step_kernel)
        self.kernels.scan_downsweep_step = self.gridctx.make_kernel(scan_downsweep_step_kernel)
        self.kernels.scan_set_root_zero = self.gridctx.make_kernel(scan_set_root_zero_kernel)
        self.kernels.scan_make_inclusive_and_copy = self.gridctx.make_kernel(
            scan_make_inclusive_and_copy_kernel
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
