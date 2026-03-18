from ..context import ContextFactory
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
    Grid-bound general-algorithm specialization context.

    Author: B.G (03/2026)
    """

    def __init__(self, gridctx):
        self.gridctx = gridctx
        self.n_flat = self.gridctx.n_flat
        self.scan_work_size = 1
        while self.scan_work_size < self.n_flat:
            self.scan_work_size *= 2

        self.gridctx.ga = self

        self._factory = ContextFactory(
            self,
            bindings={
                "gridctx": self.gridctx,
                "gactx": self,
            },
            n_flat=self.n_flat,
        )

        self._factory.compile_block(
            [
                {"target": "tfunc", "name": "atan", "template": atan, "kind": "func"},
                {"target": "tfunc", "name": "get_src", "template": getSrc, "kind": "func"},
                {"target": "tfunc", "name": "update_src", "template": updateSrc, "kind": "func"},
            ]
        )
        self._factory.compile_block(
            [
                {"target": "kernels", "name": "swap_flat", "template": swap_flat_kernel, "kind": "kernel"},
                {"target": "kernels", "name": "add_to_flat", "template": add_to_flat_kernel, "kind": "kernel"},
                {
                    "target": "kernels",
                    "name": "add_weighted_to_flat",
                    "template": add_weighted_to_flat_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "weighted_mean_into_flat",
                    "template": weighted_mean_into_flat_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "init_flat_arange",
                    "template": init_flat_arange_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "multiply_flat_by_scalar",
                    "template": multiply_flat_by_scalar_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "scan_copy_input_to_work",
                    "template": scan_copy_input_to_work_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "scan_upsweep_step",
                    "template": scan_upsweep_step_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "scan_downsweep_step",
                    "template": scan_downsweep_step_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "scan_set_root_zero",
                    "template": scan_set_root_zero_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels",
                    "name": "scan_make_inclusive_and_copy",
                    "template": scan_make_inclusive_and_copy_kernel,
                    "kind": "kernel",
                },
            ]
        )
        self._factory.export(
            {
                "swap_flat": "kernels.swap_flat",
                "add_to_flat": "kernels.add_to_flat",
                "add_weighted_to_flat": "kernels.add_weighted_to_flat",
                "weighted_mean_into_flat": "kernels.weighted_mean_into_flat",
                "init_flat_arange": "kernels.init_flat_arange",
                "multiply_flat_by_scalar": "kernels.multiply_flat_by_scalar",
            }
        )

    def inclusive_scan_flat(self, input_arr, output_arr, work_arr):
        """
        Compute an inclusive scan over the bound flat grid size.

        ``work_arr`` must have at least ``scan_work_size`` elements.

        Author: B.G (03/2026)
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
