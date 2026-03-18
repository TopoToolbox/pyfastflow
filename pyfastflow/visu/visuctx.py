import math
from types import SimpleNamespace

from .. import constants as cte
from .. import pool as ppool
from ..context import require_flat_field
from .hillshading import (
    gradient_x_flat,
    gradient_y_flat,
    hillshade_at_flat,
    hillshading_flat_kernel,
    multishading_flat_kernel,
)


class VisuContext:
    """
    Flat grid-bound hillshading context.

    The compiled API is flat-only. Optional 2D output formatting happens only
    at the numpy/TPField boundary.

    Author: B.G (03/2026)
    """

    def __init__(self, gridctx):
        self.gridctx = gridctx
        self.tfunc = SimpleNamespace()
        self.kernels = SimpleNamespace()
        self._compile_helpers()
        self._compile_kernels()

    def _compile_helpers(self):
        self.tfunc.gradient_x = self.gridctx.make_func(gradient_x_flat)
        self.tfunc.gradient_y = self.gridctx.make_func(gradient_y_flat)
        self.tfunc.hillshade_at = self.gridctx.make_func(
            hillshade_at_flat,
            gradient_x_flat=self.tfunc.gradient_x,
            gradient_y_flat=self.tfunc.gradient_y,
        )

    def _compile_kernels(self):
        self.kernels.hillshading = self.gridctx.make_kernel(
            hillshading_flat_kernel,
            hillshade_at_flat=self.tfunc.hillshade_at,
        )
        self.kernels.multishading = self.gridctx.make_kernel(
            multishading_flat_kernel,
            hillshade_at_flat=self.tfunc.hillshade_at,
        )
        self.hillshading = self.kernels.hillshading
        self.multishading = self.kernels.multishading

    def _allocate_output(self):
        return ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.n_flat))

    def _format_numpy_output(self, out_field, output_layout):
        arr = out_field.to_numpy().reshape(-1)
        if output_layout == "flat":
            return arr
        return arr.reshape((self.gridctx.ny, self.gridctx.nx))

    def generate_hillshade(
        self,
        z,
        altitude_deg: float = 45.0,
        azimuth_deg: float = 315.0,
        z_factor: float = 1.0,
        output_layout="2d",
    ):
        """
        Compute one hillshade image from a flat field.

        Author: B.G (03/2026)
        """
        z_field = require_flat_field(z, "z")
        hillshade = self._allocate_output()
        try:
            self.hillshading(
                z_field,
                hillshade.field,
                math.radians(90.0 - altitude_deg),
                math.radians(azimuth_deg),
                z_factor,
            )
            return self._format_numpy_output(hillshade.field, str(output_layout).lower())
        finally:
            hillshade.release()

    def generate_multishade(
        self,
        z,
        altitude_deg: float = 45.0,
        z_factor: float = 1.0,
        azimuths_deg=None,
        output_layout="2d",
    ):
        """
        Compute a four-direction averaged hillshade image from a flat field.

        Author: B.G (03/2026)
        """
        z_field = require_flat_field(z, "z")
        hillshade = self._allocate_output()
        try:
            if azimuths_deg is None:
                azimuths_deg = [315.0, 45.0, 135.0, 225.0]
            if len(azimuths_deg) != 4:
                raise ValueError("generate_multishade expects exactly 4 azimuths")

            self.multishading(
                z_field,
                hillshade.field,
                math.radians(90.0 - altitude_deg),
                math.radians(float(azimuths_deg[0])),
                math.radians(float(azimuths_deg[1])),
                math.radians(float(azimuths_deg[2])),
                math.radians(float(azimuths_deg[3])),
                z_factor,
            )
            return self._format_numpy_output(hillshade.field, str(output_layout).lower())
        finally:
            hillshade.release()
