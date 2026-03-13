import math
from types import SimpleNamespace

from .. import constants as cte
from .. import pool as ppool
from .hillshading import (
    gradient_x_2d,
    gradient_x_flat,
    gradient_y_2d,
    gradient_y_flat,
    hillshade_at_2d,
    hillshade_at_flat,
    hillshading_2d_kernel,
    hillshading_flat_kernel,
    multishading_2d_kernel,
    multishading_flat_kernel,
)


class VisuContext:
    """
    Grid-bound visualization context.

    This context binds streamlined flat hillshading kernels to one GridContext
    and exposes small wrapper methods that manage pooled temporary output.

    Author: B.G (02/2026)
    """

    def __init__(self, gridctx):
        """
        Initialize the visualization context for one GridContext.

        Author: B.G (02/2026)
        """
        self.gridctx = gridctx
        self.tfunc = SimpleNamespace()
        self.kernels = SimpleNamespace()
        self._compile_helpers()
        self._compile_kernels()

    def _compile_helpers(self):
        """
        Bind raw visualization helper funcs to this context.

        Author: B.G (02/2026)
        """
        self.tfunc.gradient_x_flat = self.gridctx.make_func(gradient_x_flat)
        self.tfunc.gradient_y_flat = self.gridctx.make_func(gradient_y_flat)
        self.tfunc.hillshade_at_flat = self.gridctx.make_func(
            hillshade_at_flat,
            gradient_x_flat=self.tfunc.gradient_x_flat,
            gradient_y_flat=self.tfunc.gradient_y_flat,
        )
        self.tfunc.gradient_x_2d = self.gridctx.make_func(gradient_x_2d)
        self.tfunc.gradient_y_2d = self.gridctx.make_func(gradient_y_2d)
        self.tfunc.hillshade_at_2d = self.gridctx.make_func(
            hillshade_at_2d,
            gradient_x_2d=self.tfunc.gradient_x_2d,
            gradient_y_2d=self.tfunc.gradient_y_2d,
        )
        self.tfunc.gradient_x = self.tfunc.gradient_x_flat
        self.tfunc.gradient_y = self.tfunc.gradient_y_flat
        self.tfunc.hillshade_at = self.tfunc.hillshade_at_flat

    def _compile_kernels(self):
        """
        Bind raw visualization kernels to this context.

        Author: B.G (02/2026)
        """
        self.kernels.hillshading_flat = self.gridctx.make_kernel(
            hillshading_flat_kernel,
            hillshade_at_flat=self.tfunc.hillshade_at_flat,
        )
        self.kernels.multishading_flat = self.gridctx.make_kernel(
            multishading_flat_kernel,
            hillshade_at_flat=self.tfunc.hillshade_at_flat,
        )
        self.kernels.hillshading_2d = self.gridctx.make_kernel(
            hillshading_2d_kernel,
            hillshade_at_2d=self.tfunc.hillshade_at_2d,
        )
        self.kernels.multishading_2d = self.gridctx.make_kernel(
            multishading_2d_kernel,
            hillshade_at_2d=self.tfunc.hillshade_at_2d,
        )
        self.kernels.hillshading = self.kernels.hillshading_flat
        self.kernels.multishading = self.kernels.multishading_flat

    def _unwrap_field(self, z):
        """
        Return the Taichi field handle from a raw field or TPField.

        Author: B.G (02/2026)
        """
        return z.field if hasattr(z, "field") else z

    def _infer_layout(self, z_field):
        """
        Infer whether a field is flat or 2D from its shape.

        Author: B.G (02/2026)
        """
        shape = tuple(z_field.shape)
        if len(shape) == 1:
            return "flat"
        if len(shape) == 2:
            return "2d"
        raise ValueError("Only flat and 2D Taichi fields are supported")

    def _normalize_output_layout(self, output_layout, input_layout):
        """
        Normalize the requested output layout.

        Author: B.G (02/2026)
        """
        if output_layout is None:
            return input_layout
        value = str(output_layout).lower()
        if value not in {"flat", "2d"}:
            raise ValueError("output_layout must be 'flat', '2d', or None")
        return value

    def _allocate_output(self, layout):
        """
        Allocate a pooled hillshade field in the requested layout.

        Author: B.G (02/2026)
        """
        if layout == "flat":
            return ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx * self.gridctx.ny))
        return ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.ny, self.gridctx.nx))

    def _format_numpy_output(self, out_field, output_layout):
        """
        Convert the internal Taichi output field to a numpy array in the requested layout.

        Author: B.G (02/2026)
        """
        arr = out_field.to_numpy()
        if output_layout == "flat":
            return arr.reshape(-1)
        return arr.reshape((self.gridctx.ny, self.gridctx.nx))

    def generate_hillshade(
        self,
        z,
        altitude_deg: float = 45.0,
        azimuth_deg: float = 315.0,
        z_factor: float = 1.0,
        output_layout=None,
    ):
        """
        Compute a hillshade image and return it as a numpy array.

        Temporary memory is allocated from the pool and released internally.
        The input field may be flat or 2D. ``output_layout`` controls whether
        the returned numpy array is flat or 2D and defaults to the input layout.

        Author: B.G (02/2026)
        """
        z_field = self._unwrap_field(z)
        input_layout = self._infer_layout(z_field)
        output_layout = self._normalize_output_layout(output_layout, input_layout)
        hillshade = self._allocate_output(input_layout)

        try:
            zenith_rad = math.radians(90.0 - altitude_deg)
            azimuth_rad = math.radians(azimuth_deg)
            kernel = self.kernels.hillshading_flat if input_layout == "flat" else self.kernels.hillshading_2d
            kernel(
                z_field,
                hillshade.field,
                zenith_rad,
                azimuth_rad,
                z_factor,
            )
            return self._format_numpy_output(hillshade.field, output_layout)
        finally:
            hillshade.release()

    def generate_multishade(
        self,
        z,
        altitude_deg: float = 45.0,
        z_factor: float = 1.0,
        azimuths_deg=None,
        output_layout=None,
    ):
        """
        Compute a four-direction averaged hillshade image and return it as a numpy array.

        Temporary memory is allocated from the pool and released internally.
        The input field may be flat or 2D. ``output_layout`` controls whether
        the returned numpy array is flat or 2D and defaults to the input layout.

        Author: B.G (02/2026)
        """
        z_field = self._unwrap_field(z)
        input_layout = self._infer_layout(z_field)
        output_layout = self._normalize_output_layout(output_layout, input_layout)
        hillshade = self._allocate_output(input_layout)

        try:
            if azimuths_deg is None:
                azimuths_deg = [315.0, 45.0, 135.0, 225.0]
            if len(azimuths_deg) != 4:
                raise ValueError("generate_multishade expects exactly 4 azimuths")

            zenith_rad = math.radians(90.0 - altitude_deg)
            azimuth0_rad = math.radians(float(azimuths_deg[0]))
            azimuth1_rad = math.radians(float(azimuths_deg[1]))
            azimuth2_rad = math.radians(float(azimuths_deg[2]))
            azimuth3_rad = math.radians(float(azimuths_deg[3]))

            kernel = self.kernels.multishading_flat if input_layout == "flat" else self.kernels.multishading_2d
            kernel(
                z_field,
                hillshade.field,
                zenith_rad,
                azimuth0_rad,
                azimuth1_rad,
                azimuth2_rad,
                azimuth3_rad,
                z_factor,
            )
            return self._format_numpy_output(hillshade.field, output_layout)
        finally:
            hillshade.release()
