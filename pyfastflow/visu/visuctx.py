import math
from types import SimpleNamespace

import numpy as np

from .. import constants as cte
from .. import pool as ppool
from .hillshading import gradient_x, gradient_y, hillshade_at, hillshading_kernel, multishading_kernel


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
        self.tfunc.gradient_x = self.gridctx.make_func(gradient_x)
        self.tfunc.gradient_y = self.gridctx.make_func(gradient_y)
        self.tfunc.hillshade_at = self.gridctx.make_func(
            hillshade_at,
            gradient_x=self.tfunc.gradient_x,
            gradient_y=self.tfunc.gradient_y,
        )

    def _compile_kernels(self):
        """
        Bind raw visualization kernels to this context.

        Author: B.G (02/2026)
        """
        self.kernels.hillshading = self.gridctx.make_kernel(
            hillshading_kernel,
            hillshade_at=self.tfunc.hillshade_at,
        )
        self.kernels.multishading = self.gridctx.make_kernel(
            multishading_kernel,
            hillshade_at=self.tfunc.hillshade_at,
        )

    def _unwrap_field(self, z):
        """
        Return the Taichi field handle from a raw field or TPField.

        Author: B.G (02/2026)
        """
        return z.field if hasattr(z, "field") else z

    def generate_hillshade(
        self,
        z,
        altitude_deg: float = 45.0,
        azimuth_deg: float = 315.0,
        z_factor: float = 1.0,
    ):
        """
        Compute a hillshade image and return it as a 2D numpy array.

        Temporary memory is allocated from the pool and released internally.

        Author: B.G (02/2026)
        """
        z_field = self._unwrap_field(z)
        hillshade = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx * self.gridctx.ny))

        try:
            zenith_rad = math.radians(90.0 - altitude_deg)
            azimuth_rad = math.radians(azimuth_deg)
            self.kernels.hillshading(
                z_field,
                hillshade.field,
                zenith_rad,
                azimuth_rad,
                z_factor,
            )
            return hillshade.field.to_numpy().reshape((self.gridctx.ny, self.gridctx.nx))
        finally:
            hillshade.release()

    def generate_multishade(
        self,
        z,
        altitude_deg: float = 45.0,
        z_factor: float = 1.0,
        azimuths_deg=None,
    ):
        """
        Compute a four-direction averaged hillshade image and return it as a 2D numpy array.

        Temporary memory is allocated from the pool and released internally.

        Author: B.G (02/2026)
        """
        z_field = self._unwrap_field(z)
        hillshade = ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.nx * self.gridctx.ny))

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

            self.kernels.multishading(
                z_field,
                hillshade.field,
                zenith_rad,
                azimuth0_rad,
                azimuth1_rad,
                azimuth2_rad,
                azimuth3_rad,
                z_factor,
            )
            return hillshade.field.to_numpy().reshape((self.gridctx.ny, self.gridctx.nx))
        finally:
            hillshade.release()
