import math

from .. import constants as cte
from .. import pool as ppool
from ..context import ContextFactory, ContextRef, flat_field_to_numpy, format_flat_numpy, require_flat_field
from .hillshading import (
    gradient_x_flat,
    gradient_y_flat,
    hillshade_at_flat,
    hillshading_flat_kernel,
    multishading_flat_kernel,
)


class VisuContext:
    """
    Flat grid-bound hillshading API context.

    Author: B.G (03/2026)
    """

    def __init__(self, gridctx):
        self.gridctx = gridctx
        self._factory = ContextFactory(
            self,
            bindings={"gridctx": self.gridctx, "visuctx": self},
            n_flat=self.gridctx.n_flat,
        )
        self._factory.compile_block(
            [
                {"target": "tfunc", "name": "gradient_x", "template": gradient_x_flat, "kind": "func"},
                {"target": "tfunc", "name": "gradient_y", "template": gradient_y_flat, "kind": "func"},
                {
                    "target": "tfunc",
                    "name": "hillshade_at",
                    "template": hillshade_at_flat,
                    "kind": "func",
                    "bindings": {
                        "gradient_x_flat": ContextRef("tfunc.gradient_x"),
                        "gradient_y_flat": ContextRef("tfunc.gradient_y"),
                    },
                },
                {
                    "target": "kernels",
                    "name": "hillshading",
                    "template": hillshading_flat_kernel,
                    "kind": "kernel",
                    "bindings": {"hillshade_at_flat": ContextRef("tfunc.hillshade_at")},
                },
                {
                    "target": "kernels",
                    "name": "multishading",
                    "template": multishading_flat_kernel,
                    "kind": "kernel",
                    "bindings": {"hillshade_at_flat": ContextRef("tfunc.hillshade_at")},
                },
            ]
        )
        self._factory.export(
            {
                "hillshading": "kernels.hillshading",
                "multishading": "kernels.multishading",
            }
        )

    def _allocate_output(self):
        return ppool.taipool.get_tpfield(dtype=cte.FLOAT_TYPE_TI, shape=(self.gridctx.n_flat))

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
            return format_flat_numpy(
                flat_field_to_numpy(hillshade.field),
                self.gridctx.rshp,
                output_layout=output_layout,
            )
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
        Compute one four-direction averaged hillshade image from a flat field.

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
            return format_flat_numpy(
                flat_field_to_numpy(hillshade.field),
                self.gridctx.rshp,
                output_layout=output_layout,
            )
        finally:
            hillshade.release()
