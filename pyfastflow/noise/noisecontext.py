from types import SimpleNamespace

import numpy as np
import taichi as ti

from .. import constants as cte
from .. import pool as ppool
from .perlin_noise import perlin_noise_kernel
from .white_noise import white_noise_kernel


class NoiseContext:
    """
    Grid-bound noise generation context.

    This context binds generic Taichi noise kernels to one GridContext through
    ``gridctx.make_kernel`` and exposes both the raw kernels and small pooled
    allocation helpers.

    Author: B.G (02/2026)
    """

    def __init__(self, gridctx):
        """
        Initialize the noise context for one GridContext.

        Author: B.G (02/2026)
        """
        self.gridctx = gridctx
        self.kernels = SimpleNamespace()
        self._compile_kernels()

    def _compile_kernels(self):
        """
        Bind raw generic kernels to this context.

        Author: B.G (02/2026)
        """
        self.kernels.white_noise = self.gridctx.make_kernel(white_noise_kernel)
        self.kernels.perlin_noise = self.gridctx.make_kernel(perlin_noise_kernel)

    def _fisher_yates_permutation(self, seed: int) -> np.ndarray:
        """
        Build a 512-entry Perlin permutation table.

        Author: B.G (02/2026)
        """
        rng = np.random.default_rng(seed)
        perm = np.arange(256, dtype=np.int32)
        for i in range(255, 0, -1):
            j = rng.integers(0, i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        return np.concatenate([perm, perm])

    def generate_white_noise(
        self, amplitude: float = 1.0, seed: int = 42
    ):
        """
        Allocate and fill a pooled field with white noise.

        The returned TPField remains owned by the caller and must be released by
        the caller once it is no longer needed.

        Author: B.G (02/2026)
        """
        noise_field = ppool.taipool.get_tpfield(
            cte.FLOAT_TYPE_TI, (self.gridctx.ny, self.gridctx.nx)
        )
        self.kernels.white_noise(noise_field.field, amplitude, seed)
        return noise_field

    def generate_perlin_noise(
        self,
        frequency: float = 8.0,
        octaves: int = 4,
        persistence: float = 0.5,
        amplitude: float = 1.0,
        seed: int = 42,
        frequency_x: float | None = None,
        frequency_y: float | None = None,
    ):
        """
        Allocate and fill a pooled field with Perlin noise.

        The returned TPField remains owned by the caller and must be released by
        the caller once it is no longer needed.

        Author: B.G (02/2026)
        """
        noise_field = ppool.taipool.get_tpfield(
            cte.FLOAT_TYPE_TI, (self.gridctx.ny, self.gridctx.nx)
        )
        perm_field = ppool.taipool.get_tpfield(ti.i32, (512,))

        try:
            perm_field.from_numpy(self._fisher_yates_permutation(seed))

            fx = float(frequency_x if frequency_x is not None else frequency)
            fy = float(frequency_y if frequency_y is not None else frequency)

            self.kernels.perlin_noise(
                noise_field.field,
                fx,
                fy,
                int(octaves),
                persistence,
                amplitude,
                perm_field.field,
            )
        finally:
            perm_field.release()

        return noise_field
