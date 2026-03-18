from types import SimpleNamespace

import numpy as np
import taichi as ti

from .. import constants as cte
from .. import pool as ppool
from .perlin_noise import perlin_noise_flat_kernel
from .white_noise import white_noise_flat_kernel


class NoiseContext:
    """
    Flat grid-bound noise generation context.

    The compiled API is flat-only. Optional 2D formatting is applied only when
    returning numpy arrays at the edge.

    Author: B.G (03/2026)
    """

    def __init__(self, gridctx):
        self.gridctx = gridctx
        self.kernels = SimpleNamespace()
        self.kernels.white_noise = self.gridctx.make_kernel(white_noise_flat_kernel)
        self.kernels.perlin_noise = self.gridctx.make_kernel(perlin_noise_flat_kernel)
        self.white_noise = self.kernels.white_noise
        self.perlin_noise = self.kernels.perlin_noise

    def _allocate_noise_field(self):
        return ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (self.gridctx.n_flat))

    def _fisher_yates_permutation(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        perm = np.arange(256, dtype=np.int32)
        for i in range(255, 0, -1):
            j = rng.integers(0, i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        return np.concatenate([perm, perm])

    def generate_white_noise(
        self,
        amplitude: float = 1.0,
        seed: int = 42,
        output_layout: str = "flat",
        as_numpy: bool = False,
        layout: str | None = None,
    ):
        """
        Allocate and fill one flat noise field.

        Author: B.G (03/2026)
        """
        if layout is not None:
            output_layout = layout
        noise_field = self._allocate_noise_field()
        self.white_noise(noise_field.field, amplitude, seed)
        if as_numpy:
            arr = noise_field.field.to_numpy().reshape(-1)
            noise_field.release()
            if str(output_layout).lower() == "2d":
                return arr.reshape((self.gridctx.ny, self.gridctx.nx))
            return arr
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
        output_layout: str = "flat",
        as_numpy: bool = False,
        layout: str | None = None,
    ):
        """
        Allocate and fill one flat Perlin-noise field.

        Author: B.G (03/2026)
        """
        if layout is not None:
            output_layout = layout
        noise_field = self._allocate_noise_field()
        perm_field = ppool.taipool.get_tpfield(ti.i32, (512,))

        try:
            perm_field.from_numpy(self._fisher_yates_permutation(seed))
            fx = float(frequency_x if frequency_x is not None else frequency)
            fy = float(frequency_y if frequency_y is not None else frequency)
            self.perlin_noise(
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

        if as_numpy:
            arr = noise_field.field.to_numpy().reshape(-1)
            noise_field.release()
            if str(output_layout).lower() == "2d":
                return arr.reshape((self.gridctx.ny, self.gridctx.nx))
            return arr
        return noise_field
