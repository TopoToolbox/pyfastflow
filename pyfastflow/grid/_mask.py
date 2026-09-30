"""Public access to optional grid-owned masks in a Program."""

import numpy as np

from pyfastflow.core.context.slot import ProgramError


class GridMaskAccessor:
    __slots__ = ("_program", "_leaf", "_option")

    def __init__(self, program, leaf, option):
        self._program = program
        self._leaf = leaf
        self._option = option

    def _parameter(self):
        self._program._check_open()
        try:
            return self._program._bundle_params["grid"][self._leaf]
        except KeyError as exc:
            raise ProgramError(
                f"{self._leaf.lower()!r} is unavailable; construct the "
                f"program with {self._option}"
            ) from exc

    def from_numpy(self, array):
        array = np.asarray(array)
        shape = (self._program.ny, self._program.nx)
        if array.shape != shape:
            raise ProgramError(
                f"{self._leaf.lower()!r}: expected shape {shape}, "
                f"got {array.shape}"
            )
        self._parameter().set(array.astype(np.uint8, copy=False).reshape(-1))

    def to_numpy(self):
        return self._parameter().handle().to_numpy().reshape(
            self._program.ny, self._program.nx,
        )

    @property
    def array(self):
        return self._parameter().handle().array

    @property
    def shape(self):
        return (self._program.ny, self._program.nx)

    @property
    def dtype(self):
        return np.dtype(np.uint8)
