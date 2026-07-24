"""
Cupy backend implementation of DataHandle.

Author: B.G (07/2026)
"""

from typing import Any

import cupy as cp

from .base import DataHandle, new_uid


class CupyDataHandle(DataHandle):
    """
    DataHandle backed by one cupy ndarray.

    Author: B.G (07/2026)
    """

    _next_id = 0

    def __init__(self, dtype: Any, shape: tuple[int, ...]):
        """
        Allocate a cupy ndarray of the given dtype/shape.

        Author: B.G (07/2026)
        """
        CupyDataHandle._next_id += 1
        self.id = CupyDataHandle._next_id
        self._uid = new_uid()
        self.dtype = dtype
        self.shape = tuple(shape)
        self.in_use = False
        self._array = cp.empty(self.shape, dtype=dtype)

    @property
    def data(self):
        """
        Return the underlying cupy ndarray, for passing straight into a
        RawKernel launch.

        Author: B.G (07/2026)
        """
        return self._array

    def acquire(self) -> None:
        self.in_use = True

    def release(self) -> None:
        self.in_use = False

    def destroy(self) -> None:
        """
        Drop the reference; cupy's own memory pool reclaims the block for
        reuse. Unusable afterwards.

        Author: B.G (07/2026)
        """
        self._array = None

    def to_numpy(self):
        return cp.asnumpy(self._array)

    def from_numpy(self, arr) -> None:
        self._array[...] = cp.asarray(arr)
