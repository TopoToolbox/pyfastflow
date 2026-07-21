"""
Quadrants backend implementation of DataHandle.

Mirrors taichi_handle.py - Quadrants' FieldsBuilder/SNodeTree lifecycle is
identical to Taichi's. Composition, not inheritance, same rationale as the
Taichi backend: kernels take the raw field via `.data`.

Author: B.G (07/2026)
"""

from typing import Any

import quadrants as qd

from .base import DataHandle


class QuadrantsDataHandle(DataHandle):
    """
    DataHandle backed by one Quadrants field.

    Author: B.G (07/2026)
    """

    _next_id = 0

    def __init__(self, dtype: Any, shape: tuple[int, ...]):
        """
        Allocate a Quadrants field of the given dtype/shape via FieldsBuilder.

        shape=() allocates a 0D scalar field, indexed as field[None].

        Author: B.G (07/2026)
        """
        QuadrantsDataHandle._next_id += 1
        self.id = QuadrantsDataHandle._next_id
        self.dtype = dtype
        self.shape = tuple(shape)
        self.in_use = False

        self._builder = qd.FieldsBuilder()
        self._field = qd.field(dtype)

        if len(self.shape) == 0:
            self._builder.place(self._field)
        elif len(self.shape) == 1:
            self._builder.dense(qd.i, self.shape).place(self._field)
        elif len(self.shape) == 2:
            self._builder.dense(qd.ij, self.shape).place(self._field)
        else:
            raise ValueError(f"Unsupported field dimensionality: {len(self.shape)}D. Only 0D, 1D, 2D supported.")

        self._snodetree = self._builder.finalize()

    @property
    def data(self):
        """
        Return the underlying qd.field, for passing straight into kernels
        or binding as a global (ndarrays cannot be bound as globals -
        Parameter/DeviceFunction wiring stays field-backed for that reason).

        Author: B.G (07/2026)
        """
        return self._field

    def acquire(self) -> None:
        self.in_use = True

    def release(self) -> None:
        self.in_use = False

    def destroy(self) -> None:
        """
        Free the field's GPU memory. Unusable afterwards.

        Author: B.G (07/2026)
        """
        if self._snodetree is not None:
            self._snodetree.destroy()
            self._snodetree = None

    def to_numpy(self):
        return self._field.to_numpy()

    def from_numpy(self, arr) -> None:
        self._field.from_numpy(arr)
