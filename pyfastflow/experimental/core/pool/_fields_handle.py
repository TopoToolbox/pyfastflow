"""
Shared DataHandle implementation for FieldsBuilder-based backends.

Taichi and Quadrants expose an identical field/FieldsBuilder API; subclasses
only pin `_backend` to their module (ti or qd).

Author: B.G (07/2026)
"""

from typing import Any, ClassVar

from .base import DataHandle


class FieldsBuilderDataHandle(DataHandle):
    """
    DataHandle backed by one field allocated via FieldsBuilder.

    Composition, not inheritance: kernels take the raw field via `.data`,
    not the handle itself - see pool/base.py design notes on why
    subclassing a field type was rejected.

    Author: B.G (07/2026)
    """

    _backend: ClassVar[Any]
    _next_id = 0

    def __init__(self, dtype: Any, shape: tuple[int, ...]):
        """
        Allocate a field of the given dtype/shape via FieldsBuilder.

        shape=() allocates a 0D scalar field, indexed as field[None].

        Author: B.G (07/2026)
        """
        cls = type(self)
        cls._next_id += 1
        self.id = cls._next_id
        self.dtype = dtype
        self.shape = tuple(shape)
        self.in_use = False

        backend = self._backend
        self._builder = backend.FieldsBuilder()
        self._field = backend.field(dtype)

        if len(self.shape) == 0:
            self._builder.place(self._field)
        elif len(self.shape) == 1:
            self._builder.dense(backend.i, self.shape).place(self._field)
        elif len(self.shape) == 2:
            self._builder.dense(backend.ij, self.shape).place(self._field)
        else:
            raise ValueError(f"Unsupported field dimensionality: {len(self.shape)}D. Only 0D, 1D, 2D supported.")

        self._snodetree = self._builder.finalize()

    @property
    def data(self):
        """
        Return the underlying field, for passing straight into kernels or
        binding as a global.

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
