"""
Backend-agnostic pool contracts.

Defines the blueprint that every pool backend (Taichi fields, ndarrays,
quadrants, cupy, ...) must implement. No allocation logic here
- this is the interface only.

Author: B.G (07/2026)
"""

import itertools
from abc import ABC, abstractmethod
from typing import Any

_uid_counter = itertools.count()


def new_uid() -> int:
    """
    Return the next value from the process-wide identity counter.

    Every Parameter, Bag, Helper (device-function builder and its compiled
    artifact) and pool DataHandle is assigned one of these at construction,
    exposed as a read-only `uid` property. uids are plain integers drawn from
    this single shared counter - not stable across processes, and
    deliberately so: they identify an object within one running process and
    must never appear in generated code or a cache key.

    Author: B.G (07/2026)
    """
    return next(_uid_counter)


class DataHandle(ABC):
    """
    Opaque handle to one pooled backend resource (a Taichi field, ndarray, ...).

    Owns the acquire/release lifecycle: `release()` returns the handle to its
    pool for reuse without freeing memory; `destroy()` actually frees it.

    Attributes:
        alloc_id: Per-backend allocation counter, assigned by the backend - used
            for pool bookkeeping and not unique across backends.
        uid: Process-wide identity from the shared counter (new_uid()) - unique
            across every Parameter, Bag, Helper and DataHandle regardless of
            backend. Concrete handles set self._uid in their own __init__.
        dtype: Backend-native or common dtype tag for this resource.
        shape: Resource dimensions. () for a scalar.
        in_use: True between acquire() and release().

    Two handles from different pools can share an alloc_id; only uid identifies
    a handle on its own.

    Author: B.G (07/2026)
    """

    alloc_id: int
    dtype: Any
    shape: tuple[int, ...]
    in_use: bool

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction. See new_uid().

        Author: B.G (07/2026)
        """
        return self._uid

    @property
    @abstractmethod
    def data(self):
        """
        Return the raw backend object (ti.field, np.ndarray, ...).

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def acquire(self) -> None:
        """
        Mark this handle in_use. Called by the owning pool on checkout.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """
        Mark this handle available for reuse. Backend memory is kept.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def destroy(self) -> None:
        """
        Free the underlying backend memory. Handle is unusable afterwards.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def to_numpy(self):
        """
        Copy the resource out to a numpy array.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def from_numpy(self, arr) -> None:
        """
        Copy a numpy array into the resource in place.

        Author: B.G (07/2026)
        """
        ...


class Pool(ABC):
    """
    Blueprint for a backend-specific pool manager.

    Implementations keep handles bucketed by (dtype, shape) and reuse
    released handles before allocating new ones.

    Author: B.G (07/2026)
    """

    @abstractmethod
    def get_data(self, dtype, shape) -> DataHandle:
        """
        Return an available handle matching (dtype, shape), allocating one if needed.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def release_data(self, handle: DataHandle) -> None:
        """
        Return a handle to the pool for reuse.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def clear_unused(self) -> None:
        """
        Destroy and drop all handles currently not in_use.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def clear_all(self) -> None:
        """
        Destroy and drop every handle, regardless of in_use state.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def stats(self) -> dict:
        """
        Return {"total", "in_use", "available"} handle counts.

        Author: B.G (07/2026)
        """
        ...
