"""
Backend-agnostic pool contracts.

Defines the blueprint that every pool backend (Taichi fields, ndarrays,
quadrants, cupy, ...) must implement. No allocation logic here
- this is the interface only.

Author: B.G (07/2026)
"""

import itertools
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

_uid_counter = itertools.count()


class PoolError(RuntimeError):
    """
    Raised on a pool lifecycle misuse: releasing a foreign or already-free
    handle, or clearing a pool that still has handles in use.

    Author: B.G (08/2026)
    """


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
        uid: Process-wide identity from the shared counter (new_uid()) - unique
            across every Parameter, Bag, Helper and DataHandle regardless of
            backend. Concrete handles set self._uid in their own __init__.
        dtype: Backend-native or common dtype tag for this resource.
        shape: Resource dimensions. () for a scalar.
        in_use: True between acquire() and release().

    Author: B.G (07/2026)
    """

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

        Raises on a handle this pool never minted, and on a double release (a
        handle already marked available) - either would let the same backing
        buffer be handed out to two owners at once.

        Author: B.G (07/2026)
        """
        ...

    @contextmanager
    def data(self, dtype, shape):
        """
        Scoped acquire/release: `with pool.data(dtype, shape) as h:` checks a
        handle out and returns it on block exit, including on exception. The
        primary acquisition API - use `get_data`/`release_data` directly only
        when a handle must outlive the acquiring scope.

        Author: B.G (08/2026)
        """
        handle = self.get_data(dtype, shape)
        try:
            yield handle
        finally:
            self.release_data(handle)

    @abstractmethod
    def clear_unused(self) -> None:
        """
        Destroy and drop all handles currently not in_use.

        Author: B.G (07/2026)
        """
        ...

    @abstractmethod
    def clear_all(self, force: bool = False) -> None:
        """
        Destroy and drop every handle.

        Raises if any handle is still `in_use` unless `force=True` is passed -
        destroying a live handle leaves its holder with a dangling device
        resource.

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
