"""
Backend-agnostic pool contracts.

Defines the blueprint that every pool backend (Taichi fields, ndarrays,
quadrants, cupy, ...) must implement. No allocation logic here
- this is the interface only.

Author: B.G (07/2026)
"""

from abc import ABC, abstractmethod
from typing import Any


class DataHandle(ABC):
    """
    Opaque handle to one pooled backend resource (a Taichi field, ndarray, ...).

    Owns the acquire/release lifecycle: `release()` returns the handle to its
    pool for reuse without freeing memory; `destroy()` actually frees it.

    Attributes:
        id: Unique handle identifier, assigned by the backend.
        dtype: Backend-native or common dtype tag for this resource.
        shape: Resource dimensions. () for a scalar.
        in_use: True between acquire() and release().

    Author: B.G (07/2026)
    """

    id: int
    dtype: Any
    shape: tuple[int, ...]
    in_use: bool

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
