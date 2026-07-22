"""
Shared bucketed Pool implementation, parameterized by a DataHandle subclass.

Author: B.G (07/2026)
"""

from typing import Any, ClassVar

from .base import DataHandle, Pool


class BucketedPool(Pool):
    """
    Pool manager bucketed by (dtype, shape). Subclasses only pin
    `_handle_cls` to the DataHandle implementation they allocate.

    Author: B.G (07/2026)
    """

    _handle_cls: ClassVar[type]

    def __init__(self):
        self._buckets: dict[tuple[Any, tuple[int, ...]], list[DataHandle]] = {}

    def get_data(self, dtype, shape) -> DataHandle:
        key = (dtype, tuple(shape))
        bucket = self._buckets.setdefault(key, [])
        for handle in bucket:
            if not handle.in_use:
                handle.acquire()
                return handle
        handle = self._handle_cls(dtype, key[1])
        handle.acquire()
        bucket.append(handle)
        return handle

    def release_data(self, handle: DataHandle) -> None:
        handle.release()

    def clear_unused(self) -> None:
        for bucket in self._buckets.values():
            for handle in bucket[:]:
                if not handle.in_use:
                    handle.destroy()
                    bucket.remove(handle)

    def clear_all(self) -> None:
        for bucket in self._buckets.values():
            for handle in bucket[:]:
                handle.destroy()
                bucket.remove(handle)

    def stats(self) -> dict:
        total = sum(len(bucket) for bucket in self._buckets.values())
        in_use = sum(1 for bucket in self._buckets.values() for h in bucket if h.in_use)
        return {"total": total, "in_use": in_use, "available": total - in_use}
