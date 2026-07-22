"""
Cupy backend implementation of Pool.

Author: B.G (07/2026)
"""

from ._bucketed_pool import BucketedPool
from .cupy_handle import CupyDataHandle


class CupyPool(BucketedPool):
    """
    Pool manager for CupyDataHandle, bucketed by (dtype, shape).

    Author: B.G (07/2026)
    """

    _handle_cls = CupyDataHandle
