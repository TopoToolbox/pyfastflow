"""
Quadrants backend implementation of Pool.

Author: B.G (07/2026)
"""

from ._bucketed_pool import BucketedPool
from .quadrants_handle import QuadrantsDataHandle


class QuadrantsPool(BucketedPool):
    """
    Pool manager for QuadrantsDataHandle, bucketed by (dtype, shape).

    Author: B.G (07/2026)
    """

    _handle_cls = QuadrantsDataHandle
