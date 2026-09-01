"""
Taichi backend implementation of Pool.

Author: B.G (07/2026)
"""

from ._bucketed_pool import BucketedPool
from .taichi_handle import TaichiDataHandle


class TaichiPool(BucketedPool):
    """
    Pool manager for TaichiDataHandle, bucketed by (dtype, shape).

    Author: B.G (07/2026)
    """

    _handle_cls = TaichiDataHandle
