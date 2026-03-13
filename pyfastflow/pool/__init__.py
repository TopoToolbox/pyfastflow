"""
GPU field pooling tools for PyFastFlow.

The public interface is intentionally reduced to the global ``taipool`` and the
two core types behind it. Callers should access pooled fields through:

    pyfastflow.pool.taipool.get_tpfield(...)

Author: B.G (02/2026)
"""

from .pool import TaiPool, TPField, taipool

__all__ = ["TPField", "TaiPool", "taipool"]
