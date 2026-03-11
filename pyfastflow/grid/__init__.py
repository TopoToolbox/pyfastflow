"""
Grid tools for PyFastFlow.

The package root only exposes the current non-legacy grid context layer. Older
grid containers and neighbourer implementations remain available under the
``pyfastflow.grid.legacy`` package but are not re-exported here.

Author: B.G.
"""

from .gridcontext import GridContext

__all__ = ["GridContext"]
