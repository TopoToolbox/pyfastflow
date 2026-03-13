"""
Landscape-evolution tools for PyFastFlow.

The cleaned package root exposes the context-driven API through ``LEMContext``.
Legacy SPL and uplift experiments remain available under
``pyfastflow.erodep.legacy`` until the migration is complete.

Author: B.G (02/2026)
"""

from .lemcontext import LEMContext

__all__ = ["LEMContext"]

# LEGACY:
# - pyfastflow.erodep.legacy
