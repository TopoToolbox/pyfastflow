"""
Landscape-evolution tools for PyFastFlow.

The cleaned package root exposes the context-driven API through ``LEMContext``.
Legacy SPL and uplift experiments remain available under
``pyfastflow.erodep.legacy`` until the migration is complete.

Author: B.G (02/2026)
"""

from .lemcontext import LEMContext
from . import runtime

__all__ = ["LEMContext", "runtime"]

# LEGACY:
# - pyfastflow.erodep.legacy
