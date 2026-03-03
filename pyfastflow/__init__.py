"""
PyFastFlow package root.

At this stage the top-level surface only exposes the modules that have already
been migrated to the new context-driven API. Older subpackages remain in the
repository but are intentionally left out of the root namespace until they are
reworked.

Author: B.G (02/2026)
"""

__version__ = "0.1.0"
__author__ = "B.G."

# Lazy submodule loading to avoid heavy side effects at import time.
_LAZY_SUBMODULES = [
    "grid",
    "noise",
]

__all__ = list(_LAZY_SUBMODULES)

# LEGACY:
# "cli",
# "constants",
# "erodep",
# "flood",
# "flow",
# "general_algorithms",
# "io",
# "misc",
# "pool",
# "rastermanip",
# "visu",
# "visuGL",


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
