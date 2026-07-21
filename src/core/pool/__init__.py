"""
New backend-agnostic pool architecture (DataHandle/Pool ABCs + backends).

Physically lives under ./src/core/pool, exposed at import time as
pyfastflow.experimental.core.pool via the path shim in
pyfastflow/experimental/core/__init__.py.

Author: B.G (07/2026)
"""
