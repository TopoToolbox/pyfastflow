"""
Shared context factories for the cleaned context architecture.

Author: B.G (03/2026)
"""

from .factories import (
    APINamespace,
    CallableFactory,
    ContextFactory,
    ContextRef,
    ParameterFactory,
    flat_field_to_numpy,
    format_flat_numpy,
    require_flat_field,
    unwrap_field,
)

__all__ = [
    "APINamespace",
    "CallableFactory",
    "ContextFactory",
    "ContextRef",
    "ParameterFactory",
    "flat_field_to_numpy",
    "format_flat_numpy",
    "require_flat_field",
    "unwrap_field",
]
