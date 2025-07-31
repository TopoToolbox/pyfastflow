"""
Taichi Field Pool Module

Efficient pooling system for temporary Taichi fields to minimize allocation overhead.
Provides automatic field reuse and memory management for GPU-accelerated computations.

Classes:
    TPField: Wrapper for pooled Taichi fields with usage tracking
    TaiPool: Pool manager for temporary fields

Functions:
    get_temp_field: Get temporary field from global pool
    release_temp_field: Release field back to global pool
    pool_stats: Get pool usage statistics
    clear_pool: Clear unused fields

Usage:
    ```python
    import taichi as ti
    from pyfastflow.pool import temp_field, get_temp_field, release_temp_field
    
    # Recommended: Context manager for automatic release
    with temp_field(ti.f32, (100, 100)) as field:
        # Use field...
        some_kernel(field)
    # Field automatically released
    
    # Manual management
    temp = get_temp_field(ti.f32, (100, 100))
    # Use field...
    release_temp_field(temp)
    ```

Author: B. Gailleton
"""

from .pool import (
    TPField,
    TaiPool, 
    get_temp_field,
    release_temp_field,
    pool_stats,
    clear_pool,
    temp_field,
    taipool
)

__all__ = [
    "TPField",
    "TaiPool",
    "get_temp_field", 
    "release_temp_field",
    "pool_stats",
    "clear_pool",
    "temp_field",
    "taipool"
]