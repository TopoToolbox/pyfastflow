"""
Taichi Field Pool Module

Efficient pooling system for temporary Taichi fields to minimize GPU memory
allocation/deallocation overhead. Provides automatic field reuse and proper
memory management using the FieldsBuilder pattern.

The pool organizes fields by data type and shape, enabling O(1) lookup
performance and efficient field reuse across computations.

Supports 0D (scalar), 1D, and 2D fields with appropriate Taichi indexing:
- 0D fields: Single scalar values, no indexing required
- 1D fields: Linear arrays with ti.i indexing  
- 2D fields: Matrices with ti.ij indexing

Author: B. Gailleton
"""

import taichi as ti
from typing import Tuple, Any, Optional
import weakref


class TPField:
    """
    Temporary Pooled Field wrapper for Taichi fields.
    
    Provides automatic memory management for temporary GPU fields using the
    FieldsBuilder pattern. Tracks usage state and enables field reuse through
    pooling to minimize allocation overhead.
    
    Supports 0D (scalar), 1D, and 2D fields with appropriate indexing patterns.
    
    Attributes:
        id: Unique field identifier
        field: Underlying Taichi field
        in_use: Current usage status
        dtype: Field data type
        shape: Field dimensions (empty tuple () for 0D scalars)
        snodetree: Finalized field structure for memory management
        
    Author: B. Gailleton
    """
    
    _next_id = 0
    
    def __init__(self, dtype: Any, shape: Tuple[int, ...]):
        """
        Initialize TPField with specified data type and shape.
        
        Creates a Taichi field using the FieldsBuilder pattern for proper
        GPU memory management. The field is initially marked as available.
        
        Args:
            dtype: Taichi data type (ti.f32, ti.i32, etc.)
            shape: Field dimensions as tuple, int, or empty tuple/0 for scalar fields
                   - () or 0: 0D scalar field
                   - (n,): 1D field with n elements
                   - (n, m): 2D field with n×m elements
            
        Author: B. Gailleton
        """

        # Handle different shape input types and normalize to tuple
        if isinstance(shape, int):
            shape = (shape,) if shape > 0 else ()
        elif not isinstance(shape, tuple):
            shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
        
        # Handle empty tuple or zero values for 0-dimensional fields
        if not shape or (len(shape) == 1 and shape[0] == 0):
            shape = ()

        TPField._next_id += 1
        self.id = TPField._next_id
        self.in_use = False
        self.dtype = dtype
        self.shape = shape
        
        # Create field using FieldsBuilder approach for proper memory management
        self.fb = ti.FieldsBuilder()
        self.field = ti.field(dtype)
        
        # Use appropriate indexing based on dimensionality
        if len(shape) == 0:
            # 0-dimensional field (scalar) - no indexing needed
            self.fb.place(self.field)
        elif len(shape) == 1:
            # 1-dimensional field
            self.fb.dense(ti.i, shape).place(self.field)
        elif len(shape) == 2:
            # 2-dimensional field
            self.fb.dense(ti.ij, shape).place(self.field)
        else:
            raise ValueError(f"Unsupported field dimensionality: {len(shape)}D. Only 0D, 1D and 2D fields supported.")
            
        self.snodetree = self.fb.finalize()
        
        # Note: Automatic cleanup via weakref is unreliable due to Python GC timing
        # Users should explicitly call release() or use context manager
    
    def acquire(self):
        """
        Mark field as in use and unavailable for other requests.
        
        Must be called when getting a field from the pool to prevent
        double allocation of the same field.
        
        Author: B. Gailleton
        """
        self.in_use = True
    
    def release(self):
        """
        Mark field as available for reuse in the pool.
        
        Should be called when finished using a pooled field to make
        it available for future requests. Does not destroy the field.
        
        Author: B. Gailleton
        """
        self.in_use = False
    
    def destroy(self):
        """
        Destroy field and free GPU memory.
        
        Calls the snodetree destroy method to properly deallocate GPU
        memory. Should only be called when permanently removing fields
        from the pool.
        
        Author: B. Gailleton
        """
        if hasattr(self, 'snodetree') and self.snodetree is not None:
            self.snodetree.destroy()
            self.snodetree = None


    def to_numpy(self):
        return self.field.to_numpy()
    def from_numpy(self, val):
        return self.field.from_numpy(val)

    def __enter__(self):
        """Context manager entry - return the field for use."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically release field."""
        self.release()
        return False

    def __repr__(self):
        """
        Return the underlying Taichi field for direct usage.
        
        Enables transparent usage of TPField as if it were the actual
        Taichi field, allowing seamless integration with existing code.
        
        Returns:
            ti.Field: The underlying Taichi field
            
        Author: B. Gailleton
        """
        return self.field

    def __str__(self):
        return f"Taichi field from the temp pool id:{self.id} - in_use:{self.in_use} - dtype:{self.dtype} - shape:{self.shape}"

class TaiPool:
    """
    Efficient pool manager for temporary Taichi fields.
    
    Manages pools of TPField objects organized by data type and shape.
    Minimizes field allocation/deallocation overhead by reusing existing
    fields when possible. Uses dictionary-based organization for O(1)
    pool lookup performance.
    
    Attributes:
        _pools: Dictionary mapping (dtype, shape) tuples to lists of TPField objects
        
    Usage:
        pool = TaiPool()
        # 2D field
        temp_field_2d = pool.get_tpfield(ti.f32, (100, 100))
        # 1D field
        temp_field_1d = pool.get_tpfield(ti.i32, (1000,))
        # 0D scalar field
        temp_scalar = pool.get_tpfield(ti.f32, ())
        # Use temp_field...
        pool.release_field(temp_field_2d)
        
    Author: B. Gailleton
    """
    
    def __init__(self):
        """
        Initialize empty field pool.
        
        Creates dictionary structure for organizing fields by type and shape.
        No fields are pre-allocated - they are created on demand.
        
        Author: B. Gailleton
        """
        self._pools = {}  # (dtype, shape) -> [TPField]
    
    def get_tpfield(self, dtype: Any, shape: Tuple[int, ...]) -> TPField:
        """
        Get available TPField or create new one.
        
        Searches for an unused field with matching type and shape.
        If none found, creates a new TPField and adds it to the pool.
        The returned field is automatically marked as in use.
        
        Args:
            dtype: Taichi data type (ti.f32, ti.i32, etc.)
            shape: Field dimensions as tuple, int, or empty tuple for scalar
                   - (): 0D scalar field
                   - (n,): 1D field with n elements  
                   - (n, m): 2D field with n×m elements
            
        Returns:
            TPField: Ready-to-use field marked as in use
            
        Author: B. Gailleton
        """
        # Normalize shape to match TPField's shape handling
        if isinstance(shape, int):
            shape = (shape,) if shape > 0 else ()
        elif not isinstance(shape, tuple):
            shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
        
        # Handle empty tuple or zero values for 0-dimensional fields
        if not shape or (len(shape) == 1 and shape[0] == 0):
            shape = ()
            
        key = (dtype, shape)
        
        # Get or create pool for this type/shape
        if key not in self._pools:
            self._pools[key] = []
        
        pool = self._pools[key]
        
        # Try to find available field
        for tpfield in pool:
            if not tpfield.in_use:
                tpfield.acquire()
                return tpfield
        
        # Create new field if none available
        tpfield = TPField(dtype, shape)
        pool.append(tpfield)
        tpfield.acquire()
        return tpfield
    
    def release_field(self, tpfield: TPField):
        """
        Release TPField back to pool for reuse.
        
        Marks the field as available for future requests. The field
        remains in memory and is not destroyed.
        
        Args:
            tpfield: TPField to release back to pool
            
        Author: B. Gailleton
        """
        tpfield.release()
    
    def add_N_fields(self, dtype: Any, shape: Tuple[int, ...], count: int, check: bool = True):
        """
        Pre-allocate fields of given type and shape.
        
        Useful for pre-warming the pool when you know how many fields
        of a specific type will be needed. Can either ensure a minimum
        count exists or add exactly N new fields.
        
        Args:
            dtype: Taichi data type (ti.f32, ti.i32, etc.)
            shape: Field dimensions as tuple
            count: Number of fields to ensure/add
            check: If True, only add if below count. If False, add count fields regardless
            
        Author: B. Gailleton
        """
        key = (dtype, shape)
        
        if key not in self._pools:
            self._pools[key] = []
        
        pool = self._pools[key]
        existing = len(pool)
        
        for _ in range(max(0, (count - existing) if check else count)):
            pool.append(TPField(dtype, shape))
    
    def clear_unused(self):
        """
        Remove unused fields and free their GPU memory.
        
        Destroys all fields that are not currently in use and removes
        them from the pool. This frees GPU memory but requires new
        allocation for future requests of those field types.
        
        Author: B. Gailleton
        """
        for pool in self._pools.values():
            for tpfield in pool[:]:
                if not tpfield.in_use:
                    tpfield.destroy()
                    pool.remove(tpfield)

    def clear_all(self):
        """
        Forced removal of all fields and free their GPU memory.
        
        Destroys all fields and removes them from the pool. 
        This frees GPU memory EVEN IF POTENTIALLY STILL IN USE.
        
        Author: B. Gailleton
        """
        for pool in self._pools.values():
            for tpfield in pool[:]:
                tpfield.destroy()
                pool.remove(tpfield)
    
    def stats(self) -> dict:
        """
        Get comprehensive pool usage statistics.
        
        Returns:
            dict: Statistics containing:
                - total: Total number of fields across all pools
                - in_use: Number of fields currently in use
                - available: Number of fields available for use
                
        Author: B. Gailleton
        """
        total = sum(len(pool) for pool in self._pools.values())
        in_use = sum(1 for pool in self._pools.values() for tpf in pool if tpf.in_use)
        return {"total": total, "in_use": in_use, "available": total - in_use}


# Global pool instance
taipool = TaiPool()

def get_temp_field(dtype: Any, shape: Tuple[int, ...]) -> TPField:
    """
    Get temporary TPField from global pool.
    
    Convenience function that uses the global pool instance.
    Returns a TPField that can be used directly as a Taichi field
    due to its __repr__ method.
    
    Args:
        dtype: Taichi data type (ti.f32, ti.i32, etc.)
        shape: Field dimensions as tuple, int, or empty tuple for scalar
               - (): 0D scalar field
               - (n,): 1D field with n elements
               - (n, m): 2D field with n×m elements
        
    Returns:
        TPField: Ready-to-use temporary field
        
    Author: B. Gailleton
    """
    return taipool.get_tpfield(dtype, shape)

def release_temp_field(tpfield: TPField):
    """
    Release temporary TPField back to global pool.
    
    Convenience function that uses the global pool instance.
    Should be called when finished with a temporary field to
    make it available for reuse.
    
    Args:
        tpfield: TPField to release
        
    Author: B. Gailleton
    """
    taipool.release_tpfield(tpfield)

def pool_stats() -> dict:
    """
    Get statistics from the global pool.
    
    Returns:
        dict: Pool usage statistics
        
    Author: B. Gailleton
    """
    return taipool.stats()

def clear_pool():
    """
    Clear unused fields from global pool to free memory.
    
    Calls clear_unused() on the global pool instance to destroy
    all unused fields and free their GPU memory.
    
    Author: B. Gailleton
    """
    taipool.clear_unused()

def temp_field(dtype: Any, shape: Tuple[int, ...]) -> TPField:
    """
    Get temporary TPField as context manager for automatic release.
    
    Recommended usage pattern that ensures proper cleanup:
    
    with temp_field(ti.f32, (100, 100)) as field:  # 2D field
        # Use field...
        some_kernel(field)
    # Field automatically released here
    
    with temp_field(ti.i32, ()) as scalar:  # 0D scalar field
        # Use scalar...
        scalar[None] = 42  # Access scalar with [None]
    
    Args:
        dtype: Taichi data type (ti.f32, ti.i32, etc.)
        shape: Field dimensions as tuple, int, or empty tuple for scalar
               - (): 0D scalar field
               - (n,): 1D field with n elements
               - (n, m): 2D field with n×m elements
        
    Returns:
        TPField: Ready-to-use temporary field (context manager)
        
    Author: B. Gailleton
    """
    return taipool.get_tpfield(dtype, shape)