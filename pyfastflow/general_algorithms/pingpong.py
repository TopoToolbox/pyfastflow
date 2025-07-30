"""
Ping-Pong Buffer Management

This module provides utilities for managing ping-pong (double) buffering in
iterative algorithms. Ping-pong buffering is essential for parallel algorithms
where data dependencies prevent in-place updates.

Concept:
    In iterative algorithms, reading from and writing to the same buffer creates
    race conditions in parallel execution. Ping-pong buffering alternates between
    two buffers: one for reading (source) and one for writing (destination),
    swapping roles each iteration.

Key Features:
    - Automatic buffer role determination based on iteration state
    - Thread-safe state tracking for parallel execution
    - Efficient buffer copying with selective updates
    - Optimized for GPU execution with Taichi

Applications:
    - Cellular automata (Conway's Game of Life, etc.)
    - Iterative solvers (Jacobi, Gauss-Seidel variations)
    - Wave propagation simulations
    - Flood routing and flow accumulation
    - Any algorithm requiring temporal data dependencies

Author: B. Gailleton
"""
import taichi as ti



@ti.func
def getSrc(src: ti.template(), tid: int, iteration: int) -> bool:
    """
    Determine which buffer to read from based on ping-pong state.
    
    This function implements the core ping-pong logic by examining the state
    array to determine whether a thread should read from the primary or
    alternate buffer. The state encodes both the last update iteration and
    the buffer selection as a signed value.
    
    Args:
        src: Ping-pong state array storing signed iteration values
        tid: Thread/node ID for the current processing element
        iteration: Current iteration number (0-based)
        
    Returns:
        bool: Buffer selection flag
              - True = read from alternate buffer
              - False = read from primary buffer
              
    State Encoding:
        - Positive values: last updated using alternate buffer
        - Negative values: last updated using primary buffer
        - Absolute value: iteration number when last updated + 1
        
    Algorithm Logic:
        1. Check if entry is negative (initially assume flip)
        2. If |entry| == iteration+1, this thread was updated this iteration
        3. Invert flip decision if updated this iteration
        
    Author: B. Gailleton
    """
    entry = src[tid]  # Get current ping-pong state value
    flip = entry < 0  # Initially assume we should flip if negative
    # If absolute value matches current iteration+1, invert the flip decision
    flip = (not flip) if (abs(entry) == (iteration + 1)) else flip
    return flip  # True = use alternate buffer, False = use primary buffer


@ti.func
def updateSrc(src: ti.template(), tid: int, iteration: int, flip: bool):
    """
    Update ping-pong state after processing a node.
    
    Records which buffer was used for writing at this iteration, enabling
    correct buffer selection in subsequent iterations. The state encoding
    combines iteration information with buffer selection to minimize memory
    usage while maintaining thread safety.
    
    Args:
        src: Ping-pong state array to update
        tid: Thread/node ID for the current processing element
        iteration: Current iteration number (0-based)
        flip: Buffer selection flag from getSrc()
              - True = wrote to alternate buffer
              - False = wrote to primary buffer
              
    State Update:
        Stores signed (iteration + 1):
        - Positive: if flip is True (used alternate buffer)
        - Negative: if flip is False (used primary buffer)
        
    Thread Safety:
        Each thread updates only its own state entry, ensuring no race
        conditions in parallel execution.
        
    Author: B. Gailleton
    """
    # Store signed iteration+1: positive if using alternate buffer, negative if using primary
    src[tid] = (1 if flip else -1) * (iteration + 1)


@ti.kernel
def fuse(A: ti.template(), src: ti.template(), B: ti.template(), iteration: ti.template()):
    """
    Selectively copy values from alternate buffer to primary buffer.
    
    This kernel implements the buffer fusion step in ping-pong algorithms,
    copying only those elements that were updated in the alternate buffer
    back to the primary buffer. This maintains data consistency while
    allowing efficient parallel updates.
    
    Args:
        A: Primary/destination array to update
        src: Ping-pong state array tracking buffer usage
        B: Alternate/source array with potential updates
        iteration: Current iteration number (scalar field or value)
        
    Algorithm:
        For each element tid:
        1. Check if getSrc(src, tid, iteration) returns True
        2. If True, copy B[tid] to A[tid] (element was updated in B)
        3. If False, leave A[tid] unchanged (element not updated)
        
    Use Case:
        Typically called at the end of each iteration to consolidate
        updates from the alternate buffer back to the primary buffer,
        preparing for the next iteration cycle.
        
    Performance:
        Minimizes memory bandwidth by copying only updated elements,
        rather than performing a full buffer copy each iteration.
        
    Author: B. Gailleton
    """
    for tid in A:
        # Only copy elements that were updated in the alternate buffer this iteration
        if getSrc(src, tid, iteration):
            A[tid] = B[tid]