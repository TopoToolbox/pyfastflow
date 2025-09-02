"""
Perlin noise generation for PyFastFlow.

Provides GPU-accelerated Perlin noise generation with proper permutation tables
and gradient vectors for natural-looking procedural terrain and texture generation
in geomorphological and hydrological modeling applications.

Author: B.G.
"""

import taichi as ti
import numpy as np
from .. import pool


def fisher_yates_permutation(seed: int) -> np.ndarray:
    """
    Generate a permutation table using Fisher-Yates shuffle algorithm.
    
    Args:
        seed: Random seed for reproducible permutation
        
    Returns:
        512-element permutation array (256 values duplicated)
    """
    np.random.seed(seed)
    
    # Create initial sequence [0, 1, 2, ..., 255]
    perm = np.arange(256, dtype=np.int32)
    
    # Fisher-Yates shuffle
    for i in range(255, 0, -1):
        j = np.random.randint(0, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    
    # Duplicate to 512 elements for easier wrapping
    perm_512 = np.concatenate([perm, perm])
    
    return perm_512


# 8-direction 2D gradient vectors
GRADIENTS_2D = np.array([
    [1, 1], [-1, 1], [1, -1], [-1, -1],  # Diagonal gradients
    [1, 0], [-1, 0], [0, 1], [0, -1]      # Axis-aligned gradients
], dtype=np.float32)


@ti.func
def fade(t: ti.f32) -> ti.f32:
    """Perlin fade function: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@ti.func
def lerp(t: ti.f32, a: ti.f32, b: ti.f32) -> ti.f32:
    """Linear interpolation between a and b by factor t"""
    return a + t * (b - a)


@ti.func
def grad(hash_val: ti.i32, dx: ti.f32, dy: ti.f32, gradients: ti.template()) -> ti.f32:
    """Compute dot product of gradient vector and distance vector"""
    idx = hash_val & 7  # Use lower 3 bits to select from 8 gradients
    gx = gradients[idx, 0]
    gy = gradients[idx, 1]
    return gx * dx + gy * dy


@ti.func
def perlin_noise_at(x: ti.f32, y: ti.f32, perm: ti.template(), gradients: ti.template()) -> ti.f32:
    """
    Generate Perlin noise value at specific coordinates using proper permutation table.
    
    Args:
        x, y: Coordinates for noise evaluation
        perm: 512-element permutation table (read-only Taichi field)
        gradients: 8x2 gradient vector table (read-only Taichi field)
        
    Returns:
        Perlin noise value in range approximately [-1, 1]
    """
    # Find unit grid cell containing point
    X = ti.cast(ti.floor(x), ti.i32) & 255
    Y = ti.cast(ti.floor(y), ti.i32) & 255
    
    # Find relative x,y of point in cube
    x -= ti.floor(x)
    y -= ti.floor(y)
    
    # Compute fade curves for x,y
    u = fade(x)
    v = fade(y)
    
    # Hash coordinates of 4 cube corners using permutation table
    A = perm[X] + Y
    B = perm[(X + 1) & 255] + Y
    AA = perm[A & 255]
    AB = perm[(A + 1) & 255]  
    BA = perm[B & 255]
    BB = perm[(B + 1) & 255]
    
    # Add blended results from 4 corners of cube using proper gradients
    return lerp(v, 
                lerp(u, grad(AA, x, y, gradients), grad(BA, x - 1, y, gradients)),
                lerp(u, grad(AB, x, y - 1, gradients), grad(BB, x - 1, y - 1, gradients)))


@ti.kernel
def perlin_noise_kernel(noise_field: ti.template(), frequency: ti.f32, octaves: ti.i32, 
                       persistence: ti.f32, amplitude: ti.f32, perm: ti.template(), 
                       gradients: ti.template()):
    """
    Generate Perlin noise with multiple octaves in a Taichi field.
    
    Args:
        noise_field: Taichi field to fill with Perlin noise
        frequency: Base frequency of the noise (higher = more detail)
        octaves: Number of octaves to combine (more = more detail levels)  
        persistence: How much each octave contributes (0.0-1.0)
        amplitude: Maximum amplitude of the final noise
        perm: 512-element permutation table (read-only Taichi field)
        gradients: 8x2 gradient vector table (read-only Taichi field)
    """
    ny, nx = noise_field.shape
    
    for j, i in noise_field:
        # Convert grid coordinates to noise space
        x = ti.cast(i, ti.f32) * frequency / ti.cast(nx, ti.f32)
        y = ti.cast(j, ti.f32) * frequency / ti.cast(ny, ti.f32)
        
        total = 0.0
        max_value = 0.0
        current_amplitude = 1.0  # Start with unit amplitude for proper normalization
        current_frequency = 1.0
        
        # Sum octaves using the same permutation table for all octaves
        for octave in range(octaves):
            noise_value = perlin_noise_at(x * current_frequency, y * current_frequency, perm, gradients)
            total += noise_value * current_amplitude
            max_value += current_amplitude
            
            current_amplitude *= persistence
            current_frequency *= 2.0
        
        # Normalize to [-1, 1] range then scale by requested amplitude
        if max_value > 0.0:
            noise_field[j, i] = (total / max_value) * amplitude
        else:
            noise_field[j, i] = 0.0


def perlin_noise(nx: int, ny: int, frequency: float = 8.0, octaves: int = 4, 
                persistence: float = 0.5, amplitude: float = 1.0, seed: int = 42, 
                return_field: bool = False):
    """
    Generate Perlin noise with multiple octaves using proper permutation tables.
    
    Creates coherent noise with natural-looking patterns by combining multiple
    scales of noise. Uses proper permutation tables and gradient vectors for
    artifact-free, deterministic results suitable for procedural terrain generation.
    
    Args:
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        frequency: Base frequency of noise patterns (default: 8.0)
                  Higher values create more detailed, smaller-scale features
        octaves: Number of noise layers to combine (default: 4)
                More octaves add finer detail at computational cost
        persistence: Amplitude ratio between octaves (default: 0.5)
                    Controls how much each octave contributes to final result
                    Values near 0.0 emphasize large-scale features
                    Values near 1.0 emphasize fine-scale details
        amplitude: Maximum amplitude of final noise values (default: 1.0)
        seed: Random seed for reproducible permutation table (default: 42)
        return_field: If True, return Taichi field; if False, return numpy array (default: False)
    
    Returns:
        numpy.ndarray or taichi.Field: Perlin noise array/field of shape (ny, nx)
        Values range approximately from -amplitude to +amplitude
        
    Example:
        # Generate terrain-like noise with multiple detail levels
        terrain = perlin_noise(512, 512, frequency=16.0, octaves=6, 
                              persistence=0.6, amplitude=100.0, seed=456)
        
        # Generate fine-grained texture noise  
        texture = perlin_noise(256, 256, frequency=32.0, octaves=2,
                              persistence=0.3, return_field=True)
    """
    # Generate permutation table using Fisher-Yates shuffle
    perm_array = fisher_yates_permutation(seed)
    
    # Create Taichi fields for permutation table and gradients
    perm_field = ti.field(ti.i32, shape=(512,))
    gradients_field = ti.field(ti.f32, shape=(8, 2))
    
    # Copy data to Taichi fields
    perm_field.from_numpy(perm_array)
    gradients_field.from_numpy(GRADIENTS_2D)
    
    # Get noise field from pool
    noise_field = pool.get_temp_field(ti.f32, (ny, nx))
    
    # Note: Taichi random state is global, seeding happens at permutation generation level
    perlin_noise_kernel(noise_field.field, frequency, octaves, 
                       persistence, amplitude, perm_field, gradients_field)
    
    if return_field:
        return noise_field.field
    else:
        result = noise_field.field.to_numpy()
        noise_field.release()
        return result