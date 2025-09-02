"""
Perlin noise generation for PyFastFlow.

Provides GPU-accelerated Perlin noise generation with multiple octaves for
natural-looking procedural terrain and texture generation in geomorphological
and hydrological modeling applications.

Author: B.G.
"""

import taichi as ti
import numpy as np
from .. import pool


@ti.func
def fade(t: ti.f32) -> ti.f32:
    """Perlin fade function: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@ti.func
def lerp(t: ti.f32, a: ti.f32, b: ti.f32) -> ti.f32:
    """Linear interpolation between a and b by factor t"""
    return a + t * (b - a)


@ti.func
def grad(hash_val: ti.i32, x: ti.f32, y: ti.f32) -> ti.f32:
    """Compute dot product of gradient vector and distance vector"""
    h = hash_val & 3
    u = x if h < 2 else y
    v = y if h < 2 else x
    return (-u if (h & 1) != 0 else u) + (-v if (h & 2) != 0 else v)


@ti.func
def perlin_noise_at(x: ti.f32, y: ti.f32, seed: ti.i32) -> ti.f32:
    """
    Generate Perlin noise value at specific coordinates.
    
    Args:
        x, y: Coordinates for noise evaluation
        seed: Seed for permutation table generation
        
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
    
    # Hash coordinates of 4 cube corners
    # Use seed to vary the permutation table
    base_seed = seed * 31
    A = (X + base_seed * 17) & 255
    B = (X + 1 + base_seed * 17) & 255
    AA = (A + Y + base_seed * 23) & 255  
    AB = (A + Y + 1 + base_seed * 23) & 255
    BA = (B + Y + base_seed * 23) & 255
    BB = (B + Y + 1 + base_seed * 23) & 255
    
    # Add blended results from 4 corners of cube
    return lerp(v, 
                lerp(u, grad(AA, x, y), grad(BA, x - 1, y)),
                lerp(u, grad(AB, x, y - 1), grad(BB, x - 1, y - 1)))


@ti.kernel
def perlin_noise_kernel(noise_field: ti.template(), frequency: ti.f32, octaves: ti.i32, 
                       persistence: ti.f32, amplitude: ti.f32, seed: ti.i32):
    """
    Generate Perlin noise with multiple octaves in a Taichi field.
    
    Args:
        noise_field: Taichi field to fill with Perlin noise
        frequency: Base frequency of the noise (higher = more detail)
        octaves: Number of octaves to combine (more = more detail levels)  
        persistence: How much each octave contributes (0.0-1.0)
        amplitude: Maximum amplitude of the final noise
        seed: Random seed for reproducible results
    """
    ny, nx = noise_field.shape
    
    for j, i in noise_field:
        # Convert grid coordinates to noise space
        x = ti.cast(i, ti.f32) * frequency / ti.cast(nx, ti.f32)
        y = ti.cast(j, ti.f32) * frequency / ti.cast(ny, ti.f32)
        
        total = 0.0
        max_value = 0.0
        current_amplitude = amplitude
        current_frequency = 1.0
        
        # Sum octaves
        for octave in range(octaves):
            octave_seed = seed + octave * 1009  # Prime offset for each octave
            noise_value = perlin_noise_at(x * current_frequency, y * current_frequency, octave_seed)
            total += noise_value * current_amplitude
            max_value += current_amplitude
            
            current_amplitude *= persistence
            current_frequency *= 2.0
        
        # Normalize to [-amplitude, amplitude] range
        if max_value > 0.0:
            noise_field[j, i] = total / max_value * amplitude
        else:
            noise_field[j, i] = 0.0


def perlin_noise(nx: int, ny: int, frequency: float = 8.0, octaves: int = 4, 
                persistence: float = 0.5, amplitude: float = 1.0, seed: int = 42, 
                return_field: bool = False):
    """
    Generate Perlin noise with multiple octaves.
    
    Creates coherent noise with natural-looking patterns by combining multiple
    scales of noise. Widely used for procedural terrain generation, texture
    synthesis, and natural phenomena modeling.
    
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
        seed: Random seed for reproducible results (default: 42)
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
    noise_field = pool.get_temp_field(ti.f32, (ny, nx))
    
    # Note: Taichi random state is global, seeding happens at ti.init() level
    perlin_noise_kernel(noise_field.field, frequency, octaves, 
                       persistence, amplitude, seed)
    
    if return_field:
        return noise_field.field
    else:
        result = noise_field.field.to_numpy()
        noise_field.release()
        return result