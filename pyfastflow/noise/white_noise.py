"""
White noise generation for PyFastFlow.

Provides GPU-accelerated white noise generation with seeded random number generation
for reproducible results in geomorphological and hydrological modeling.

Author: B.G.
"""

import taichi as ti
import numpy as np
from .. import pool
from .. import constants as cte


@ti.kernel
def white_noise_kernel(noise_field: ti.template(), amplitude: cte.FLOAT_TYPE_TI, seed: ti.i32):
    """
    Generate white noise in a Taichi field.
    
    White noise has equal intensity at all frequencies with no correlation between
    adjacent values. Each value is independently and identically distributed.
    
    Args:
        noise_field: Taichi field to fill with white noise
        amplitude: Maximum amplitude of the noise (noise range: [-amplitude, amplitude])
        seed: Random seed for reproducible results
    """
    # Note: Taichi seeds the random generator globally, per-element seeding not supported in struct for
    for j, i in noise_field:
        # Generate uniform random value in [-amplitude, amplitude]
        noise_field[j, i] = (ti.random(cte.FLOAT_TYPE_TI) - 0.5) * 2.0 * amplitude


def white_noise(nx: int, ny: int, amplitude: float = 1.0, seed: int = 42, return_field: bool = False):
    """
    Generate white noise with specified dimensions.
    
    Creates a 2D array of white noise values with independent random values at each point.
    White noise is characterized by having equal power at all frequencies and no spatial
    correlation between neighboring values.
    
    Args:
        nx: Number of cells in x direction
        ny: Number of cells in y direction  
        amplitude: Maximum amplitude of noise values (default: 1.0)
                  Noise values range from -amplitude to +amplitude
        seed: Random seed for reproducible results (default: 42)
        return_field: If True, return Taichi field; if False, return numpy array (default: False)
    
    Returns:
        numpy.ndarray or taichi.Field: White noise array/field of shape (ny, nx)
        
    Example:
        # Generate 100x100 white noise with amplitude 0.5
        noise = white_noise(100, 100, amplitude=0.5, seed=123)
        
        # Get Taichi field instead of numpy array
        noise_field = white_noise(100, 100, return_field=True)
    """
    noise_field = pool.get_temp_field(cte.FLOAT_TYPE_TI, (ny, nx))
    
    # Note: Taichi random state is global, seeding happens at ti.init() level
    white_noise_kernel(noise_field.field, amplitude, seed)
    
    if return_field:
        return noise_field.field
    else:
        result = noise_field.field.to_numpy()
        noise_field.release()
        return result