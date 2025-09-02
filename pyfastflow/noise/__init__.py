"""
Noise generation module for PyFastFlow.

Provides GPU-accelerated noise generation functions for procedural content creation
in geomorphological and hydrological modeling. All noise functions support seeded
generation for reproducible results and can return either Taichi fields or numpy arrays.

Noise Types:
- White Noise: Independent random values with flat frequency spectrum
- Red Noise: Correlated noise with 1/f² power spectrum for smooth variations  
- Perlin Noise: Coherent noise with natural patterns using gradient interpolation

Core Features:
- GPU acceleration via Taichi kernels
- Memory pool integration for efficient field management
- Seeded random generation for reproducibility
- Flexible output formats (numpy arrays or Taichi fields)
- Customizable parameters for each noise type

Usage:
    import pyfastflow as pf
    
    # Generate white noise for random perturbations
    white = pf.noise.white_noise(100, 100, amplitude=0.5, seed=42)
    
    # Generate smooth red noise for terrain base
    red = pf.noise.red_noise(256, 256, amplitude=10.0, decay_factor=0.9)
    
    # Generate detailed Perlin noise for realistic terrain
    perlin = pf.noise.perlin_noise(512, 512, frequency=16.0, octaves=6, 
                                  persistence=0.6, amplitude=100.0)

Author: B.G.
"""

from .white_noise import white_noise, white_noise_kernel
from .red_noise import red_noise, red_noise_kernel  
from .perlin_noise import perlin_noise, perlin_noise_kernel

# Export all noise generation functions
__all__ = [
    "white_noise", "white_noise_kernel",
    "red_noise", "red_noise_kernel", 
    "perlin_noise", "perlin_noise_kernel"
]