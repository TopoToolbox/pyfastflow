"""
Red noise (Brownian noise) generation for PyFastFlow.

Provides GPU-accelerated red noise generation with true 1/f² power spectrum using
spectral shaping via FFT. Creates isotropic, smooth, correlated variations suitable
for natural phenomena modeling without directional artifacts.

Author: B.G.
"""

import taichi as ti
import numpy as np
from .. import pool

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def generate_red_noise_spectral(ny: int, nx: int, seed: int, amplitude: float, eps: float = 1e-12):
    """
    Generate true 2D red noise with 1/f² power spectrum using spectral shaping.
    
    Creates isotropic red noise by filtering white noise in the frequency domain.
    The power spectral density follows PSD ∝ 1/‖k‖², producing smooth, naturally
    correlated variations without directional bias.
    
    Args:
        ny: Grid height
        nx: Grid width  
        seed: Random seed for reproducible white noise generation
        amplitude: Target amplitude for final scaling
        eps: Small value to avoid division by zero at DC component
        
    Returns:
        numpy.ndarray: Red noise array of shape (ny, nx)
    """
    # Set random seed for reproducible white noise
    np.random.seed(seed)
    
    # Generate white noise W ∈ R^(ny×nx)
    W = np.random.randn(ny, nx).astype(np.float32)
    
    # Compute 2D FFT of white noise
    F = np.fft.fft2(W)
    
    # Build radial frequency grid
    kx = np.fft.fftfreq(nx, d=1.0)  # Normalized frequencies [-0.5, 0.5)
    ky = np.fft.fftfreq(ny, d=1.0)
    
    # Create 2D frequency magnitude grid: K = sqrt(kx² + ky²)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')  
    K = np.sqrt(KX**2 + KY**2)
    
    # Define amplitude filter A = 1 / max(eps, K) for 1/f² spectrum
    # Set DC component (0,0) to 0 to remove mean
    A = 1.0 / np.maximum(eps, K)
    A[0, 0] = 0.0  # Kill DC component
    
    # Apply spectral filter in frequency domain
    F *= A
    
    # Inverse FFT to get red noise in spatial domain
    R = np.fft.ifft2(F).real.astype(np.float32)
    
    # Zero-mean the result (should already be close to zero due to DC=0)
    R -= np.mean(R)
    
    # Scale to target amplitude range
    R_max = np.max(np.abs(R))
    if R_max > eps:
        R *= amplitude / R_max
    
    return R


@ti.kernel
def copy_to_field(source: ti.types.ndarray(), target: ti.template()):
    """Copy numpy array data to Taichi field."""
    for j, i in target:
        target[j, i] = source[j, i]


def red_noise(nx: int, ny: int, amplitude: float = 1.0, decay_factor: float = 0.8, 
              seed: int = 42, return_field: bool = False):
    """
    Generate red noise with true 1/f² power spectrum using spectral shaping.
    
    Creates isotropic red noise with power spectral density proportional to 1/f².
    Unlike AR(1) filtering, this method produces truly isotropic correlations
    without directional artifacts, making it ideal for natural phenomena modeling
    such as terrain elevation, temperature variations, or flow disturbances.
    
    The spectral shaping approach generates white noise, transforms to frequency
    domain, applies 1/f amplitude filter, and transforms back to spatial domain.
    This ensures the correct power spectrum across all frequencies and directions.
    
    Args:
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        amplitude: Maximum amplitude of noise values (default: 1.0)
        decay_factor: Legacy parameter, ignored (kept for API compatibility)
                     Red noise characteristics are determined by 1/f² spectrum
        seed: Random seed for reproducible results (default: 42)
        return_field: If True, return Taichi field; if False, return numpy array (default: False)
    
    Returns:
        numpy.ndarray or taichi.Field: Red noise array/field of shape (ny, nx)
        
    Example:
        # Generate smooth red noise for terrain perturbation
        terrain_noise = red_noise(256, 256, amplitude=10.0, seed=789)
        
        # Generate isotropic red noise field
        flow_noise = red_noise(100, 100, amplitude=2.0, return_field=True)
        
    Note:
        The decay_factor parameter is ignored in this implementation as the
        correlation structure is determined by the 1/f² power spectrum.
        This ensures proper isotropic behavior without directional bias.
    """
    # Generate red noise using spectral shaping
    red_array = generate_red_noise_spectral(ny, nx, seed, amplitude)
    
    if return_field:
        # Create Taichi field and copy data
        noise_field = pool.get_temp_field(ti.f32, (ny, nx))
        
        # Convert numpy array to Taichi ndarray for efficient copying
        red_ndarray = ti.ndarray(ti.f32, shape=(ny, nx))
        red_ndarray.from_numpy(red_array)
        
        # Copy to field using kernel
        copy_to_field(red_ndarray, noise_field.field)
        
        return noise_field.field
    else:
        return red_array


# Legacy kernel for backwards compatibility (not used in spectral implementation)
@ti.kernel
def red_noise_kernel(noise_field: ti.template(), amplitude: ti.f32, decay_factor: ti.f32, seed: ti.i32):
    """
    Legacy red noise kernel using AR(1) filtering (deprecated).
    
    This kernel is kept for backwards compatibility but is not used by the
    main red_noise() function, which now uses proper spectral shaping for
    true 1/f² power spectrum generation.
    
    Args:
        noise_field: Taichi field to fill with red noise
        amplitude: Maximum amplitude of the noise output
        decay_factor: Controls correlation decay (0.0-1.0, higher = more correlation)
        seed: Random seed for reproducible results (unused in new implementation)
    """
    ny, nx = noise_field.shape
    
    # First pass: generate white noise and apply horizontal correlation
    for j in range(ny):
        prev_value = 0.0
        for i in range(nx):
            # Note: Using global random state (seeded before kernel call)
            white_sample = (ti.random(ti.f32) - 0.5) * 2.0
            
            # Apply recursive filter for correlation
            filtered_value = decay_factor * prev_value + (1.0 - decay_factor) * white_sample
            noise_field[j, i] = filtered_value
            prev_value = filtered_value
    
    # Second pass: apply vertical correlation
    for i in range(nx):
        prev_value = noise_field[0, i]
        for j in range(1, ny):
            current_value = noise_field[j, i]
            filtered_value = decay_factor * prev_value + (1.0 - decay_factor) * current_value
            noise_field[j, i] = filtered_value
            prev_value = filtered_value
    
    # Normalize and scale to requested amplitude
    for j, i in noise_field:
        noise_field[j, i] *= amplitude