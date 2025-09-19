import numpy as np

def create_tiled_array(shape, tile_size, shift=(0, 0)):
    """
    Creates a 2D array with tiles of size tile_size x tile_size.
    Each tile is filled with a single incrementing number (1, 2, 3, ...).
    
    Parameters:
    -----------
    shape : tuple
        Shape of the output array (height, width)
    tile_size : int
        Size of each square tile (N for N×N tiles)
    shift : tuple, optional
        Periodic shift in (y, x) direction. Default is (0, 0)
    
    Returns:
    --------
    np.ndarray
        uint32 array with tiled pattern where each tile has a single value
    """
    height, width = shape
    
    # Calculate how many tiles we need in each dimension
    tiles_y = (height + tile_size - 1) // tile_size  # Ceiling division
    tiles_x = (width + tile_size - 1) // tile_size
    
    # Create array to hold tile numbers
    tile_numbers = np.arange(1, tiles_y * tiles_x + 1, dtype=np.uint32)
    tile_numbers = tile_numbers.reshape(tiles_y, tiles_x)
    
    # Expand each tile number to fill a tile_size x tile_size block
    # Using repeat to expand each element
    expanded = np.repeat(np.repeat(tile_numbers, tile_size, axis=0), tile_size, axis=1)
    
    # Crop to exact desired shape
    result = expanded[:height, :width]
    
    # Apply periodic shift if specified
    if shift != (0, 0):
        shift_y, shift_x = shift
        result = np.roll(result, shift_y, axis=0)
        result = np.roll(result, shift_x, axis=1)
    
    return result