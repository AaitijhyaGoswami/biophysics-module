"""
Shared utility functions for biophysics simulations.
Optimized for performance using vectorized NumPy operations.
"""
import numpy as np


def laplacian(arr):
    """
    Compute discrete 5-point stencil Laplacian for diffusion.
    
    Uses vectorized NumPy operations for optimal performance.
    Handles boundary conditions with zero-padding.
    
    Args:
        arr: 2D numpy array
        
    Returns:
        2D numpy array with Laplacian computed
    """
    lap = np.zeros_like(arr)
    lap[1:-1, 1:-1] = (
        arr[:-2, 1:-1] + arr[2:, 1:-1] +
        arr[1:-1, :-2] + arr[1:-1, 2:] -
        4 * arr[1:-1, 1:-1]
    )
    return lap


def create_circular_mask(size, center=None, radius=None):
    """
    Create a circular mask using vectorized operations.
    
    Args:
        size: Grid size (assumes square grid)
        center: Center coordinates (default: grid center)
        radius: Circle radius (default: size/2 - 2)
        
    Returns:
        Boolean mask array
    """
    if center is None:
        center = size // 2
    if radius is None:
        radius = size // 2 - 2
        
    y, x = np.ogrid[-center:size-center, -center:size-center]
    mask = x**2 + y**2 <= radius**2
    return mask


def compute_neighbor_field(arr):
    """
    Compute 4-neighbor average field efficiently.
    
    Args:
        arr: 2D numpy array
        
    Returns:
        2D numpy array with neighbor averages
    """
    return (
        np.roll(arr, 1, 0) + np.roll(arr, -1, 0) +
        np.roll(arr, 1, 1) + np.roll(arr, -1, 1)
    ) / 4.0


def get_neighbor_mask(condition_arr, offset=1):
    """
    Get mask of cells that have neighbors satisfying a condition.
    
    Optimized to avoid redundant roll operations.
    
    Args:
        condition_arr: Boolean array indicating cells satisfying condition
        offset: Neighbor offset distance (default: 1 for immediate neighbors)
        
    Returns:
        Boolean mask of cells with neighbors satisfying condition
    """
    return (
        np.roll(condition_arr, offset, 0) |
        np.roll(condition_arr, -offset, 0) |
        np.roll(condition_arr, offset, 1) |
        np.roll(condition_arr, -offset, 1)
    )
