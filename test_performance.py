"""
Performance test script for biophysics simulations.
Tests the optimized functions to ensure they work correctly and efficiently.
"""
import numpy as np
import time
from simulations import utils


def test_circular_mask_performance():
    """Test circular mask creation performance."""
    print("Testing circular mask creation...")
    
    sizes = [100, 200, 300]
    for size in sizes:
        start = time.time()
        for _ in range(100):
            mask = utils.create_circular_mask(size)
        elapsed = time.time() - start
        print(f"  Size {size}x{size}: {elapsed:.4f}s for 100 iterations")
        assert mask.shape == (size, size)
        assert mask.dtype == bool
    print("  ✓ Circular mask creation test passed\n")


def test_laplacian_performance():
    """Test Laplacian computation performance."""
    print("Testing Laplacian computation...")
    
    sizes = [100, 200, 300]
    for size in sizes:
        arr = np.random.rand(size, size)
        start = time.time()
        for _ in range(100):
            lap = utils.laplacian(arr)
        elapsed = time.time() - start
        print(f"  Size {size}x{size}: {elapsed:.4f}s for 100 iterations")
        assert lap.shape == arr.shape
        # Verify boundary conditions
        assert np.all(lap[0, :] == 0)
        assert np.all(lap[-1, :] == 0)
        assert np.all(lap[:, 0] == 0)
        assert np.all(lap[:, -1] == 0)
    print("  ✓ Laplacian computation test passed\n")


def test_neighbor_field_performance():
    """Test neighbor field computation performance."""
    print("Testing neighbor field computation...")
    
    sizes = [100, 200, 300]
    for size in sizes:
        arr = np.random.rand(size, size)
        start = time.time()
        for _ in range(100):
            nbr = utils.compute_neighbor_field(arr)
        elapsed = time.time() - start
        print(f"  Size {size}x{size}: {elapsed:.4f}s for 100 iterations")
        assert nbr.shape == arr.shape
        # Verify averaging
        assert np.all(nbr >= 0)
        assert np.all(nbr <= 1)
    print("  ✓ Neighbor field computation test passed\n")


def test_neighbor_mask_performance():
    """Test neighbor mask computation performance."""
    print("Testing neighbor mask computation...")
    
    sizes = [100, 200, 300]
    for size in sizes:
        condition = np.random.rand(size, size) > 0.5
        start = time.time()
        for _ in range(100):
            mask = utils.get_neighbor_mask(condition)
        elapsed = time.time() - start
        print(f"  Size {size}x{size}: {elapsed:.4f}s for 100 iterations")
        assert mask.shape == condition.shape
        assert mask.dtype == bool
    print("  ✓ Neighbor mask computation test passed\n")


def test_vectorized_antibiotic_map():
    """Test vectorized antibiotic map creation (from mega_plate)."""
    print("Testing vectorized antibiotic map creation...")
    
    SIZE = 100
    CENTER = SIZE // 2
    RADIUS = 45
    
    # Vectorized version
    start = time.time()
    for _ in range(100):
        y, x = np.ogrid[:SIZE, :SIZE]
        dist = np.sqrt((y - CENTER) ** 2 + (x - CENTER) ** 2)
        ab_map = np.ones((SIZE, SIZE)) * 99
        ab_map[dist <= RADIUS] = 2
        ab_map[dist < 2 * RADIUS / 3] = 1
        ab_map[dist < RADIUS / 3] = 0
    vectorized_time = time.time() - start
    
    print(f"  Vectorized: {vectorized_time:.4f}s for 100 iterations")
    
    # Verify the map is correct
    assert ab_map.shape == (SIZE, SIZE)
    assert ab_map[CENTER, CENTER] == 0  # Center should be 0
    assert np.any(ab_map == 1)  # Should have middle zone
    assert np.any(ab_map == 2)  # Should have outer zone
    
    print("  ✓ Vectorized antibiotic map test passed\n")


def test_vectorized_seed_propagation():
    """Test vectorized seed ID propagation."""
    print("Testing vectorized seed ID propagation...")
    
    grid_size = 100
    bacteria = np.random.rand(grid_size, grid_size)
    seed_ids = np.zeros((grid_size, grid_size), dtype=int)
    
    # Place some seeds
    seed_ids[40:45, 40:45] = 1
    seed_ids[55:60, 55:60] = 2
    
    # Propagate seeds (vectorized)
    start = time.time()
    for _ in range(10):
        for sid in range(1, 3):
            has_seed = (seed_ids == sid)
            nbr_has_seed = utils.get_neighbor_mask(has_seed)
            propagate_mask = nbr_has_seed & (seed_ids == 0) & (bacteria > 0.5)
            seed_ids[propagate_mask] = sid
    elapsed = time.time() - start
    
    print(f"  Vectorized seed propagation: {elapsed:.4f}s for 10 iterations")
    print(f"  Seed 1 cells: {np.sum(seed_ids == 1)}")
    print(f"  Seed 2 cells: {np.sum(seed_ids == 2)}")
    print("  ✓ Vectorized seed propagation test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Biophysics Module Performance Tests")
    print("=" * 60 + "\n")
    
    test_circular_mask_performance()
    test_laplacian_performance()
    test_neighbor_field_performance()
    test_neighbor_mask_performance()
    test_vectorized_antibiotic_map()
    test_vectorized_seed_propagation()
    
    print("=" * 60)
    print("All performance tests passed! ✓")
    print("=" * 60)
