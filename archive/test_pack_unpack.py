#!/usr/bin/env python3
import taichi as ti
import numpy as np
import sys

# Import the functions to test
from fastflow.kernels.lakeflow_bg import pack_float_index, unpack_float_index

ti.init(arch=ti.cpu)

def test_basic_pack_unpack():
    """Test basic pack/unpack functionality"""
    print("Testing basic pack/unpack...")
    
    # Test data: (float_val, int_val)
    test_cases = [
        (0.0, 0),
        (1.0, 1),
        (-1.0, 2),
        (3.14159, 42),
        (-2.71828, 100),
        (1e6, 999),
        (-1e6, 1000),
        (1e-6, 5),
        (-1e-6, 10)
    ]
    
    n = len(test_cases)
    vals = ti.field(ti.f32, shape=n)
    indices = ti.field(ti.i32, shape=n)
    unpacked_vals = ti.field(ti.f32, shape=n)
    unpacked_indices = ti.field(ti.i32, shape=n)
    
    # Fill test data
    for i, (f, idx) in enumerate(test_cases):
        vals[i] = f
        indices[i] = idx
    
    @ti.kernel
    def test_pack_unpack():
        for i in range(n):
            packed = pack_float_index(vals[i], indices[i])
            f_val, i_val = unpack_float_index(packed)
            unpacked_vals[i] = f_val
            unpacked_indices[i] = i_val
    
    test_pack_unpack()
    
    success = True
    for i, (orig_f, orig_i) in enumerate(test_cases):
        unp_f, unp_i = unpacked_vals[i], unpacked_indices[i]
        if not (np.isclose(orig_f, unp_f) and orig_i == unp_i):
            print(f"FAIL: ({orig_f}, {orig_i}) -> ({unp_f}, {unp_i})")
            success = False
        else:
            print(f"SUCCESS: ({orig_f}, {orig_i}) -> ({unp_f}, {unp_i})")

    
    if success:
        print("✓ Basic pack/unpack test passed")
    return success

def test_lexicographic_ordering():
    """Test lexicographic ordering: float first, then int"""
    print("Testing lexicographic ordering...")
    
    # Test cases in CORRECT lexicographic order
    test_cases = [
        (-10.0, 0),   # Smallest float
        (-1.0, 0),    # Same float, smaller index first  
        (-1.0, 999),  # Same float, larger index
        (0.0, 100),   # Zero
        (1.0, 0),     # Positive, smaller index first
        (1.0, 50),    # Same float, middle index
        (1.0, 999),   # Same float, largest index
        (10.0, 0),    # Largest float
    ]
    
    n = len(test_cases)
    vals = ti.field(ti.f32, shape=n)
    indices = ti.field(ti.i32, shape=n)
    packed = ti.field(ti.i64, shape=n)
    
    # Fill test data
    for i, (f, idx) in enumerate(test_cases):
        vals[i] = f
        indices[i] = idx
    
    @ti.kernel
    def get_packed_values():
        for i in range(n):
            packed[i] = pack_float_index(vals[i], indices[i])
    
    get_packed_values()
    
    # Get packed values as numpy array for checking
    packed_values = packed.to_numpy()
    
    # Check if packed values are in ascending order
    is_sorted = np.all(packed_values[:-1] <= packed_values[1:])
    
    print(f"Input (float, int) -> Packed value:")
    for i, (f, idx) in enumerate(test_cases):
        print(f"  ({f:6.1f}, {idx:3d}) -> {packed_values[i]:20d}")
    
    # Additional checks for same float values
    same_float_indices = []
    for i in range(len(test_cases)):
        for j in range(i+1, len(test_cases)):
            if np.isclose(test_cases[i][0], test_cases[j][0]):
                same_float_indices.append((i, j))
    
    same_float_correct = True
    for i, j in same_float_indices:
        if test_cases[i][1] < test_cases[j][1] and packed_values[i] >= packed_values[j]:
            print(f"FAIL: Same float {test_cases[i][0]}, but index {test_cases[i][1]} < {test_cases[j][1]} should give smaller packed value")
            same_float_correct = False
        elif test_cases[i][1] > test_cases[j][1] and packed_values[i] <= packed_values[j]:
            print(f"FAIL: Same float {test_cases[i][0]}, but index {test_cases[i][1]} > {test_cases[j][1]} should give larger packed value")
            same_float_correct = False
    
    if is_sorted and same_float_correct:
        print("✓ Lexicographic ordering test passed")
    else:
        print("✗ Lexicographic ordering test failed")
        if not is_sorted:
            print("  Packed values are not in ascending order")
    
    return is_sorted and same_float_correct

def test_simple_cases():
    """Test simple cases"""
    print("Testing simple cases...")
    
    # Simple test cases
    test_cases = [
        (1.0, 1),
        (2.0, 2),
        (-1.0, 3),
        (0.0, 4),
    ]
    
    n = len(test_cases)
    vals = ti.field(ti.f32, shape=n)
    indices = ti.field(ti.i32, shape=n)
    unpacked_vals = ti.field(ti.f32, shape=n)
    unpacked_indices = ti.field(ti.i32, shape=n)
    
    # Fill test data
    for i, (f, idx) in enumerate(test_cases):
        vals[i] = f
        indices[i] = idx
    
    @ti.kernel
    def run_test():
        for i in range(n):
            packed = pack_float_index(vals[i], indices[i])
            f_val, i_val = unpack_float_index(packed)
            unpacked_vals[i] = f_val
            unpacked_indices[i] = i_val
    
    run_test()
    
    success = True
    for i, (orig_f, orig_i) in enumerate(test_cases):
        unp_f, unp_i = unpacked_vals[i], unpacked_indices[i]
        if not (np.isclose(orig_f, unp_f) and orig_i == unp_i):
            print(f"FAIL: ({orig_f}, {orig_i}) -> ({unp_f}, {unp_i})")
            success = False
    
    if success:
        print("✓ Simple cases test passed")
    return success

def test_massive_array():
    """Test with massive array and compare to numpy lexicographic sort"""
    print("Testing massive array (10000 elements)...")
    
    np.random.seed(42)
    n = 10000
    
    # Generate random test data
    floats = np.random.uniform(-1e6, 1e6, n).astype(np.float32)
    indices = np.random.randint(0, 750000, n, dtype=np.int32)
    
    # Taichi approach
    vals = ti.field(ti.f32, shape=n)
    idx_field = ti.field(ti.i32, shape=n)
    packed_field = ti.field(ti.i64, shape=n)
    
    # Fill fields
    vals.from_numpy(floats)
    idx_field.from_numpy(indices)
    
    @ti.kernel
    def pack_all():
        for i in range(n):
            packed_field[i] = pack_float_index(vals[i], idx_field[i])
    
    pack_all()
    packed_values = packed_field.to_numpy()
    
    # Numpy approach - create tuples and sort
    pairs = list(zip(floats, indices))
    numpy_sorted_pairs = sorted(pairs)
    
    # Sort our packed values and unpack them
    sorted_indices = np.argsort(packed_values)
    taichi_sorted_pairs = [(floats[i], indices[i]) for i in sorted_indices]
    
    # Compare the results
    matches = 0
    for i in range(n):  # Check first 100 to avoid spam
        if numpy_sorted_pairs[i] == taichi_sorted_pairs[i]:
            matches += 1
        elif i < 10:  # Show first few mismatches
            print(f"  Mismatch {i}: numpy={numpy_sorted_pairs[i]}, taichi={taichi_sorted_pairs[i]}")
    
    success = (numpy_sorted_pairs == taichi_sorted_pairs)
    
    if success:
        print(f"✓ Massive array test passed ({n} elements, all match)")
    else:
        print(f"✗ Massive array test failed ({matches}/{min(100, n)} first elements match)")
        
        # Show some stats
        print(f"  First 5 numpy: {numpy_sorted_pairs[:5]}")
        print(f"  First 5 taichi: {taichi_sorted_pairs[:5]}")
    
    return success

def test_positive_values():
    """Test lexicographic ordering with only positive values"""
    print("Testing with positive values only...")
    
    test_cases = [
        (0., 1),     # Smallest float
        (0., 5),     # Smallest float
        (0.1, 1),     # Smallest float
        (1.0, 1),     # Same float, smaller index first
        (1.0, 50),    
        (1.0, 999),   
        (2.0, 1),     
        (5.0, 100),   
        (10.0, 1),    
        (100.0, 1),   # Largest float
    ]
    
    n = len(test_cases)
    vals = ti.field(ti.f32, shape=n)
    indices = ti.field(ti.i32, shape=n)
    packed = ti.field(ti.i64, shape=n)
    
    # Fill test data
    for i, (f, idx) in enumerate(test_cases):
        vals[i] = f
        indices[i] = idx
    
    @ti.kernel
    def get_packed_values():
        for i in range(n):
            packed[i] = pack_float_index(vals[i], indices[i])
    
    get_packed_values()
    
    # Get packed values as numpy array for checking
    packed_values = packed.to_numpy()
    
    # Check if packed values are in ascending order
    is_sorted = all(packed_values[i] <= packed_values[i+1] for i in range(len(packed_values)-1))
    
    print(f"Input (float, int) -> Packed value:")
    for i, (f, idx) in enumerate(test_cases):
        print(f"  ({f:6.1f}, {idx:3d}) -> {packed_values[i]:20d}")
    
    if is_sorted:
        print("✓ Positive values test passed")
    else:
        print("✗ Positive values test failed")
    
    return is_sorted

def main():
    print("Testing pack_float_index and unpack_float_index functions")
    print("=" * 60)
    
    tests = [
        test_basic_pack_unpack,
        test_lexicographic_ordering,
        test_positive_values,
        test_simple_cases,
        test_massive_array
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)