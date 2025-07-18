import taichi as ti
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parallel_scan import inclusive_scan


def test_inclusive_scan():
    ti.init(arch=ti.gpu)
    
    # Test cases - including larger arrays to test GPU parallelism
    test_cases = [
        [1, 0, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1],
        [5, 3, 2, 8, 1, 4],
        [1, 2, 3, 4, 5, 6, 7, 8],
        list(range(1, 65)),  # 64 elements
        list(range(1, 257)), # 256 elements  
        list(range(1, 1025)) # 1024 elements
    ]
    
    for i, test_input in enumerate(test_cases):
        n = len(test_input)
        if n <= 10:
            print(f"\nTest case {i+1}: {test_input}")
        else:
            print(f"\nTest case {i+1}: Array of size {n} (first 5: {test_input[:5]})")
        
        # Taichi version
        input_field = ti.field(ti.i64, shape=n)
        output_field = ti.field(ti.i64, shape=n)
        
        # Fill input
        for j in range(n):
            input_field[j] = test_input[j]
        
        # Run inclusive scan
        inclusive_scan(input_field, output_field, n)
        
        # Get results
        taichi_result = [output_field[j] for j in range(n)]
        
        # NumPy version (reference)
        numpy_result = np.cumsum(test_input).tolist()
        
        if n <= 10:
            print(f"Taichi result: {taichi_result}")
            print(f"NumPy result:  {numpy_result}")
        else:
            print(f"Taichi result (first 5): {taichi_result[:5]}")
            print(f"NumPy result (first 5):  {numpy_result[:5]}")
        
        # Compare
        if taichi_result == numpy_result:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            if n <= 20:
                print(f"Full Taichi: {taichi_result}")
                print(f"Full NumPy:  {numpy_result}")
            return False
    
    print("\n🔥 All tests passed! Inclusive scan is working correctly!")
    return True


if __name__ == "__main__":
    test_inclusive_scan()