import sys
import os
import numpy as np

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath('src'))

# Import our newly appended 2D classes
from refinet.algebra.system import TensorProductSystem
from refinet.cascade.exact import TensorBlockCascadeEngine

def main():
    print("--- Level 3: 2D Tensor-Product Matrix Compilation Test ---\n")
    
    # 1. Define a 2D Bilinear Mask
    # A standard 1D hat function has mask coefficients: c_0=0.5, c_1=1.0, c_2=0.5
    # The 2D tensor-product mask is simply c_{j,k} = c_j * c_k
    mask_1d = {0: 0.5, 1: 1.0, 2: 0.5}
    mask_2d = {}
    for j, c_j in mask_1d.items():
        for k, c_k in mask_1d.items():
            mask_2d[(j, k)] = c_j * c_k
            
    print(f"[1/3] Generated 2D Mask with {len(mask_2d)} coefficients.")

    # 2. Initialize Level 2 System
    # For a mask spanning indices 0 to 2, the preserved support window is [0, 2] x [0, 2]
    # Therefore, L1 = 2 and L2 = 2.
    L1, L2 = 2, 2
    system = TensorProductSystem(mask_coeffs=mask_2d, L1=L1, L2=L2)
    print(f"[2/3] Initialized TensorProductSystem (L1={L1}, L2={L2}).")

    # 3. Initialize Level 3 Cascade Engine
    print("[3/3] Compiling T_q Block Transition Matrices...")
    engine = TensorBlockCascadeEngine(system)
    
    # 4. Display the results
    expected_dim = L1 * L2
    print(f"\nSuccess! Compiled 4 matrices of shape ({expected_dim}, {expected_dim}).")
    
    for q1 in [0, 1]:
        for q2 in [0, 1]:
            q = (q1, q2)
            print(f"\n=== T_{q} ===")
            print(engine.T[q])

if __name__ == "__main__":
    main()