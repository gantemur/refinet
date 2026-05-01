import sys
import os
import numpy as np
import random

sys.path.insert(0, os.path.abspath('src'))

from refinet.algebra.system import TensorProductSystem
from refinet.cascade.exact import TensorBlockCascadeEngine
from refinet.cascade.surrogate2d import SurrogateCascadeEngine2D

def main():
    print("--- Level 4: 2D Tensor-Product Continuous Surrogate Test ---\n")
    
    # 1. Define Mask and System
    mask_1d = {0: 0.5, 1: 1.0, 2: 0.5}
    mask_2d = {(j, k): c_j * c_k for j, c_j in mask_1d.items() for k, c_k in mask_1d.items()}
    system = TensorProductSystem(mask_coeffs=mask_2d, L1=2, L2=2)
    engine3 = TensorBlockCascadeEngine(system)
    
    # Initialize the new Level 4 Engine
    engine4 = SurrogateCascadeEngine2D(engine3)
    
    # 2. Define a generic Special Atom H(x, y)
    # Must be supported strictly inside [rho, 1-rho]^2
    rho = 0.15
    def H_atom(x, y):
        if x < rho or x > 1 - rho or y < rho or y > 1 - rho:
            return 0.0
        # A simple asymmetric pyramid inside the bounds
        return max(0.0, 1.0 - 2.0 * abs(x - 0.5) - 3.0 * abs(y - 0.5))

    print("[1/2] Engines compiled. Testing across 10 random 2D points...")
    
    n_depth = 3
    success = True
    
    for _ in range(10):
        # Generate random coordinates in [0, 1]^2
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        # --- Evaluate Level 3 (Discrete) ---
        # Get digits exactly via math.floor
        P_n = np.eye(4)
        rx, ry = x, y
        for _ in range(n_depth):
            q1, q2 = int(rx * 2), int(ry * 2)
            if q1 == 2: q1 = 1
            if q2 == 2: q2 = 1
            P_n = P_n @ engine3.T[(q1, q2)]
            rx = rx * 2 - q1
            ry = ry * 2 - q2
            
        phi3 = P_n @ np.array([H_atom(rx, ry), 0, 0, 0])
        
        # --- Evaluate Level 4 (Continuous Loop Math) ---
        phi4 = engine4.evaluate_cascade(x, y, n_depth, H_atom)
        
        diff = np.linalg.norm(phi3 - phi4)
        if diff > 1e-9:
            print(f"  FAIL at ({x:.3f}, {y:.3f})! Diff: {diff}")
            success = False
        else:
            print(f"  PASS at ({x:.3f}, {y:.3f})")

    if success:
        print("\n[2/2] SUCCESS! Level 4 Surrogate Math exactly matches Discrete Algebra.")

if __name__ == "__main__":
    main()