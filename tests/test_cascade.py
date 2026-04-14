import sys
import os
import numpy as np

print("[SYSTEM] Script triggered successfully. Loading imports...")

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.cpwl import CPwLCurve
from refinet.algebra.system import AffineRefinementSystem
from refinet.cascade.exact import BlockCascadeEngine

def main():
    print("\n--- Level 3 Exact Cascade Validation (Block Vectorization) ---")
    
    M = 2
    A_0 = [[0.5, -0.1], [0.1, 0.5]]
    A_1 = [[0.5, 0.1], [-0.1, 0.5]]
    A_matrices = [A_0, A_1]
    
    # Support spans [-0.5, 1.5]
    gamma_0_bps = [(-0.5, [0.0, 0.0]), (0.0, [0.0, 0.0]), (0.5, [1.0, 2.0]), (1.0, [0.0, 0.0]), (1.5, [0.0, 0.0])]
    gamma_0 = CPwLCurve(gamma_0_bps, p=2)
    
    B_bps = [(-0.5, [0.0, 0.0]), (0.0, [0.0, 0.0]), (0.3, [-0.5, 0.8]), (0.7, [0.5, -0.4]), (1.0, [0.0, 0.0]), (1.5, [0.0, 0.0])]
    B_curve = CPwLCurve(B_bps, p=2)
    
    n_depth = 4
    c_weights = [1.0, 0.5, 0.25, 0.125] 
    
    # Level 2 Baseline
    system = AffineRefinementSystem(A_matrices, M=M, B_forcing=B_curve, p=2)
    
    # Level 3 Block Cascade (Support bounds: K_min = -1, K_max = 2)
    engine = BlockCascadeEngine(A_matrices, M, K_min=-1, K_max=2)
    
    t_targets = [0.123, 0.5, 0.777, 0.991]
    
    print(f"Evaluating depth n={n_depth} with target dimension p=2...")
    print(f"{'Target (t)':<12} | {'Level 2 (Recursive)':<35} | {'Level 3 (Cascade)':<35} | {'Diff'}")
    print("-" * 100)
    
    passed_all = True
    for t in t_targets:
        # Construct Level 2 exact weighted affine iterate
        v_gamma_L2 = system.evaluate_V_n(gamma_0, n_depth, t)
        s_n_L2 = [0.0, 0.0]
        for r in range(n_depth):
            term = system.evaluate_V_n(B_curve, r, t)
            weighted_term = [c_weights[r] * term[0], c_weights[r] * term[1]]
            s_n_L2 = [s_n_L2[0] + weighted_term[0], s_n_L2[1] + weighted_term[1]]
            
        pt_L2 = np.array(v_gamma_L2) + np.array(s_n_L2)
        
        # Evaluate Level 3
        pt_L3 = engine.evaluate_affine(t, n_depth, gamma_0, B_curve, c_weights)
        
        diff = np.linalg.norm(pt_L2 - pt_L3)
        if diff > 1e-9: passed_all = False
        
        l2_str = f"[{pt_L2[0]:.6f}, {pt_L2[1]:.6f}]"
        l3_str = f"[{pt_L3[0]:.6f}, {pt_L3[1]:.6f}]"
        print(f"{t:<12.5f} | {l2_str:<35} | {l3_str:<35} | {diff:.2e}")

    print("-" * 100)
    if passed_all:
        print("SUCCESS! Level 3 O(n) Block Cascade perfectly matches Level 2 recursion.")
    else:
        print("FAILED. Numerical discrepancies detected.")

if __name__ == "__main__":
    main()