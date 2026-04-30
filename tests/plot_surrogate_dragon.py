import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('src'))
from refinet.geometry.library import create_dragon
from refinet.cascade.exact import BlockCascadeEngine
from refinet.cascade.surrogate import SurrogateCascadeEngine

def main():
    print("--- Proving Level 4 Continuous Topology ---")
    
    system = create_dragon('heighway')
    base_engine = BlockCascadeEngine(system, K_min=0, K_max=1)
    surrogate_engine = SurrogateCascadeEngine(base_engine)
    
    n_depth = 8
    num_points = (system.M ** n_depth) + 1
    t_vals = np.linspace(0, 1, num_points)
    
    # Base straight line profile
    def gamma_0(t):
        if t <= 0.0: return [0.0, 0.0]
        if t >= 1.0: return [1.0, 0.0]
        return [t, 0.0]

    # The stationary affine forcing defect D(t)
    def D_curve(t):
        res = np.zeros(system.p)
        for j in range(len(system.A)):  # <--- FIXED THIS LINE
            res += system.A[j] @ np.array(gamma_0(system.M * t - j))
        return (res - np.array(gamma_0(t))).tolist()        

    # The truly zero vector to initialize the purely affine recursion
    def zero_curve(t): return [0.0, 0.0]

    x_vals, y_vals = [], []
    
    print(f"Rendering Continuous Surrogate Dragon (Depth {n_depth}, {num_points} vertices)...")
    t0 = time.time()
    
    for t in t_vals:
        # Pass D_curve as an accumulated forcing term (B_curve), not the base state!
        pt_cascade = surrogate_engine.evaluate_affine(
            t, n_depth, 
            gamma_0=zero_curve, 
            B_curves=[D_curve]
        )
        
        # Add the affine result back to the global anchor profile
        pt = np.array(gamma_0(t)) + np.array(pt_cascade)
        
        x_vals.append(pt[0])
        y_vals.append(pt[1])
        
    print(f"Completed in {time.time() - t0:.3f} seconds.")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_vals, y_vals, 'r-', linewidth=0.5)
    ax.set_title(f"Level 4: Continuous Loop Surrogate Dragon ($n={n_depth}$)")
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("surrogate_dragon.png", dpi=600, bbox_inches='tight')
    print("Saved surrogate_dragon.png!")

if __name__ == "__main__":
    main()