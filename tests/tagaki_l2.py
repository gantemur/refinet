import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_knopp

def main():
    print("Fetching Takagi (Blancmange) geometry from library...")
    
    system = create_knopp(.7)
    print(f"Loaded system with {system.num_matrices} matrices, M={system.M}, L={system.L}.")

    # 1. The initial state for Takagi is the zero function
    def gamma_0(t):
        return [0.0]

    # 2. Evaluate the FULL affine sum (V^n \gamma + S_n)
    n_depth = 9
    num_points = 8000
    
    print(f"Evaluating exact full iterate W^n \\gamma(t) at depth {n_depth}...")
    t_values = [i * system.L / (num_points - 1) for i in range(num_points)]
    
    y_vals = []
    
    for t in t_values:
        # CRITICAL FIX: Must use compute_full_iterate to include the forcing term!
        point = system.compute_full_iterate(gamma_0, n_depth, t)
        y_vals.append(point[0]) 

    # 3. Plot and save to PNG
    print("Rendering plot...")
    plt.figure(figsize=(10, 5))
    
    plt.plot(t_values, y_vals, 'k-', linewidth=1.0)
    
    plt.title(f"Takagi / Blancmange Curve (Level 2 Exact Algebra) - Depth {n_depth}")
    plt.grid(True, alpha=0.3)
    
    output_filename = "takagi_level2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()