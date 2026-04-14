import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_square_bridge, initial_unit_segment

def main():
    print("Fetching Square Bridge geometry from library...")
    
    system = create_square_bridge()
    print(f"Loaded system with {system.num_matrices} matrices, M={system.M}.")

    # 1. Fetch the standard endpoint-extended initial curve \gamma_0(t)
    gamma_0 = initial_unit_segment(p=2)

    # 2. Evaluate the FULL affine sum (V^n \gamma + S_n)
    n_depth = 5
    num_points = 2000
    
    print(f"Evaluating exact full iterate W^n \\gamma(t) at depth {n_depth}...")
    t_values = [i / (num_points - 1) for i in range(num_points)]
    
    x_vals = []
    y_vals = []
    
    for t in t_values:
        point = system.compute_full_iterate(gamma_0, n_depth, t)
        x_vals.append(point[0])
        y_vals.append(point[1])

    # 3. Plot and save to PNG
    print("Rendering plot...")
    plt.figure(figsize=(10, 4))
    
    plt.plot(x_vals, y_vals, 'k-', linewidth=1.0)
    plt.axis('equal')
    plt.axis('off') 
    
    plt.title(f"Stationary Affine Bridge (Level 2 Exact Algebra) - Depth {n_depth}")
    
    output_filename = "bridge_stationary_level2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()