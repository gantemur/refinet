import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_morton_like

def main():
    # You can change scale to 0.4, 0.3, or 0.49!
    system, gamma_0 = create_morton_like(scale=0.45)
    
    print(f"Loaded system with {system.num_matrices} matrices, M={system.M}.")

    n_depth = 4
    M = system.M
    gamma_0_segments = 3
    num_segments = M ** n_depth * gamma_0_segments
    num_points = num_segments + 1
    
    print(f"Evaluating exact ANCHORED full iterate W^n \\gamma(t) at depth {n_depth}...")
    print(f"Evaluating exactly at the {num_points} combined grid points...")
    t_values = [i / (num_points - 1) for i in range(num_points)]
    
    x_vals = []
    y_vals = []
    
    for t in t_values:
        point = system.compute_full_iterate(gamma_0, n_depth, t)
        x_vals.append(point[0])
        y_vals.append(point[1])

    # Plot and save to PNG
    print("Rendering plot...")
    plt.figure(figsize=(8, 8))
    
    plt.plot(x_vals, y_vals, 'k-', linewidth=0.8)
    
    # Fix the bounding box to see the global drift
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off') 
    
    plt.title(f"Morton-like Stationary (Level 2 Exact) - Depth {n_depth}")
    
    output_filename = "morton_like_stationary_level2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()