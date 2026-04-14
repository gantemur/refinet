import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_hilbert_like

def main():
    scale=0.4
    height=1
    # Unpack the system AND the U-shaped initial curve
    system, gamma_0 = create_hilbert_like(scale=scale, height=height) 
    
    print(f"Loaded Hilbert-like system with {system.num_matrices} matrices, M={system.M}.")

    n_depth = 3
    M = system.M 
    
    # gamma_0 has 3 segments. We multiply the grid to resolve every corner!
    gamma_0_segments = 3 
    num_segments = gamma_0_segments * (M ** n_depth)
    num_points = num_segments + 1
    
    print(f"Evaluating exact ANCHORED full iterate W^n \\gamma(t) at depth {n_depth} with {num_points} points...")
    t_values = [i / num_segments for i in range(num_points)]
    
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
    
    plt.axis('equal')
    plt.axis('off') 
    
    plt.title(f"Hilbert-like Stationary (Scale {scale}, Height {height}) - Depth {n_depth}")
    
    output_filename = "hilbert_like_stationary_level2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()