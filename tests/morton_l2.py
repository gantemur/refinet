import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_morton

def main():
    n_depth = 4 # Try 1, 2, and 3 to watch the dot expand!
    
    system, Gamma_n = create_morton()
    
    print(f"Loaded Stage-Dependent Morton system. M={system.M}.")

    # The initial curve is a single dot at (0.5, 0.5)
    gamma_0 = Gamma_n(0)

    # AUTOMATIC POINT CALCULATION:
    # Since Gamma_0(t) is a constant point, it has NO internal corners.
    base_segments = 1 
    num_segments = base_segments * (system.M ** n_depth)
    num_points = num_segments + 1
    
    print(f"Evaluating EXACT NATIVE W^n \gamma(t) at depth {n_depth} with {num_points} points...")
    t_values = [i / num_segments for i in range(num_points)]
    
    x_vals = []
    y_vals = []
    
    for t in t_values:
        point = system.compute_full_iterate(gamma_0, n_depth, t)
        x_vals.append(point[0])
        y_vals.append(point[1])

    print("Rendering plot...")
    plt.figure(figsize=(8, 8))
    
    plt.plot(x_vals, y_vals, 'k-', linewidth=0.8)
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off') 
    
    plt.title(f"Morton (Infinite Finite-Span) - Depth {n_depth}")
    
    output_filename = f"morton_infinite_level2_d{n_depth}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()