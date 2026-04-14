import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_hilbert

def main():
    n_depth = 3 # Try changing this to 2, 3, or 4
    
    system, Gamma_n = create_hilbert()
    
    print(f"Loaded Stage-Dependent Hilbert system. M={system.M}.")

    # The geometric initial curve is exactly the stage-0 anchor
    gamma_0 = Gamma_n(0)

    # AUTOMATIC POINT CALCULATION:
    # Since Gamma_0(t) is a constant point, it has NO internal corners.
    # Therefore, the base curve is effectively just 1 segment.
    base_segments = 1 
    
    # The grid is purely defined by the M-ary subdivisions at the target depth
    num_segments = base_segments * (system.M ** n_depth)
    
    # Add 1 to get the exact number of boundary points
    num_points = num_segments + 1
    
    print(f"Evaluating EXACT NATIVE W^n \gamma(t) at depth {n_depth} with {num_points} points...")
    t_values = [i / num_segments for i in range(num_points)]
    
    x_vals = []
    y_vals = []
    
    for t in t_values:
        # Warning: This is O(M^n) per point right now!
        point = system.compute_full_iterate(gamma_0, n_depth, t)
        x_vals.append(point[0])
        y_vals.append(point[1])

    print("Rendering plot...")
    plt.figure(figsize=(8, 8))
    
    plt.plot(x_vals, y_vals, 'k-', linewidth=0.8)
    
    plt.axis('equal')
    plt.axis('off') 
    
    plt.title(f"Hilbert (Infinite Finite-Span) - Depth {n_depth}")
    
    output_filename = f"hilbert_infinite_level2_d{n_depth}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()