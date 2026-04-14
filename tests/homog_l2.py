import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.homogeneous import HomogeneousBuilder
from refinet.geometry.library import initial_unit_segment

def main():
    
    #builder = HomogeneousBuilder([(0,0),(1,0),(1,1),(2,1),(2,0),(3,0)],orientations=[1,1,1,1,1]) triangle-ulzii
    builder = HomogeneousBuilder([(0,0),(1,0),(1,1),(3,0),(4,0)])
    system = builder.to_affine_system()
    print(f"Loaded system with {system.M} transition matrices (A_j).")

    gamma_0 = initial_unit_segment(p=2)

    n_depth = 4  # Depth 4 yields 256 line segments (4^4)
    num_points = 2000 # High resolution to capture sharp corners
    
    print(f"Evaluating exact V^n \\gamma(t) at depth {n_depth}...")
    t_values = [i / (num_points - 1) for i in range(num_points)]
    
    x_vals = []
    y_vals = []
    
    for t in t_values:
        # evaluate_V_n executes the exact recursive sum
        point = system.evaluate_V_n(gamma_0, n_depth, t)
        x_vals.append(point[0])
        y_vals.append(point[1])

    # 4. Plot and save to PNG
    print("Rendering plot...")
    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, y_vals, 'k-', linewidth=1.0)
    
    # Ensure geometry isn't distorted
    plt.axis('equal')
    plt.axis('off') 
    plt.title(f"Hom Curve (Level 2 Exact Algebra) - Depth {n_depth}")
    
    output_filename = "homog2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()