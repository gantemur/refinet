import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_sierpinski_arrowhead, initial_stacked_unit_segment

def main():
    print("Fetching Stacked 2-State Sierpinski Arrowhead geometry from library...")
    
    # 1. Compile Level 1 to Level 2 using the library factory
    system = create_sierpinski_arrowhead()
    print(f"Loaded system with {system.M} stacked block transition matrices (A_j).")

    # 2. Fetch the stacked endpoint-extended initial curve \gamma_0(t)
    # p=2 (plane curve) and r=2 (states A and B)
    gamma_0 = initial_stacked_unit_segment(p=2, r=2)

    # 3. Evaluate the Level 2 exact algebraic sum
    n_depth = 6  # Depth 6 yields 729 segments (3^6)
    num_points = 730 
    
    print(f"Evaluating exact V^n \\gamma(t) at depth {n_depth}...")
    t_values = [i / (num_points - 1) for i in range(num_points)]
    
    x_vals = []
    y_vals = []
    
    for t in t_values:
        point = system.evaluate_V_n(gamma_0, n_depth, t)
        # We render State A (the main curve), which occupies indices 0 and 1
        x_vals.append(point[0])
        y_vals.append(point[1])

    # 4. Plot and save to PNG
    print("Rendering plot...")
    plt.figure(figsize=(8, 7))
    
    plt.plot(x_vals, y_vals, 'k-', linewidth=0.8)
    
    plt.axis('equal')
    plt.axis('off') 
    plt.title(f"Sierpinski Arrowhead (Stacked 2-State Algebra) - Depth {n_depth}")
    
    output_filename = "sierpinski_arrowhead_level2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()