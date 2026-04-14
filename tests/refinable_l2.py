import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_bspline, create_daubechies_d4, initial_box_function

def main():
    print("Fetching Refinable Function geometry from library...")
    
    # --- TOGGLE YOUR FUNCTION HERE ---
    # system = create_bspline(degree=3)  # Cubic B-spline
    system = create_daubechies_d4()      # Daubechies D4
    
    print(f"Loaded system with {system.M} matrices, support window L={system.L}.")

    # 2. Fetch the standard box function
    gamma_0 = initial_box_function(p=1)

    # 3. Evaluate the Level 2 exact algebraic sum
    n_depth = 4
    num_points = system.L * 1000 # High resolution across the whole support
    
    print(f"Evaluating exact V^n \\gamma(t) at depth {n_depth}...")
    t_values = [i * system.L / (num_points - 1) for i in range(num_points)]
    
    y_vals = []
    
    for t in t_values:
        point = system.evaluate_V_n(gamma_0, n_depth, t)
        y_vals.append(point[0]) # Extract the 1D scalar value

    # 4. Plot and save to PNG
    print("Rendering plot...")
    plt.figure(figsize=(10, 4))
    
    # Plotting t against the 1D scalar value
    plt.plot(t_values, y_vals, 'k-', linewidth=1.5)
    
    # We do not use plt.axis('equal') here because x and y are independent units
    plt.title(f"Refinable Function (Level 2 Exact Algebra) - Depth {n_depth}")
    plt.grid(True, alpha=0.3)
    
    output_filename = "refinable_function_level2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()