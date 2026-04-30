import sys
import os
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_dragon, initial_unit_segment

# --- TOGGLE YOUR DRAGON HERE ---
# Options: 'levy' or 'heighway'
DRAGON_TYPE = 'heighway'

def main():
    print(f"Fetching {DRAGON_TYPE.capitalize()} Dragon geometry from library...")
    
    # 1. Compile Level 1 to Level 2 using the library factory
    system = create_dragon(dragon_type=DRAGON_TYPE)
    print(f"Loaded system with {system.M} transition matrices (A_j).")

    # 2. Fetch the standard endpoint-extended initial curve \gamma_0(t)
    gamma_0 = initial_unit_segment(p=2)

    # 3. Evaluate the Level 2 exact algebraic sum
    n_depth = 10  # Depth 12 = 4096 segments
    num_points = 4097 # One point per vertex
    
    print(f"Evaluating exact V^n \\gamma(t) at depth {n_depth}...")
    t_values = [i / (num_points - 1) for i in range(num_points)]
    
    x_vals = []
    y_vals = []
    
    for t in t_values:
        point = system.evaluate_V_n(gamma_0, n_depth, t)
        x_vals.append(point[0])
        y_vals.append(point[1])

    # 4. Plot and save to PNG
    print("Rendering plot...")
    plt.figure(figsize=(8, 6))
    
    # Plot using a thinner line for the dense dragon curve
    plt.plot(x_vals, y_vals, 'm-', linewidth=1)
    
    plt.axis('equal')
    plt.axis('off') 
    plt.title(f"{DRAGON_TYPE.capitalize()} Dragon (Level 2 Exact Algebra) - Depth {n_depth}")
    
    output_filename = f"dragon_{DRAGON_TYPE}_d{n_depth}_l2.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()