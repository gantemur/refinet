import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import create_sierpinski_stage_dependent

def main():
    n_depth = 4
    
    system, Gamma_n = create_sierpinski_stage_dependent()
    
    # The geometric initial curve is simply the stage-0 anchor
    gamma_0 = Gamma_n(0)

    num_points = 2000
    print(f"Evaluating EXACT NATIVE W^n \gamma(t) at depth {n_depth}...")
    t_values = [i / (num_points - 1) for i in range(num_points)]
    
    x_vals = []
    y_vals = []
    
    for t in t_values:
        point = system.compute_full_iterate(gamma_0, n_depth, t)
        x_vals.append(point[0])
        y_vals.append(point[1])

    print("Rendering plot...")
    plt.figure(figsize=(8, 7))
    plt.plot(x_vals, y_vals, 'k-', linewidth=0.8)
    plt.axis('equal')
    plt.axis('off') 
    plt.title(f"Sierpinski (Infinite Finite-Span) - Depth {n_depth}")
    
    plt.savefig("sierpinski_infinite_level2.png", dpi=300, bbox_inches='tight')
    print("Success! Saved.")

if __name__ == "__main__":
    main()