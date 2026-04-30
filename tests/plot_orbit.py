import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('src'))
from refinet.cascade.surrogate import LoopController

def main():
    print("--- Rendering Intrinsic Orbit Trace on the Loop ---")
    
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    M = 2
    N_STEPS = 15
    
    # Try swapping these!
    # T_START = 3.0 / 8.0              # Rational: Collapses to 0 in exactly 3 steps
    T_START = 1.0 / math.sqrt(2)       # Irrational: Chaotic dense orbit
    # ---------------------------------------------------------
    
    controller = LoopController(M)
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Plot the Main Loop (\Gamma)
    verts = controller.vertices + [controller.vertices[0]]
    vx = [v[0] for v in verts]
    vy = [v[1] for v in verts]
    ax.plot(vx, vy, 'k-', linewidth=3, zorder=1, label="Loop $\Gamma$")

    # 2. Generate the Orbit
    z_orbit = []
    z_curr = controller.E(T_START)
    for _ in range(N_STEPS):
        z_orbit.append(z_curr)
        z_curr = controller.F(z_curr)

    # 3. Plot the Orbit Path
    ox = [z[0] for z in z_orbit]
    oy = [z[1] for z in z_orbit]
    
    # Draw dashed lines to represent the discrete hops
    ax.plot(ox, oy, 'b--', alpha=0.4, zorder=2)
    
    # Plot the exact landing points on the perimeter with a time-progression color gradient
    colors = plt.cm.plasma(np.linspace(0, 1, N_STEPS))
    ax.scatter(ox, oy, color=colors, s=120, zorder=3, edgecolors='k')
    
    # Annotate the steps (0, 1, 2...) so we can track the sequence
    for i, (x, y) in enumerate(zip(ox, oy)):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(8,8), 
                    fontsize=12, fontweight='bold', color='navy')

    # Aesthetics
    ax.set_aspect('equal')
    ax.axis('off')
    title = f"Intrinsic Loop Dynamics: Orbit Trace ($M={M}$)\nStart $t_0={T_START:.4f}$"
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    output_filename = "loop_orbit.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Orbit saved to {output_filename}")

if __name__ == "__main__":
    main()