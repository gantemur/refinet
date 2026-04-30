import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.cascade.surrogate import LoopController

def main():
    print("--- Rendering Exact Loop Controller Vector Field ---")
    
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    M = 2
    N_POINTS = 60          # Density of the vectors along the loop
    VECTOR_SCALE = 0.25    # Scales down the jump F(z) - z so arrows don't overlap
    # ---------------------------------------------------------
    
    # Equilateral triangle centered at origin (radius 1)
    R = 1.0
    equilateral_verts = [
        [-R, 0],
        [R * math.cos(math.pi/3), R * math.sin(math.pi/3)],
        [R * math.cos(math.pi/3), - R * math.sin(math.pi/3)]
    ]

    # Square centered at origin (side length 2)
    square_verts = [
        [-1.0, -1.0],  # Bottom Left
        [ 1.0, -1.0],  # Bottom Right
        [ 1.0,  1.0],  # Top Right
        [-1.0,  1.0]   # Top Left
    ]

#    controller = LoopController(M)
    controller = LoopController(M=M, vertices=equilateral_verts)
#    controller = LoopController(M=M, vertices=square_verts)
    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Plot the Main Loop (\Gamma)
    verts = controller.vertices + [controller.vertices[0]] # Close the loop
    vx = [v[0] for v in verts]
    vy = [v[1] for v in verts]
    
    # Thinned out the main loop line slightly for a cleaner look
    ax.plot(vx, vy, 'k-', linewidth=1.5, zorder=2, label="Loop $\Gamma$")

    # 2. Plot the Barycenter and the Fan Partition (Appendix A)
    cx, cy = controller.c
    ax.plot(cx, cy, 'ko', markersize=4, zorder=3, label="Barycenter $c$")
    
    # Draw the rays defining the K * M sectors
    total_sectors = controller.K * controller.M
    for m in range(total_sectors):
        t_m = m / total_sectors
        p_m = controller.E(t_m)
        ax.plot([cx, p_m[0]], [cy, p_m[1]], color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)

    # 3. Generate the Vector Field along the perimeter
    t_vals = np.linspace(0, 1, N_POINTS, endpoint=False)
    
    X, Y, U, V = [], [], [], []
    colors = []

    for t in t_vals:
        z = controller.E(t)
        z_next = controller.F(z)
        
        # Calculate the direction of the exact residual jump
        dz = z_next # - z
        
        X.append(z[0])
        Y.append(z[1])
        U.append(dz[0] * VECTOR_SCALE)
        V.append(dz[1] * VECTOR_SCALE)
        
        # Color by t to show the flow around the loop
        colors.append(t)

    # 4. Plot the Vectors using a Quiver plot (Tuned for an Academic Look)
    q = ax.quiver(X, Y, U, V, colors, cmap='hsv', 
                  angles='xy', scale_units='xy', scale=1, 
                  width=0.002,           # Needle-thin shafts
                  headwidth=3.5,         # Sharp, narrow arrowheads
                  headlength=4.5,        # Elongated heads
                  headaxislength=4.0,    # Clean sweep angle on the head
                  edgecolor='black', linewidth=0.2,  # Microscopic black outline for crispness
                  alpha=0.95, zorder=4)

    # ---------------------------------------------------------
    # Dynamically pad the canvas to fit the vector tips
    # ---------------------------------------------------------
    tip_x = np.array(X) + np.array(U)
    tip_y = np.array(Y) + np.array(V)
    
    min_x, max_x = min(min(X), min(tip_x)), max(max(X), max(tip_x))
    min_y, max_y = min(min(Y), min(tip_y)), max(max(Y), max(tip_y))
    
    # Add a 15% margin around the edges
    margin_x = (max_x - min_x) * 0.15
    margin_y = (max_y - min_y) * 0.15
    
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)
    # ---------------------------------------------------------

    # Aesthetics
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Exact Loop Controller Vector Field ($M={M}$)\nJump scaled to {VECTOR_SCALE*100}%", fontsize=14, pad=20)
    
    # Add a colorbar to indicate the parameter t
    cbar = fig.colorbar(q, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Curve Parameter $t \in [0, 1)$', rotation=270, labelpad=15)

    plt.tight_layout()
    output_filename = "loop_vector_field.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to {output_filename}")

if __name__ == "__main__":
    main()