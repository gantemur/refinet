import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.cascade.surrogate import LoopController, LoopSelector

def main():
    print("--- Rendering Loop-State Matrix Selectors ---")
    
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    M = 3
    n_depth = 0      # Set to 0 to magnify the transition ramp \delta_n so it is visible
    rho = 0.15
    delta_bar = 0.5
    N_POINTS = 2000
    # ---------------------------------------------------------
    
    # Initialize the system
    controller = LoopController(M)
    selector = LoopSelector(controller, n_depth=n_depth, rho=rho, delta_bar=delta_bar)

    # Evaluate the selectors by driving the parameter t around the exact loop E(t)
    t_vals = np.linspace(0, 1, N_POINTS)
    chi_vals = {q: [] for q in range(M)}

    for t in t_vals:
        z = controller.E(t)
        for q in range(M):
            chi_vals[q].append(selector.evaluate_chi(q, z))

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ['#1f77b4', '#2ca02c', '#9467bd'] # Blue, Green, Purple
    
    # Plot each chi_q function with a transparent fill to show the partition of unity
    for q in range(M):
        ax.plot(t_vals, chi_vals[q], color=colors[q], linewidth=2.5, label=fr"Selector $\chi_{{{q},n}}(z)$")
        ax.fill_between(t_vals, 0, chi_vals[q], color=colors[q], alpha=0.3)

    # Highlight the transition zones (J_n) and exact mathematical knots
    delta_n = selector.delta_n
    knot_positions = []
    knot_labels = []

    for k in range(M):
        cell_start = k / M
        
        # 1. Draw the exact knot boundary as a crisp dashed line
        line_label = "Exact Knot $k/M$" if k == 0 else ""
        ax.axvline(x=cell_start, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=line_label, zorder=4)
        
        # 2. Draw the transition zone spilling to the right
        span_label = r"Transition Zone $J_n$" if k == 0 else ""
        ax.axvspan(cell_start, cell_start + delta_n, color='red', alpha=0.15, label=span_label, zorder=1)
        
        # 3. Record the exact fractional labels for the x-axis
        knot_positions.append(cell_start)
        knot_labels.append("0" if k == 0 else f"{k}/{M}")

    # Don't forget the final knot at t=1
    knot_positions.append(1.0)
    knot_labels.append("1")

    # Aesthetics
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Loop Parameter $t \in [0, 1]$ (via $z = E(t)$)", fontsize=12)
    ax.set_ylabel(r"Matrix Weight $\chi_{q,n}$", fontsize=12)
    ax.set_title(rf"Loop-State Matrix Selectors: Partition of Unity ($M={M}$)", fontsize=14, pad=15)
    
    # Apply the custom exact fractional X-ticks
    ax.set_xticks(knot_positions)
    ax.set_xticklabels(knot_labels, fontsize=12)
    
    # Place legend outside to avoid obscuring the graph
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

    plt.tight_layout()
    output_filename = "loop_selectors.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to {output_filename}")

if __name__ == "__main__":
    main()