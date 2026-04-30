import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.cascade.surrogate import LoopController, TerminalReadout

def main():
    print("--- Rendering Terminal Readout Functions ---")
    
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    M = 2
    rho = 0.15      # Special hat support margin (Exaggerated for visibility)
    epsilon = 0.10  # Folding margin (Must be strictly < rho)
    N_POINTS = 1000
    # ---------------------------------------------------------
    
    # Initialize the system
    controller = LoopController(M)
    readout = TerminalReadout(controller, epsilon=epsilon)

    # Evaluate the readouts by driving the parameter t around the exact loop E(t)
    t_vals = np.linspace(0, 1, N_POINTS)
    rho_minus_vals = []
    rho_plus_vals = []

    for t in t_vals:
        z = controller.E(t)
        rho_minus_vals.append(readout.rho_minus(z))
        rho_plus_vals.append(readout.rho_plus(z))

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Common Aesthetics for both plots
    for ax in (ax1, ax2):
        # Plot the true topological identity map
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="True Identity ($y=t$)")
        
        # Highlight the "Safe Zone" where the special hat is non-zero
        ax.axhspan(rho, 1 - rho, color='green', alpha=0.1, 
                   label=r"Hat Support $h(t) > 0$")
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel("Loop Parameter $t \in [0, 1]$ (via $z = E(t)$)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)

    # --- Left Plot: Right-Seam Fold (rho_minus) ---
    ax1.plot(t_vals, rho_minus_vals, 'b-', linewidth=2.5, label=r"Readout $\rho^-(z)$")
    ax1.axvspan(1 - epsilon, 1.0, color='red', alpha=0.1, 
                label=r"Fold Zone $[1-\epsilon, 1]$")
    ax1.set_title(r"Right-Seam Fold: $\rho^-(E(t))$", fontsize=14, pad=15)
    ax1.set_ylabel("Readout Value", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)

    # --- Right Plot: Left-Seam Fold (rho_plus) ---
    ax2.plot(t_vals, rho_plus_vals, 'm-', linewidth=2.5, label=r"Readout $\rho^+(z)$")
    ax2.axvspan(0.0, epsilon, color='red', alpha=0.1, 
                label=r"Fold Zone $[0, \epsilon]$")
    ax2.set_title(r"Left-Seam Fold: $\rho^+(E(t))$", fontsize=14, pad=15)
    ax2.legend(loc="lower right", fontsize=10)

    plt.suptitle("Terminal Readouts & The Seam Ambiguity (Lemma 3.3)", fontsize=16, y=1.05)
    plt.tight_layout()

    output_filename = "terminal_readouts.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to {output_filename}")

if __name__ == "__main__":
    main()