import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.cascade.exact import MaryRouter

def plot_rigorous_function(ax, t, y, is_sawtooth, color, ylabel, title, ylim=None, yticks=None):
    """
    Plots a discontinuous piecewise function with absolute mathematical rigor:
    - Breaks the solid line at jumps.
    - Draws thin dotted vertical lines at the discontinuity.
    - Uses open circles for the limit approach.
    - Uses filled dots for the evaluated, right-continuous state.
    - Anchors the absolute global boundaries t=0 and t=1 with filled dots.
    """
    y_clean = np.array(y, dtype=float)
    
    for i in range(1, len(t)):
        # Detect any jump > 0.5 (catches both integer steps and sawtooth resets)
        if abs(y[i] - y[i-1]) > 0.5:
            t_jump = t[i]
            
            # Enforce exact theoretical limits for the sawtooth
            if is_sawtooth:
                y_left_limit = 1.0
                y_actual = 0.0
            else:
                y_left_limit = y[i-1]
                y_actual = y[i]
                
            # Break the continuous line so matplotlib leaves a gap
            y_clean[i] = np.nan
            
            # 1. Thin dotted vertical line bridging the gap
            ax.vlines(x=t_jump, ymin=min(y_left_limit, y_actual), ymax=max(y_left_limit, y_actual), 
                      colors=color, linestyles=':', linewidth=1.2, alpha=0.6)
            
            # 2. Left limit (open circle)
            ax.plot(t_jump, y_left_limit, marker='o', markerfacecolor='white', 
                    markeredgecolor=color, markersize=5, zorder=10)
            
            # 3. Right-continuous actual value (filled dot)
            ax.plot(t_jump, y_actual, marker='o', color=color, markersize=5, zorder=10)

    # Plot the clean, broken continuous segments
    ax.plot(t, y_clean, color=color, linewidth=1.5)
    
    # 4. Absolute Domain Boundaries (Filled dots at t=0 and t=1 for completeness)
    ax.plot(t[0], y[0], marker='o', color=color, markersize=5, zorder=10)
    ax.plot(t[-1], y[-1], marker='o', color=color, markersize=5, zorder=10)
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(ylim)
    if yticks is not None: ax.set_yticks(yticks)
    ax.grid(True, linestyle=':', alpha=0.6)


def main():
    M = 3
    depth = 2
    router = MaryRouter(M)
    
    print(f"Generating rigorously styled Router plots for M={M}, Depth={depth}...")

    t_vals = np.linspace(0, 1, 5000)
    
    q1, r1 = [], []
    q2, r2 = [], []
    
    for t in t_vals:
        digits, residuals = router.decode(t, depth)
        q1.append(digits[0])
        r1.append(residuals[0])
        q2.append(digits[1])
        r2.append(residuals[1])

    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"M-ary Topological Router Functions ($M={M}$)", fontsize=16)
    
    # [Row 1] Depth 1
    plot_rigorous_function(axs[0, 0], t_vals, q1, False, 'blue', "Matrix Index", f"Depth 1 Digit: $q_1(t)$", yticks=range(M))
    plot_rigorous_function(axs[0, 1], t_vals, r1, True, 'red', "Local Coordinate", f"Depth 1 Residual: $r_1(t)$", ylim=(-0.05, 1.05))

    # [Row 2] Depth 2
    plot_rigorous_function(axs[1, 0], t_vals, q2, False, 'blue', "Matrix Index", f"Depth 2 Digit: $q_2(t)$", yticks=range(M))
    axs[1, 0].set_xlabel("Global $t$")
    
    plot_rigorous_function(axs[1, 1], t_vals, r2, True, 'red', "Local Coordinate", f"Depth 2 Residual: $r_2(t)$", ylim=(-0.05, 1.05))
    axs[1, 1].set_xlabel("Global $t$")

    plt.tight_layout()
    output_filename = "test_router_plots.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Saved publication-grade plot to {output_filename}")

if __name__ == "__main__":
    main()