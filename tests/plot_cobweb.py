import numpy as np
import matplotlib.pyplot as plt

def main():
    print("--- Rendering Residual Dynamics Cobweb Plots ---")
    
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    M = 2
    N_STEPS = 10
    
    # "Good" Orbit: A chaotic path that naturally avoids J_n
    # 0.37 -> 0.74 -> 0.48 -> 0.96 -> 0.92 -> 0.84 -> 0.68 ...
    T_GOOD = 0.37 
    
    # "Bad" Orbit: Starts 2 iterations before disaster.
    # 0.376 -> 0.752 -> 0.504 (Lands in J_n!) -> 0.008 -> 0.016 -> 0.032 ...
    T_BAD = 0.376 
    
    # Zones for visual reference
    delta_visual = 0.05  # Matrix Transition J_n
    rho_visual = 0.15    # Special Hat Zero-Zone
    # ---------------------------------------------------------

    def R_map(x):
        """The exact continuous modulo residual map R(x) = Mx (mod 1)."""
        return (M * x) % 1.0

    def generate_cobweb(t_start, n_steps):
        """Generates the (x, y) coordinate pairs for a cobweb plot."""
        t_orbit = [t_start]
        cobweb_x, cobweb_y = [t_start], [0.0]
        
        t_curr = t_start
        for _ in range(n_steps):
            t_next = R_map(t_curr)
            
            # Vertical line to the function
            cobweb_x.append(t_curr)
            cobweb_y.append(t_next)
            # Horizontal line to y=x
            cobweb_x.append(t_next)
            cobweb_y.append(t_next)
            
            t_orbit.append(t_next)
            t_curr = t_next
            
        return t_orbit, cobweb_x, cobweb_y

    # Generate the orbits
    orbit_good, cw_x_good, cw_y_good = generate_cobweb(T_GOOD, N_STEPS)
    orbit_bad, cw_x_bad, cw_y_bad = generate_cobweb(T_BAD, N_STEPS)

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

    # Base function lines for plotting the modulo map
    x_vals = np.linspace(0, 1, 1000)
    # Disconnect the line at the knot 0.5 so it doesn't draw a vertical wall
    x_vals[500] = np.nan 
    y_vals = (M * x_vals) % 1.0

    for ax, title, cw_x, cw_y, orbit in zip(
        [ax1, ax2], 
        ["Good Orbit: Chaotic & Safe", "Bad Orbit: Falling into the Trap"],
        [cw_x_good, cw_x_bad],
        [cw_y_good, cw_y_bad],
        [orbit_good, orbit_bad]
    ):
        # 1. Background Math
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Identity ($y=x$)")
        ax.plot(x_vals, y_vals, 'k-', linewidth=2.5, label=r"$R(t) = 2t$ (mod 1)")
        
        # 2. The Special Hat Zero-Zones [\rho]
        ax.axvspan(0.0, rho_visual, color='gray', alpha=0.15, hatch='//', label=r"Hat Zero-Zone ($\rho$)")
        ax.axvspan(1.0 - rho_visual, 1.0, color='gray', alpha=0.15, hatch='//')

        # 3. Transition Zones J_n (Right-aligned to knots)
        ax.axvspan(0.0, delta_visual, color='red', alpha=0.25, label=r"Transition Ramp ($J_n$)")
        ax.axvspan(0.5, 0.5 + delta_visual, color='red', alpha=0.25)
        
        # 4. The Cobweb Trace
        ax.plot(cw_x, cw_y, 'b-', alpha=0.7, linewidth=1.5)
        
        # Highlight the discrete states with a color gradient indicating time
        colors = plt.cm.plasma(np.linspace(0, 1, N_STEPS + 1))
        ax.scatter(orbit, orbit, color=colors, s=30, zorder=5, edgecolors='k')

        # Aesthetics
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=15, pad=15)
        ax.set_xlabel("State $t_k$", fontsize=12)
        ax.set_ylabel("Next State $t_{k+1}$", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if ax == ax1:
            ax.legend(loc="upper left", fontsize=11)

    plt.suptitle("Residual Dynamics, Transition Ramps, and the Topological Hat", fontsize=18, y=1.02)
    plt.tight_layout()
    
    output_filename = "residual_cobweb.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to {output_filename}")

if __name__ == "__main__":
    main()