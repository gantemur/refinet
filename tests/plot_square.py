import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_fundamental_square():
    # --- Parameters ---
    n = 13  # Depth of the orbit
    rho = 0.15
    delta_bar = 0.5
    
    # ARTIFICIAL VISUAL THICKNESS for the paper
    visual_n = 1 
    delta_n = delta_bar * rho * (2 ** -visual_n)

    # --- Setup Plot ---
    plt.rcParams.update({'text.usetex': False, 'mathtext.fontset': 'cm'})
    fig, ax = plt.subplots(figsize=(8, 8))

    transition_color = '#E63946' 
    atom_color = '#1D3557'       

    # --- 1. Draw Transition Regions (J_n) ---
    ax.add_patch(patches.Rectangle((0, 0), delta_n, 1, facecolor=transition_color, alpha=0.35, edgecolor='none'))
    ax.add_patch(patches.Rectangle((0.5, 0), delta_n, 1, facecolor=transition_color, alpha=0.35, edgecolor='none'))
    ax.add_patch(patches.Rectangle((0, 0), 1, delta_n, facecolor=transition_color, alpha=0.35, edgecolor='none'))
    ax.add_patch(patches.Rectangle((0, 0.5), 1, delta_n, facecolor=transition_color, alpha=0.35, edgecolor='none'))

    # --- 2. Draw Special Atom Support ---
    atom_support = patches.Rectangle(
        (rho, rho), 1 - 2*rho, 1 - 2*rho, 
        linewidth=2.5, linestyle='--', edgecolor=atom_color, facecolor='none', zorder=5
    )
    ax.add_patch(atom_support)

    # --- 3. Sample Iteration Sequence (Trajectory) ---
#    z_0 = np.array([0.22, 0.38])
    z_0 = np.array([0.72, 0.63])
    orbit = [z_0]
    
    for _ in range(n):
        next_z = (2.0 * orbit[-1]) % 1.0
        orbit.append(next_z)
        
    orbit = np.array(orbit)
    
    # Draw arrows FIRST so they sit underneath the dots (zorder=6)
    for i in range(len(orbit) - 1):
        start = orbit[i]
        end = orbit[i+1]
        
        # We add connectionstyle="arc3,rad=0.15" to curve the arrows!
        # This prevents lines from awkwardly crossing over each other's paths.
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color='#6C7A89', alpha=0.85, 
                                    lw=1.2, shrinkA=8, shrinkB=8,
                                    connectionstyle="arc3,rad=0.05"),
                    zorder=6)
        
    # Plot points LAST (zorder=7)
    ax.plot(orbit[:, 0], orbit[:, 1], 'o', color='black', markersize=8, 
            markeredgecolor='white', markeredgewidth=1.5, zorder=7)
    
    # Label ONLY the first and last points to prevent clutter
    ax.text(z_0[0] - 0.04, z_0[1] + 0.02, r'$z_0$', 
            fontsize=16, color='black', weight='bold', zorder=8)
            
    final_z = orbit[-1]
    ax.text(final_z[0] + 0.02, final_z[1] - 0.03, rf'$z_{{{n}}}$', 
            fontsize=16, color='black', weight='bold', zorder=8)

    # --- 4. Formatting & Ticks ---
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    ticks = [0, rho, 0.5, 1-rho, 1]
    tick_labels = ['0', r'$\rho$', '0.5', r'$1-\rho$', '1']
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=16)
    
    ax.grid(True, linestyle=':', color='gray', alpha=0.6, zorder=1)

    # --- 5. Legend ---
    legend_elements = [
        patches.Patch(facecolor=transition_color, alpha=0.5, label=r'Transition Set $J_n$'),
        patches.Patch(edgecolor=atom_color, facecolor='none', linestyle='--', 
                      linewidth=2.5, label=r'Atom Support $[\rho, 1-\rho]^2$')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, framealpha=1.0)

    # --- Save to High-Res PNG ---
    plt.title(r'Fundamental Square Geometry & Residual Orbit', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig('fundamental_square.png', format='png', dpi=300, bbox_inches='tight')
    print("Saved graphic to 'fundamental_square.png'")
#    plt.show()

if __name__ == "__main__":
    draw_fundamental_square()