import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.cpwl import CPwLCurve
from refinet.cascade.decomposition import SpecialHatDecomposition

def evaluate_reconstructed_curve(t, decomp):
    """Reconstructs the curve using the shifted local hats to prove lossless data."""
    result = [0.0] * decomp.p
    for hat in decomp.hats:
        # Cascade mapping: Global t -> Local tau
        tau = t - hat['shift']
        weight = decomp.evaluate_local_hat(hat['index'], tau)
        if weight > 0.0:
            result = [result[d] + weight * hat['coeff'][d] for d in range(decomp.p)]
    return result

def main():
    rho = 0.15 # Using a large rho to make the padding visually obvious
    
    # 1. Create a highly irregular 1D test curve spanning well outside [0,1]
    # We use p=1 so we can easily plot it as a standard function y = f(t)
    breakpoints = [
        (-1.0, [0.0]),
        (0.8,  [2.0]),  
        (1.2,  [-1.5]),
        (2.5,  [1.0]),
        (2.7,  [0.0])   # Added closure to ensure compact support!
    ]
    curve = CPwLCurve(breakpoints, p=1)
    
    print(f"--- Special Hat Decomposition (rho={rho}) ---")
    decomp = SpecialHatDecomposition(curve, rho=rho)
    
    print("\n[Decomposition Packets]")
    print(f"{'Idx':<4} | {'Peak (t)':<10} | {'Support [L, R]':<18} | {'Shift':<10} | {'Local [tau_L, tau_R]':<20} | {'Coeff'}")
    print("-" * 85)
    
    for hat in decomp.hats:
        idx = hat['index']
        peak = round(hat['global_peak'], 3)
        sup_L, sup_R = round(hat['global_support'][0], 3), round(hat['global_support'][1], 3)
        shift = round(hat['shift'], 3)
        loc_L, loc_R = round(hat['local_nodes'][0], 3), round(hat['local_nodes'][2], 3)
        coeff = [round(c, 3) for c in hat['coeff']]
        
        print(f"{idx:<4} | {peak:<10} | [{sup_L:<6}, {sup_R:<6}] | {shift:<10} | [{loc_L:<6}, {loc_R:<6}] | {coeff}")

    # --- PLOTTING ---
    t_vals = np.linspace(-1.2, 2.7, 1000)
    
    # Evaluate Original vs Reconstructed
    y_orig = [curve(t)[0] for t in t_vals]
    y_recon = [evaluate_reconstructed_curve(t, decomp)[0] for t in t_vals]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: The Curves
    axs[0].plot(t_vals, y_orig, 'k-', linewidth=4, label="Original CPwL", alpha=0.3)
    axs[0].plot(t_vals, y_recon, 'r--', linewidth=2, label="Reconstructed via Hats")
    axs[0].set_title("Lossless Reconstruction Test")
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Global Hat Basis (Partition of Unity)
    axs[1].set_title("Global Hat Basis Functions $h_j(t)$")
    for hat in decomp.hats:
        # Evaluate this specific hat across the global domain
        y_hat = [decomp.evaluate_local_hat(hat['index'], t - hat['shift']) for t in t_vals]
        axs[1].plot(t_vals, y_hat, label=f"h_{hat['index']}")
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    # Plot 3: Shifted Local Hats
    tau_vals = np.linspace(0, 1, 500)
    axs[2].set_title(f"Locally Shifted Hats $\\tau \\in [0, 1]$ (Safe Zone $\\rho={rho}$)")
    axs[2].axvspan(0, rho, color='red', alpha=0.1, label="Danger Zone (0 to \u03c1)")
    axs[2].axvspan(1-rho, 1, color='red', alpha=0.1, label="Danger Zone (1-\u03c1 to 1)")
    
    for hat in decomp.hats:
        y_local = [decomp.evaluate_local_hat(hat['index'], tau) for tau in tau_vals]
        axs[2].plot(tau_vals, y_local)
        
    axs[2].axvline(rho, color='r', linestyle='--')
    axs[2].axvline(1-rho, color='r', linestyle='--')
    axs[2].legend(loc="upper right")
    
    plt.tight_layout()
    output_filename = "test_decomposition_shifts.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nSuccess! Plot saved to {output_filename}")

if __name__ == "__main__":
    main()