import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from refinet.cascade.surrogate import LoopController, TerminalReadout

def mock_special_hat(t, rho=0.1):
    """A standard CPwL hat function supported strictly in [rho, 1 - rho]."""
    if t <= rho or t >= 1.0 - rho:
        return 0.0
    elif t <= 0.5:
        return (t - rho) / (0.5 - rho)
    else:
        return (1.0 - rho - t) / (0.5 - rho)

def test_terminal_readout():
    print("--- Validating Terminal Readout & Min-Gadget (Lemma 3.3) ---")
    
    M = 3
    n_depth = 4       # Test deep in the cascade
    rho = 0.1         # Hat support margin
    epsilon = 0.05    # Folding margin (Must be < rho)
    
    controller = LoopController(M)
    readout = TerminalReadout(controller, epsilon=epsilon)
    
    errors = []
    t_vals = np.linspace(0, 1, 5000)
    
    for x in t_vals:
        # 1. Exact discontinuous math: h(R^n(x))
        # (Calculate R^n(x) carefully to avoid compounding float errors)
        R_n_x = (x * (M ** n_depth)) % 1.0
        exact_h_val = mock_special_hat(R_n_x, rho)
        
        # 2. Level 4 Surrogate Pipeline: min( h(rho-(z)), h(rho+(z)) )
        z_n = controller.evaluate_loop_state(n_depth, x)
        simulated_h_val = readout.evaluate_scalar_factor(z_n, lambda t: mock_special_hat(t, rho))
        
        errors.append(abs(exact_h_val - simulated_h_val))
        
    max_err = max(errors)
    print(f"Max deviation for special hat scalar factor at depth {n_depth}: {max_err:.3e}")
    assert max_err < 1e-10, "Min-Gadget failed to perfectly recover the scalar factor!"
    
    print("Success! The terminal readouts absorb the seam ambiguity flawlessly.")

if __name__ == "__main__":
    test_terminal_readout()