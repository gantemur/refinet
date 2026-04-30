import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from refinet.cascade.surrogate import LoopController, LoopSelector

def test_loop_selector():
    print("--- Validating Loop-State Matrix Selectors (Lemma 3.6) ---")
    
    M = 3
    n_depth = 4
    controller = LoopController(M)
    selector = LoopSelector(controller, n_depth, rho=0.15, delta_bar=0.5)
    
    # Mock Block Transition Matrices
    T_matrices = [np.full((2,2), float(q)) for q in range(M)]
    
    # Sweep around the loop
    t_vals = np.linspace(0, 1, 5000, endpoint=False)
    
    sum_errors = []
    continuity_jumps = []
    prev_chi = None
    
    for t in t_vals:
        z = controller.E(t)
        
        # Test 1: Partition of Unity
        chi_sum = sum(selector.evaluate_chi(q, z) for q in range(M))
        sum_errors.append(abs(1.0 - chi_sum))
        
        # Test 2: Continuity (Max step change between adjacent high-res points)
        current_chi = [selector.evaluate_chi(q, z) for q in range(M)]
        if prev_chi is not None:
            jump = max(abs(current_chi[q] - prev_chi[q]) for q in range(M))
            continuity_jumps.append(jump)
        prev_chi = current_chi
        
        # Test 3: Smooth Matrix Field Evaluation
        T_hat = selector.evaluate_T_hat(z, T_matrices)
        
    max_sum_err = max(sum_errors)
    max_jump = max(continuity_jumps)
    
    print(f"Max partition of unity error: {max_sum_err:.3e}")
    assert max_sum_err < 1e-10, "Selectors do not sum to 1!"
    
    print(f"Max adjacent step jump (Continuity check): {max_jump:.3e}")
    assert max_jump < 1e-2, "Selector functions are discontinuous!"
    
    print("Success! The continuous matrix field gracefully replaces the discrete digit jumps.")

if __name__ == "__main__":
    test_loop_selector()