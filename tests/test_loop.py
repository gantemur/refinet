import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from refinet.cascade.surrogate import LoopController

def test_exact_loop_controller():
    print("--- Validating Exact Loop Controller (Lemma 3.1) ---")
    M = 3
    
    # Test 1: Default Triangle
    controller_tri = LoopController(M)
    
    # Test 2: Custom Hexagon (Generalization check)
    hex_verts = [
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ]
    controller_hex = LoopController(M, vertices=hex_verts)
    
    for name, controller in [("Triangle", controller_tri), ("Hexagon", controller_hex)]:
        errors = []
        t_vals = np.linspace(0, 1, 1000)
        for t in t_vals:
            # Standard discontinuous Modulo math
            R_t = (M * t) % 1.0 if t < 1.0 else 1.0
            exact_E_R_t = controller.E(R_t)
            
            # New Continuous Loop controller
            z_0 = controller.E(t)
            simulated_z_1 = controller.F(z_0)
            
            error = np.linalg.norm(exact_E_R_t - simulated_z_1)
            errors.append(error)
            
        max_err = max(errors)
        print(f"{name} | Max deviation between F(E(t)) and E(R(t)): {max_err:.3e}")
        assert max_err < 1e-10, f"Failed on {name}"
        
    print("Success! The loop controller exactly transports the residual for arbitrary polygons.")

if __name__ == "__main__":
    test_exact_loop_controller()