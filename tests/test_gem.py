import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from refinet.network.model import ExactCascade

def main():
    print("--- Testing General M Exact Cascade (M=3) ---")
    
    # 1. Define the Fractal/Curve Parameters for M=3
    M = 3
    depth_n = 3          # 3 levels of recursive depth
    a = 5.0              
    rho = 0.15           
    epsilon = 0.05       
    delta_bar = 0.1      
    
    # Initial vector state
    u_mu = np.array([1.0, 0.0], dtype=np.float32)
    N = len(u_mu)
    
    # Transition Matrices (We need M=3 matrices now: T_0, T_1, T_2)
    # Let's use 1/3 scaling to mathematically verify the depth recursion
    scale = 1.0 / 3.0
    T_base = np.array([[scale, 0.0], 
                       [0.0, scale]], dtype=np.float32)
                    
    T_matrices = [T_base, T_base, T_base]

    # 2. Initialize the PyTorch Network
    print(f"\n[1/3] Compiling 2-Layer Radial Tent Engine for M={M}...")
    try:
        model = ExactCascade(
            M=M, 
            depth_n=depth_n, 
            T_matrices=T_matrices, 
            a=a, 
            rho=rho, 
            epsilon=epsilon, 
            delta_bar=delta_bar, 
            u_mu=u_mu
        )
        print("  -> Success! 2-Layer Boolean Intersection Engine hardcoded.")
    except Exception as e:
        print(f"  -> Error initializing model: {e}")
        return

    # 3. Generate a Batch of Input Parameters
    print("\n[2/3] Generating 1,000 continuous parameter states t in [0, 1]...")
    batch_size = 1000
    t_np = np.linspace(0, 1, batch_size, dtype=np.float32).reshape(-1, 1)
    t_tensor = torch.from_numpy(t_np)

    # 4. Execute the Forward Pass
    print("[3/3] Executing batched forward pass through the Exact Loop...")
    try:
        with torch.no_grad():
            Phi_out = model(t_tensor)
            
        print("  -> Success! Cascade evaluated flawlessly.")
        print(f"\nFinal Output Shape: {Phi_out.shape} (Expected: [{batch_size}, {N}])")
        
        # Since we used T = 1/3 and depth n = 3, the final coordinate 
        # should be exactly (1/3)^3 = 1/27 = 0.037037...
        out_np = Phi_out.numpy()
        print("\nSample Curve Coordinates \u03a6(t):")
        print(f"  t = {t_np[0][0]:.3f}  ->  {out_np[0]}")
        print(f"  t = {t_np[250][0]:.3f}  ->  {out_np[250]}")
        print(f"  t = {t_np[500][0]:.3f}  ->  {out_np[500]}")
        print(f"  t = {t_np[750][0]:.3f}  ->  {out_np[750]}")
        print(f"  t = {t_np[-1][0]:.3f}  ->  {out_np[-1]}")
        
    except Exception as e:
        print(f"  -> Error during forward pass: {e}")

if __name__ == "__main__":
    main()