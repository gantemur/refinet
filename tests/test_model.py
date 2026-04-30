import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from refinet.network.model import ExactCascade

def main():
    print("--- Testing Full Exact Cascade (Level 5) ---")
    
    # 1. Define the Fractal/Curve Parameters (M=2)
    M = 2
    depth_n = 4          # 4 levels of recursive depth
    a = 5.0              # Product gadget boundary
    rho = 0.15           # Special hat zero-zone
    epsilon = 0.05       # Seam proximity threshold
    delta_bar = 0.1      # Cross-fade width
    
    # Initial vector state (e.g., a starting point in 2D space)
    u_mu = np.array([1.0, 0.0], dtype=np.float32)
    N = len(u_mu)
    
    # Transition Matrices (T_0 and T_1 for M=2)
    # Using simple affine scaling/translation matrices for a generic 2D curve
    T_0 = np.array([[0.5, 0.0], 
                    [0.0, 0.5]], dtype=np.float32)
                    
    T_1 = np.array([[0.5, 0.0], 
                    [0.0, 0.5]], dtype=np.float32)
                    
    T_matrices = [T_0, T_1]

    # 2. Initialize the PyTorch Network
    print("\n[1/3] Compiling offline tensors and building PyTorch Cascade...")
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
        print("  -> Success! Network topology successfully hardcoded.")
    except Exception as e:
        print(f"  -> Error initializing model: {e}")
        return

    # 3. Generate a Batch of Input Parameters
    print("\n[2/3] Generating 1,000 continuous parameter states t in [0, 1]...")
    batch_size = 1000
    # Shape must be [batch_size, 1] for PyTorch
    t_np = np.linspace(0, 1, batch_size, dtype=np.float32).reshape(-1, 1)
    t_tensor = torch.from_numpy(t_np)

    # 4. Execute the Forward Pass
    print("[3/3] Executing batched forward pass through the Exact Loop...")
    try:
        with torch.no_grad(): # No training, pure mathematical evaluation
            Phi_out = model(t_tensor)
            
        print("  -> Success! Cascade evaluated flawlessly.")
        print(f"\nFinal Output Shape: {Phi_out.shape} (Expected: [{batch_size}, {N}])")
        
        # Print a few sample coordinates to prove it calculated dynamic values
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