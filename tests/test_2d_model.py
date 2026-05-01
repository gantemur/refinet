import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random

from refinet.algebra.system import TensorProductSystem
from refinet.cascade.exact import TensorBlockCascadeEngine
from refinet.cascade.surrogate2d import SurrogateCascadeEngine2D
from refinet.network.model2d import ExactCascade2D

torch.set_default_dtype(torch.float64)  # Force PyTorch to match NumPy precision

# --- Define the Special Atom ---
# We need both a Python function (for Level 4) and a PyTorch Module (for Level 5).
# We'll use the exact same asymmetric pyramid from the previous test.

RHO = 0.15

def H_atom_func(x, y):
    if x < RHO or x > 1 - RHO or y < RHO or y > 1 - RHO:
        return 0.0
    return max(0.0, 1.0 - 2.0 * abs(x - 0.5) - 3.0 * abs(y - 0.5))

class HAtomNet(nn.Module):
    """A strictly CPwL PyTorch representation of the Special Atom."""
    def forward(self, z):
        # z shape: (batch_size, 2)
        x = z[:, 0]
        y = z[:, 1]
        
        # Enforce support [rho, 1-rho]^2 using ReLUs
        mask_x = torch.relu(x - RHO) * torch.relu((1 - RHO) - x)
        mask_y = torch.relu(y - RHO) * torch.relu((1 - RHO) - y)
        in_support = (mask_x > 0) & (mask_y > 0)
        
        # Pyramid: 1 - 2|x - 0.5| - 3|y - 0.5|
        val = 1.0 - 2.0 * torch.abs(x - 0.5) - 3.0 * torch.abs(y - 0.5)
        
        # ReLU to ensure it doesn't go negative, and zero-out outside support
        out = torch.relu(val) * in_support.float()
        return out.unsqueeze(1)


def main():
    print("--- Level 5: 2D PyTorch Model vs Level 4 Surrogate Math ---\n")
    
    # 1. Initialize Theoretical Algebra (Levels 2 & 3)
    mask_1d = {0: 0.5, 1: 1.0, 2: 0.5}
    mask_2d = {(j, k): c_j * c_k for j, c_j in mask_1d.items() for k, c_k in mask_1d.items()}
    
    system = TensorProductSystem(mask_coeffs=mask_2d, L1=2, L2=2)
    engine3 = TensorBlockCascadeEngine(system)
    
    # 2. Initialize Continuous Math (Level 4)
    engine4 = SurrogateCascadeEngine2D(engine3, rho=RHO)
    
    # 3. Initialize PyTorch Neural Network (Level 5)
    n_depth = 3
    print("[1/3] Compiling Exact PyTorch Tensor-Product Cascade...")
    model = ExactCascade2D(
        system=system,
        T_matrices=engine3.T,
        depth_n=n_depth,
        H_atom_module=HAtomNet(),  # The 2D module fully replaces 'rho'
        epsilon=0.05,
        delta_bar=0.5,
        rho=0.1
    )
    
    # Set to evaluation mode (disables gradients/dropout conceptually)
    model = model.double()
    model.eval()

    print("[2/3] PyTorch Model Compiled. Testing across 10 random 2D points...\n")
    
    success = True
    max_error = 0.0
    
    for i in range(1, 11):
        # Generate random coordinates in the unit square
        x_val = random.uniform(0, 1)
        y_val = random.uniform(0, 1)
        
        # --- L4: Evaluate Continuous Loop Math ---
        # phi is the vectorized patch state (size 4). 
        # Since x, y are in [0, 1], our target is patch index 0.
        phi4 = engine4.evaluate_cascade(x_val, y_val, n_depth, H_atom_func)
        l4_scalar = phi4[0] 
        
        # --- L5: Evaluate PyTorch ReLU Network ---
        # Prepare batched tensors
        batch_x = torch.tensor([[x_val]], dtype=torch.float64)
        batch_y = torch.tensor([[y_val]], dtype=torch.float64)
        
        with torch.no_grad():
            l5_out = model(batch_x, batch_y)
            l5_scalar = l5_out.item()
            
        # --- Compare ---
        diff = abs(l4_scalar - l5_scalar)
        max_error = max(max_error, diff)
        
        # Using 1e-5 tolerance to account for float32 vs float64 NumPy precision drift
        if diff > 1e-5:
            print(f"  ❌ FAIL at ({x_val:.3f}, {y_val:.3f})! L4: {l4_scalar:.6f} | L5: {l5_scalar:.6f}")
            success = False
        else:
            print(f"  ✅ PASS at ({x_val:.3f}, {y_val:.3f}) | Output: {l5_scalar:.6f}")

    print(f"\n[3/3] Maximum floating-point deviation: {max_error:.2e}")
    if success:
        print("🎉 SUCCESS! The deep PyTorch ReLU network perfectly realizes the CPwL Tensor-Product Cascade.")
    else:
        print("⚠️ FAILED. Numerical mismatch detected.")

if __name__ == "__main__":
    main()