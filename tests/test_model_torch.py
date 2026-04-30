import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from refinet.network.model import SpecialHat, MinGadget, ProductGadget, ResidualLoop

# ---------------------------------------------------------
# Exact Mathematical Ground Truths (NumPy)
# ---------------------------------------------------------
def exact_hat(t, rho):
    slope = 1.0 / (0.5 - rho)
    return slope * (np.maximum(0, t - rho) - 2 * np.maximum(0, t - 0.5) + np.maximum(0, t - (1 - rho)))

def exact_min(u, v):
    return u - np.maximum(0, u - v)

def exact_gadget(lam, y, a):
    T1 = np.maximum(0, lam * a - y)
    T2 = np.maximum(0, a - lam * a - np.maximum(0, -y))
    return -T1 - T2 + a

def exact_E(t):
    t = t % 1.0
    if t < 1/3: return np.array([3*t, 3*t])
    elif t < 2/3: return np.array([1.0, 2.0 - 3*t])
    else: return np.array([3.0 - 3*t, 0.0])

# ---------------------------------------------------------
# PyTorch Test Runner
# ---------------------------------------------------------
def main():
    print("--- Testing PyTorch Runtime (Batch Processing) ---")
    torch.manual_seed(42)
    np.random.seed(42)
    num_tests = 1000

    # 1. Test Special Hat
    rho = 0.15
    hat_model = SpecialHat(rho)
    
    t_np = np.random.uniform(0, 1, (num_tests, 1)).astype(np.float32)
    t_pt = torch.from_numpy(t_np)
    
    out_pt = hat_model(t_pt).detach().numpy()
    out_np = exact_hat(t_np, rho)
    
    err_hat = np.max(np.abs(out_pt - out_np))
    print(f"SpecialHat Max Error:    {err_hat:.4e} {'✅' if err_hat < 1e-6 else '❌'}")

    # 2. Test Min Gadget
    min_model = MinGadget()
    
    u_np = np.random.uniform(-5, 5, (num_tests, 1)).astype(np.float32)
    v_np = np.random.uniform(-5, 5, (num_tests, 1)).astype(np.float32)
    u_pt, v_pt = torch.from_numpy(u_np), torch.from_numpy(v_np)
    
    out_pt = min_model(u_pt, v_pt).detach().numpy()
    out_np = exact_min(u_np, v_np)
    
    err_min = np.max(np.abs(out_pt - out_np))
    print(f"MinGadget Max Error:     {err_min:.4e} {'✅' if err_min < 1e-6 else '❌'}")

    # 3. Test Product Gadget
    N, a = 3, 5.0
    prod_model = ProductGadget(N, a)
    
    lam_np = np.random.uniform(0, 1, (num_tests, 1)).astype(np.float32)
    y_np = np.random.uniform(-a, a, (num_tests, N)).astype(np.float32)
    lam_pt, y_pt = torch.from_numpy(lam_np), torch.from_numpy(y_np)
    
    out_pt = prod_model(lam_pt, y_pt).detach().numpy()
    out_np = exact_gadget(lam_np, y_np, a)
    
    err_prod = np.max(np.abs(out_pt - out_np))
    print(f"ProductGadget Max Error: {err_prod:.4e} {'✅' if err_prod < 1e-6 else '❌'}")

    # 4. Test Exact Loop Controller (M=2)
    M = 2
    loop_model = ResidualLoop(M)
    
    t_vals = np.random.uniform(0, 1, num_tests)
    z_in_np = np.array([exact_E(t) for t in t_vals], dtype=np.float32)
    z_out_np = np.array([exact_E((M * t) % 1.0) for t in t_vals], dtype=np.float32)
    
    z_in_pt = torch.from_numpy(z_in_np)
    out_pt = loop_model(z_in_pt).detach().numpy()
    
    err_loop = np.max(np.abs(out_pt - z_out_np))
    print(f"ResidualLoop Max Error:  {err_loop:.4e} {'✅' if err_loop < 1e-6 else '❌'}")

if __name__ == "__main__":
    main()