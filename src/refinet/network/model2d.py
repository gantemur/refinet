import torch
import torch.nn as nn
import numpy as np

from refinet.network.tensor2d import TensorCompiler2D
from refinet.network.model import (
    create_pytorch_block, 
    InitialEmbedding, 
    ProductGadget, 
    MinGadget, 
    SpecialHat
)

# ---------------------------------------------------------
# 2D Sub-Modules (Compiled via tensor2d.py)
# ---------------------------------------------------------

class TorusController(nn.Module):
    """PyTorch module for the 2D Exact Torus Controller F(z_x, z_y)."""
    def __init__(self, M: int):
        super().__init__()
        self.net = create_pytorch_block(TensorCompiler2D.build_torus_controller(M))
    def forward(self, z): return self.net(z)

class TensorReadouts(nn.Module):
    """Outputs the 4 independent seam branches for the Torus state."""
    def __init__(self, epsilon: float):
        super().__init__()
        self.net = create_pytorch_block(TensorCompiler2D.build_tensor_readouts(epsilon))
    def forward(self, z): return self.net(z)

class TensorSelectors(nn.Module):
    """Outputs 2M partition weights (M for x, M for y)."""
    def __init__(self, M: int, n: int, delta_bar: float, rho: float):
        super().__init__()
        self.net = create_pytorch_block(TensorCompiler2D.build_tensor_selectors(M, n, delta_bar, rho))
    def forward(self, z): return self.net(z)

class ExactCascade2D(nn.Module):
    def __init__(self, system, T_matrices, depth_n: int, H_atom_module: nn.Module, epsilon: float, delta_bar: float, rho: float = 0.15):
        super().__init__()
        self.M = system.M
        self.n = depth_n
        self.L1 = system.L1
        self.L2 = system.L2
        self.dim = self.L1 * self.L2
        
        self.embed_1d = InitialEmbedding()
        self.torus_controller = TorusController(self.M)
        self.selectors = TensorSelectors(self.M, self.n, delta_bar, rho)
        self.readouts = TensorReadouts(epsilon)
        self.H_atom = H_atom_module
        self.min_gadget = MinGadget()
        
        # 1. Define B (The maximum matrix element)
        B = max([np.max(np.abs(T)) for T in T_matrices.values()])
        
        # 2. Safe mathematical ceiling for the 2D CPwL Product Gadget
        a_n = float((self.M**2)**self.n * (max(1.0, B)**self.n)) * 10.0 
        self.product_gadget = ProductGadget(N=self.dim, a=a_n)
        
        # CRITICAL FIX: NO TRANSPOSE. 
        # PyTorch row-vectors require the raw matrix to correctly compute w_{next}^T = w_{prev}^T @ T
        for q_tuple, T_mat in T_matrices.items():
            q_name = f"T_{q_tuple[0]}_{q_tuple[1]}"
            self.register_buffer(q_name, torch.tensor(T_mat, dtype=torch.float64))

    def forward(self, x, y):
        batch_size = x.shape[0]
        
        u = x - torch.floor(x)
        v = y - torch.floor(y)
        
        z_x = self.embed_1d(u)
        z_y = self.embed_1d(v)
        z = torch.cat([z_x, z_y], dim=1)
        
        # 1. Initialize Left Co-Vector at the TARGET patch
        a_idx = torch.clamp(torch.floor(x).long(), 0, self.L1 - 1)
        b_idx = torch.clamp(torch.floor(y).long(), 0, self.L2 - 1)
        block_idx = (a_idx * self.L2 + b_idx).view(batch_size, 1)
        
        # Enforce 64-bit state vector
        Phi = torch.zeros(batch_size, self.dim, device=x.device, dtype=torch.float64)
        Phi.scatter_(1, block_idx, 1.0)
        
        # 2. Concurrent Forward O(n) Trajectory
        for _ in range(self.n):
            chi = self.selectors(z)
            chi_x = chi[:, :self.M]
            chi_y = chi[:, self.M:]
            
            Phi_next = 0
            for q1 in range(self.M):
                for q2 in range(self.M):
                    T_q = getattr(self, f"T_{q1}_{q2}")
                    
                    # Row-vector multiplication precisely mirrors the Surrogate
                    y_state = Phi @ T_q
                    
                    y_state = self.product_gadget(chi_x[:, q1:q1+1], y_state)
                    y_state = self.product_gadget(chi_y[:, q2:q2+1], y_state)
                    
                    Phi_next = Phi_next + y_state
                    
            Phi = Phi_next
            z = self.torus_controller(z)
            
        # 3. Terminal State Readout
        reads = self.readouts(z)
        
        b1 = self.H_atom(reads[:, 0:2])
        b2 = self.H_atom(reads[:, 2:4])
        b3 = self.H_atom(reads[:, 4:6])
        b4 = self.H_atom(reads[:, 6:8])
        
        min_12 = self.min_gadget(b1, b2)
        min_34 = self.min_gadget(b3, b4)
        scalar_factor = self.min_gadget(min_12, min_34)
        
        # 4. Extract the SOURCE patch (index 0)
        output = Phi[:, 0:1] * scalar_factor
        
        return output