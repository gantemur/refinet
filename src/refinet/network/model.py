import torch
import torch.nn as nn
import numpy as np
from .tensor import TensorCompiler

def create_pytorch_block(numpy_layers):
    """
    Automated assembly line: Converts a list of numpy layer dictionaries 
    into a frozen PyTorch nn.Sequential block.
    """
    modules = []
    for layer in numpy_layers:
        W_np = layer['weight']
        b_np = layer['bias']
        out_features, in_features = W_np.shape

        linear = nn.Linear(in_features, out_features)
        
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(W_np, dtype=torch.float32))
            linear.bias.copy_(torch.tensor(b_np, dtype=torch.float32))
            
            # Lock the topology to prevent gradient updates from destroying the math
            linear.weight.requires_grad = False
            linear.bias.requires_grad = False

        modules.append(linear)

        if layer['activation'] == 'relu':
            modules.append(nn.ReLU())
            
    return nn.Sequential(*modules)


# ---------------------------------------------------------
# Sub-Modules (Compiled via tensor.py)
# ---------------------------------------------------------

class ResidualLoop(nn.Module):
    """PyTorch module for the Exact Loop Controller F(z)."""
    def __init__(self, M: int):
        super().__init__()
        offline_layers = TensorCompiler.build_loop_controller(M)
        self.net = create_pytorch_block(offline_layers)

    def forward(self, z):
        return self.net(z)

class SpecialHat(nn.Module):
    """PyTorch module for the Special Hat readout h(t)."""
    def __init__(self, rho: float):
        super().__init__()
        offline_layers = TensorCompiler.build_special_hat(rho)
        self.net = create_pytorch_block(offline_layers)

    def forward(self, t):
        return self.net(t)

class MinGadget(nn.Module):
    """PyTorch module for the Exact Min-Intersection min(u, v)."""
    def __init__(self):
        super().__init__()
        offline_layers = TensorCompiler.build_min_gadget()
        self.net = create_pytorch_block(offline_layers)

    def forward(self, u, v):
        x = torch.cat([u, v], dim=-1)
        return self.net(x)

class ProductGadget(nn.Module):
    """PyTorch module for the Cascade Mixing Valve \\Pi_a(\\lambda, y)."""
    def __init__(self, N: int, a: float):
        super().__init__()
        offline_layers = TensorCompiler.build_product_gadget(N, a)
        self.net = create_pytorch_block(offline_layers)

    def forward(self, lam, y):
        x = torch.cat([lam, y], dim=-1)
        return self.net(x)

# --- Assumption Stubs (Now Fully Implemented) ---

class InitialEmbedding(nn.Module):
    """Embeds scalar t into loop state z_0 = E(t)."""
    def __init__(self):
        super().__init__()
        offline_layers = TensorCompiler.build_initial_embedding()
        self.net = create_pytorch_block(offline_layers)
        
    def forward(self, t):
        return self.net(t)

class SeamReadouts(nn.Module):
    """Outputs [rho_minus(z), rho_plus(z)] to handle seam ambiguity."""
    def __init__(self, epsilon: float):
        super().__init__()
        offline_layers = TensorCompiler.build_seam_readouts(epsilon)
        self.net = create_pytorch_block(offline_layers)
        
    def forward(self, z):
        return self.net(z)

class LoopSelectors(nn.Module):
    """Outputs partition of unity weights chi_q,n(z)."""
    def __init__(self, M: int, n: int, delta_bar: float):
        super().__init__()
        offline_layers = TensorCompiler.build_selectors(M, n, delta_bar)
        self.net = create_pytorch_block(offline_layers)
        
    def forward(self, z):
        return self.net(z)

# ---------------------------------------------------------
# The Full Exact Cascade Network
# ---------------------------------------------------------

class ExactCascade(nn.Module):
    """
    The full O(n) depth continuous ReLU realization of the Homogeneous Curve Refinement.
    Implements Proposition 3.8 and Lemma 3.11.
    """
    def __init__(self, M: int, depth_n: int, T_matrices: list, a: float, rho: float, epsilon: float, delta_bar: float, u_mu: np.ndarray):
        super().__init__()
        self.M = M
        self.n = depth_n
        
        # Format the initial vector u_mu as a PyTorch tensor [1, N]
        self.register_buffer('u_mu', torch.tensor(u_mu, dtype=torch.float32).view(1, -1))
        N = self.u_mu.shape[-1]
        
        # 1. Forward Topological Loop
        self.embed = InitialEmbedding()
        self.loop_controller = ResidualLoop(M)
        
        # 2. Partition of Unity Selectors
        self.selectors = LoopSelectors(M, depth_n, delta_bar)
        
        # 3. Terminal Safety (Readouts & Hat)
        self.readouts = SeamReadouts(epsilon)
        self.hat = SpecialHat(rho)
        self.min_gadget = MinGadget()
        
        # 4. Matrix Field & Mixing Valve
        self.product_gadget = ProductGadget(N=N, a=a)
        
        # Register the transposed transition matrices T_q^T as frozen buffers
        for q, T_mat in enumerate(T_matrices):
            self.register_buffer(f"T_{q}", torch.tensor(T_mat.T, dtype=torch.float32))

    def forward(self, x):
        """
        Executes the cascade. 
        Input 'x' shape: [batch, 1]
        Output 'Phi' shape: [batch, N]
        """
        # --- STAGE 1: Forward Loop Trajectory ---
        # z_0 = E(x)
        z = self.embed(x)
        z_history = [z]
        
        # z_j = F(z_{j-1})
        for _ in range(self.n):
            z = self.loop_controller(z)
            z_history.append(z)
            
        # --- STAGE 2: Terminal Topological Safety Mechanism ---
        z_n = z_history[-1]
        
        # Extract the dual readouts and split into separate vectors
        rho_out = self.readouts(z_n)
        rho_minus, rho_plus = torch.split(rho_out, 1, dim=-1)
        
        h_minus = self.hat(rho_minus)
        h_plus = self.hat(rho_plus)
        
        # scalar_factor = min(h(rho-), h(rho+))
        scalar_factor = self.min_gadget(h_minus, h_plus)
        
        # --- STAGE 3: Backward Vector State Recursion ---
        # Initialize Phi_0 = u_mu (Batched across all input points)
        Phi = self.u_mu.expand(x.shape[0], -1)

        for j in range(1, self.n + 1):
            # z_{j-1} dictates the selector field
            z_prev = z_history[j - 1]
            chi_vals = self.selectors(z_prev)  # Shape: [batch, M]
            
            Phi_next = 0
            for q in range(self.M):
                # Isolate the selector for digit q: chi_q,n(z_{j-1}(x))
                chi_q = chi_vals[:, q:q+1]
                
                # Retrieve T_q^T
                T_q = getattr(self, f"T_{q}")
                
                # Compute y_q = Phi_{j-1} @ T_q^T (Batched vector-matrix multiplication)
                y_q = Phi @ T_q  
                
                # Accumulate via the Product Gadget: Pi_a(chi_q, y_q)
                Phi_next = Phi_next + self.product_gadget(chi_q, y_q)
                
            Phi = Phi_next
            
        return Phi