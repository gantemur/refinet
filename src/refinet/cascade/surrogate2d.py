import numpy as np
from refinet.cascade.surrogate import LoopController

class TorusController:
    """Implements the Exact Torus Controller from Proposition 3.4."""
    def __init__(self, M=2):
        self.M = M
        self.ctrl_x = LoopController(M)
        self.ctrl_y = LoopController(M)

    def E_torus(self, x, y):
        return np.concatenate([self.ctrl_x.E(x), self.ctrl_y.E(y)])

    def F_torus(self, z_torus):
        return np.concatenate([self.ctrl_x.F(z_torus[:2]), self.ctrl_y.F(z_torus[2:])])

class TerminalReadout2D:
    """Forces L4 to match the PyTorch CPwL reflection logic."""
    def rho_minus(self, z): return z[0]
    def rho_plus(self, z): return 1.0 - z[0]

class SurrogateCascadeEngine2D:
    def __init__(self, block_engine_2d, rho=0.15, delta_bar=0.5, epsilon=0.05):
        self.engine = block_engine_2d
        self.T = self.engine.T
        self.M = self.engine.system.M
        
        self.torus = TorusController(self.M)
        self.readout_x = TerminalReadout2D()
        self.readout_y = TerminalReadout2D()
        
        self.rho = rho
        self.delta_bar = delta_bar

    def evaluate_special_atom(self, z_torus, H_func):
        z_x, z_y = z_torus[:2], z_torus[2:]
        rx_minus, rx_plus = self.readout_x.rho_minus(z_x), self.readout_x.rho_plus(z_x)
        ry_minus, ry_plus = self.readout_y.rho_minus(z_y), self.readout_y.rho_plus(z_y)
        
        b1, b2 = H_func(rx_minus, ry_minus), H_func(rx_minus, ry_plus)
        b3, b4 = H_func(rx_plus, ry_minus), H_func(rx_plus, ry_plus)
        return min(b1, b2, b3, b4)

    def _product_gadget(self, lam, y, a):
        """Exact numpy equivalent of the PyTorch CPwL Product Gadget."""
        term1 = np.maximum(0.0, a * lam - y)
        term2 = np.maximum(0.0, a * (1.0 - lam) - np.maximum(0.0, -y))
        return a - term1 - term2

    def evaluate_cascade(self, x, y, n_depth, H_atom):
        z = self.torus.E_torus(x, y)
        
        # 1. Initialize ROW Vector at the TARGET patch
        phi = np.zeros(self.engine.dim, dtype=float)
        a_idx = int(np.clip(np.floor(x), 0, self.engine.L1 - 1))
        b_idx = int(np.clip(np.floor(y), 0, self.engine.L2 - 1))
        target_idx = a_idx * self.engine.L2 + b_idx
        phi[target_idx] = 1.0
        
        # 2. Match the exact PyTorch a_n ceiling parameter
        B = max([np.max(np.abs(T)) for T in self.T.values()])
        a_n = float((self.M**2)**n_depth * (max(1.0, B)**n_depth)) * 10.0
        
        # 3. Import the exact offline weights used by PyTorch L5
        from refinet.network.tensor import TensorCompiler
        selector_layers = TensorCompiler.build_selectors(self.M, n_depth, self.delta_bar, self.rho)
        
        # Helper to run the Numpy-equivalent forward pass for chi
        def eval_cpwl_net(z_in):
            out = z_in
            for layer in selector_layers:
                W, b = layer['weight'], layer['bias']
                out = out @ W.T + b
                if layer['activation'] == 'relu': 
                    out = np.maximum(0, out)
            return out
        
        # 4. Exact O(n) Forward ROW-Vector Propagation
        for j in range(n_depth):
            z_x, z_y = z[:2], z[2:]
            
            # Evaluate identical CPwL polyhedral geometry
            chi_x_all = eval_cpwl_net(z_x)
            chi_y_all = eval_cpwl_net(z_y)
            
            phi_next = np.zeros_like(phi)
            
            for q1 in range(self.M):
                for q2 in range(self.M):
                    chi_x = chi_x_all[q1]
                    chi_y = chi_y_all[q2]
                    
                    # Row-vector multiplication, exactly matching Phi @ T_q
                    y_state = phi @ self.T[(q1, q2)]
                    
                    # Strictly apply the CPwL Product Gadgets
                    y_state = self._product_gadget(chi_x, y_state, a_n)
                    y_state = self._product_gadget(chi_y, y_state, a_n)
                    
                    phi_next += y_state
                    
            phi = phi_next
            z = self.torus.F_torus(z)
            
        scalar_factor = self.evaluate_special_atom(z, H_atom)
        
        # 5. Extract the SOURCE patch
        return [phi[0] * scalar_factor]