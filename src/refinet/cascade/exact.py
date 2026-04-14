import math
import numpy as np
from .decomposition import SpecialHatDecomposition

class MaryRouter:
    """
    Implements Lemma 4.3: The M-ary Topological Router.
    Decodes a global parameter t in [0, 1] into its local M-ary digits q_j 
    and local fractional residuals r_j.
    """
    def __init__(self, M):
        self.M = M

    def decode(self, t, depth):
        # Clamp t to [0, 1] to prevent array bounds errors at the extremes
        t_safe = max(0.0, min(1.0, t))
        digits = []
        residuals = []
        current_r = t_safe
        
        for _ in range(depth):
            scaled = current_r * self.M
            q = math.floor(scaled)
            if q == self.M:
                q = self.M - 1
            digits.append(q)
            current_r = scaled - q
            residuals.append(current_r)
            
        return digits, residuals
        
    def get_active_branch(self, t, depth):
        digits, residuals = self.decode(t, depth)
        return digits, residuals[-1] if residuals else t

class BlockCascadeEngine:
    """
    Level 3: The Exact Vectorized Cascade Algorithm.
    Evaluates V^n, S_n, and stage-dependent non-stationary W^n natively in O(n) time.
    """
    def __init__(self, system, K_min=0, K_max=None):
        self.system = system
        self.A = np.array(system.A, dtype=float)
        self.M = system.M
        self.p = system.p
        self.num_A = len(self.A)
        
        self.K_min = K_min
        self.K_max = K_max if K_max is not None else K_min + system.L
        self.L = self.K_max - self.K_min
        
        self.router = MaryRouter(self.M)
        self.T = self._compile_T_matrices()

    def _compile_T_matrices(self):
        T = np.zeros((self.M, self.p * self.L, self.p * self.L))
        for q in range(self.M):
            for k in range(self.L):
                for l in range(self.L):
                    j = q + self.M * (self.K_min + k) - (self.K_min + l)
                    if 0 <= j < self.num_A:
                        row_start = k * self.p
                        row_end = (k + 1) * self.p
                        col_start = l * self.p
                        col_end = (l + 1) * self.p
                        T[q, row_start:row_end, col_start:col_end] = self.A[j]
        return T

    def _vectorize(self, curve, x):
        vec = np.zeros(self.p * self.L)
        for k in range(self.L):
            global_t = x + self.K_min + k
            vec[k*self.p : (k+1)*self.p] = curve(global_t)
        return vec

    def evaluate_affine(self, t, n_depth, gamma_0, B_curves=None, c_weights_list=None):
        """Standard stationary and fixed-weight affine evaluation."""
        if B_curves is None: B_curves = []
        if c_weights_list is None: c_weights_list = []
            
        k_target = int(math.floor(t))
        x = t - k_target
        if k_target < self.K_min or k_target >= self.K_max:
            return [0.0] * self.p 
            
        block_idx = k_target - self.K_min
        digits, residuals = self.router.decode(x, n_depth)
        
        P_n = np.eye(self.p * self.L)
        for q in digits:
            P_n = P_n @ self.T[q]
            
        G_0 = self._vectorize(gamma_0, residuals[-1])
        G_n = P_n @ G_0
        v_gamma = G_n[block_idx*self.p : (block_idx+1)*self.p]
        
        R_array = [x] + residuals
        U = np.zeros(self.p * self.L) 
        
        for k in range(n_depth - 1, -1, -1):
            q_val = digits[k]
            r_val = R_array[k]
            U = self.T[q_val] @ U
            
            for alpha, B_curve in enumerate(B_curves):
                if alpha < len(c_weights_list) and k < len(c_weights_list[alpha]):
                    c_val = c_weights_list[alpha][k]
                else:
                    c_val = 1.0
                U += c_val * self._vectorize(B_curve, r_val)
                
        s_n = U[block_idx*self.p : (block_idx+1)*self.p]
        return (v_gamma + s_n).tolist()

    def evaluate_nonstationary(self, t, n_depth):
        """
        Natively evaluates W^n \Gamma_0(t) = \Gamma_n(t) + \eta_n(t) for stage-dependent systems.
        Delegates the geometric forcing directly to the AffineRefinementSystem's exact compute_E,
        which natively encapsulates all scaled bases and connectors.
        """
        k_target = int(math.floor(t))
        x = t - k_target
        if k_target < self.K_min or k_target >= self.K_max:
            # If evaluating outside bounds, gracefully return the anchor profile
            final_anchor = self.system._get_anchor(n_depth)
            return final_anchor(t) if final_anchor else [0.0] * self.p 
            
        block_idx = k_target - self.K_min
        digits, residuals = self.router.decode(x, n_depth)

        # Affine Forcing Sum \eta_n via exact backward recursion
        R_array = [x] + residuals
        U = np.zeros(self.p * self.L) 
        
        for k in range(n_depth - 1, -1, -1):
            stage_r = n_depth - 1 - k  
            q_val = digits[k]
            r_val = R_array[k]
            
            U = self.T[q_val] @ U
            
            # Request the true geometric defect curve directly from the Level 2 system
            vec_B = np.zeros(self.p * self.L)
            for block in range(self.L):
                global_tau = r_val + self.K_min + block
                vec_B[block*self.p : (block+1)*self.p] = self.system.compute_E(stage_r, global_tau)
                
            U += vec_B
            
        s_n = U[block_idx*self.p : (block_idx+1)*self.p]
        
        # Add the final global geometric anchor profile (\Gamma_n)
        final_anchor = self.system._get_anchor(n_depth)
        gamma_base = final_anchor(t) if final_anchor else [0.0] * self.p
        
        return (np.array(gamma_base) + s_n).tolist()