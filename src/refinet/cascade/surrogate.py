import math
import numpy as np

# ==================================================================================================
# Level 4: Continuous Surrogate Cascade
# Implements CPwL exact loop controllers, readouts, and the product gadget.
# ==================================================================================================

class LoopController:
    """
    Implements the Exact Loop Controller from Section 3.2 and Appendix A.
    Transports the residual orbit exactly on the polygonal loop \\Gamma.
    """
    def __init__(self, M, vertices=None):
        self.M = M
        
        # Default to the paper's exact triangle if no vertices are provided
        if vertices is None:
            vertices = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
            
        self.vertices = [np.array(v, dtype=float) for v in vertices]
        self.K = len(self.vertices)
        
        # The central anchor for the fan partition (Barycenter)
        self.c = np.mean(self.vertices, axis=0)
        
        # Precompile the K * M barycentric sectors for the CPwL map F(z)
        self._sectors = self._compile_sectors()

    def E(self, t):
        """Maps t in [0, 1] proportionally onto the perimeter of the polygon."""
        t = max(0.0, min(1.0, t))
        scaled_t = t * self.K
        j = int(math.floor(scaled_t))
        
        if j == self.K:
            return self.vertices[0]
            
        s = scaled_t - j
        v_start = self.vertices[j]
        v_end = self.vertices[(j + 1) % self.K]
        
        return (1.0 - s) * v_start + s * v_end

    def _compile_sectors(self):
        sectors = []
        for m in range(self.K * self.M):
            t_m = m / (self.K * self.M)
            t_next = (m + 1) / (self.K * self.M)
            
            p_m = self.E(t_m)
            p_next = self.E(t_next)
            
            j = m % self.K
            q_m = self.vertices[j]
            q_next = self.vertices[(j + 1) % self.K]
                
            V = np.column_stack((p_m - self.c, p_next - self.c))
            V_inv = np.linalg.inv(V)
            
            sectors.append({
                'V_inv': V_inv,
                'q_m': q_m,
                'q_next': q_next
            })
        return sectors

    def F(self, z):
        """The Exact CPwL Controller Map F: R^2 -> R^2."""
        z_vec = np.array(z, dtype=float)
        rel_z = z_vec - self.c
        
        best_sector = None
        min_violation = float('inf')
        best_coords = (0.0, 0.0)
        
        for sector in self._sectors:
            coords = sector['V_inv'] @ rel_z
            alpha, beta = coords[0], coords[1]
            
            # Perfect fit
            if alpha >= -1e-9 and beta >= -1e-9:
                return self.c + alpha * (sector['q_m'] - self.c) + beta * (sector['q_next'] - self.c)
                
            # Track the closest sector in case numerical float drift pushed z slightly outside
            violation = max(0, -alpha) + max(0, -beta)
            if violation < min_violation:
                min_violation = violation
                best_sector = sector
                best_coords = (alpha, beta)
                
        # Graceful fallback for numerical drift
        alpha, beta = best_coords
        return self.c + alpha * (best_sector['q_m'] - self.c) + beta * (best_sector['q_next'] - self.c)

    def evaluate_loop_state(self, n, x):
        """Recursively evaluates z_n(x) = F^n(E(x))."""
        z_current = self.E(x)
        for _ in range(n):
            z_current = self.F(z_current)
        return z_current


class TerminalReadout:
    """
    Implements the Terminal Readouts and Min-Gadget from Section 3.3.
    """
    def __init__(self, controller, epsilon=0.05):
        self.controller = controller
        self.epsilon = epsilon

    def E_inv(self, z):
        """
        Calculates the inverse mapping E^{-1}: \\Gamma -> [0, 1).
        Rigorously finds the physically closest boundary segment to avoid aliasing.
        """
        z = np.array(z, dtype=float)
        K = self.controller.K
        
        best_t = 0.0
        min_dist = float('inf')
        
        for j in range(K):
            v_start = self.controller.vertices[j]
            v_end = self.controller.vertices[(j + 1) % K]
            
            edge_vec = v_end - v_start
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq < 1e-14: continue
                
            # Project z onto the edge to find local parameter s
            s = np.dot(z - v_start, edge_vec) / edge_len_sq
            s_clamped = max(0.0, min(1.0, s))
            
            # Calculate true orthogonal distance to the bounded segment
            projection = v_start + s_clamped * edge_vec
            dist = np.linalg.norm(z - projection)
            
            if dist < min_dist:
                min_dist = dist
                t = (j + s_clamped) / K
                # Enforce the seam constraint
                best_t = t if t < 1.0 else 0.0
                
        return best_t

    def r_minus(self, t):
        if t <= 1.0 - self.epsilon:
            return t
        return ((1.0 - self.epsilon) / self.epsilon) * (1.0 - t)

    def r_plus(self, t):
        if t <= self.epsilon:
            return 1.0 - ((1.0 - self.epsilon) / self.epsilon) * t
        return t

    def rho_minus(self, z):
        return self.r_minus(self.E_inv(z))

    def rho_plus(self, z):
        return self.r_plus(self.E_inv(z))

    def evaluate_scalar_factor(self, z, h_func):
        val_minus = h_func(self.rho_minus(z))
        val_plus = h_func(self.rho_plus(z))
        return min(val_minus, val_plus)

class LoopSelector:
    """
    Implements the Loop-State Matrix Selectors from Section 3.4.
    Generates the continuous CPwL matrix field \hat{T}_n(z) to replace discrete digit jumps.
    """
    def __init__(self, controller, n_depth, rho=0.15, delta_bar=0.5):
        self.controller = controller
        self.M = controller.M
        self.n_depth = n_depth
        self.rho = rho
        self.delta_bar = delta_bar
        
        # Eq 27: The exact transition width
        self.delta_n = delta_bar * rho * (self.M ** -(n_depth + 1))
        
        # Helper to map z back to parameter t rigorously
        self._readout_helper = TerminalReadout(controller)

    def evaluate_chi(self, q, z):
        """
        Evaluates the loop-state selector \chi_{q,n}(z).
        Ramps continuously between matrices in the transition zones J_n.
        """
        t = self._readout_helper.E_inv(z)
        
        # Enforce strict topological closure
        if t >= 1.0: t = 0.0 
            
        k = int(math.floor(t * self.M))
        if k == self.M: k = self.M - 1
        
        cell_start = k / self.M
        s = t - cell_start
        
        if s <= self.delta_n:
            # We are inside J_n (The Transition Ramp)
            prev_q = (k - 1) % self.M
            alpha = s / self.delta_n
            
            if q == prev_q: return 1.0 - alpha
            if q == k: return alpha
            return 0.0
        else:
            # We are inside the Stable Indicator Region
            if q == k: return 1.0
            return 0.0

    def evaluate_T_hat(self, z, T_matrices):
        """
        Constructs the modified continuous matrix field \hat{T}_n(z) (Eq 29).
        """
        # Initialize with a zero matrix of the correct block shape
        T_hat = np.zeros_like(T_matrices[0], dtype=float)
        
        for q in range(self.M):
            chi = self.evaluate_chi(q, z)
            if chi > 0.0:
                T_hat += chi * T_matrices[q]
                
        return T_hat

class SurrogateCascadeEngine:
    """
    Level 4: The Continuous Surrogate Cascade Engine (Section 3.5 & 4.1).
    Replaces the exact discrete M-ary router with the continuous CPwL exact loop math,
    and natively accumulates affine forcing terms.
    """
    def __init__(self, block_engine, rho=0.15, delta_bar=0.5, epsilon=0.05):
        self.block_engine = block_engine
        self.p = block_engine.p
        self.M = block_engine.M
        self.T_matrices = block_engine.T
        
        self.controller = LoopController(self.M)
        self.readout = TerminalReadout(self.controller, epsilon=epsilon)
        
        self.rho = rho
        self.delta_bar = delta_bar

    def evaluate_affine(self, t, n_depth, gamma_0, B_curves=None):
        """
        Evaluates the continuous surrogate cascade using exact loop transport
        and the continuous matrix field.
        """
        if B_curves is None: B_curves = []
            
        k_target = int(math.floor(t))
        x = t - k_target
        if k_target < self.block_engine.K_min or k_target >= self.block_engine.K_max:
            return [0.0] * self.p 
            
        block_idx = k_target - self.block_engine.K_min
        
        # -------------------------------------------------------------
        # 1. Forward Pass (Transport the exact continuous loop state)
        # -------------------------------------------------------------
        z_orbit = [self.controller.E(x)]
        for _ in range(n_depth):
            z_orbit.append(self.controller.F(z_orbit[-1]))
            
        # -------------------------------------------------------------
        # 2. Terminal State Parameter
        # -------------------------------------------------------------
        # We extract the fractional parameter directly from the loop using E_inv.
        # Because the forcing curves (like D(t)) vanish at 0 and 1, the topological 
        # seam jump 1 -> 0 is naturally absorbed without the min-gadget!
        r_array = [self.readout.E_inv(z) for z in z_orbit]
        
        # Initialize the base state
        phi = self.block_engine._vectorize(gamma_0, r_array[-1])
        
        # -------------------------------------------------------------
        # 3. Backward Pass (Matrix Field + Affine Forcing Accumulation)
        # -------------------------------------------------------------
        for j in range(n_depth - 1, -1, -1):
            # CORRECT: Pass the global cascade depth so delta_n is universally tiny!
            selector = LoopSelector(self.controller, n_depth=n_depth, rho=self.rho, delta_bar=self.delta_bar)
    
            # Evaluate the matrix field exactly on the loop
            T_hat = selector.evaluate_T_hat(z_orbit[j], self.T_matrices)
            
            # Product Gadget: Multiply the state by the continuous field
            phi = T_hat @ phi
            
            # Affine Accumulation (Lemma 4.1): Add the continuous forcing terms
            for B_curve in B_curves:
                phi += self.block_engine._vectorize(B_curve, r_array[j])
            
        # Extract the local p-dimensional block for the target coordinate x
        s_n = phi[block_idx*self.p : (block_idx+1)*self.p]
        return s_n.tolist()