import numpy as np

class TensorCompiler:
    """
    Offline compiler for Level 5. Translates exact CPwL mathematical objects 
    into raw, hardcoded numpy weight and bias tensors.
    """
    
    @staticmethod
    def build_product_gadget(N: int, a: float):
        """
        Compiles the Product Gadget \Pi_a(\lambda, y) from Lemma 3.10.
        Input dimension: 1 + N (where index 0 is lambda, and 1:N+1 is y).
        Output dimension: N.
        
        Returns a list of layer dictionaries: [{'weight': W, 'bias': b}, ...]
        Layers 1 and 2 are ReLU layers. Layer 3 is a linear output layer.
        """
        # --- Layer 1 ---
        # Computes: [\lambda, ReLU(-y), ReLU(a * \lambda * 1 - y)]^T
        W1 = np.zeros((2 * N + 1, N + 1), dtype=float)
        b1 = np.zeros(2 * N + 1, dtype=float)
        
        # \lambda -> \lambda
        W1[0, 0] = 1.0
        
        # y -> -y
        W1[1 : N + 1, 1 : N + 1] = -np.eye(N)
        
        # (\lambda, y) -> a * \lambda * 1 - y
        W1[N + 1 : 2 * N + 1, 0] = a
        W1[N + 1 : 2 * N + 1, 1 : N + 1] = -np.eye(N)
        
        # --- Layer 2 ---
        # Computes: [T1, T2]^T
        # where T1 = ReLU(a * \lambda * 1 - y)
        # and   T2 = ReLU(a * 1 - a * \lambda * 1 - ReLU(-y))
        W2 = np.zeros((2 * N, 2 * N + 1), dtype=float)
        b2 = np.zeros(2 * N, dtype=float)
        
        # Pass T1 through: ReLU(T1) = T1 (since T1 >= 0)
        W2[0 : N, N + 1 : 2 * N + 1] = np.eye(N)
        
        # Compute T2
        W2[N : 2 * N, 0] = -a              # -a * \lambda
        W2[N : 2 * N, 1 : N + 1] = -np.eye(N)  # -ReLU(-y)
        b2[N : 2 * N] = a                  # +a * 1
        
        # --- Layer 3 (Linear Output) ---
        # Computes: -T1 - T2 + a * 1
        W3 = np.zeros((N, 2 * N), dtype=float)
        b3 = np.full(N, a, dtype=float)
        
        W3[0 : N, 0 : N] = -np.eye(N)      # -T1
        W3[0 : N, N : 2 * N] = -np.eye(N)  # -T2
        
        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'relu'},
            {'weight': W3, 'bias': b3, 'activation': 'linear'}
        ]

    @staticmethod
    def build_loop_controller(M: int):
        """
        Compiles the Exact Loop Controller F(z) from Appendix A.
        Input dimension: 2 (The loop state z)
        Output dimension: 2 (The next loop state F(z))
        
        Sets up the global CPwL gradient compatibility system across all 3M sectors
        and solves for the exact 1-hidden-layer ReLU weights.
        """
        c = np.array([2/3, 1/3])
        
        def E(t):
            t = t % 1.0
            if t < 1/3: return np.array([3*t, 3*t])
            elif t < 2/3: return np.array([1.0, 2.0 - 3*t])
            else: return np.array([3.0 - 3*t, 0.0])

        num_rays = 3 * M
        r, n, p = [], [], []
        
        # 1. Geometry of the sectors
        for m in range(num_rays):
            pm = E(m / num_rays)
            p.append(pm)
            rm = pm - c
            r.append(rm)
            n.append(np.array([rm[1], -rm[0]])) # Inward normal

        # 2. Extract the local affine transformations F_m(z) = M_m(z - c) + c
        M_mats = []
        for m in range(num_rays):
            j = m % 3
            tgt1 = E(j / 3)
            tgt2 = E((j + 1) / 3) if j < 2 else E(0.0)
            
            rm = r[m]
            rm_next = p[(m + 1) % num_rays] - c
            
            V_m = np.column_stack((rm, rm_next))
            U_m = np.column_stack((tgt1 - c, tgt2 - c))
            Mm = U_m @ np.linalg.inv(V_m)
            M_mats.append(Mm)

        # 3. Build activation matrix for the sectors
        # active[m, k] = 1 if ReLU k is active in sector m
        active = np.zeros((num_rays, num_rays))
        for m in range(num_rays):
            r_mid = r[m] + r[(m+1) % num_rays]
            for k in range(num_rays):
                if np.dot(n[k], r_mid) > 1e-9:
                    active[m, k] = 1.0

        # 4. Set up the linear system to find the global Base Gradient (A) and ReLU Weights (W_vecs)
        # M_m = A + sum_k (active[m, k] * W_vecs[k] * n[k]^T)
        num_eqs = 2 * num_rays
        num_vars = 2 + num_rays

        C = np.zeros((num_eqs, num_vars))
        Bx = np.zeros(num_eqs)
        By = np.zeros(num_eqs)

        for m in range(num_rays):
            C[2*m, 0] = 1.0
            C[2*m, 1] = 0.0
            for k in range(num_rays):
                C[2*m, 2+k] = active[m, k] * n[k][0]
            Bx[2*m] = M_mats[m][0, 0]
            By[2*m] = M_mats[m][1, 0]

            C[2*m+1, 0] = 0.0
            C[2*m+1, 1] = 1.0
            for k in range(num_rays):
                C[2*m+1, 2+k] = active[m, k] * n[k][1]
            Bx[2*m+1] = M_mats[m][0, 1]
            By[2*m+1] = M_mats[m][1, 1]

        # Solve exactly!
        Xx, _, _, _ = np.linalg.lstsq(C, Bx, rcond=None)
        Xy, _, _, _ = np.linalg.lstsq(C, By, rcond=None)

        A = np.array([[Xx[0], Xx[1]], [Xy[0], Xy[1]]])
        W_vecs = np.zeros((num_rays, 2))
        for k in range(num_rays):
            W_vecs[k, 0] = Xx[2+k]
            W_vecs[k, 1] = Xy[2+k]

        # 5. Assemble the Tensors (Layer 1: The 3M ReLUs + 4 ReLUs to pass the linear part)
        N_hidden = num_rays + 4
        W1 = np.zeros((N_hidden, 2))
        b1 = np.zeros(N_hidden)
        
        for m in range(num_rays):
            W1[m, :] = n[m]
            b1[m] = -np.dot(n[m], c)
            
        W1[num_rays, :]   = [1, 0]
        W1[num_rays+1, :] = [-1, 0]
        W1[num_rays+2, :] = [0, 1]
        W1[num_rays+3, :] = [0, -1]

        # 6. Assemble the Tensors (Layer 2: Linear Output)
        W2 = np.zeros((2, N_hidden))
        for m in range(num_rays):
            W2[:, m] = W_vecs[m]
            
        W2[:, num_rays]   = A[:, 0]
        W2[:, num_rays+1] = -A[:, 0]
        W2[:, num_rays+2] = A[:, 1]
        W2[:, num_rays+3] = -A[:, 1]

        b2 = c - A @ c

        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'linear'}
        ]

    @staticmethod
    def build_special_hat(rho: float):
        """
        Compiles the Special Hat function h(t) from Section 2.7.
        Input dimension: 1 (The scalar parameter t)
        Output dimension: 1 (The hat value h(t))
        
        Returns a 1-hidden-layer ReLU network.
        """
        slope = 1.0 / (0.5 - rho)
        
        # --- Layer 1 ---
        # Computes: [ReLU(t - rho), ReLU(t - 0.5), ReLU(t - (1-rho))]^T
        W1 = np.array([[1.0], [1.0], [1.0]], dtype=float)
        b1 = np.array([-rho, -0.5, -(1.0 - rho)], dtype=float)
        
        # --- Layer 2 (Linear Output) ---
        # Computes: slope * (ReLU_1 - 2*ReLU_2 + ReLU_3)
        W2 = np.array([[slope, -2.0 * slope, slope]], dtype=float)
        b2 = np.zeros(1, dtype=float)
        
        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'linear'}
        ]

    @staticmethod
    def build_min_gadget():
        """
        Compiles the terminal exact min-gadget: min(u, v)
        Used to merge the left and right seam readouts: min(h(rho-), h(rho+))
        Input dimension: 2 (u, v)
        Output dimension: 1
        """
        # min(u, v) = u - ReLU(u - v)
        # Since 'u' can be negative, we must decompose it to pass through the ReLU:
        # u = ReLU(u) - ReLU(-u)
        
        # --- Layer 1 ---
        # Computes: [ReLU(u - v), ReLU(u), ReLU(-u)]^T
        W1 = np.array([[ 1.0, -1.0], 
                       [ 1.0,  0.0],
                       [-1.0,  0.0]], dtype=float)
        b1 = np.zeros(3, dtype=float)
        
        # --- Layer 2 (Linear Output) ---
        # Computes: -ReLU(u - v) + ReLU(u) - ReLU(-u)
        W2 = np.array([[-1.0, 1.0, -1.0]], dtype=float)
        b2 = np.zeros(1, dtype=float)
        
        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'linear'}
        ]

    @staticmethod
    def build_initial_embedding():
        """
        Compiles the Initial Embedding E(t).
        Input dimension: 1 (The scalar parameter t)
        Output dimension: 2 (The loop state z_0 = [z_x, z_y])
        """
        # --- Layer 1 ---
        # Computes: [ReLU(t), ReLU(-t), ReLU(t - 1/3), ReLU(t - 2/3)]^T
        # ReLU(-t) is included to safely pass t if it ever slightly dips below 0
        W1 = np.array([[ 1.0], 
                       [-1.0], 
                       [ 1.0], 
                       [ 1.0]], dtype=float)
        b1 = np.array([0.0, 0.0, -1.0/3.0, -2.0/3.0], dtype=float)
        
        # --- Layer 2 (Linear Output) ---
        # z_x = 3t - 3*ReLU(t - 1/3) - 3*ReLU(t - 2/3)
        # z_y = 3t - 6*ReLU(t - 1/3) + 3*ReLU(t - 2/3)
        W2 = np.zeros((2, 4), dtype=float)
        
        # z_x coefficients
        W2[0, :] = [3.0, -3.0, -3.0, -3.0]
        # z_y coefficients
        W2[1, :] = [3.0, -3.0, -6.0,  3.0]
        
        b2 = np.zeros(2, dtype=float)
        
        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'linear'}
        ]

    @staticmethod
    def build_seam_readouts(epsilon: float):
        """
        Compiles the Seam Readouts rho_minus and rho_plus.
        Input dimension: 2 (The loop state z)
        Output dimension: 2 ([rho_minus, rho_plus])
        """
        # --- Layer 1 ---
        # We extract distance to the horizontal and vertical boundaries
        # Computes: [ReLU(z_x / epsilon), ReLU((1 - z_x) / epsilon)]^T
        W1 = np.zeros((2, 2), dtype=float)
        b1 = np.zeros(2, dtype=float)
        
        W1[0, 0] = 1.0 / epsilon
        W1[1, 0] = -1.0 / epsilon
        b1[1] = 1.0 / epsilon
        
        # --- Layer 2 (Linear Output) ---
        # Pass the ReLUs through directly
        W2 = np.eye(2, dtype=float)
        b2 = np.zeros(2, dtype=float)
        
        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'linear'}
        ]

    @staticmethod
    def build_selectors(M: int, n: int, delta_bar: float):
        """
        Compiles the Partition of Unity Selectors \chi_{q,n}(z).
        Input dimension: 2 (The loop state z)
        Output dimension: M (The selector weights, strictly summing to 1 outside the transition zone)
        
        Note: The exact 1-layer CPwL formulation here is strictly valid for M=2.
        """
        if M != 2:
            raise NotImplementedError("Exact 1-layer CPwL selectors are geometrically restricted to M=2.")
            
        delta = delta_bar / (M ** n)
        
        # We project z onto a diagonal decision boundary: d = z_x + z_y - 1
        # --- Layer 1 ---
        # Computes components for the cross-fade ramps
        # [ReLU(1 + d/delta), ReLU(d/delta), ReLU(1 - d/delta), ReLU(-d/delta)]
        W1 = np.zeros((4, 2), dtype=float)
        b1 = np.zeros(4, dtype=float)
        
        factor = 1.0 / delta
        
        # d/delta = (z_x + z_y)/delta - 1/delta
        w_d = np.array([factor, factor])
        b_d = -factor
        
        W1[0, :] = w_d;  b1[0] = 1.0 + b_d
        W1[1, :] = w_d;  b1[1] = b_d
        W1[2, :] = -w_d; b1[2] = 1.0 - b_d
        W1[3, :] = -w_d; b1[3] = -b_d
        
        # --- Layer 2 (Linear Output) ---
        # chi_0 = ReLU(1 + d/delta) - ReLU(d/delta)
        # chi_1 = ReLU(1 - d/delta) - ReLU(-d/delta)
        W2 = np.zeros((2, 4), dtype=float)
        
        W2[0, 0] = 1.0; W2[0, 1] = -1.0
        W2[1, 2] = 1.0; W2[1, 3] = -1.0
        
        b2 = np.zeros(2, dtype=float)
        
        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'linear'}
        ]

    