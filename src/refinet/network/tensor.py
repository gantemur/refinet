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
    def build_loop_controller(M: int):
        """
        Compiles the Exact Loop Controller F(z).
        - For M=2: Uses the geometric miracle 1-hidden-layer network.
        - For M>2: Uses the universal 2-hidden-layer Radial Tent Engine.
        """
        c = np.array([2/3, 1/3], dtype=float)
        
        def E(t):
            t = t % 1.0
            if t < 1/3: return np.array([3*t, 3*t])
            elif t < 2/3: return np.array([1.0, 2.0 - 3*t])
            else: return np.array([3.0 - 3*t, 0.0])

        if M == 2:
            # --- FAST PATH: 1-Layer Geometric Miracle (M=2 Only) ---
            num_rays = 6
            r, n, p = [], [], []
            for m in range(num_rays):
                pm = E(m / num_rays)
                p.append(pm)
                rm = pm - c
                r.append(rm)
                n.append(np.array([rm[1], -rm[0]])) 

            M_mats = []
            for m in range(num_rays):
                j = m % 3
                tgt1 = E(j / 3)
                tgt2 = E((j + 1) / 3) if j < 2 else E(0.0)
                V_m = np.column_stack((r[m], p[(m + 1) % num_rays] - c))
                U_m = np.column_stack((tgt1 - c, tgt2 - c))
                M_mats.append(U_m @ np.linalg.inv(V_m))

            active = np.zeros((num_rays, num_rays))
            for m in range(num_rays):
                r_mid = r[m] + r[(m+1) % num_rays]
                for k in range(num_rays):
                    if np.dot(n[k], r_mid) > 1e-9:
                        active[m, k] = 1.0

            C, Bx, By = np.zeros((12, 8)), np.zeros(12), np.zeros(12)
            for m in range(num_rays):
                C[2*m, 0], C[2*m, 1] = 1.0, 0.0
                C[2*m, 2:] = active[m] * np.array([nk[0] for nk in n])
                Bx[2*m], By[2*m] = M_mats[m][0, 0], M_mats[m][1, 0]

                C[2*m+1, 0], C[2*m+1, 1] = 0.0, 1.0
                C[2*m+1, 2:] = active[m] * np.array([nk[1] for nk in n])
                Bx[2*m+1], By[2*m+1] = M_mats[m][0, 1], M_mats[m][1, 1]

            Xx, _, _, _ = np.linalg.lstsq(C, Bx, rcond=None)
            Xy, _, _, _ = np.linalg.lstsq(C, By, rcond=None)
            A = np.array([[Xx[0], Xx[1]], [Xy[0], Xy[1]]])
            W_vecs = np.column_stack((Xx[2:], Xy[2:]))

            N_hidden = num_rays + 4
            W1, b1 = np.zeros((N_hidden, 2)), np.zeros(N_hidden)
            for m in range(num_rays):
                W1[m, :] = n[m]
                b1[m] = -np.dot(n[m], c)
            W1[num_rays:num_rays+4, :] = [[1, 0], [-1, 0], [0, 1], [0, -1]]

            W2 = np.zeros((2, N_hidden))
            W2[:, :num_rays] = W_vecs.T
            W2[:, num_rays:] = np.column_stack((A[:, 0], -A[:, 0], A[:, 1], -A[:, 1]))
            b2 = c - A @ c

            return [
                {'weight': W1, 'bias': b1, 'activation': 'relu'},
                {'weight': W2, 'bias': b2, 'activation': 'linear'}
            ]

        # --- GENERAL M: 2-Layer Exact Radial Tent Engine ---
        num_rays = 3 * M
        r, p = [], []
        
        for m in range(num_rays):
            pm = E(m / num_rays)
            p.append(pm)
            r.append(pm - c)
            
        # Target vectors for each ray F(p_m) - c
        tgt = []
        for m in range(num_rays):
            tgt_pm = E((M * (m / num_rays)) % 1.0)
            tgt.append(tgt_pm - c)

        # Layer 1: Computes the bounds for the Tent functions
        # For each ray m, we need 3 neurons
        W1 = np.zeros((3 * num_rays, 2), dtype=float)
        b1 = np.zeros(3 * num_rays, dtype=float)
        
        for m in range(num_rays):
            rm = r[m]
            rm_prev = r[(m - 1) % num_rays]
            rm_next = r[(m + 1) % num_rays]
            
            # alpha_1: 0 on prev ray, 1 on current ray
            R1 = np.vstack((rm_prev, rm))
            alpha_m1 = np.linalg.solve(R1, np.array([0.0, 1.0]))
            
            # alpha_2: 0 on next ray, 1 on current ray
            R2 = np.vstack((rm_next, rm))
            alpha_m2 = np.linalg.solve(R2, np.array([0.0, 1.0]))
            
            idx = 3 * m
            # Neuron A: (alpha_1 - alpha_2) * (z - c)
            W1[idx, :] = alpha_m1 - alpha_m2
            b1[idx] = -np.dot(alpha_m1 - alpha_m2, c)
            
            # Neuron B: alpha_1 * (z - c)  [Positive part]
            W1[idx+1, :] = alpha_m1
            b1[idx+1] = -np.dot(alpha_m1, c)
            
            # Neuron C: -alpha_1 * (z - c) [Negative part]
            W1[idx+2, :] = -alpha_m1
            b1[idx+2] = np.dot(alpha_m1, c)

        # Layer 2: Applies exact Min logic to complete the Tent T_m(z)
        W2 = np.zeros((num_rays, 3 * num_rays), dtype=float)
        b2 = np.zeros(num_rays, dtype=float)
        
        for m in range(num_rays):
            idx = 3 * m
            W2[m, idx] = -1.0      # - ReLU(u1 - u2)
            W2[m, idx+1] = 1.0     # + ReLU(u1)
            W2[m, idx+2] = -1.0    # - ReLU(-u1)

        # Layer 3: Maps T_m(z) to the final target coordinates
        W3 = np.zeros((2, num_rays), dtype=float)
        for m in range(num_rays):
            W3[:, m] = tgt[m]
        b3 = c

        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'relu'},
            {'weight': W3, 'bias': b3, 'activation': 'linear'}
        ]

    @staticmethod
    def build_seam_readouts(epsilon: float):
        """Extracts the exact coordinates [z_x, 1 - z_x] without scaling."""
        W1 = np.zeros((2, 2), dtype=float)
        b1 = np.zeros(2, dtype=float)
        W1[0, 0] = 1.0
        W1[1, 0] = -1.0
        b1[1] = 1.0
        
        W2 = np.eye(2, dtype=float)
        b2 = np.zeros(2, dtype=float)
        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'linear'}
        ]

    @staticmethod
    def build_selectors(M: int, n: int, delta_bar: float, rho: float = 0.15):
        # FIXED: Added the rho multiplier to exactly match the L4 transition width
        delta = delta_bar * rho / (M ** n)
        
        c = np.array([2/3, 1/3], dtype=float)
        num_rays = max(12, 6 * M)
        r = []
        for m in range(num_rays):
            t_m = m / num_rays
            t_mod = t_m % 1.0
            if t_mod < 1/3: pm = np.array([3*t_mod, 3*t_mod])
            elif t_mod < 2/3: pm = np.array([1.0, 2.0 - 3*t_mod])
            else: pm = np.array([3.0 - 3*t_mod, 0.0])
            r.append(pm - c)

        W1 = np.zeros((3 * num_rays, 2), dtype=float)
        b1 = np.zeros(3 * num_rays, dtype=float)
        for m in range(num_rays):
            rm = r[m]
            rm_prev, rm_next = r[(m - 1) % num_rays], r[(m + 1) % num_rays]
            
            R1 = np.vstack((rm_prev, rm))
            alpha_m1 = np.linalg.solve(R1, np.array([0.0, 1.0]))
            R2 = np.vstack((rm_next, rm))
            alpha_m2 = np.linalg.solve(R2, np.array([0.0, 1.0]))
            
            idx = 3 * m
            W1[idx, :], b1[idx] = alpha_m1 - alpha_m2, -np.dot(alpha_m1 - alpha_m2, c)
            W1[idx+1, :], b1[idx+1] = alpha_m1, -np.dot(alpha_m1, c)
            W1[idx+2, :], b1[idx+2] = -alpha_m1, np.dot(alpha_m1, c)

        W2 = np.zeros((num_rays, 3 * num_rays), dtype=float)
        b2 = np.zeros(num_rays, dtype=float)
        for m in range(num_rays):
            idx = 3 * m
            W2[m, idx], W2[m, idx+1], W2[m, idx+2] = -1.0, 1.0, -1.0

        W3 = np.zeros((M, num_rays), dtype=float)
        b3 = np.zeros(M, dtype=float)
        for m in range(num_rays):
            t_m = m / num_rays
            for q in range(M):
                center = (q + 0.5) / M
                dist = abs((t_m - center + 0.5) % 1.0 - 0.5)
                radius = 0.5 / M
                
                if dist <= radius - delta/2: chi_val = 1.0
                elif dist >= radius + delta/2: chi_val = 0.0
                else: chi_val = 0.5 - (dist - radius) / delta
                W3[q, m] = chi_val

        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'relu'},
            {'weight': W3, 'bias': b3, 'activation': 'linear'}
        ]