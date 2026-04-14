class AffineRefinementSystem:
    """
    Level 2: Evaluates the global affine refinement equations exactly.
    Natively handles stationary and infinite stage-dependent systems.
    """
    def __init__(self, A_matrices, M=2, B_forcing=None, p=2, L=1, anchor_profile=None, forcing_is_defect=False):
        self.A = A_matrices
        self.num_matrices = len(A_matrices)
        self.M = M 
        self.B = B_forcing
        self.p = p
        self.L = L
        self.anchor_profile = anchor_profile
        self.forcing_is_defect = forcing_is_defect

        # Robustly detect if B is a stage-dependent generator or a static curve
        self._B_is_stage_dependent = False
        if isinstance(self.B, list):
            self._B_is_stage_dependent = True
        elif callable(self.B):
            try:
                # If evaluating it returns a callable, it's a generator returning a curve
                if callable(self.B(0)):
                    self._B_is_stage_dependent = True
            except Exception:
                pass

        # Robustly detect if anchor_profile is a stage-dependent generator
        self._anchor_is_stage_dependent = False
        if isinstance(self.anchor_profile, list):
            self._anchor_is_stage_dependent = True
        elif callable(self.anchor_profile):
            try:
                if callable(self.anchor_profile(0)):
                    self._anchor_is_stage_dependent = True
            except Exception:
                pass

    def _matrix_vector_mult(self, matrix, vector):
        result = [0.0] * self.p
        for row in range(self.p):
            for col in range(self.p):
                result[row] += matrix[row][col] * vector[col]
        return result

    def _add_vectors(self, v1, v2):
        return [a + b for a, b in zip(v1, v2)]

    def _subtract_vectors(self, v1, v2):
        return [a - b for a, b in zip(v1, v2)]

    def evaluate_V_n(self, gamma_0, n, t):
        """Evaluates the homogeneous iterate (V^n \gamma_0)(t) exactly."""
        if n == 0:
            return gamma_0(t)
        
        result = [0.0] * self.p
        for j in range(self.num_matrices):
            arg = self.M * t - j
            prev_val = self.evaluate_V_n(gamma_0, n - 1, arg)
            term = self._matrix_vector_mult(self.A[j], prev_val)
            result = self._add_vectors(result, term)
                
        return result

    def _get_B(self, r):
        if self.B is None: return None
        if self._B_is_stage_dependent:
            return self.B[r] if isinstance(self.B, list) else self.B(r)
        return self.B

    def _get_anchor(self, r):
        if self.anchor_profile is None: return None
        if self._anchor_is_stage_dependent:
            return self.anchor_profile[r] if isinstance(self.anchor_profile, list) else self.anchor_profile(r)
        return self.anchor_profile

    def compute_forcing_sum(self, n, t):
        if self.B is None:
            return [0.0] * self.p
            
        result = [0.0] * self.p
        
        for r in range(n):
            current_B = self._get_B(r)
            power = n - 1 - r if self._B_is_stage_dependent else r
            term = self.evaluate_V_n(current_B, power, t)
            result = self._add_vectors(result, term)
            
        return result

    def compute_E(self, r, t):
        # If the builder already computed the compact defect, use it directly!
        if self.forcing_is_defect:
            current_E = self._get_B(r)
            return current_E(t) if current_E else [0.0] * self.p
            
        # Otherwise, compute standard E = V\Gamma + B - \Gamma
        current_B = self._get_B(r)
        current_anchor = self._get_anchor(r)
        next_anchor = self._get_anchor(r+1)
        
        if current_anchor is None:
            return [0.0] * self.p
            
        v_gamma = self.evaluate_V_n(current_anchor, 1, t)
        b_val = current_B(t) if current_B else [0.0] * self.p
        res = self._add_vectors(v_gamma, b_val)
        return self._subtract_vectors(res, next_anchor(t))

    def compute_defect_forcing_sum(self, n, t):
        result = [0.0] * self.p
        
        for r in range(n):
            power = n - 1 - r if self._B_is_stage_dependent else r
            
            def E_r(x, stage=r):
                return self.compute_E(stage, x)
                
            term = self.evaluate_V_n(E_r, power, t)
            result = self._add_vectors(result, term)
            
        return result

    def compute_full_iterate(self, gamma_0, n, t):
        initial_anchor = self._get_anchor(0)
        final_anchor = self._get_anchor(n)
        
        if self.anchor_profile is None:
            homogeneous_part = self.evaluate_V_n(gamma_0, n, t)
            forcing_part = self.compute_forcing_sum(n, t)
            return self._add_vectors(homogeneous_part, forcing_part)
        else:
            def eta_0(x):
                return self._subtract_vectors(gamma_0(x), initial_anchor(x))
            
            homog_defect = self.evaluate_V_n(eta_0, n, t)
            defect_forcing = self.compute_defect_forcing_sum(n, t)
            
            anchor_val = final_anchor(t)
            res = self._add_vectors(anchor_val, homog_defect)
            return self._add_vectors(res, defect_forcing)