from .base import CurveBuilder
from .cpwl import CPwLCurve
from ..algebra.system import AffineRefinementSystem

class NonstationaryAffineBuilder(CurveBuilder):
    """
    Level 1: Handles nonstationary (stage-dependent) affine refinement.
    Computes dynamic anchor profiles and compact defect sequences E_n(t).
    """
    def __init__(self, matrices, translations, Gamma_bases, Gamma_coeffs, p=2):
        self.A = matrices
        self.u = translations
        self.l = len(matrices)
        self.M = 2 * self.l - 1
        self.p = p
        
        self.Gamma_bases = Gamma_bases
        self.Gamma_coeffs = Gamma_coeffs # Function: n -> [lambda_0, lambda_1, ...]
        
        self.A_tilde = []
        zero_mat = [[0.0] * p for _ in range(p)]
        for j in range(self.l):
            self.A_tilde.append(self.A[j])
            if j < self.l - 1:
                self.A_tilde.append(zero_mat)

    def Gamma_n(self, n):
        """Dynamically evaluates the finite span: \sum \lambda_{n,\alpha} \Gamma^{(\alpha)}"""
        coeffs = self.Gamma_coeffs(n)
        def anchor(t):
            res = [0.0] * self.p
            for alpha, c in enumerate(coeffs):
                val = self.Gamma_bases[alpha](t)
                res = [res[d] + c * val[d] for d in range(self.p)]
            return res
        return anchor

    def B_n(self, n):
        """Generates the compact defect CPwLCurve E_n(t) for stage n."""
        Gamma_curr = self.Gamma_n(n)
        Gamma_next = self.Gamma_n(n+1)
        
        a_n = Gamma_curr(0.0)
        b_n = Gamma_curr(1.0)
        
        P = [self._add(self._matvec(self.A[j], a_n), self.u[j]) for j in range(self.l)]
        Q = [self._add(self._matvec(self.A[j], b_n), self.u[j]) for j in range(self.l)]
        
        breakpoints = []
        for j in range(self.l):
            t_start = 2 * j / self.M
            t_end = (2 * j + 1) / self.M
            
            # Since P[j] and Q[j] are absolute targets, subtracting Gamma_next 
            # computes the defect directly!
            val_start = self._sub(P[j], Gamma_next(t_start))
            val_end = self._sub(Q[j], Gamma_next(t_end))
            
            if not breakpoints or abs(breakpoints[-1][0] - t_start) > 1e-9:
                breakpoints.append((t_start, val_start))
            else:
                breakpoints[-1] = (t_start, val_start)
                
            breakpoints.append((t_end, val_end))
            
            if j < self.l - 1:
                t_conn_end = (2 * j + 2) / self.M
                val_conn_end = self._sub(P[j+1], Gamma_next(t_conn_end))
                breakpoints.append((t_conn_end, val_conn_end))
                
        return CPwLCurve(breakpoints, p=self.p)

    def to_affine_system(self):
        system = AffineRefinementSystem(
            A_matrices=self.A_tilde,
            M=self.M,
            B_forcing=self.B_n, 
            p=self.p,
            L=1,
            anchor_profile=self.Gamma_n,
            forcing_is_defect=True
        )
        
        # Cleanly export the raw data needed by evaluate_nonstationary
        system.Gamma_bases = self.Gamma_bases
        system.Gamma_coeffs = self.Gamma_coeffs
        system.translations = self.u 
        
        return system

    def _matvec(self, matrix, vector):
        result = [0.0] * self.p
        for row in range(self.p):
            for col in range(self.p):
                result[row] += matrix[row][col] * vector[col]
        return result

    def _add(self, v1, v2): return [a + b for a, b in zip(v1, v2)]
    def _sub(self, v1, v2): return [a - b for a, b in zip(v1, v2)]