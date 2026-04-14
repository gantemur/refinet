from .base import CurveBuilder
from .cpwl import CPwLCurve
from ..algebra.system import AffineRefinementSystem

class StationaryAffineBuilder(CurveBuilder):
    """
    Level 1: Builds a stationary affine system with prescribed ports and connectors.
    Natively computes the compact defect E(t) to bypass global tail spillover.
    """
    def __init__(self, translation_vectors, matrices, connectors, global_start, global_end, p=2):
        self.P = translation_vectors 
        self.A = matrices
        self.connectors = connectors
        self.l = len(matrices)
        self.M = 2 * self.l - 1
        self.p = p
        self.E_0 = global_start
        self.E_1 = global_end

        # Precompute interlaced matrices (copies and zero-matrices)
        self.A_tilde = []
        zero_matrix = [[0.0] * self.p for _ in range(self.p)]
        for j in range(self.l):
            self.A_tilde.append(self.A[j])
            if j < self.l - 1:
                self.A_tilde.append(zero_matrix)
            
    def _anchor(self, t):
        if t <= 0.0: return self.E_0
        elif t >= 1.0: return self.E_1
        else:
            return [self.E_0[d] + t * (self.E_1[d] - self.E_0[d]) for d in range(self.p)]

    def _compute_defect_forcing(self):
        """Constructs E(t) directly from the isolated piecewise geometry."""
        breakpoints = []
        
        for k in range(self.M):
            t_start = k / self.M
            t_end = (k + 1) / self.M

            if k % 2 == 0:
                j = k // 2
                # Target(t) = P_j + A_j \Gamma(Mt-2j). Since \Gamma is linear, we just evaluate endpoints.
                val_start = self._add_vectors(self.P[j], self._matvec(self.A[j], self.E_0))
                val_start = self._subtract_vectors(val_start, self._anchor(t_start))
                
                val_end = self._add_vectors(self.P[j], self._matvec(self.A[j], self.E_1))
                val_end = self._subtract_vectors(val_end, self._anchor(t_end))
                
                if len(breakpoints) == 0 or abs(t_start - breakpoints[-1][0]) > 1e-9:
                    breakpoints.append((t_start, val_start))
                else:
                    breakpoints[-1] = (t_start, val_start)
                breakpoints.append((t_end, val_end))
            else:
                j = k // 2
                zeta = self.connectors[j]
                # Target(t) = \zeta(Mt - (2j+1))
                for s, v in zeta.breakpoints:
                    t = t_start + s / self.M
                    defect_v = self._subtract_vectors(v, self._anchor(t))
                    
                    if len(breakpoints) == 0 or abs(t - breakpoints[-1][0]) > 1e-9:
                        breakpoints.append((t, defect_v))
                    else:
                        breakpoints[-1] = (t, defect_v)

        return CPwLCurve(breakpoints=breakpoints, p=self.p)

    def to_affine_system(self):
        # Pass the pre-computed compact defect directly to the engine
        E_forcing = self._compute_defect_forcing()

        return AffineRefinementSystem(
            A_matrices=self.A_tilde,
            M=self.M,
            B_forcing=E_forcing, 
            p=self.p,
            L=1,
            anchor_profile=self._anchor,
            forcing_is_defect=True  # FLAG: Do not add V\Gamma to this curve!
        )

    def _matvec(self, matrix, vector):
        result = [0.0] * self.p
        for row in range(self.p):
            for col in range(self.p):
                result[row] += matrix[row][col] * vector[col]
        return result

    def _add_vectors(self, v1, v2):
        return [a + b for a, b in zip(v1, v2)]

    def _subtract_vectors(self, v1, v2):
        return [a - b for a, b in zip(v1, v2)]