from .base import CurveBuilder
from .cpwl import CPwLCurve
from ..algebra.system import AffineRefinementSystem

class HomogeneousBuilder(CurveBuilder):
    """
    Level 1: Builds a homogeneous affine system from a polygonal chain.
    Enforces strict C0 continuity by natively computing the longitudinal axis from the vertices.
    Users only supply the transversal mappings (the action on e^\perp).
    """
    def __init__(self, vertices, transversals=None, orientations=None):
        """
        vertices: List of P_0, ..., P_M in R^p.
        transversals: Optional list of p x (p-1) matrices defining the orthogonal freedom.
        orientations: Optional list of 'preserving' or 'reversing' for automatic 2D transversals.
        """
        self.p = len(vertices[0])
        self.M = len(vertices) - 1

        # --- NORMALIZATION BLOCK ---
        P0 = vertices[0]
        PM = vertices[-1]
        
        if self.p == 2:
            # 2D similarity transform: translation + rotation + scaling
            # Mathematically equivalent to complex division: (z - P0) / (PM - P0)
            Vx = PM[0] - P0[0]
            Vy = PM[1] - P0[1]
            V_sq = Vx**2 + Vy**2
            
            self.P = []
            for pt in vertices:
                dx = pt[0] - P0[0]
                dy = pt[1] - P0[1]
                
                nx = (dx * Vx + dy * Vy) / V_sq
                ny = (dy * Vx - dx * Vy) / V_sq
                self.P.append([nx, ny])
        else:
            # For p > 2, aligning an arbitrary vector to e_1 requires Householder 
            # reflections. Left as raw input, assuming user pre-normalized if p > 2.
            self.P = vertices
        # ------------------------------------
        
        self.A = []
        if orientations is None and self.p == 2:
            orientations = [1] * self.M

        for j in range(self.M):
            # 1. The longitudinal part (Column 1) is strictly fixed by the geometry
            longitudinal = [self.P[j+1][d] - self.P[j][d] for d in range(self.p)]
            
            # 2. Determine the transversal part (Columns 2 through p)
            if self.p == 2 and transversals is None:
                # In 2D, the transversal space is 1D. We auto-generate it via similarities.
                dx, dy = longitudinal[0], longitudinal[1]
                # Backward-compatible check supporting both integers and strings
                if orientations[j] in ('preserving', 1):
                    trans_j = [[-dy], [dx]]
                elif orientations[j] in ('reversing', -1):
                    trans_j = [[dy], [-dx]]
                else:
                    raise ValueError(f"Invalid orientation '{orientations[j]}' at block {j}. Use 1, -1, 'preserving', or 'reversing'.")
            else:
                if transversals is None or len(transversals) <= j:
                    raise ValueError(f"For p={self.p}, a p x (p-1) transversal matrix is required for block {j}.")
                trans_j = transversals[j]

            # 3. Assemble the full p x p matrix: [Longitudinal | Transversal]
            A_j = []
            for row in range(self.p):
                # Combine the 1D longitudinal entry with the (p-1)D transversal row
                A_j.append([longitudinal[row]] + list(trans_j[row]))
                
            self.A.append(A_j)

    def to_affine_system(self):
        B_forcing = CPwLCurve.zero(p=self.p)
        
        P_0 = self.P[0]
        P_M = self.P[-1]
        
        def anchor(t):
            if t <= 0.0: return P_0
            elif t >= 1.0: return P_M
            else:
                return [P_0[d] + t * (P_M[d] - P_0[d]) for d in range(self.p)]

        return AffineRefinementSystem(
            A_matrices=self.A,
            M=self.M,
            B_forcing=B_forcing,
            p=self.p,
            L=1,
            anchor_profile=anchor
        )
