from .base import CurveBuilder
from .cpwl import CPwLCurve
from ..algebra.system import AffineRefinementSystem

class ScalarRefinableBuilder(CurveBuilder):
    """
    Level 1: Handles scalar refinable functions f(x) = \sum c_j f(Mx - j) + B(x).
    Natively validates the exact Lemma 5.2 open curve condition using tail evaluations.
    """
    def __init__(self, mask_coefficients, M=2, forcing_curve=None, anchor_profile=None):
        self.c = mask_coefficients
        self.M = M
        self.p = 1
        self.forcing_curve = forcing_curve
        self.anchor_profile = anchor_profile
        
        N = len(mask_coefficients) - 1
        self.L = int(N / (self.M - 1))

        self._validate_lemma_5_2()

    def _validate_lemma_5_2(self):
        """Evaluates the tails at t << 0 and t >> 0 to check the anchor mismatch."""
        S = sum(self.c)
        
        # Evaluate far outside the expected support window [0, L]
        t_minus, t_plus = -1000.0, 1000.0 
        
        # B(t) evaluates to [val], extract the scalar. Default to 0.0 if B is None
        B_minus = self.forcing_curve(t_minus)[0] if self.forcing_curve else 0.0
        B_plus  = self.forcing_curve(t_plus)[0] if self.forcing_curve else 0.0
        
        # \Gamma(t) evaluates to [val], extract the scalar. Default to 0.0 if Gamma is None
        Gamma_minus = self.anchor_profile(t_minus)[0] if self.anchor_profile else 0.0
        Gamma_plus  = self.anchor_profile(t_plus)[0] if self.anchor_profile else 0.0
        
        # Exact Lemma 5.2 conditions: S*\Gamma_± + B_± = \Gamma_±
        left_match = abs(S * Gamma_minus + B_minus - Gamma_minus) < 1e-9
        right_match = abs(S * Gamma_plus + B_plus - Gamma_plus) < 1e-9
        
        if not (left_match and right_match):
            print(f"Warning: Lemma 5.2 anchor mismatch condition failed!")
            print(f"  Left tail  (t={t_minus}): S*{Gamma_minus:.4f} + {B_minus:.4f} != {Gamma_minus:.4f}")
            print(f"  Right tail (t={t_plus}): S*{Gamma_plus:.4f} + {B_plus:.4f} != {Gamma_plus:.4f}")
            print(f"  The geometric defect will not be compactly supported.")

    def _compute_matrices(self):
        return [[[c_j]] for c_j in self.c]

    def to_affine_system(self):
        A_matrices = self._compute_matrices()
        B_forcing = self.forcing_curve if self.forcing_curve is not None else CPwLCurve.zero(p=self.p)
        
        return AffineRefinementSystem(
            A_matrices=A_matrices,
            M=self.M,
            B_forcing=B_forcing,
            p=self.p,
            L=self.L,
            anchor_profile=self.anchor_profile
        )
