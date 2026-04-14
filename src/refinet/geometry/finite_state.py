from .base import CurveBuilder
from .cpwl import CPwLCurve
from ..algebra.system import AffineRefinementSystem

class FiniteStateAffineBuilder(CurveBuilder):
    """
    Level 1: Implements Theorem 5.7 for Finite-State Refinement.
    Stacks r states of dimension p into a single affine system of dimension p*r.
    """
    def __init__(self, states_data, p=2, B_forcing=None, anchor_profile=None):
        """
        states_data: A dictionary mapping state_index -> state_info.
        Format:
        {
            0: {'vertices': [(x,y), ...], 'transitions': [0, 1, 1, ...]},
            1: {'vertices': [(x,y), ...], 'transitions': [0, 1, 0, ...]}
        }
        B_forcing: Optional stacked CPwLCurve of dimension p*r.
        anchor_profile: Optional stacked callable \Gamma(t) of dimension p*r.
        """
        self.states_data = states_data
        self.r = len(states_data) # Number of states
        self.p = p
        self.B_forcing = B_forcing
        self.anchor_profile = anchor_profile
        
        # Assume all states have the same number of segments (M)
        self.M = len(states_data[0]['vertices']) - 1 

    def _compute_stacked_matrices(self):
        stacked_matrices = []
        for j in range(self.M):
            # Initialize a (pr x pr) matrix with zeros
            mat = [[0.0] * (self.p * self.r) for _ in range(self.p * self.r)]
            
            for state_idx, data in self.states_data.items():
                verts = data['vertices']
                targets = data['transitions']
                
                target_state = targets[j]
                
                # The segment vector E = P_{j+1} - P_j
                dx = verts[j+1][0] - verts[j][0]
                dy = verts[j+1][1] - verts[j][1]
                
                # Calculate block offsets for the stacked state vector
                row_offset = state_idx * self.p
                col_offset = target_state * self.p
                
                # Insert the orientation-preserving similarity block
                # (Can be upgraded later to accept orientation arrays like HomogeneousBuilder)
                mat[row_offset][col_offset] = dx
                mat[row_offset][col_offset+1] = -dy
                mat[row_offset+1][col_offset] = dy
                mat[row_offset+1][col_offset+1] = dx
                
            stacked_matrices.append(mat)
        return stacked_matrices

    def to_affine_system(self):
        """Compiles the stacked system into Level 2."""
        A_matrices = self._compute_stacked_matrices()
        
        # Default to exactly zero if no stacked affine forcing is provided
        B = self.B_forcing if self.B_forcing is not None else CPwLCurve.zero(p=self.p * self.r)
        
        return AffineRefinementSystem(
            A_matrices=A_matrices,
            M=self.M,
            B_forcing=B,
            p=self.p * self.r,
            L=1,
            anchor_profile=self.anchor_profile
        )