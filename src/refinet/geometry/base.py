class CurveBuilder:
    """
    Level 1: Abstract base class for all geometric curve builders.
    """
    def to_affine_system(self):
        """
        Compiles the geometric specification down into the standard 
        algebraic matrices (A_j) and forcing terms (B_n).
        
        Returns:
            AffineRefinementSystem
        """
        raise NotImplementedError("Subclasses must implement to_affine_system.")
