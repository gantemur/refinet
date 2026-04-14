import math
from .cpwl import CPwLCurve
from .refinable import ScalarRefinableBuilder
from .homogeneous import HomogeneousBuilder
from .finite_state import FiniteStateAffineBuilder
from .stationary import StationaryAffineBuilder
from .nonstationary import NonstationaryAffineBuilder

#==================================================================================================
#   Initial Curves
#==================================================================================================

def initial_box_function(p=1):
    """
    The standard initial state for refinable functions is often 
    the characteristic function of [0, 1).
    """
    def gamma_0(t):
        if 0.0 <= t < 1.0:
            return [1.0] * p
        else:
            return [0.0] * p
    return gamma_0

def initial_unit_segment(p=2):
    """
    Returns the standard endpoint-extended initial curve \gamma_0(t).
    It travels from the origin to the first standard basis vector e_1.
    """
    def gamma_0(t):
        if t <= 0.0:
            return [0.0] * p
        elif t >= 1.0:
            res = [0.0] * p
            res[0] = 1.0
            return res
        else:
            res = [0.0] * p
            res[0] = t
            return res
            
    return gamma_0

def initial_stacked_unit_segment(p=2, r=2):
    """
    Returns the standard endpoint-extended initial curve for a stacked finite-state system.
    Replicates the p-dimensional segment across r states.
    """
    def gamma_0(t):
        if t <= 0.0:
            return [0.0] * (p * r)
        elif t >= 1.0:
            res = [0.0] * (p * r)
            for i in range(r):
                res[i * p] = 1.0
            return res
        else:
            res = [0.0] * (p * r)
            for i in range(r):
                res[i * p] = t
            return res
            
    return gamma_0

#==================================================================================================
#   Scalar Reinable Functions
#==================================================================================================

def create_bspline(degree=3):
    """
    Returns a Level 2 system for B-splines of a given degree.
    degree=0: Haar scaling function (box)
    degree=1: Linear hat function
    degree=2: Quadratic B-spline
    degree=3: Cubic B-spline
    """
    # The mask coefficients for B-splines are binomial coefficients / 2^(d-1)
    if degree == 0:
        mask = [1.0, 1.0]
    elif degree == 1:
        mask = [0.5, 1.0, 0.5]
    elif degree == 2:
        mask = [0.25, 0.75, 0.75, 0.25]
    elif degree == 3:
        mask = [0.125, 0.5, 0.75, 0.5, 0.125]
    else:
        raise ValueError("Degree not supported in standard library yet.")
        
    builder = ScalarRefinableBuilder(mask_coefficients=mask, M=2)
    return builder.to_affine_system()

def create_daubechies_d4():
    """Returns a Level 2 system for the Daubechies D4 scaling function."""
    sqrt3 = math.sqrt(3)
    # Standard D4 scaling mask
    mask = [
        (1 + sqrt3) / 4,
        (3 + sqrt3) / 4,
        (3 - sqrt3) / 4,
        (1 - sqrt3) / 4
    ]
    builder = ScalarRefinableBuilder(mask_coefficients=mask, M=2)
    return builder.to_affine_system()

def create_knopp(a=0.5):
    """
    Returns a Level 2 system for the Knopp function.
    For a=0.5, this is exactly the Takagi (Blancmange) curve.
    """
    mask = [a, a]
    
    # Direct CPwL triangle wave s(t) on [0, 1]
    triangle_breakpoints = [(0.0, [0.0]), (0.5, [0.5]), (1.0, [0.0])]
    B_forcing = CPwLCurve(breakpoints=triangle_breakpoints, p=1)
    
    builder = ScalarRefinableBuilder(mask_coefficients=mask, M=2, forcing_curve=B_forcing)
    return builder.to_affine_system()

def create_takagi():
    """Returns a Level 2 system for the Takagi (Blancmange) curve."""
    return create_knopp(a=0.5)

#==================================================================================================
#   Homogenous Polygonal Curves
#==================================================================================================

def create_koch():
    """Returns a Level 2 system for the standard Koch curve."""
    sqrt3 = math.sqrt(3)
    vertices = [(0.0, 0.0), (1/3, 0.0), (1/2, sqrt3/6), (2/3, 0.0), (1.0, 0.0)]
    builder = HomogeneousBuilder(vertices)
    return builder.to_affine_system()

def create_dragon(dragon_type='heighway'):
    """Returns a Level 2 system for the Heighway or Lévy-C Dragon."""
    vertices = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]
    orientations = [1, -1] if dragon_type == 'heighway' else [1, 1]
    builder = HomogeneousBuilder(vertices, orientations=orientations)
    return builder.to_affine_system()

def create_hilbert_homogeneous():
    """
    Returns a Level 2 system for the 4-piece self-intersecting Hilbert-type recursion.
    Constructed directly from Example 6.4 of the paper (scaled by 2).
    """
    vertices = [(0,0),(0,1),(1,1),(2,1),(2,0)]
    # A_0 is a reflection across y=x (reversing), A_1 and A_2 are identity scales (preserving), A_3 is a reflection across y=-x (reversing)
    orientations = [-1, 1, 1, -1]    
    builder = HomogeneousBuilder(vertices, orientations=orientations)
    return builder.to_affine_system()

def create_peano_homogeneous():
    """
    Returns a Level 2 system for the classic Peano curve.
    Extracted directly from the 9-segment Level 1 generator.
    """
    vertices = [(0,0),(1,0),(1,1),(2,1),(2,0),(2,-1),(1,-1),(1,0),(2,0),(3,0)]
    orientations = [1, -1, 1, -1, 1, -1, 1, -1, 1]
    builder = HomogeneousBuilder(vertices, orientations=orientations)
    return builder.to_affine_system()

#==================================================================================================
#   Finite State Systems
#==================================================================================================

def create_gosper():
    """Returns a Level 2 stacked system (dim 4) for the Gosper curve."""
    sqrt3 = math.sqrt(3)
    
    unscaled_A = [0j, 1+0j, 1.5-(sqrt3/2)*1j, 0.5-(sqrt3/2)*1j, 0.0-sqrt3*1j, 1.0-sqrt3*1j, 2.0-sqrt3*1j, 2.5-(sqrt3/2)*1j]
    unscaled_B = [0j, 0.5+(sqrt3/2)*1j, 1.5+(sqrt3/2)*1j, 2.5+(sqrt3/2)*1j, 2.0+0j, 1.0+0j, 1.5-(sqrt3/2)*1j, 2.5-(sqrt3/2)*1j]

    Z = unscaled_A[-1] 
    
    verts_A = [((v/Z).real, (v/Z).imag) for v in unscaled_A]
    verts_B = [((v/Z).real, (v/Z).imag) for v in unscaled_B]

    states_data = {
        0: {'vertices': verts_A, 'transitions': [0, 1, 1, 0, 0, 0, 1]},
        1: {'vertices': verts_B, 'transitions': [0, 1, 1, 1, 0, 0, 1]}
    }
    
    builder = FiniteStateAffineBuilder(states_data, p=2)
    return builder.to_affine_system()

def create_sierpinski_arrowhead():
    """Returns a Level 2 stacked system (dim 4) for the Sierpinski Arrowhead curve."""
    sqrt3 = math.sqrt(3)
    
    # State 0 (Rule A): Spawns B, A, B. 
    # Turns right (-60 deg), then left (+60 deg back to 0), then left (+60 deg)
    unscaled_A = [0j, 0.5-(sqrt3/2)*1j, 1.5-(sqrt3/2)*1j, 2.0+0j]
    # State 1 (Rule B): Spawns A, B, A. 
    # Turns left (+60 deg), then right (-60 deg back to 0), then right (-60 deg)
    unscaled_B = [0j, 0.5+(sqrt3/2)*1j, 1.5+(sqrt3/2)*1j, 2.0+0j]
    
    Z = unscaled_A[-1] # Z = 2.0
    
    verts_A = [((v/Z).real, (v/Z).imag) for v in unscaled_A]
    verts_B = [((v/Z).real, (v/Z).imag) for v in unscaled_B]

    states_data = {
        0: {'vertices': verts_A, 'transitions': [1, 0, 1]},
        1: {'vertices': verts_B, 'transitions': [0, 1, 0]}
    }
    
    builder = FiniteStateAffineBuilder(states_data, p=2)
    return builder.to_affine_system()

#==================================================================================================
#   Stationary Affine Systems
#==================================================================================================

def create_square_bridge():
    """
    Returns a Level 2 system using the Stationary Affine extension.
    It consists of two scaled copies joined by a square pulse connector.
    """
    p = 2
    
    # 1. Two copies scaled by 1/3
    A_0 = [[1/3, 0.0], [0.0, 1/3]]
    A_1 = [[1/3, 0.0], [0.0, 1/3]]
    matrices = [A_0, A_1]
    
    # 2. Prescribe the entry ports
    entry_ports = [
        [0.0, 0.0],      # P_0
        [2/3, 0.0]       # P_1
    ]
    
    # 3. Create the connector joining Q_0 to P_1
    # Q_0 = P_0 + A_0 e_1 = (1/3, 0)
    # The connector jumps up by 1/3, across by 1/3, and down by 1/3.
    zeta_breakpoints = [
        (0.0, [1/3, 0.0]),
        (0.2, [1/3, 1/3]),
        (0.8, [2/3, 1/3]),
        (1.0, [2/3, 0.0])
    ]
    zeta_0 = CPwLCurve(breakpoints=zeta_breakpoints, p=2)
    connectors = [zeta_0]

    global_start = [0.0, 0.0]
    global_end = [1.0, 0.0]
    
    builder = StationaryAffineBuilder(
        translation_vectors=entry_ports, 
        matrices=matrices, 
        connectors=connectors, 
        global_start=global_start, # Pass the start point
        global_end=global_end,     # Pass the end point
        p=p
    )
    return builder.to_affine_system()

from .stationary import StationaryAffineBuilder
from .cpwl import CPwLCurve

def create_morton_like(scale=0.4):
    """
    Returns a Level 2 system for a stationary Morton-like (Z-order) curve,
    along with a Z-shaped initial curve gamma_0 to ensure proper base traversal.
    """
    s = scale
    A = [[s, 0.0], [0.0, s]]
    matrices = [A, A, A, A]
    
    C = [
        (0.25, 0.75),  # 0: Top-Left
        (0.75, 0.75),  # 1: Top-Right
        (0.25, 0.25),  # 2: Bottom-Left
        (0.75, 0.25)   # 3: Bottom-Right
    ]
    
    P = [[c[0] - s/2, c[1] - s/2] for c in C]
    
    # TRUE Global Fixed Points: E = (I-A)^(-1) P
    E_0 = [P[0][0] / (1 - s), P[0][1] / (1 - s)]
    E_1 = [P[3][0] / (1 - s), P[3][1] / (1 - s)]
    
    # Calculate the fixed points for the other two corners for our base curve
    E_TR = [P[1][0] / (1 - s), P[1][1] / (1 - s)]
    E_BL = [P[2][0] / (1 - s), P[2][1] / (1 - s)]
    
    # The true Z-shape initial curve tracing all 4 corners
    gamma_0 = CPwLCurve(breakpoints=[
        (0.0, E_0), 
        (1/3, E_TR), 
        (2/3, E_BL), 
        (1.0, E_1)
    ], p=2)
    
    Q = [[s * E_1[0] + P[j][0], s * E_1[1] + P[j][1]] for j in range(4)]
    R = [[s * E_0[0] + P[j][0], s * E_0[1] + P[j][1]] for j in range(4)]
    
    zeta_0 = CPwLCurve(breakpoints=[(0.0, Q[0]), (1.0, R[1])], p=2)
    zeta_1 = CPwLCurve(breakpoints=[(0.0, Q[1]), (1.0, R[2])], p=2)
    zeta_2 = CPwLCurve(breakpoints=[(0.0, Q[2]), (1.0, R[3])], p=2)
    
    builder = StationaryAffineBuilder(
        translation_vectors=P, 
        matrices=matrices, 
        connectors=[zeta_0, zeta_1, zeta_2], 
        global_start=E_0,
        global_end=E_1, 
        p=2
    )
    
    # Return both the system AND the Z-shaped base curve
    return builder.to_affine_system(), gamma_0

def create_hilbert_like(scale=0.4, height=1.0):
    """
    Returns a Level 2 system for a stationary Hilbert-like curve.
    Uses anisotropic reflection to perfectly tile any arbitrary 1 x h bounding box.
    """
    s = scale
    h = height
    
    # Anisotropic reflections to preserve the 1:h block aspect ratio!
    A_0 = [[0.0, s/h], [s*h, 0.0]]
    A_1 = [[s, 0.0], [0.0, s]]
    A_2 = [[s, 0.0], [0.0, s]]
    A_3 = [[0.0, -s/h], [-s*h, 0.0]]
    
    matrices = [A_0, A_1, A_2, A_3]
    
    E_0 = [0.0, 0.0]
    E_1 = [1.0, 0.0]
    
    # Bottom-left anchors for the local blocks
    P = [
        [0.0, 0.0],           # 0: Bottom-Left
        [0.0, h * (1 - s)],   # 1: Top-Left
        [1.0 - s, h * (1 - s)],# 2: Top-Right
        [1.0, s * h]          # 3: Bottom-Right (Scaled strictly by h)
    ]
    
    # Helper functions to compute exact Entry (R) and Exit (Q) ports dynamically
    def matvec(M, v): return [M[0][0]*v[0] + M[0][1]*v[1], M[1][0]*v[0] + M[1][1]*v[1]]
    def add(v, w): return [v[0]+w[0], v[1]+w[1]]
    
    Q = [add(P[i], matvec(matrices[i], E_1)) for i in range(4)]
    R = [add(P[i], matvec(matrices[i], E_0)) for i in range(4)]
    
    # The connectors perfectly bridge the dynamically calculated ports
    zeta_0 = CPwLCurve(breakpoints=[(0.0, Q[0]), (1.0, R[1])], p=2)
    zeta_1 = CPwLCurve(breakpoints=[(0.0, Q[1]), (1.0, R[2])], p=2)
    zeta_2 = CPwLCurve(breakpoints=[(0.0, Q[2]), (1.0, R[3])], p=2)
    
    # A perfect, rigid square staple for the base curve up to y=h
    U_TL = [0.0, h]
    U_TR = [1.0, h]
    
    gamma_0 = CPwLCurve(breakpoints=[
        (0.0, E_0), 
        (1/3, U_TL), 
        (2/3, U_TR), 
        (1.0, E_1)
    ], p=2)
    
    builder = StationaryAffineBuilder(
        translation_vectors=P, 
        matrices=matrices, 
        connectors=[zeta_0, zeta_1, zeta_2], 
        global_start=E_0,
        global_end=E_1, 
        p=2
    )
    
    return builder.to_affine_system(), gamma_0

#==================================================================================================
#   Non-Stationary Affine Systems
#==================================================================================================

def create_hilbert():
    """
    Returns a Level 2 system for the Stage-Dependent Hilbert Curve (Example 6.11).
    Demonstrates the infinite finite-span mechanism where gaps shrink by 2^{-n}.
    """
    # The 4 Hilbert matrices (0 and 3 are reflections)
    A_0 = [[0.0, 0.5], [0.5, 0.0]]
    A_1 = [[0.5, 0.0], [0.0, 0.5]]
    A_2 = [[0.5, 0.0], [0.0, 0.5]]
    A_3 = [[0.0, -0.5], [-0.5, 0.0]]
    matrices = [A_0, A_1, A_2, A_3]
    
    # Translations u_j
    translations = [
        [0.0, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [1.0, 0.5]
    ]
    
    # Anchor Profile Bases
    def Gamma_0(t):
        return [t, 0.0]
        
    def Gamma_1(t):
        return [0.5 - t, 0.5]
        
    Gamma_bases = [Gamma_0, Gamma_1]
    
    # Coefficients lambda(n) = [1.0, 2^{-n}]
    def Gamma_coeffs(n):
        return [1.0, 2**(-n)]
    
    builder = NonstationaryAffineBuilder(
        matrices, 
        translations, 
        Gamma_bases, 
        Gamma_coeffs, 
        p=2
    )
    return builder.to_affine_system(), builder.Gamma_n

def create_morton():
    """
    Returns a Level 2 system for the Stage-Dependent Morton Curve.
    Initial curve is a single dot at (0.5, 0.5), which dynamically 
    splits and expands to fill the dyadic squares in Z-order.
    """
    # Pure scalings for Morton
    A = [[0.5, 0.0], [0.0, 0.5]]
    matrices = [A, A, A, A]
    
    # Z-order translations: TL, TR, BL, BR
    translations = [
        [0.0, 0.5],
        [0.5, 0.5],
        [0.0, 0.0],
        [0.5, 0.0]
    ]
    
    # Basis 0: The asymptotic limit profile (0,1) to (1,0)
    def Gamma_0(t):
        return [t, 1.0 - t]
        
    # Basis 1: The vanishing defect that collapses the line to a dot
    def Gamma_1(t):
        return [0.5 - t, t - 0.5]
        
    Gamma_bases = [Gamma_0, Gamma_1]
    
    # Coefficients shrink by 2^{-n} at each stage
    def Gamma_coeffs(n):
        return [1.0, 2**(-n)]
        
    builder = NonstationaryAffineBuilder(
        matrices=matrices, 
        translations=translations, 
        Gamma_bases=Gamma_bases, 
        Gamma_coeffs=Gamma_coeffs, 
        p=2
    )
    
    return builder.to_affine_system(), builder.Gamma_n

def create_peano():
    """
    Returns a Level 2 system for the Stage-Dependent Peano Curve.
    Starts from a single dot at (0.5, 0.5) and expands on a 3x3 grid.
    """
    s = 1.0 / 3.0
    A = [[s, 0.0], [0.0, s]]
    B = [[-s, 0.0], [0.0, s]]
    C = [[s, 0.0], [0.0, -s]]
    D = [[-s, 0.0], [0.0, -s]]
    matrices = [A, B, A, C, D, C, A, B, A]
    translations = [[0.0,0.0], [s,s], [0.0,s*2], [s,s*3], [s*2, s*2], [s,s], [s*2,0.0], [s*3,s], [s*2, s*2]]
    
    # Basis 0: The asymptotic limit profile (0,0) to (1,1)
    def Gamma_0(t):
        return [t, t]
        
    # Basis 1: The vanishing defect that collapses the diagonal to a central dot
    def Gamma_1(t):
        return [0.5 - t, 0.5 - t]
        
    Gamma_bases = [Gamma_0, Gamma_1]
    
    # Coefficients shrink by 3^{-n} because the Peano grid scales by 1/3
    def Gamma_coeffs(n):
        return [1.0, 3**(-n)]
        
    builder = NonstationaryAffineBuilder(
        matrices=matrices, 
        translations=translations, 
        Gamma_bases=Gamma_bases, 
        Gamma_coeffs=Gamma_coeffs, 
        p=2
    )
    
    return builder.to_affine_system(), builder.Gamma_n