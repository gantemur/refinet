import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['agg.path.chunksize'] = 20000
sys.path.insert(0, os.path.abspath('src'))

from refinet.geometry.library import (
    create_dragon, create_koch, create_gosper, create_hilbert,
    create_peano, create_peano_homogeneous, create_morton_like,
    initial_unit_segment, initial_stacked_unit_segment
)
from refinet.cascade.exact import BlockCascadeEngine

def get_stationary_anchored_data(engine, gamma_0, extra_B=None):
    def D_curve(t):
        res = np.zeros(engine.p)
        for j in range(engine.num_A):
            arg = engine.M * t - j
            res += engine.A[j] @ np.array(gamma_0(arg))
        return (res - np.array(gamma_0(t))).tolist()
    B_curves = [D_curve] if extra_B is None else [D_curve, extra_B]
    return gamma_0, B_curves, [[] for _ in B_curves]

def render_curve(name, engine, base_curve, n_depth, B_curves=None, c_weights=None):
    num_points = (engine.M ** n_depth) + 1
    print(f"Rendering {name} (Depth {n_depth}, M={engine.M}, {num_points} exact vertices)...")
    start_time = time.time()
    
    t_vals = np.linspace(0, 1, num_points)
    x_vals, y_vals = [], []
    def zero_curve(t): return [0.0] * engine.p
    
    for t in t_vals:
        # 1. Native Non-Stationary Routing
        if hasattr(engine.system, '_anchor_is_stage_dependent') and engine.system._anchor_is_stage_dependent:
            pt = engine.evaluate_nonstationary(t, n_depth)
            
        # 2. Standard Affine Routing
        else:
            pt_cascade = engine.evaluate_affine(
                t, n_depth, zero_curve, 
                B_curves=B_curves or [], 
                c_weights_list=c_weights or []
            )
            pt = np.array(base_curve(t)) + np.array(pt_cascade)
            
        x_vals.append(pt[0])
        y_vals.append(pt[1])
        
    print(f"  -> Completed in {time.time() - start_time:.3f} seconds.")
    return x_vals, y_vals

def main():
    print("--- Single Curve Cascade Rendering ---")
    
    ACTIVE_CURVE = 'peano' 
    N_DEPTH = 4
    
    base_curve, B_curves, c_weights = None, None, None

    if ACTIVE_CURVE == 'hilbert':
        system, _ = create_hilbert()
        engine = BlockCascadeEngine(system, K_min=0, K_max=1)
        name, color = "Hilbert Curve", 'r-'
        # Look how clean this is! Everything is handled internally by the engine.

    elif ACTIVE_CURVE == 'peano':
        system, _ = create_peano()
        engine = BlockCascadeEngine(system, K_min=0, K_max=1)
        name, color = "Peano Curve", 'g-'

    elif ACTIVE_CURVE == 'peano_homog':
        system = create_peano_homogeneous()
        engine = BlockCascadeEngine(system, K_min=0, K_max=1)
        name, color = "Peano Curve (Homogeneous)", 'g-'
        def gamma_0(t):
            if t <= 0.0: return [0.0, 0.0]
            elif t >= 1.0: return [3.0, 0.0]
            else: return [3.0 * t, 0.0]
        base_curve, B_curves, c_weights = get_stationary_anchored_data(engine, gamma_0)

    elif ACTIVE_CURVE == 'gosper':
        system = create_gosper()
        engine = BlockCascadeEngine(system, K_min=0, K_max=1)
        name, color = "Gosper Curve", 'm-'
        base_curve, B_curves, c_weights = get_stationary_anchored_data(engine, initial_stacked_unit_segment(p=2, r=2))

    elif ACTIVE_CURVE == 'morton':
        system, gamma_0 = create_morton_like(scale=0.45)
        engine = BlockCascadeEngine(system, K_min=0, K_max=1)
        name, color = "Morton-like Affine", 'r-'
        base_curve, B_curves, c_weights = get_stationary_anchored_data(engine, gamma_0, extra_B=system.B)

    # -------------------------------------
    x_vals, y_vals = render_curve(name, engine, base_curve, N_DEPTH, B_curves=B_curves, c_weights=c_weights)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_vals, y_vals, color, linewidth=max(0.05, 1.5 / (N_DEPTH + 1)))
    ax.set_title(f"{name} ($n={N_DEPTH}$)")
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    output_filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_n{N_DEPTH}.png"
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"Success! Plot saved to {output_filename}\n")

if __name__ == "__main__":
    main()