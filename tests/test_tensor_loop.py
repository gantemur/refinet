import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from refinet.network.tensor import TensorCompiler

def exact_E(t):
    t = t % 1.0
    if t < 1/3: return np.array([3*t, 3*t])
    elif t < 2/3: return np.array([1.0, 2.0 - 3*t])
    else: return np.array([3.0 - 3*t, 0.0])

def run_compiled_net(layers, x):
    out = x
    for layer in layers:
        out = layer['weight'] @ out + layer['bias']
        if layer['activation'] == 'relu':
            out = np.maximum(0, out)
    return out

def main():
    print("--- Testing Tensor Compiler: Exact Loop Controller ---")
    M = 2
    
    layers = TensorCompiler.build_loop_controller(M)
    print(f"Compiled 2 layers for M={M}.")
    
    np.random.seed(42)
    t_vals = np.random.uniform(0, 1, 1000)
    max_error = 0.0
    
    for t in t_vals:
        z_n = exact_E(t)
        
        # The expected next state is exactly E(R(t))
        expected_z_next = exact_E((M * t) % 1.0)
        
        # Pass the raw coordinate through our compiled ReLU matrices
        actual_z_next = run_compiled_net(layers, z_n)
        
        error = np.max(np.abs(expected_z_next - actual_z_next))
        max_error = max(max_error, error)
        
    print(f"\nMax deviation across 1000 random loop states: {max_error:.4e}")
    if max_error < 1e-10:
        print("Success! F(z) exact topology correctly flattened to affine matrices!")
    else:
        print("Failure! Matrix arithmetic did not maintain loop topology.")

if __name__ == "__main__":
    main()