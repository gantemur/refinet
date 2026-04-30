import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from refinet.network.tensor import TensorCompiler

def exact_hat(t, rho):
    """The exact mathematical definition of the special hat h(t)."""
    slope = 1.0 / (0.5 - rho)
    return slope * (np.maximum(0, t - rho) - 2 * np.maximum(0, t - 0.5) + np.maximum(0, t - (1 - rho)))

def exact_min(u, v):
    """The exact mathematical definition of the min gadget."""
    return u - np.maximum(0, u - v)

def run_compiled_net(layers, x):
    """A minimal numpy runtime to evaluate the compiled tensors."""
    out = x
    for layer in layers:
        W = layer['weight']
        b = layer['bias']
        # Compute W * x + b (Standard column-vector math)
        out = W @ out + b
        
        if layer['activation'] == 'relu':
            out = np.maximum(0, out)
    return out

def main():
    print("--- Testing Tensor Compiler: Terminal Readouts ---")
    
    np.random.seed(42)
    num_tests = 1000
    
    # ---------------------------------------------------------
    # 1. Test the Special Hat h(t)
    # ---------------------------------------------------------
    rho = 0.15
    hat_layers = TensorCompiler.build_special_hat(rho)
    print(f"\nCompiled {len(hat_layers)} layers for Special Hat (rho={rho}).")
    
    t_vals = np.random.uniform(0, 1, num_tests)
    max_error_hat = 0.0
    
    for t in t_vals:
        x_in = np.array([t])
        expected = exact_hat(t, rho)
        # The network outputs a 1D array of size 1, so we extract the scalar
        actual = run_compiled_net(hat_layers, x_in)[0] 
        
        error = abs(expected - actual)
        max_error_hat = max(max_error_hat, error)
        
    print(f"Max deviation for Special Hat across {num_tests} trials: {max_error_hat:.4e}")
    if max_error_hat < 1e-10:
        print(" -> Success! Special Hat topology flattened perfectly.")
    else:
        print(" -> Failure in Special Hat compilation.")

    # ---------------------------------------------------------
    # 2. Test the Min Gadget min(u, v)
    # ---------------------------------------------------------
    min_layers = TensorCompiler.build_min_gadget()
    print(f"\nCompiled {len(min_layers)} layers for Min Gadget.")
    
    # Generate random pairs of (u, v)
    u_vals = np.random.uniform(-5, 5, num_tests)
    v_vals = np.random.uniform(-5, 5, num_tests)
    max_error_min = 0.0
    
    for u, v in zip(u_vals, v_vals):
        x_in = np.array([u, v])
        expected = exact_min(u, v)
        actual = run_compiled_net(min_layers, x_in)[0]
        
        error = abs(expected - actual)
        max_error_min = max(max_error_min, error)
        
    print(f"Max deviation for Min Gadget across {num_tests} trials: {max_error_min:.4e}")
    if max_error_min < 1e-10:
        print(" -> Success! Min Gadget logic flattened perfectly.")
    else:
        print(" -> Failure in Min Gadget compilation.")

if __name__ == "__main__":
    main()