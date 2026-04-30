import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from refinet.network.tensor import TensorCompiler

def exact_math_gadget(lam, y, a):
    """The exact mathematical definition of the product gadget."""
    T1 = np.maximum(0, lam * a - y)
    T2 = np.maximum(0, a - lam * a - np.maximum(0, -y))
    return -T1 - T2 + a

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
    print("--- Testing Tensor Compiler: Product Gadget ---")
    N = 3
    a = 5.0
    
    # 1. Compile the tensors!
    layers = TensorCompiler.build_product_gadget(N, a)
    
    print(f"Compiled {len(layers)} layers.")
    for i, l in enumerate(layers):
        print(f"  Layer {i+1}: W {l['weight'].shape}, b {l['bias'].shape}, act: {l['activation']}")
        
    # 2. Test over random inputs
    np.random.seed(42)
    num_tests = 1000
    max_error = 0.0
    
    for _ in range(num_tests):
        # lambda must be in [0, 1], y must be in [-a, a]
        lam = np.random.uniform(0, 1)
        y = np.random.uniform(-a, a, size=N)
        
        # Build the input vector [lambda, y1, y2, ...]
        x_in = np.concatenate(([lam], y))
        
        # Evaluate both pathways
        expected = exact_math_gadget(lam, y, a)
        actual = run_compiled_net(layers, x_in)
        
        error = np.max(np.abs(expected - actual))
        max_error = max(max_error, error)
        
    print(f"\nMax deviation across {num_tests} random trials: {max_error:.4e}")
    if max_error < 1e-10:
        print("Success! The raw compiled tensors mathematically match the exact gadget.")
    else:
        print("Failure! Tensor arithmetic diverged from exact math.")

if __name__ == "__main__":
    main()