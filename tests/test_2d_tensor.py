import sys
import os

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath('src'))

from refinet.network.tensor2d import TensorCompiler2D

def main():
    print("--- Level 5: 2D Offline Tensor Compiler Test ---\n")
    
    # Standard test parameters
    M = 2
    n = 3
    delta_bar = 0.5
    epsilon = 0.05
    L = 2

    compiler = TensorCompiler2D()

    # 1. Test Torus Controller
    print("[1/4] Compiling Torus Controller...")
    torus_layers = compiler.build_torus_controller(M)
    for i, layer in enumerate(torus_layers):
        w_shape = layer['weight'].shape
        b_shape = layer['bias'].shape
        print(f"  Layer {i+1} ({layer['activation']}): Weight {w_shape}, Bias {b_shape}")

    # 2. Test 4-Branch Readouts
    print("\n[2/4] Compiling 4-Branch Tensor Readouts...")
    readout_layers = compiler.build_tensor_readouts(epsilon)
    for i, layer in enumerate(readout_layers):
        w_shape = layer['weight'].shape
        b_shape = layer['bias'].shape
        print(f"  Layer {i+1} ({layer['activation']}): Weight {w_shape}, Bias {b_shape}")

    # 3. Test 2D Selectors
    print("\n[3/4] Compiling 2D Tensor Selectors...")
    selector_layers = compiler.build_tensor_selectors(M, n, delta_bar)
    for i, layer in enumerate(selector_layers):
        w_shape = layer['weight'].shape
        b_shape = layer['bias'].shape
        print(f"  Layer {i+1} ({layer['activation']}): Weight {w_shape}, Bias {b_shape}")

    # 4. Test Clamped Gluing Ramps
    print(f"\n[4/4] Compiling Clamped Gluing Ramps (L={L})...")
    ramp_layers = compiler.build_gluing_ramps(L)
    for i, layer in enumerate(ramp_layers):
        w_shape = layer['weight'].shape
        b_shape = layer['bias'].shape
        print(f"  Layer {i+1} ({layer['activation']}): Weight {w_shape}, Bias {b_shape}")

    print("\nSUCCESS: All 2D neural tensors compiled without shape errors.")

if __name__ == "__main__":
    main()