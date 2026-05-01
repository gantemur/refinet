import numpy as np
from refinet.network.tensor import TensorCompiler

class TensorCompiler2D:
    """
    Level 5 (2D): Offline compiler for the 2D Tensor-Product Exact Cascade.
    Translates the Torus Controller, 2D Selectors, and Clamped Gluing into
    raw PyTorch weights.
    """

    @staticmethod
    def build_torus_controller(M: int):
        """
        Compiles the Exact Torus Controller F(z_1, z_2).
        Input dimension: 4 (z_1x, z_1y, z_2x, z_2y)
        Output dimension: 4
        """
        # Get the 1D exact loop controller layers
        layers_1d = TensorCompiler.build_loop_controller(M)
        layers_2d = []

        for layer in layers_1d:
            W1D = layer['weight']
            b1D = layer['bias']
            act = layer['activation']
            
            # The 1D weight matrix has shape (out_features, in_features)
            out_f, in_f = W1D.shape
            
            # Create a Block Diagonal matrix for the Torus
            W2D = np.zeros((2 * out_f, 2 * in_f), dtype=float)
            W2D[:out_f, :in_f] = W1D
            W2D[out_f:, in_f:] = W1D
            
            # Concatenate the biases
            b2D = np.concatenate([b1D, b1D])
            
            layers_2d.append({
                'weight': W2D,
                'bias': b2D,
                'activation': act
            })
            
        return layers_2d

    @staticmethod
    def build_tensor_readouts(epsilon: float):
        """
        Compiles the 4-Branch Terminal Readouts for the Torus state.
        Outputs rho^(-,-), rho^(-,+), rho^(+,-), rho^(+,+)
        Input dimension: 4 (z_1, z_2)
        Output dimension: 4 (The four distinct seam evaluations)
        """
        # Get the 1D seam readouts [rho_minus, rho_plus]
        layers_1d = TensorCompiler.build_seam_readouts(epsilon)
        layers_2d = []

        for i, layer in enumerate(layers_1d):
            W1D = layer['weight']
            b1D = layer['bias']
            act = layer['activation']
            
            out_f, in_f = W1D.shape
            
            # Block Diagonal exactly like the Torus Controller
            W2D = np.zeros((2 * out_f, 2 * in_f), dtype=float)
            W2D[:out_f, :in_f] = W1D
            W2D[out_f:, in_f:] = W1D
            
            b2D = np.concatenate([b1D, b1D])
            
            # Only on the final linear layer do we map the 4 independent 
            # 1D readouts into the specific 4 coordinate branches needed for the 2D atom.
            # 1D out is: [x-, x+, y-, y+]
            # We want: [(x-, y-), (x-, y+), (x+, y-), (x+, y+)]
            if act == 'linear' and i == len(layers_1d) - 1:
                # We need an extra mixing layer to pair them up
                W_mix = np.zeros((8, 4), dtype=float)
                # Branch 1: (x-, y-)
                W_mix[0, 0], W_mix[1, 2] = 1.0, 1.0
                # Branch 2: (x-, y+)
                W_mix[2, 0], W_mix[3, 3] = 1.0, 1.0
                # Branch 3: (x+, y-)
                W_mix[4, 1], W_mix[5, 2] = 1.0, 1.0
                # Branch 4: (x+, y+)
                W_mix[6, 1], W_mix[7, 3] = 1.0, 1.0
                
                W2D = W_mix @ W2D
                b2D = W_mix @ b2D
            
            layers_2d.append({
                'weight': W2D,
                'bias': b2D,
                'activation': act
            })
            
        return layers_2d

    @staticmethod
    def build_tensor_selectors(M: int, n: int, delta_bar: float, rho: float = 0.15):
        """
        Compiles the 2D Loop-State Selectors.
        Input dimension: 4 (z_1x, z_1y, z_2x, z_2y)
        Output dimension: 2M (M selectors for x, M selectors for y)
        """
        # Get the bulletproof 2-layer Radial Tent Engine selectors from 1D
        layers_1d = TensorCompiler.build_selectors(M, n, delta_bar, rho)
        layers_2d = []

        for layer in layers_1d:
            W1D = layer['weight']
            b1D = layer['bias']
            act = layer['activation']
            
            out_f, in_f = W1D.shape
            
            # Block Diagonalize to evaluate x and y selectors simultaneously
            W2D = np.zeros((2 * out_f, 2 * in_f), dtype=float)
            W2D[:out_f, :in_f] = W1D
            W2D[out_f:, in_f:] = W1D
            
            b2D = np.concatenate([b1D, b1D])
            
            layers_2d.append({
                'weight': W2D,
                'bias': b2D,
                'activation': act
            })
            
        return layers_2d

    @staticmethod
    def build_gluing_ramps(L: int):
        """
        Compiles the 1D Clamped Gluing Ramps \sigma_k(t) from Lemma 4.4.
        Evaluated twice (once for L1, once for L2) in the final PyTorch model
        to map global coordinates back to the reference square patches.
        
        Input dimension: 1 (global coordinate t in [0, L])
        Output dimension: L (the fractional ramp outputs for each patch)
        """
        # Layer 1: [ReLU(t - k + 1), ReLU(t - k)]^T for k=1..L
        W1 = np.zeros((2 * L, 1), dtype=float)
        b1 = np.zeros(2 * L, dtype=float)
        
        for k in range(1, L + 1):
            idx = 2 * (k - 1)
            # ReLU(t - (k - 1))
            W1[idx, 0] = 1.0
            b1[idx] = -(k - 1.0)
            
            # ReLU(t - k)
            W1[idx+1, 0] = 1.0
            b1[idx+1] = -float(k)
            
        # Layer 2: \sigma_k(t) = ReLU(t - k + 1) - ReLU(t - k)
        W2 = np.zeros((L, 2 * L), dtype=float)
        b2 = np.zeros(L, dtype=float)
        
        for k in range(1, L + 1):
            idx = 2 * (k - 1)
            W2[k-1, idx] = 1.0
            W2[k-1, idx+1] = -1.0
            
        return [
            {'weight': W1, 'bias': b1, 'activation': 'relu'},
            {'weight': W2, 'bias': b2, 'activation': 'linear'}
        ]