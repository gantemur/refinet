from .tensor import Matrix, Vector

class LightweightReLUNet:
    """
    Level 5: Pure Python sequential neural network engine.
    Runs natively in the browser without PyTorch or NumPy.
    """
    def __init__(self):
        self.layers = [] # List of (Matrix, Vector) tuples for W and b

    def add_layer(self, weights, biases):
        self.layers.append((weights, biases))

    def forward(self, x):
        """
        Evaluates the network: a_{k+1} = max(0, W_k a_k + b_k)
        """
        current_activation = Vector([x]) if isinstance(x, (int, float)) else x
        
        for W, b in self.layers:
            # Affine step
            z = W.matvec(current_activation) + b
            # ReLU activation
            current_activation = Vector([max(0.0, val) for val in z.data])
            
        return current_activation
