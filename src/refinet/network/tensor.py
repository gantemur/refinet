class Vector:
    """Lightweight, dependency-free 1D array."""
    def __init__(self, data):
        self.data = list(data)
        self.dim = len(self.data)
        
    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

class Matrix:
    """Lightweight, dependency-free 2D array."""
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        
    def matvec(self, vec):
        """Multiplies Matrix by Vector."""
        if self.cols != vec.dim:
            raise ValueError("Dimension mismatch")
        result = []
        for row in self.data:
            result.append(sum(a * b for a, b in zip(row, vec.data)))
        return Vector(result)
