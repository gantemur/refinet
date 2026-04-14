class CPwLCurve:
    """
    Level 1: A continuous piecewise linear curve defined entirely by breakpoints.
    The decomposition into special-hats is deferred to the network compilation levels.
    """
    def __init__(self, breakpoints, p=2):
        """
        breakpoints: List of tuples (t, value_vector).
        Example for a 1D triangle wave: [(0.0, [0.0]), (0.5, [1.0]), (1.0, [0.0])]
        """
        self.p = p
        self.breakpoints = sorted(breakpoints, key=lambda x: x[0])

    def __call__(self, t):
        if not self.breakpoints:
            return [0.0] * self.p

        t_first, v_first = self.breakpoints[0]
        if t <= t_first: 
            return v_first

        t_last, v_last = self.breakpoints[-1]
        if t >= t_last: 
            return v_last

        for i in range(len(self.breakpoints) - 1):
            t0, v0 = self.breakpoints[i]
            t1, v1 = self.breakpoints[i+1]
            if t0 <= t <= t1:
                # Linear interpolation for all target dimensions
                ratio = (t - t0) / (t1 - t0)
                return [v0[d] + ratio * (v1[d] - v0[d]) for d in range(self.p)]
                
        return [0.0] * self.p

    @classmethod
    def zero(cls, p=2):
        """Returns a completely zero forcing term."""
        return cls(breakpoints=[(0.0, [0.0] * p), (1.0, [0.0] * p)], p=p)
