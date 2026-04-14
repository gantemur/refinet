import math

class SpecialHatDecomposition:
    """
    Implements Lemma 2.7: Decomposes a CPwL curve into special hat functions.
    Dynamically subdivides the curve evenly to ensure all hat supports are <= 1 - 2*rho,
    and discards the boundary hats (which have coefficient 0 due to compact support).
    """
    def __init__(self, curve, rho=0.125):
        self.p = curve.p
        self.rho = rho
        
        max_gap = (1.0 - 2.0 * rho) / 2.0
        
        # 1. Refine the partition evenly
        raw_t = [bp[0] for bp in curve.breakpoints]
        refined_t = [raw_t[0]]
        
        for i in range(1, len(raw_t)):
            prev = refined_t[-1]
            curr = raw_t[i]
            gap = curr - prev
            
            if gap > max_gap + 1e-9:
                num_segments = math.ceil(gap / max_gap)
                step = gap / num_segments
                for k in range(1, num_segments):
                    refined_t.append(prev + k * step)
                    
            refined_t.append(curr)
            
        self.nodes = refined_t
        self.coeffs = [curve(t) for t in refined_t] 
        self.N = len(self.nodes) - 1
        
        # 2. Build the shifted hat functions (SKIPPING 0 and N)
        self.hats = []
        self._hat_map = {}
        
        for j in range(1, self.N):
            t_peak = self.nodes[j]
            t_left = self.nodes[j-1]  # No boundary checks needed!
            t_right = self.nodes[j+1] # No boundary checks needed!
            
            midpoint = (t_left + t_right) / 2.0
            shift = midpoint - 0.5
            
            hat = {
                'index': j,
                'global_peak': t_peak,
                'global_support': (t_left, t_right),
                'shift': shift,
                'local_nodes': (t_left - shift, t_peak - shift, t_right - shift),
                'coeff': self.coeffs[j]
            }
            self.hats.append(hat)
            self._hat_map[j] = hat

    def evaluate_local_hat(self, j, tau):
        # Safely ignore requests for boundary hats
        if j not in self._hat_map:
            return 0.0
            
        local_left, local_peak, local_right = self._hat_map[j]['local_nodes']
        
        if tau <= local_left or tau >= local_right:
            return 0.0
        elif tau <= local_peak:
            return (tau - local_left) / (local_peak - local_left)
        else:
            return (local_right - tau) / (local_right - local_peak)