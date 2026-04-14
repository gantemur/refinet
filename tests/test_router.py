import sys
import os

# Ensure the local src directory is in the Python path
sys.path.insert(0, os.path.abspath('src'))

from refinet.cascade.exact import MaryRouter

def main():
    M = 3  # Using M=3 (e.g., Peano Curve geometry)
    depth = 4
    
    router = MaryRouter(M)
    
    print(f"--- M-ary Topological Router Test (M={M}, Depth={depth}) ---")
    
    # Test a few interesting t-values
    test_points = [
        0.0,            # Exact start
        0.15,           # Random lower end
        1.0 / 3.0,      # Exact boundary between block 0 and 1
        0.892,          # Random upper end
        1.0             # Exact end
    ]
    
    header_q = "Digits [q_1 ... q_n]"
    header_r = "Residuals [r_1 ... r_n]"
    print(f"\n{'Target (t)':<12} | {header_q:<22} | {header_r}")
    print("-" * 80)
    
    for t in test_points:
        digits, residuals = router.decode(t, depth)
        
        # Format for clean printing
        t_str = f"{t:.6f}"
        q_str = str(digits)
        r_str = "[" + ", ".join([f"{r:.3f}" for r in residuals]) + "]"
        
        print(f"{t_str:<12} | {q_str:<22} | {r_str}")

if __name__ == "__main__":
    main()