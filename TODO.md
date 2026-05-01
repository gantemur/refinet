# refinet Development Roadmap

## ✅ Completed Milestones
- **Level 1-3:** Exact algebraic block cascades and $M$-ary topological routing in $O(n)$ time.
- **Level 4:** CPwL Surrogate Simulation unified with nested product gadgets and left co-vector row topologies (`surrogate2d.py`).
- **Level 5:** Exact Deep PyTorch Realization. Neural network mathematically unified with L4 down to the FP64 precision floor, achieving `0.00e+00` deviation (`model2d.py`).

## 🚀 Active: Validation & Generalization
Now that the L4/L5 core is mathematically sealed, the immediate priority is hardening the library and expanding dimensionality.
- [ ] **Comprehensive Testing & Edge Cases:** Execute exhaustive test suites across randomized coordinates to ensure absolute FP64 stability across all supported fractal geometries.
- [ ] **Arbitrary $n$-Dimensional Topology:** Generalize the 1D and 2D tensor-product engines to natively route and compute arbitrary $n$-D affine systems.
- [ ] **Vector-Valued Extensions:** Ensure smooth handling of broader vector-valued functions and affine matrices beyond the current testing parameters.
- [ ] **Configuration Standardization:** Clean up dependency injection for all `plot_lib_curves.py` testbeds.

## 🌐 Mid-term: Ecosystem & Tooling
- [ ] **Interactive Visualizers:** Develop in-browser interactive widgets (e.g., D3.js, Three.js) to let users visually explore the $M$-ary topological routers, transition band scaling, and polyhedral CPwL geometries in real-time.
- [ ] **Web Ecosystem:** Build the official `refinet` webpage and accessible web-based API/tools for the wider mathematical and machine learning community.

## ⚡ Long-term: Hardware Scaling
- [ ] **JAX Integration:** Build a JAX exporter for highly parallel, XLA-compiled batch curve evaluation across TPUs/GPUs.
- [ ] **Tensor Flattening Protocol:** Create an automated pipeline to dump compiled Level 5 architectures into static, portable weight/bias matrix files for deployment outside the Python ecosystem.