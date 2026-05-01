# refinet

**Neural Network Realization of Affine Refinement Iterates**

`refinet` is a specialized computational library developed to translate recursive geometric constructions into Continuous Piecewise Linear (CPwL) ReLU neural networks. It provides the exact algorithmic infrastructure to evaluate vector-valued affine refinement operators acting on CPwL curves, mapping continuous fractal geometry into discrete topological spaces.

This library is the programmatic realization of the theoretical framework established by Bolorkhuu and Gantumur.

## Mathematical Framework

The core of the library evaluates affine refinement operators acting on continuous piecewise linear curves $\gamma: \mathbb{R} \to \mathbb{R}^p$:

$$(W\gamma)(t) = \sum_{j\in\mathbb{Z}} A_{j} \gamma(Mt-j) + B(t)$$

Where:
* $M \ge 2$ is an integer topological scale.
* $A_j \in \mathbb{R}^{p \times p}$ are finite transition matrices.
* $B(t)$ is a compactly supported CPwL forcing sequence.

To bypass the naive $O(M^n)$ exponential complexity of recursive geometric evaluation, `refinet` implements an $M$-ary topological router and exact backward recursion to evaluate the iterates $W^n\gamma$ in strict **$O(n)$ time**. The framework natively scales from 1D operators to full 2D Tensor-Product matrix fields.

## Architecture

The framework is strictly separated into a 5-Level Architecture bridging abstract continuous geometry to discrete neural weights.

* **Level 1: Geometric Builders (`src/refinet/geometry/`)**
  Translates geometric rules into formal affine structures. Implements Anchored Reduction (Proposition 5.4) to handle open curves via $\gamma = \Gamma + \eta$. Computes dynamic anchor profiles $\Gamma_n$ and stage-dependent defect sequences $E_n(t) = B_n(r)$ within finite-dimensional spans.
* **Level 2: Exact Affine Systems (`src/refinet/algebra/system.py`)**
  The universal contract (`AffineRefinementSystem`). Encapsulates $A_j$, $M$, $p$, support bounds $L$, and natively houses non-stationary drifting weights and basis spans.
* **Level 3: The $O(n)$ Cascade Engine (`src/refinet/cascade/exact.py`)**
  Vectorizes the curve space $G(x) \in \mathbb{R}^{pL}$ and compiles global block transition matrices $T_q$. Implements Exact Backward Recursion (Theorem 4.16) utilizing an $M$-ary fractional residual router (Lemma 4.3). Sub-second exact rendering of deep fractals.
* **Level 4: CPwL Surrogate Simulation (`src/refinet/cascade/surrogate2d.py`)**
  Bridges abstract algorithmic logic to constrained Neural Network topology. Simulates the continuous residual surrogates, fractional decoding channels, and exact left co-vector (row-vector) forward matrix propagation. Identical in geometric behavior to the compiled deep network.
* **Level 5: Exact PyTorch Realization (`src/refinet/network/model2d.py`)**
  Strict compilation of the affine cascade into an executable, exact deep ReLU PyTorch `nn.Module`. Utilizes concurrent matrix field evaluations, $M$-ary topological loop controllers, and exact CPwL Product Gadgets. Enforces native 64-bit float precision to achieve mathematically flawless, zero-deviation evaluation of the theoretical framework.

## Supported Geometry
The builder library natively supports and accurately compiles:
* **Homogeneous Systems:** Lévy Dragon, Koch Curve, standard Peano.
* **Finite-State Affine Systems:** Gosper Curve, Sierpinski Arrowhead (via state-stacking where dim $= p \times r$).
* **Stationary Affine Systems:** Morton-like Z-order curves.
* **Stage-Dependent (Non-Stationary) Systems:** Infinite finite-span Hilbert and Peano space-filling curves.
* **2D Tensor-Product Systems:** Exact higher-dimensional evaluations utilizing polyhedral partition of unity selectors and bounded nested product gadgets.