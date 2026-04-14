# refinet Development Roadmap

## Active: Level 4 - Surrogate Simulation
The immediate priority is transitioning from the $O(n)$ exact algebraic cascade (Level 3) to the constrained CPwL neural network simulation framework.

- [ ] **CPwL Residual Surrogates (Lemma 3.1):** - Implement the `ResidualSurrogate` class to simulate $\tilde{R}_1(x)$ and $\tilde{R}_2(x)$.
  - Introduce the $\alpha$ and $\beta$ "danger zone" margin parameters to rigorously flatten the discontinuous tails of the modulo map.
- [ ] **The Coding/Decoding Maps (Lemma 4.3):**
  - Simulate the forward CPwL propagation of $c_m$.
  - Implement the backward decoding channels for $\hat{r}_{m-1}$ and $\hat{c}_{m-1}$.
- [ ] **Basis Decomposition (Lemma 2.7):**
  - Implement the `SpecialHatDecomposition` to fracture curves into strict $[0,1]$ supported basis functions, maintaining $L=1$ translation covariance for the neural network layers.
- [ ] **Surrogate Cascade Integration:**
  - Build a `SurrogateCascadeEngine` to replace `BlockCascadeEngine`.
  - Validate the $L^\infty$ error bound between the Level 3 exact geometry and the Level 4 simulated surrogate geometry.

## Short-term: Validation & Robustness
- [ ] Standardize the configuration injection for all `plot_lib_curves.py` testbeds.
- [ ] Write rigorous unit tests verifying $W^n\Gamma - \Gamma_n = 0$ at the topological boundaries for non-stationary constructs.

## Long-term: Level 5 - Hardware Acceleration
- [ ] Design the tensor flattening protocol to convert Level 4 surrogate architectures into raw weight/bias matrices.
- [ ] PyTorch exporter integration.
- [ ] JAX exporter integration for highly parallel batch curve evaluation.