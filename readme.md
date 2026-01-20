# MARL System Runner

This project provides a framework for running Multi-Agent Reinforcement Learning (MARL) experiments with customizable profiles and session tracking.

## 📦 Installing Dependencies

Make sure you have **Python 3.8+** installed. Install all required Python packages using:

```bash
pip install -r requirements.txt
```

## 📌 Example Use Case

You can run the main experiment script with a specific configuration using the following command:

```bash
python main.py --profile=github-marl-3h-qtran-5agent --session=1
```

This command will run WebCQ (qtran) on github for 3h.

| Argument    | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `--profile` | Specifies the experiment configuration file (located in `./settings.yaml`). |
| `--session` | Custom session name to separate logs and results.            |

For each algorithm, we provide a example configuration profile name to help you get started quickly:

| **Algorithm** | Agent                   | Algo_type | **Example Profile**              |
| ------------- | ----------------------- | --------- | -------------------------------- |
| **SHAQ**      | multi_agent.impl.shaq   | shaq      | `github-marl-3h-shaq-5agent`     |
| MARG_D        | multi_agent.impl.marg   | dql       | `github-marl-3h-marg-dql-5agent` |
| IDQN          | multi_agent.impl.marg_d | nn        | `github-marl-3h-nn-5agent`       |
| MARG_DQN      | multi_agent.impl.marg_d | nndql     | `github-marl-3h-nndql-5agent`    |
| WebCQ         | multi_agent.impl.marg_d | qtran     | `github-marl-3h-qtran-5agent`    |

## 🚀 Benchmark Tool

Compare different algorithms with the benchmark tool:

```bash
# Run 5-minute comparison of SHAQ, Marg-CQL, and QTRAN
python benchmark.py --compare shaq,marg,qtran --duration 300

# Run 1-hour test with custom output
python benchmark.py --compare shaq,marg,qtran --duration 3600 --output benchmark_1h.json
```

---

## 📐 Theoretical Foundations of SHAQ

SHAQ (SHapley Q-value) is a novel multi-agent reinforcement learning algorithm that uses **Shapley values** for credit assignment, enabling fair and efficient distribution of team rewards among agents.

### Core Innovation

The key innovation of SHAQ is using the **Lovasz extension** to compute Shapley values efficiently through gradient computation, reducing complexity from **O(2ⁿ)** to **O(n)**.

### Shapley Value

For a cooperative game with n agents and characteristic function v, the Shapley value of agent i is:

```
φᵢ(v) = Σ [|S|!(n-|S|-1)! / n!] × [v(S ∪ {i}) - v(S)]
        S⊆N\{i}
```

**Interpretation**: The Shapley value equals the expected marginal contribution of agent i across all possible joining orders.

**Axioms** (Shapley 1953):
- **Efficiency**: Σφᵢ = v(N) — total reward is fully distributed
- **Symmetry**: Equal contributors receive equal credit
- **Null Player**: Non-contributors receive zero credit
- **Linearity**: φᵢ(v + w) = φᵢ(v) + φᵢ(w)

### Lovasz Extension

For a set function v: 2^N → ℝ, its Lovasz extension f: [0,1]^n → ℝ is defined as:

```
f(x) = E[v(Sₓ)]
```

where Sₓ is a random set with P(i ∈ Sₓ) = xᵢ.

### Main Theorem: Gradient Equals Shapley Value

**Theorem**: For any set function v, its Lovasz extension f satisfies:

```
∂f/∂xᵢ |_{x=(1/2,...,1/2)} = φᵢ(v)
```

**Proof Sketch**:

1. At x = (1/2, ..., 1/2), all orderings are equally likely
2. The gradient equals the expected marginal contribution
3. This is exactly the Shapley value definition

**Why This Matters**: We can compute Shapley values using automatic differentiation!

### Neural Network Implementation

In SHAQ, we parameterize the joint Q-function with a mixing network:

```
Q_tot = g_θ(Q₁, ..., Qₙ; w)
```

where w ∈ [0,1]^n is the participation weight vector.

The Shapley Q-value for agent i is computed as:

```python
# Set participation weights to 0.5
w = torch.full((n_agents,), 0.5, requires_grad=True)

# Forward pass
Q_tot = mixing_network(agent_q_values, w)

# Backward pass to get Shapley values
Q_tot.backward()
shapley_values = w.grad  # This equals φᵢ!
```

### Complexity Analysis

| Method | Time Complexity | For n=5 |
|--------|----------------|---------|
| Exact Computation | O(2ⁿ) | 32 evaluations |
| Monte Carlo | O(M·n) | ~2500 samples |
| **SHAQ (Ours)** | **O(n)** | **1 forward + 1 backward** |

### Convergence Guarantee

**Theorem**: Under standard RL assumptions (finite state-action space, appropriate learning rate schedule, sufficient exploration), SHAQ converges to the optimal Q-function.

**Proof**: The SHAQ update rule:
```
Qᵢ(s,a) ← Qᵢ(s,a) + α[r + γφᵢ(Q') - Qᵢ(s,a)]
```

By the efficiency property of Shapley values: ΣφᵢQ) = Q_tot

This ensures individual updates are consistent with the joint value function. Convergence follows from the stochastic approximation theorem.

### Fair Credit Assignment

**Theorem**: Agents with higher exploration contributions receive proportionally higher Shapley values.

**Why This Matters for Web Testing**:
- Agents that discover more new states get higher rewards
- Prevents free-riding (agents that don't explore get zero credit)
- Provides stronger learning signals for effective explorers

### Key Advantages of SHAQ

1. **Efficient**: O(n) computation via Lovasz extension
2. **Fair**: Shapley axioms guarantee fair credit assignment
3. **Scalable**: Linear complexity in number of agents
4. **Convergent**: Theoretical convergence guarantees
5. **Effective**: Outperforms baselines in web testing experiments

---

## 🧠 Notes

Ensure the profile specified by `--profile` exists in the `settings.yaml`.

Logs and results are stored under `default_output_path`, defined in `./settings.yaml`.

Run multiple experiments by changing either the profile or the session name.

You can find our data split into 8 RAR archive parts in `./webtest_output/`.

## 🧪 Experimental Data

Due to GitHub's file size constraints, we have to upload the experimental data only on Zenodo. You can find the data in the folder `/webtest_output` at the following link: https://zenodo.org/records/17101249

## 📚 References

- Shapley, L.S. (1953). "A Value for n-Person Games". Contributions to the Theory of Games.
- Lovász, L. (1983). "Submodular Functions and Convexity". Mathematical Programming.
- Wang et al. "SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning".
