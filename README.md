# Avellaneda-Stoikov Market-Making Model: Analytical Reproduction

Reproduction of the Avellaneda-Stoikov (2008) market-making model with three experiments:

## Experiments

### Experiment 1: Analytical Reproduction
Symbolic derivation of all AS model formulas using SymPy:
- Frozen-inventory value function
- Finite-horizon reservation bid/ask prices
- Infinite-horizon stationary reservation prices
- Optimal quote-distance conditions under exponential intensity
- Gamma → 0 convergence to symmetric strategy

### Experiment 2: Monte Carlo Simulation
Replication of the paper's main numerical experiment:
- 1000 paths per strategy per gamma value
- Inventory-based vs symmetric benchmark strategies
- Binomial Brownian mid-price dynamics
- Exponential execution intensities

### Experiment 3: Appendix Mean-Variance Model
Analytical reproduction of the paper's appendix extension:
- Alternative geometric-like price dynamics
- Mean-variance objective function
- Separate from main simulation tables

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run all experiments
python src/run_all_experiments.py

# Run tests
pytest tests/ -v
```

## Results

Results are saved to `results/RESULTS.md` with all metrics and plots.

## Parameters

| Parameter | Value |
|-----------|-------|
| S0 | 100 |
| T | 1 |
| sigma | 2 |
| dt | 0.005 |
| N | 200 |
| A | 140 |
| k | 1.5 |
| gamma | {0.1, 0.01, 0.5} |
| n_paths | 1000 |

## Key Formulas

**Reservation price:**
```
r(s,q,t) = s - q·γ·σ²·(T-t)
```

**Total spread:**
```
Δ_t = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k)
```

**Optimal quotes:**
```
p^a = r + Δ_t/2
p^b = r - Δ_t/2
```
