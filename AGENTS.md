# Repository: Avellaneda-Stoikov Market-Making Model

## Overview
This repository implements the Avellaneda-Stoikov (2008) market-making model with:
- **Experiment 1**: Analytical/symbolic reproduction of all key formulas using SymPy
- **Experiment 2**: Monte Carlo simulation comparing inventory-based vs symmetric strategies

## Repository Structure
```
/workspace/project/
├── exp1_analytical.py      # Symbolic derivations and numerical checks (SymPy + NumPy)
├── exp2_montecarlo.py      # Monte Carlo simulation engine (NumPy + Pandas)
├── visualize.py            # All visualization code (Matplotlib)
├── run_experiments.py      # Main runner: executes both experiments, saves results
├── tests/
│   ├── test_exp1_analytical.py   # Tests for Exp 1
│   └── test_exp2_montecarlo.py   # Tests for Exp 2
├── results/
│   ├── RESULTS.md          # Experiment results and metrics
│   └── *.png               # Generated plots
└── requirements.txt
```

## Key Model Parameters (Paper-Specified)
- S0=100, q0=0, X0=0, T=1, dt=0.005, N=200
- sigma=2, A=140, k=1.5
- gamma values: {0.01, 0.1, 0.5}
- n_paths=1000 per strategy per gamma

## Core Formulas
- Reservation price: r(s,q,t) = s - q·γ·σ²·(T-t)
- Ask distance: δᵃ = ((1-2q)/2)·γ·σ²·τ + (1/γ)·ln(1+γ/k)
- Bid distance: δᵇ = ((1+2q)/2)·γ·σ²·τ + (1/γ)·ln(1+γ/k)
- Total spread: γ·σ²·τ + (2/γ)·ln(1+γ/k)

## Running the Experiments
```bash
# Run all experiments and save results
python run_experiments.py

# Run tests
pytest tests/ -v
```

## Library Choices
- **SymPy**: Symbolic algebra for Exp 1 derivations (Context7 confirmed)
- **NumPy**: Vectorized Monte Carlo simulation for Exp 2
- **Matplotlib**: All visualizations (non-interactive Agg backend)
- No external portfolio/backtesting libraries needed (custom implementation)

## Important Implementation Notes
- Price dynamics use Bernoulli ±σ√dt increments (not Gaussian), per paper
- Fill probabilities validated to be ≤1 at each step
- Correction term (1/γ)·ln(1+γ/k) uses np.log1p for numerical stability near γ=0
- Both ask and bid fills can occur simultaneously in the same time step
- Symmetric strategy centers spread on mid-price, not reservation price

## Expected Paper Reference Values (Exp 2)
| γ | Strategy | Mean Profit | Std(Profit) | Mean q | Std(q) |
|---|----------|-------------|-------------|--------|--------|
| 0.01 | inventory | 66.78 | 8.76 | -0.02 | 4.70 |
| 0.01 | symmetric | 67.36 | 13.40 | -0.31 | 8.65 |
| 0.1 | inventory | 62.94 | 5.89 | 0.10 | 2.80 |
| 0.1 | symmetric | 67.21 | 13.43 | -0.018 | 8.66 |
| 0.5 | inventory | 33.92 | 4.72 | -0.02 | 1.88 |
| 0.5 | symmetric | 66.20 | 14.53 | 0.25 | 9.06 |
