# Repository: Avellaneda-Stoikov Market-Making Model

## Overview
This repository implements the Avellaneda-Stoikov (AS) market-making model with three experiments:

1. **Exp 1** (`src/exp1_analytical.py`): Symbolic derivation of all AS model formulas using SymPy
2. **Exp 2** (`src/exp2_montecarlo.py`): Monte Carlo simulation comparing inventory vs symmetric strategies
3. **Exp 3** (`src/exp3_appendix.py`): Appendix mean-variance model (separate extension)

## Key Parameters
- S0=100, T=1, sigma=2, dt=0.005, N=200
- A=140, k=1.5 (exponential intensity)
- gamma in {0.1, 0.01, 0.5}
- n_paths=1000, random_seed=42

## Structure
```
src/
  exp1_analytical.py    # SymPy symbolic derivations
  exp2_montecarlo.py    # NumPy Monte Carlo simulation
  exp3_appendix.py      # Appendix mean-variance model
  run_all_experiments.py # Main runner
tests/
  test_exp1_analytical.py
  test_exp2_montecarlo.py
  test_exp3_appendix.py
results/
  RESULTS.md            # All metrics and findings
  *.png                 # Plots
```

## Running
```bash
# Run all experiments
python src/run_all_experiments.py

# Run tests
pytest tests/ -v

# Run individual experiments
python src/exp1_analytical.py
python src/exp2_montecarlo.py
python src/exp3_appendix.py
```

## Key Formulas
- Reservation price: r(s,q,t) = s - q*gamma*sigma^2*(T-t)
- Spread: Delta_t = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)
- Quotes: p^a = r + Delta_t/2, p^b = r - Delta_t/2
- Intensity: lambda(delta) = A*exp(-k*delta)

## Library Choices
- SymPy: symbolic math (exp1, exp3)
- NumPy: numerical computation (exp2)
- Matplotlib: all visualizations
- pytest: testing

## Notes
- Exp 3 appendix model uses geometric dynamics dS/S = sigma*dW (NOT the main model)
- Paper targets in exp2 are directional, not exact (Monte Carlo randomness)
- Spread static component (2/gamma)*ln(1+gamma/k) matches paper table values
