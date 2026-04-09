# Avellaneda-Stoikov Model: Experiment Results

Analytical reproduction and Monte Carlo replication of the Avellaneda-Stoikov (2008) market-making model.

---

## Experiment 1: Analytical Reproduction of AS Model Formulas

### Symbolic Verification Results

| Formula | Verified |
|---------|----------|
| Reservation ask price r^a = s + ((1-2q)/2)·γ·σ²·τ | ✓ PASS |
| Reservation bid price r^b = s + ((-1-2q)/2)·γ·σ²·τ | ✓ PASS |
| Reservation price r = s - q·γ·σ²·τ | ✓ PASS |
| Spread = δᵃ + δᵇ (consistency) | ✓ PASS |
| p^a = r + spread/2 = s + δᵃ | ✓ PASS |
| p^b = r - spread/2 = s - δᵇ | ✓ PASS |
| r = 0.5·(r^a + r^b) | ✓ PASS |

**All formulas verified: True**

### Key Symbolic Expressions

```
v(x,s,q,t)  = -exp(-gamma*x)*exp(-gamma*q*s)*exp(gamma**2*q**2*sigma**2*(T - t)/2)
r^a(s,q,t)  = -T*gamma*q*sigma**2 + T*gamma*sigma**2/2 + gamma*q*sigma**2*t - gamma*sigma**2*t/2 + s
r^b(s,q,t)  = -T*gamma*q*sigma**2 - T*gamma*sigma**2/2 + gamma*q*sigma**2*t + gamma*sigma**2*t/2 + s
r(s,q,t)    = -T*gamma*q*sigma**2 + gamma*q*sigma**2*t + s
spread      = gamma*sigma**2*(T - t) + 2*log(gamma/k + 1)/gamma
delta^a     = gamma*sigma**2*(1/2 - q)*(T - t) + log(gamma/k + 1)/gamma
delta^b     = gamma*sigma**2*(T - t)*(q + 1/2) + log(gamma/k + 1)/gamma
correction  = log(gamma/k + 1)/gamma
```

### Limiting Behavior Checks

#### As t → T (τ → 0)

| γ | Inventory Risk at T | Spread at T | Equals Liquidity |
|---|---------------------|-------------|-----------------|
| 0.01 | 0.000000 | 1.328909 | ✓ |
| 0.1 | 0.000000 | 1.290770 | ✓ |
| 0.5 | 0.000000 | 1.150728 | ✓ |

#### As γ → 0

- Correction term at γ=1e-6: 0.666666
- Limit 1/k: 0.666667
- Converges: ✓
- Asymmetry at γ=1e-6 (q=1): 8.00e-06
- Asymmetry vanishes: ✓

### Numerical Parameter Sweep (s=100, σ=2, k=1.5, T=1)

Selected results for t=0.0:

| γ | q | τ | r | δᵃ | δᵇ | Spread |
|---|---|---|---|----|----|--------|
| 0.01 | -2 | 1.00 | 100.0800 | 0.7645 | 0.6045 | 1.3689 |
| 0.01 | -1 | 1.00 | 100.0400 | 0.7245 | 0.6445 | 1.3689 |
| 0.01 | +0 | 1.00 | 100.0000 | 0.6845 | 0.6845 | 1.3689 |
| 0.01 | +1 | 1.00 | 99.9600 | 0.6445 | 0.7245 | 1.3689 |
| 0.01 | +2 | 1.00 | 99.9200 | 0.6045 | 0.7645 | 1.3689 |
| 0.1 | -2 | 1.00 | 100.8000 | 1.6454 | 0.0454 | 1.6908 |
| 0.1 | -1 | 1.00 | 100.4000 | 1.2454 | 0.4454 | 1.6908 |
| 0.1 | +0 | 1.00 | 100.0000 | 0.8454 | 0.8454 | 1.6908 |
| 0.1 | +1 | 1.00 | 99.6000 | 0.4454 | 1.2454 | 1.6908 |
| 0.1 | +2 | 1.00 | 99.2000 | 0.0454 | 1.6454 | 1.6908 |
| 0.5 | -2 | 1.00 | 104.0000 | 5.5754 | -2.4246 | 3.1507 |
| 0.5 | -1 | 1.00 | 102.0000 | 3.5754 | -0.4246 | 3.1507 |
| 0.5 | +0 | 1.00 | 100.0000 | 1.5754 | 1.5754 | 3.1507 |
| 0.5 | +1 | 1.00 | 98.0000 | -0.4246 | 3.5754 | 3.1507 |
| 0.5 | +2 | 1.00 | 96.0000 | -2.4246 | 5.5754 | 3.1507 |

### Liquidity Spread Components (2/γ)·ln(1+γ/k) for k=1.5

| γ | Liquidity Spread | Paper Reference |
|---|-----------------|-----------------|
| 0.01 | 1.3289 | ≈1.33 |
| 0.1 | 1.2908 | ≈1.29 |
| 0.5 | 1.1507 | ≈1.15 |


*Experiment 1 runtime: 0.65s*

---

## Experiment 2: Monte Carlo Market-Making Simulation

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| S0 | 100.0 |
| q0 | 0.0 |
| X0 | 0.0 |
| T | 1.0 |
| dt | 0.005 |
| N | 200 |
| sigma | 2.0 |
| A | 140.0 |
| k | 1.5 |
| n_paths | 1000 |

### Summary Statistics

| γ | Strategy | Liquidity Spread | Mean Profit | Std(Profit) | Mean Final q | Std(Final q) |
|---|----------|-----------------|-------------|-------------|-------------|-------------|
| 0.01 | inventory | 1.3289 | 68.35 | 9.00 | -0.177 | 5.210 |
| 0.01 | symmetric | 1.3289 | 68.80 | 13.65 | 0.400 | 8.364 |
| 0.1 | inventory | 1.2908 | 64.67 | 6.59 | -0.110 | 2.905 |
| 0.1 | symmetric | 1.2908 | 68.98 | 13.72 | 0.012 | 8.471 |
| 0.5 | inventory | 1.1507 | 48.60 | 5.58 | -0.007 | 1.886 |
| 0.5 | symmetric | 1.1507 | 59.44 | 12.42 | 0.044 | 7.105 |

### Paper Reference Values (for qualitative comparison)

| γ | Strategy | Mean Profit (paper) | Std(Profit) (paper) | Mean q (paper) | Std(q) (paper) |
|---|----------|---------------------|---------------------|----------------|----------------|
| 0.01 | inventory | 66.78 | 8.76 | -0.02 | 4.7 |
| 0.01 | symmetric | 67.36 | 13.4 | -0.31 | 8.65 |
| 0.1 | inventory | 62.94 | 5.89 | 0.1 | 2.8 |
| 0.1 | symmetric | 67.21 | 13.43 | -0.018 | 8.66 |
| 0.5 | inventory | 33.92 | 4.72 | -0.02 | 1.88 |
| 0.5 | symmetric | 66.2 | 14.53 | 0.25 | 9.06 |

### Qualitative Analysis

**γ = 0.01:**
- Profit risk reduction (inventory vs symmetric): 34.1%
- Inventory risk reduction: 37.7%
- Symmetric strategy mean profit advantage: 0.45

**γ = 0.1:**
- Profit risk reduction (inventory vs symmetric): 52.0%
- Inventory risk reduction: 65.7%
- Symmetric strategy mean profit advantage: 4.31

**γ = 0.5:**
- Profit risk reduction (inventory vs symmetric): 55.1%
- Inventory risk reduction: 73.4%
- Symmetric strategy mean profit advantage: 10.84

### Convergence Check (γ → 0)

- std(Profit) ratio inventory/symmetric at γ=0.01: 0.659 (closer to 1 = more similar)
- std(Profit) ratio inventory/symmetric at γ=0.5: 0.449 (further from 1 = more different)
- Strategies converge as γ→0: ✓


*Experiment 2 runtime: 0.14s*

---

## Generated Plots

### Experiment 1
- `exp1_reservation_price.png` — Reservation price vs τ for different q and γ
- `exp1_quote_distances.png` — Optimal quote distances δᵃ and δᵇ vs τ
- `exp1_spread_components.png` — Total spread and its components vs τ
- `exp1_limiting_behavior.png` — Limiting behavior as γ → 0

### Experiment 2
- `exp2_sample_path_gamma0.1.png` — Sample path with quotes and reservation price
- `exp2_profit_histograms.png` — Terminal profit distributions
- `exp2_inventory_histograms.png` — Terminal inventory distributions
- `exp2_summary_comparison.png` — Bar chart comparison of strategies
