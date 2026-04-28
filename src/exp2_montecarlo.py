"""
Experiment 2: Monte Carlo replication of finite-horizon inventory-based versus
symmetric market-making strategies in the Avellaneda-Stoikov model.

Simulates 1000 paths per strategy per gamma value under:
- Binomial Brownian mid-price dynamics: S_{t+dt} = S_t ± sigma*sqrt(dt)
- Exponential execution intensities: lambda(delta) = A * exp(-k * delta)
- Inventory-based strategy using finite-horizon reservation prices
- Symmetric benchmark strategy centered at mid-price

Parameters (from paper):
  S0=100, T=1, sigma=2, dt=0.005, N=200, q0=0, X0=0
  A=140, k=1.5, gamma in {0.1, 0.01, 0.5}
  n_paths=1000

Using NumPy for vectorized computation — Context7 confirmed.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

PARAMS = {
    'S0': 100.0,       # initial mid-price
    'T': 1.0,          # terminal horizon
    'sigma': 2.0,      # volatility
    'dt': 0.005,       # time step
    'N': 200,          # number of time steps (T/dt)
    'q0': 0,           # initial inventory
    'X0': 0.0,         # initial cash
    'A': 140.0,        # intensity scale
    'k': 1.5,          # intensity decay
    'n_paths': 1000,   # Monte Carlo paths per strategy per gamma
    'gamma_values': [0.1, 0.01, 0.5],
    'random_seed': 42, # documented for reproducibility
}

# Paper-reported target values for qualitative comparison
PAPER_TARGETS = {
    0.1: {
        'inventory': {'spread': 1.29, 'profit': 62.94, 'std_profit': 5.89,
                      'mean_q': 0.10, 'std_q': 2.80},
        'symmetric': {'spread': 1.29, 'profit': 67.21, 'std_profit': 13.43,
                      'mean_q': -0.018, 'std_q': 8.66},
    },
    0.01: {
        'inventory': {'spread': 1.33, 'profit': 66.78, 'std_profit': 8.76,
                      'mean_q': -0.02, 'std_q': 4.70},
        'symmetric': {'spread': 1.33, 'profit': 67.36, 'std_profit': 13.40,
                      'mean_q': -0.31, 'std_q': 8.65},
    },
    0.5: {
        'inventory': {'spread': 1.15, 'profit': 23.92, 'std_profit': 4.72,
                      'mean_q': -0.02, 'std_q': 1.88},
        'symmetric': {'spread': 1.15, 'profit': 66.20, 'std_profit': 14.53,
                      'mean_q': 0.25, 'std_q': 9.06},
    },
}


# ---------------------------------------------------------------------------
# Core formula functions
# ---------------------------------------------------------------------------

def reservation_price(S: float, q: float, gamma: float, sigma: float,
                      tau: float) -> float:
    """
    Compute the finite-horizon reservation/indifference price.

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    S : float
        Current mid-price.
    q : float
        Current inventory.
    gamma : float
        Risk aversion coefficient.
    sigma : float
        Mid-price volatility.
    tau : float
        Time to maturity (T - t).

    Returns
    -------
    float
        Reservation price.
    """
    return S - q * gamma * sigma**2 * tau


def spread_adjustment(gamma: float, k: float) -> float:
    """
    Compute the execution-intensity-dependent spread adjustment.

    spread_adj = (1/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    gamma : float
        Risk aversion coefficient.
    k : float
        Intensity decay parameter.

    Returns
    -------
    float
        Spread adjustment term.
    """
    return (1.0 / gamma) * np.log(1.0 + gamma / k)


def total_spread(gamma: float, sigma: float, tau: float, k: float) -> float:
    """
    Compute the total bid-ask spread Delta_t.

    Delta_t = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    gamma : float
        Risk aversion coefficient.
    sigma : float
        Mid-price volatility.
    tau : float
        Time to maturity (T - t).
    k : float
        Intensity decay parameter.

    Returns
    -------
    float
        Total spread.
    """
    return gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)


def execution_intensity(delta: float, A: float, k: float) -> float:
    """
    Compute the exponential execution intensity.

    lambda(delta) = A * exp(-k * delta)

    Parameters
    ----------
    delta : float
        Quote distance from mid-price.
    A : float
        Intensity scale parameter.
    k : float
        Intensity decay parameter.

    Returns
    -------
    float
        Execution intensity.
    """
    return A * np.exp(-k * delta)


# ---------------------------------------------------------------------------
# Single-path simulation
# ---------------------------------------------------------------------------

def simulate_path(
    gamma: float,
    strategy: str,
    params: Dict[str, Any],
    rng: np.random.Generator,
    store_trajectory: bool = False,
) -> Dict[str, Any]:
    """
    Simulate a single market-making path under the specified strategy.

    Parameters
    ----------
    gamma : float
        Risk aversion coefficient.
    strategy : str
        'inventory' or 'symmetric'.
    params : dict
        Simulation parameters dictionary.
    rng : np.random.Generator
        NumPy random generator for reproducibility.
    store_trajectory : bool
        If True, store full time-series trajectories.

    Returns
    -------
    dict
        Dictionary with terminal profit, final inventory, and optional trajectories.
    """
    S0 = params['S0']
    T = params['T']
    sigma = params['sigma']
    dt = params['dt']
    N = params['N']
    q0 = params['q0']
    X0 = params['X0']
    A = params['A']
    k = params['k']

    S = float(S0)
    q = float(q0)
    X = float(X0)

    if store_trajectory:
        S_traj = np.zeros(N + 1)
        r_traj = np.zeros(N + 1)
        p_a_traj = np.zeros(N + 1)
        p_b_traj = np.zeros(N + 1)
        q_traj = np.zeros(N + 1)
        S_traj[0] = S
        r_traj[0] = S  # at t=0, q=0, r=S
        p_a_traj[0] = S
        p_b_traj[0] = S
        q_traj[0] = q

    # Pre-generate random numbers for efficiency
    # Mid-price steps: +1 or -1 with equal probability
    price_steps = rng.choice([-1, 1], size=N).astype(float)
    # Execution events: uniform draws for Bernoulli sampling
    u_ask = rng.uniform(0, 1, size=N)
    u_bid = rng.uniform(0, 1, size=N)

    for n in range(N):
        t = n * dt
        tau = T - t

        # Compute spread
        Delta_t = total_spread(gamma, sigma, tau, k)

        if strategy == 'inventory':
            # Inventory-based strategy
            r = reservation_price(S, q, gamma, sigma, tau)
            p_a = r + Delta_t / 2.0
            p_b = r - Delta_t / 2.0
        else:
            # Symmetric benchmark: centered at mid-price
            p_a = S + Delta_t / 2.0
            p_b = S - Delta_t / 2.0

        # Quote distances from mid-price
        delta_a = p_a - S
        delta_b = S - p_b

        # Execution intensities
        lam_a = execution_intensity(delta_a, A, k)
        lam_b = execution_intensity(delta_b, A, k)

        # Execution probabilities (Bernoulli per side, as documented)
        prob_a = min(max(lam_a * dt, 0.0), 1.0)
        prob_b = min(max(lam_b * dt, 0.0), 1.0)

        # Simulate fills independently
        ask_fill = u_ask[n] < prob_a
        bid_fill = u_bid[n] < prob_b

        # Update state: ask fill -> sell one share
        if ask_fill:
            q -= 1.0
            X += p_a

        # Update state: bid fill -> buy one share
        if bid_fill:
            q += 1.0
            X -= p_b

        # Update mid-price: binomial Brownian step (paper specification)
        S += price_steps[n] * sigma * np.sqrt(dt)

        if store_trajectory:
            S_traj[n + 1] = S
            r_traj[n + 1] = reservation_price(S, q, gamma, sigma, T - (n + 1) * dt)
            p_a_traj[n + 1] = S + total_spread(gamma, sigma, T - (n + 1) * dt, k) / 2.0
            p_b_traj[n + 1] = S - total_spread(gamma, sigma, T - (n + 1) * dt, k) / 2.0
            q_traj[n + 1] = q

    # Terminal profit
    Pi_T = X + q * S

    result = {
        'Pi_T': Pi_T,
        'q_T': q,
        'X_T': X,
        'S_T': S,
    }

    if store_trajectory:
        result['S_traj'] = S_traj
        result['r_traj'] = r_traj
        result['p_a_traj'] = p_a_traj
        result['p_b_traj'] = p_b_traj
        result['q_traj'] = q_traj

    return result


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    gamma: float,
    strategy: str,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Run Monte Carlo simulation for a given gamma and strategy.

    Parameters
    ----------
    gamma : float
        Risk aversion coefficient.
    strategy : str
        'inventory' or 'symmetric'.
    params : dict
        Simulation parameters.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    dict
        Arrays of terminal profits and final inventories.
    """
    n_paths = params['n_paths']
    profits = np.zeros(n_paths)
    inventories = np.zeros(n_paths)

    for i in range(n_paths):
        result = simulate_path(gamma, strategy, params, rng)
        profits[i] = result['Pi_T']
        inventories[i] = result['q_T']

    return {'profits': profits, 'inventories': inventories}


def compute_summary_stats(
    profits: np.ndarray,
    inventories: np.ndarray,
    gamma: float,
    params: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute summary statistics for a set of simulated paths.

    Uses sample standard deviation (ddof=1) as documented.

    Parameters
    ----------
    profits : np.ndarray
        Array of terminal profits.
    inventories : np.ndarray
        Array of final inventories.
    gamma : float
        Risk aversion coefficient.
    params : dict
        Simulation parameters.

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    k = params['k']
    # Static spread component matching paper's table values
    spread_static = (2.0 / gamma) * np.log(1.0 + gamma / k)

    return {
        'spread': spread_static,
        'mean_profit': float(np.mean(profits)),
        'std_profit': float(np.std(profits, ddof=1)),
        'mean_q': float(np.mean(inventories)),
        'std_q': float(np.std(inventories, ddof=1)),
    }


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------

def run_diagnostics(params: Dict[str, Any]) -> None:
    """
    Run diagnostic checks to verify model properties.

    Checks:
    - r(S, 0, t) = S
    - q > 0 implies r < S
    - q < 0 implies r > S
    - r approaches S as t approaches T
    - lambda decreases with distance

    Parameters
    ----------
    params : dict
        Simulation parameters.
    """
    S = params['S0']
    sigma = params['sigma']
    gamma = 0.1
    k = params['k']
    A = params['A']
    T = params['T']

    # r(S, 0, t) = S
    r_q0 = reservation_price(S, 0, gamma, sigma, T)
    assert abs(r_q0 - S) < 1e-10, f"r(S,0,t) should equal S, got {r_q0}"

    # q > 0 implies r < S
    r_pos = reservation_price(S, 2, gamma, sigma, T)
    assert r_pos < S, f"r should be < S when q > 0, got r={r_pos}, S={S}"

    # q < 0 implies r > S
    r_neg = reservation_price(S, -2, gamma, sigma, T)
    assert r_neg > S, f"r should be > S when q < 0, got r={r_neg}, S={S}"

    # r approaches S as t approaches T (tau -> 0)
    r_at_T = reservation_price(S, 2, gamma, sigma, 0.0)
    assert abs(r_at_T - S) < 1e-10, f"r should approach S as t->T, got {r_at_T}"

    # lambda decreases with distance
    lam1 = execution_intensity(0.5, A, k)
    lam2 = execution_intensity(1.0, A, k)
    assert lam1 > lam2, "lambda should decrease with distance"

    print("  All diagnostic checks passed.")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_illustrative_path(
    gamma: float,
    params: Dict[str, Any],
    rng: np.random.Generator,
    output_dir: str,
) -> None:
    """
    Plot an illustrative single path showing mid-price, reservation price,
    and bid/ask quotes for the inventory strategy.

    Parameters
    ----------
    gamma : float
        Risk aversion coefficient.
    params : dict
        Simulation parameters.
    rng : np.random.Generator
        NumPy random generator.
    output_dir : str
        Directory to save the plot.
    """
    result = simulate_path(gamma, 'inventory', params, rng, store_trajectory=True)

    N = params['N']
    dt = params['dt']
    T = params['T']
    time_grid = np.linspace(0, T, N + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Price trajectories
    ax1 = axes[0]
    ax1.plot(time_grid, result['S_traj'], 'k-', linewidth=1.5, label='Mid-price $S_t$')
    ax1.plot(time_grid, result['r_traj'], 'b--', linewidth=1.5, label='Reservation price $r_t$')
    ax1.plot(time_grid, result['p_a_traj'], 'r-', linewidth=1.0, alpha=0.7, label='Ask quote $p^a_t$')
    ax1.plot(time_grid, result['p_b_traj'], 'g-', linewidth=1.0, alpha=0.7, label='Bid quote $p^b_t$')
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Illustrative Path — Inventory Strategy ($\\gamma={gamma}$)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Inventory trajectory
    ax2 = axes[1]
    ax2.plot(time_grid, result['q_traj'], 'purple', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Inventory $q_t$')
    ax2.set_title(f'Inventory Trajectory ($\\gamma={gamma}$)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, f'exp2_illustrative_path_gamma{gamma}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_profit_histograms(
    results: Dict[float, Dict[str, Dict[str, np.ndarray]]],
    output_dir: str,
) -> None:
    """
    Plot overlaid profit histograms comparing inventory vs symmetric strategies
    for each gamma value.

    Parameters
    ----------
    results : dict
        Nested dict: results[gamma][strategy]['profits'].
    output_dir : str
        Directory to save the plot.
    """
    gamma_values = sorted(results.keys())
    n_plots = len(gamma_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), squeeze=False)
    axes = axes[0]  # flatten to 1D array

    for ax, gamma in zip(axes, gamma_values):
        inv_profits = results[gamma]['inventory']['profits']
        sym_profits = results[gamma]['symmetric']['profits']

        ax.hist(inv_profits, bins=50, alpha=0.6, color='steelblue',
                label='Inventory', density=True)
        ax.hist(sym_profits, bins=50, alpha=0.6, color='tomato',
                label='Symmetric', density=True)

        ax.axvline(np.mean(inv_profits), color='steelblue', linestyle='--',
                   linewidth=2, label=f'Mean inv: {np.mean(inv_profits):.1f}')
        ax.axvline(np.mean(sym_profits), color='tomato', linestyle='--',
                   linewidth=2, label=f'Mean sym: {np.mean(sym_profits):.1f}')

        ax.set_xlabel('Terminal Profit $\\Pi_T$')
        ax.set_ylabel('Density')
        ax.set_title(f'$\\gamma = {gamma}$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Terminal Profit Distributions: Inventory vs Symmetric Strategy',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fname = os.path.join(output_dir, 'exp2_profit_histograms.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_summary_comparison(
    summary_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Plot bar charts comparing key statistics between strategies.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics DataFrame.
    output_dir : str
        Directory to save the plot.
    """
    metrics = ['mean_profit', 'std_profit', 'std_q']
    titles = ['Mean Terminal Profit', 'Std Dev of Profit', 'Std Dev of Final Inventory']
    colors = {'inventory': 'steelblue', 'symmetric': 'tomato'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        for strategy in ['inventory', 'symmetric']:
            subset = summary_df[summary_df['strategy'] == strategy]
            ax.bar(
                [str(g) for g in subset['gamma']],
                subset[metric],
                alpha=0.7,
                label=strategy.capitalize(),
                color=colors[strategy],
                width=0.35,
            )
        ax.set_xlabel('$\\gamma$')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Strategy Comparison by Risk Aversion', fontsize=13)
    plt.tight_layout()
    fname = os.path.join(output_dir, 'exp2_strategy_comparison.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment_2(
    params: Dict[str, Any] = None,
    output_dir: str = 'results',
) -> Dict[str, Any]:
    """
    Run the full Experiment 2: Monte Carlo replication of finite-horizon
    inventory-based versus symmetric market-making strategies.

    Parameters
    ----------
    params : dict, optional
        Simulation parameters. Defaults to PARAMS.
    output_dir : str
        Directory to save results and plots.

    Returns
    -------
    dict
        Dictionary containing all simulation results and summary statistics.
    """
    if params is None:
        params = PARAMS

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 2: Monte Carlo Replication of AS Market-Making Strategies")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  S0={params['S0']}, T={params['T']}, sigma={params['sigma']}")
    print(f"  dt={params['dt']}, N={params['N']}, n_paths={params['n_paths']}")
    print(f"  A={params['A']}, k={params['k']}")
    print(f"  gamma values: {params['gamma_values']}")
    print(f"  Random seed: {params['random_seed']}")

    # Diagnostics
    print("\n--- Diagnostic Checks ---")
    run_diagnostics(params)

    # Initialize random generator with documented seed
    rng = np.random.default_rng(params['random_seed'])

    all_results = {}
    summary_rows = []

    for gamma in params['gamma_values']:
        print(f"\n--- Simulating gamma = {gamma} ---")
        all_results[gamma] = {}

        for strategy in ['inventory', 'symmetric']:
            print(f"  Running {strategy} strategy ({params['n_paths']} paths)...")

            # Use separate RNG state per (gamma, strategy) for independence
            strategy_rng = np.random.default_rng(
                params['random_seed'] + int(gamma * 1000) + (0 if strategy == 'inventory' else 500)
            )

            mc_results = run_monte_carlo(gamma, strategy, params, strategy_rng)
            all_results[gamma][strategy] = mc_results

            stats = compute_summary_stats(
                mc_results['profits'],
                mc_results['inventories'],
                gamma,
                params,
            )

            print(f"    Spread (static): {stats['spread']:.4f}")
            print(f"    Mean profit:     {stats['mean_profit']:.4f}")
            print(f"    Std profit:      {stats['std_profit']:.4f}")
            print(f"    Mean final q:    {stats['mean_q']:.4f}")
            print(f"    Std final q:     {stats['std_q']:.4f}")

            summary_rows.append({
                'gamma': gamma,
                'strategy': strategy,
                'spread': stats['spread'],
                'mean_profit': stats['mean_profit'],
                'std_profit': stats['std_profit'],
                'mean_q': stats['mean_q'],
                'std_q': stats['std_q'],
            })

    # Build summary DataFrame
    summary_df = pd.DataFrame(summary_rows)

    print("\n--- Summary Table ---")
    print(summary_df.to_string(index=False, float_format='{:.4f}'.format))

    # Save summary CSV
    csv_path = os.path.join(output_dir, 'exp2_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV: {csv_path}")

    # Qualitative comparison against paper targets
    print("\n--- Qualitative Comparison with Paper Targets ---")
    for gamma in params['gamma_values']:
        print(f"\n  gamma = {gamma}:")
        for strategy in ['inventory', 'symmetric']:
            row = summary_df[
                (summary_df['gamma'] == gamma) & (summary_df['strategy'] == strategy)
            ].iloc[0]
            target = PAPER_TARGETS[gamma][strategy]
            print(f"    {strategy.capitalize()}:")
            print(f"      Spread:     simulated={row['spread']:.4f}, paper={target['spread']:.2f}")
            print(f"      Profit:     simulated={row['mean_profit']:.2f}, paper={target['profit']:.2f}")
            print(f"      Std Profit: simulated={row['std_profit']:.2f}, paper={target['std_profit']:.2f}")
            print(f"      Mean q:     simulated={row['mean_q']:.4f}, paper={target['mean_q']:.3f}")
            print(f"      Std q:      simulated={row['std_q']:.4f}, paper={target['std_q']:.2f}")

    # Verify key hypotheses
    print("\n--- Hypothesis Verification ---")
    for gamma in params['gamma_values']:
        inv_row = summary_df[
            (summary_df['gamma'] == gamma) & (summary_df['strategy'] == 'inventory')
        ].iloc[0]
        sym_row = summary_df[
            (summary_df['gamma'] == gamma) & (summary_df['strategy'] == 'symmetric')
        ].iloc[0]

        h1 = inv_row['std_profit'] < sym_row['std_profit']
        h2 = inv_row['std_q'] < sym_row['std_q']
        print(f"  gamma={gamma}: inv_std_profit < sym_std_profit: {h1} "
              f"({inv_row['std_profit']:.2f} < {sym_row['std_profit']:.2f})")
        print(f"  gamma={gamma}: inv_std_q < sym_std_q: {h2} "
              f"({inv_row['std_q']:.2f} < {sym_row['std_q']:.2f})")

    # Visualizations
    print("\n--- Generating Visualizations ---")

    # Illustrative path for gamma=0.1
    plot_rng = np.random.default_rng(params['random_seed'] + 9999)
    plot_illustrative_path(0.1, params, plot_rng, output_dir)

    # Profit histograms
    plot_profit_histograms(all_results, output_dir)

    # Summary comparison
    plot_summary_comparison(summary_df, output_dir)

    print("\n" + "=" * 70)
    print("Experiment 2 completed successfully.")
    print("=" * 70)

    return {
        'params': params,
        'results': all_results,
        'summary': summary_df,
    }


if __name__ == '__main__':
    run_experiment_2()
