"""
Experiment 2: Monte Carlo replication of inventory-based versus symmetric
market-making strategies across risk aversion levels.

Reproduces the paper's main simulation experiment comparing:
- Inventory-based quoting strategy (AS model)
- Symmetric benchmark strategy (same spread, centered on mid-price)

Over 1000 Monte Carlo paths for gamma in {0.01, 0.1, 0.5}.

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
           Quantitative Finance, 8(3), 217-224.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model parameters (paper-specified)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: Dict = {
    "S0": 100.0,      # Initial mid-price
    "q0": 0.0,        # Initial inventory
    "X0": 0.0,        # Initial cash
    "T": 1.0,         # Terminal horizon
    "dt": 0.005,      # Time step
    "N": 200,         # Number of time steps (T/dt)
    "sigma": 2.0,     # Volatility
    "A": 140.0,       # Intensity scale parameter
    "k": 1.5,         # Intensity decay parameter
    "n_paths": 1000,  # Monte Carlo paths per strategy per gamma
}

GAMMA_VALUES: List[float] = [0.01, 0.1, 0.5]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def stable_correction_term(gamma: float, k: float) -> float:
    """
    Numerically stable evaluation of (1/gamma)*ln(1+gamma/k).

    Uses np.log1p for stability near gamma=0.

    Parameters
    ----------
    gamma : float
        Risk aversion parameter (> 0).
    k : float
        Intensity decay parameter (> 0).

    Returns
    -------
    float
        Value of (1/gamma)*ln(1+gamma/k).
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    return float(np.log1p(gamma / k) / gamma)


def compute_spread(tau: float, gamma: float, sigma: float, k: float) -> float:
    """
    Compute the total operational spread at time-to-maturity tau.

    spread = gamma*sigma^2*tau + (2/gamma)*ln(1+gamma/k)

    Parameters
    ----------
    tau : float
        Time to maturity (T - t).
    gamma : float
        Risk aversion.
    sigma : float
        Volatility.
    k : float
        Intensity decay parameter.

    Returns
    -------
    float
        Total spread.
    """
    return gamma * sigma**2 * tau + 2.0 * stable_correction_term(gamma, k)


def compute_reservation_price(
    S: np.ndarray,
    q: np.ndarray,
    tau: float,
    gamma: float,
    sigma: float,
) -> np.ndarray:
    """
    Compute the reservation/indifference price for an array of states.

    r(s,q,t) = s - q * gamma * sigma^2 * tau

    Parameters
    ----------
    S : np.ndarray
        Current mid-prices, shape (n_paths,).
    q : np.ndarray
        Current inventories, shape (n_paths,).
    tau : float
        Time to maturity.
    gamma : float
        Risk aversion.
    sigma : float
        Volatility.

    Returns
    -------
    np.ndarray
        Reservation prices, shape (n_paths,).
    """
    return S - q * gamma * sigma**2 * tau


def compute_quote_distances_inventory(
    q: np.ndarray,
    tau: float,
    gamma: float,
    sigma: float,
    k: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inventory-strategy quote distances from mid-price.

    delta^a = ((1-2q)/2)*gamma*sigma^2*tau + (1/gamma)*ln(1+gamma/k)
    delta^b = ((1+2q)/2)*gamma*sigma^2*tau + (1/gamma)*ln(1+gamma/k)

    Parameters
    ----------
    q : np.ndarray
        Current inventories, shape (n_paths,).
    tau : float
        Time to maturity.
    gamma : float
        Risk aversion.
    sigma : float
        Volatility.
    k : float
        Intensity decay parameter.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (delta_a, delta_b), each shape (n_paths,).
    """
    correction = stable_correction_term(gamma, k)
    delta_a = ((1.0 - 2.0 * q) / 2.0) * gamma * sigma**2 * tau + correction
    delta_b = ((1.0 + 2.0 * q) / 2.0) * gamma * sigma**2 * tau + correction
    return delta_a, delta_b


def compute_quote_distances_symmetric(
    tau: float,
    gamma: float,
    sigma: float,
    k: float,
) -> Tuple[float, float]:
    """
    Compute symmetric benchmark quote distances from mid-price.

    Both sides use half the total spread, centered on mid-price (not reservation price).

    delta^{a,sym} = delta^{b,sym} = spread/2

    Parameters
    ----------
    tau : float
        Time to maturity.
    gamma : float
        Risk aversion.
    sigma : float
        Volatility.
    k : float
        Intensity decay parameter.

    Returns
    -------
    Tuple[float, float]
        (delta_a_sym, delta_b_sym)
    """
    half_spread = compute_spread(tau, gamma, sigma, k) / 2.0
    return half_spread, half_spread


def compute_liquidity_spread(gamma: float, k: float) -> float:
    """
    Compute the liquidity component of the spread: (2/gamma)*ln(1+gamma/k).

    This is the paper-style displayed spread number (time-independent component).

    Parameters
    ----------
    gamma : float
        Risk aversion.
    k : float
        Intensity decay parameter.

    Returns
    -------
    float
        Liquidity spread component.
    """
    return 2.0 * stable_correction_term(gamma, k)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_paths(
    strategy: str,
    gamma: float,
    n_paths: int,
    params: Dict,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Simulate n_paths Monte Carlo paths for a given strategy and risk aversion.

    Simulation follows the paper's discrete-time approximation:
    - Price: S_{t+dt} = S_t + epsilon_t * sigma * sqrt(dt), epsilon_t in {-1, +1}
    - Fills: independent Bernoulli with probabilities lambda^a*dt and lambda^b*dt
    - Update order: observe state -> compute quotes -> simulate fills ->
                    update cash/inventory -> update price

    Parameters
    ----------
    strategy : str
        Either "inventory" or "symmetric".
    gamma : float
        Risk aversion parameter.
    n_paths : int
        Number of Monte Carlo paths.
    params : Dict
        Model parameters dictionary.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with arrays:
        - "terminal_profit": shape (n_paths,)
        - "terminal_inventory": shape (n_paths,)
        - "sample_S": shape (N+1,) for path 0
        - "sample_r": shape (N+1,) for path 0
        - "sample_pa": shape (N,) for path 0
        - "sample_pb": shape (N,) for path 0
        - "sample_q": shape (N+1,) for path 0
    """
    S0 = params["S0"]
    q0 = params["q0"]
    X0 = params["X0"]
    T = params["T"]
    dt = params["dt"]
    N = params["N"]
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]

    assert strategy in ("inventory", "symmetric"), f"Unknown strategy: {strategy}"
    assert abs(N - T / dt) < 1e-9, f"N={N} inconsistent with T/dt={T/dt}"

    sqrt_dt = np.sqrt(dt)

    # Initialize state arrays: shape (n_paths,)
    S = np.full(n_paths, S0, dtype=np.float64)
    X = np.full(n_paths, X0, dtype=np.float64)
    q = np.full(n_paths, q0, dtype=np.float64)

    # Storage for sample path (path index 0)
    sample_S = np.zeros(N + 1)
    sample_r = np.zeros(N + 1)
    sample_pa = np.zeros(N)
    sample_pb = np.zeros(N)
    sample_q = np.zeros(N + 1)

    sample_S[0] = S0
    sample_r[0] = float(compute_reservation_price(
        np.array([S0]), np.array([q0]), T, gamma, sigma
    )[0])
    sample_q[0] = q0

    fill_prob_violations = 0

    for n in range(N):
        tau = T - n * dt

        # --- Compute quote distances ---
        if strategy == "inventory":
            delta_a, delta_b = compute_quote_distances_inventory(q, tau, gamma, sigma, k)
        else:  # symmetric
            half_spread = compute_spread(tau, gamma, sigma, k) / 2.0
            delta_a = np.full(n_paths, half_spread)
            delta_b = np.full(n_paths, half_spread)

        # --- Compute ask/bid prices ---
        p_a = S + delta_a
        p_b = S - delta_b

        # --- Compute fill intensities ---
        lambda_a = A * np.exp(-k * delta_a)
        lambda_b = A * np.exp(-k * delta_b)

        # --- Convert to fill probabilities ---
        prob_a = lambda_a * dt
        prob_b = lambda_b * dt

        # Validate: probabilities must be <= 1
        if np.any(prob_a > 1.0) or np.any(prob_b > 1.0):
            fill_prob_violations += 1
            warnings.warn(
                f"Fill probability exceeds 1 at step {n} (tau={tau:.4f}). "
                f"max(prob_a)={prob_a.max():.4f}, max(prob_b)={prob_b.max():.4f}. "
                "This indicates a discretization/parameter issue.",
                RuntimeWarning,
                stacklevel=2,
            )

        # --- Simulate fills (independent Bernoulli) ---
        ask_fill = rng.random(n_paths) < prob_a  # True if ask filled
        bid_fill = rng.random(n_paths) < prob_b  # True if bid filled

        # --- Update cash and inventory ---
        # Ask fill: sell 1 unit -> q -= 1, X += p_a
        # Bid fill: buy 1 unit  -> q += 1, X -= p_b
        X += ask_fill * p_a - bid_fill * p_b
        q += bid_fill.astype(np.float64) - ask_fill.astype(np.float64)

        # --- Update mid-price (Bernoulli +/- sigma*sqrt(dt)) ---
        epsilon = rng.choice(np.array([-1.0, 1.0]), size=n_paths)
        S = S + epsilon * sigma * sqrt_dt

        # --- Store sample path data (path 0) ---
        sample_S[n + 1] = S[0]
        sample_q[n + 1] = q[0]
        tau_next = T - (n + 1) * dt
        if strategy == "inventory":
            sample_r[n + 1] = float(compute_reservation_price(
                np.array([S[0]]), np.array([q[0]]), tau_next, gamma, sigma
            )[0])
        else:
            sample_r[n + 1] = S[0]  # symmetric: center = mid-price
        sample_pa[n] = p_a[0]
        sample_pb[n] = p_b[0]

    # --- Terminal profit ---
    terminal_profit = X + q * S

    if fill_prob_violations > 0:
        warnings.warn(
            f"Total fill probability violations: {fill_prob_violations} steps.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "terminal_profit": terminal_profit,
        "terminal_inventory": q,
        "sample_S": sample_S,
        "sample_r": sample_r,
        "sample_pa": sample_pa,
        "sample_pb": sample_pb,
        "sample_q": sample_q,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_statistics(
    terminal_profit: np.ndarray,
    terminal_inventory: np.ndarray,
) -> Dict[str, float]:
    """
    Compute aggregate statistics for terminal profit and inventory.

    Parameters
    ----------
    terminal_profit : np.ndarray
        Terminal marked-to-market profit for each path.
    terminal_inventory : np.ndarray
        Terminal inventory for each path.

    Returns
    -------
    Dict[str, float]
        Dictionary with mean/std of profit and inventory.
    """
    return {
        "mean_profit": float(np.mean(terminal_profit)),
        "std_profit": float(np.std(terminal_profit, ddof=1)),
        "mean_final_q": float(np.mean(terminal_inventory)),
        "std_final_q": float(np.std(terminal_inventory, ddof=1)),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment_2(
    gamma_values: Optional[List[float]] = None,
    params: Optional[Dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run the full Experiment 2: Monte Carlo comparison of inventory vs symmetric strategies.

    Parameters
    ----------
    gamma_values : Optional[List[float]]
        Risk aversion values to test. Defaults to [0.01, 0.1, 0.5].
    params : Optional[Dict]
        Model parameters. Defaults to DEFAULT_PARAMS.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress and results.

    Returns
    -------
    Dict
        Dictionary containing simulation results, statistics, and sample paths.
    """
    if gamma_values is None:
        gamma_values = GAMMA_VALUES
    if params is None:
        params = DEFAULT_PARAMS.copy()

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 70)
        print("EXPERIMENT 2: Monte Carlo Market-Making Simulation")
        print(f"  Paths per config: {params['n_paths']}")
        print(f"  Gamma values: {gamma_values}")
        print(f"  Seed: {seed}")
        print("=" * 70)

    all_results: Dict = {}
    summary_rows: List[Dict] = []

    for gamma in gamma_values:
        if verbose:
            print(f"\n--- gamma = {gamma} ---")

        liq_spread = compute_liquidity_spread(gamma, params["k"])

        for strategy in ["inventory", "symmetric"]:
            if verbose:
                print(f"  Simulating {strategy} strategy...", end=" ", flush=True)

            sim = simulate_paths(
                strategy=strategy,
                gamma=gamma,
                n_paths=params["n_paths"],
                params=params,
                rng=rng,
            )

            stats = compute_statistics(
                sim["terminal_profit"],
                sim["terminal_inventory"],
            )

            key = (gamma, strategy)
            all_results[key] = {**sim, **stats, "gamma": gamma, "strategy": strategy}

            summary_rows.append({
                "gamma": gamma,
                "strategy": strategy,
                "liquidity_spread": liq_spread,
                "mean_profit": stats["mean_profit"],
                "std_profit": stats["std_profit"],
                "mean_final_q": stats["mean_final_q"],
                "std_final_q": stats["std_final_q"],
            })

            if verbose:
                print(
                    f"mean_profit={stats['mean_profit']:.2f}, "
                    f"std_profit={stats['std_profit']:.2f}, "
                    f"mean_q={stats['mean_final_q']:.3f}, "
                    f"std_q={stats['std_final_q']:.3f}"
                )

    summary_df = pd.DataFrame(summary_rows)
    all_results["summary"] = summary_df

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return all_results


if __name__ == "__main__":
    results = run_experiment_2(verbose=True)
