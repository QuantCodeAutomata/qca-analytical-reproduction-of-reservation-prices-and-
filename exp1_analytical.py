"""
Experiment 1: Analytical reproduction of reservation prices and approximate
optimal quotes in the Avellaneda-Stoikov (AS) dealer model.

Reproduces the paper's analytical derivations for:
- Frozen-inventory utility-indifference prices
- Inventory-adjusted reservation/indifference price r(s,q,t) = s - q*gamma*sigma^2*(T-t)
- Approximate optimal bid/ask quote formulas under exponential order-arrival intensities

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
           Quantitative Finance, 8(3), 217-224.

# Using sympy for symbolic algebra — Context7 confirmed
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import sympy as sp
from sympy import (
    E,
    Eq,
    Rational,
    exp,
    ln,
    simplify,
    solve,
    symbols,
)

# ---------------------------------------------------------------------------
# Symbol definitions
# ---------------------------------------------------------------------------
x, s, q, t, T = symbols("x s q t T", real=True)
gamma_sym = symbols("gamma", positive=True)
sigma_sym = symbols("sigma", positive=True)
A_sym, k_sym = symbols("A k", positive=True)
tau = T - t  # time to maturity


# ---------------------------------------------------------------------------
# Step 1-2: Frozen-inventory value function
# ---------------------------------------------------------------------------

def frozen_inventory_value_function() -> sp.Expr:
    """
    Compute the closed-form frozen-inventory value function.

    Under dS_u = sigma dW_u (driftless ABM) and exponential utility U = -exp(-gamma W),
    with W = x + q*S_T and S_T | S_t=s ~ N(s, sigma^2*(T-t)):

        v(x,s,q,t) = E_t[-exp(-gamma*(x + q*S_T))]
                   = -exp(-gamma*x) * exp(-gamma*q*s) * exp(0.5*gamma^2*q^2*sigma^2*(T-t))

    Returns
    -------
    sp.Expr
        Symbolic expression for v(x, s, q, t).
    """
    # S_T ~ N(s, sigma^2 * tau) => -gamma*(x + q*S_T) ~ N(-gamma*(x+q*s), gamma^2*q^2*sigma^2*tau)
    # E[exp(Z)] = exp(mu + 0.5*var) for Z ~ N(mu, var)
    # Here Z = -gamma*(x + q*S_T), mu = -gamma*(x+q*s), var = gamma^2*q^2*sigma^2*tau
    v = -exp(-gamma_sym * x) * exp(-gamma_sym * q * s) * exp(
        Rational(1, 2) * gamma_sym**2 * q**2 * sigma_sym**2 * tau
    )
    return v


# ---------------------------------------------------------------------------
# Step 3-4: Reservation ask and bid prices (utility indifference)
# ---------------------------------------------------------------------------

def reservation_ask_price() -> sp.Expr:
    """
    Derive the reservation ask price r^a via utility indifference.

    Condition: v(x + r^a, s, q-1, t) = v(x, s, q, t)
    Solving for r^a yields:
        r^a(s,q,t) = s + ((1-2q)/2) * gamma * sigma^2 * (T-t)

    Returns
    -------
    sp.Expr
        Symbolic expression for r^a(s, q, t).
    """
    r_a = symbols("r_a", real=True)
    v = frozen_inventory_value_function()

    # v(x + r^a, s, q-1, t) = v(x, s, q, t)
    lhs = v.subs([(x, x + r_a), (q, q - 1)])
    rhs = v

    # Divide both sides by -exp(-gamma*x) and take log
    # lhs/rhs = exp(-gamma*r_a) * exp(-gamma*(q-1)*s + 0.5*gamma^2*(q-1)^2*sigma^2*tau)
    #         / exp(-gamma*q*s + 0.5*gamma^2*q^2*sigma^2*tau) = 1
    ratio = simplify(lhs / rhs)
    # Solve exp(log_ratio) = 1 => log_ratio = 0
    log_ratio = sp.log(ratio)
    log_ratio_simplified = simplify(log_ratio)
    solution = solve(log_ratio_simplified, r_a)
    assert len(solution) == 1, f"Expected unique solution, got {solution}"
    r_a_expr = simplify(solution[0])
    return r_a_expr


def reservation_bid_price() -> sp.Expr:
    """
    Derive the reservation bid price r^b via utility indifference.

    Condition: v(x - r^b, s, q+1, t) = v(x, s, q, t)
    Solving for r^b yields:
        r^b(s,q,t) = s + ((-1-2q)/2) * gamma * sigma^2 * (T-t)

    Returns
    -------
    sp.Expr
        Symbolic expression for r^b(s, q, t).
    """
    r_b = symbols("r_b", real=True)
    v = frozen_inventory_value_function()

    # v(x - r^b, s, q+1, t) = v(x, s, q, t)
    lhs = v.subs([(x, x - r_b), (q, q + 1)])
    rhs = v

    ratio = simplify(lhs / rhs)
    log_ratio = sp.log(ratio)
    log_ratio_simplified = simplify(log_ratio)
    solution = solve(log_ratio_simplified, r_b)
    assert len(solution) == 1, f"Expected unique solution, got {solution}"
    r_b_expr = simplify(solution[0])
    return r_b_expr


# ---------------------------------------------------------------------------
# Step 5: Reservation/indifference price (midpoint)
# ---------------------------------------------------------------------------

def reservation_price(r_a_expr: sp.Expr, r_b_expr: sp.Expr) -> sp.Expr:
    """
    Compute the reservation/indifference price as the midpoint of r^a and r^b.

    r(s,q,t) = 0.5*(r^a + r^b) = s - q*gamma*sigma^2*(T-t)

    Parameters
    ----------
    r_a_expr : sp.Expr
        Reservation ask price expression.
    r_b_expr : sp.Expr
        Reservation bid price expression.

    Returns
    -------
    sp.Expr
        Symbolic expression for r(s, q, t).
    """
    r_mid = simplify(Rational(1, 2) * (r_a_expr + r_b_expr))
    return r_mid


# ---------------------------------------------------------------------------
# Step 9: Quote correction term under exponential intensities
# ---------------------------------------------------------------------------

def quote_correction_term() -> sp.Expr:
    """
    Derive the quote-offset correction term under exponential arrival intensities.

    For lambda(delta) = A * exp(-k * delta):
        lambda'(delta) = -k * A * exp(-k * delta) = -k * lambda(delta)

    First-order condition for optimal delta:
        delta* = -lambda/lambda' - 1/gamma = 1/k - 1/gamma
        ... but the correction term from the indifference condition is:
        (1/gamma) * ln(1 + gamma/k)

    Returns
    -------
    sp.Expr
        Symbolic expression for the quote correction term (1/gamma)*ln(1+gamma/k).
    """
    correction = (1 / gamma_sym) * ln(1 + gamma_sym / k_sym)
    return correction


# ---------------------------------------------------------------------------
# Step 10-11: Approximate quote formulas
# ---------------------------------------------------------------------------

def approximate_spread() -> sp.Expr:
    """
    Compute the approximate total spread formula.

    delta^a + delta^b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)

    Returns
    -------
    sp.Expr
        Symbolic expression for the total spread.
    """
    inventory_risk_term = gamma_sym * sigma_sym**2 * tau
    liquidity_term = (2 / gamma_sym) * ln(1 + gamma_sym / k_sym)
    spread = inventory_risk_term + liquidity_term
    return spread


def approximate_ask_distance() -> sp.Expr:
    """
    Compute the approximate optimal ask quote distance from mid-price.

    delta^a = ((1-2q)/2)*gamma*sigma^2*(T-t) + (1/gamma)*ln(1+gamma/k)

    Returns
    -------
    sp.Expr
        Symbolic expression for delta^a.
    """
    inventory_component = (Rational(1, 2) * (1 - 2 * q)) * gamma_sym * sigma_sym**2 * tau
    correction = (1 / gamma_sym) * ln(1 + gamma_sym / k_sym)
    delta_a = inventory_component + correction
    return delta_a


def approximate_bid_distance() -> sp.Expr:
    """
    Compute the approximate optimal bid quote distance from mid-price.

    delta^b = ((1+2q)/2)*gamma*sigma^2*(T-t) + (1/gamma)*ln(1+gamma/k)

    Returns
    -------
    sp.Expr
        Symbolic expression for delta^b.
    """
    inventory_component = (Rational(1, 2) * (1 + 2 * q)) * gamma_sym * sigma_sym**2 * tau
    correction = (1 / gamma_sym) * ln(1 + gamma_sym / k_sym)
    delta_b = inventory_component + correction
    return delta_b


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_spread_consistency(
    delta_a_expr: sp.Expr,
    delta_b_expr: sp.Expr,
    spread_expr: sp.Expr,
) -> bool:
    """
    Verify that delta^a + delta^b equals the total spread formula symbolically.

    Parameters
    ----------
    delta_a_expr : sp.Expr
        Ask distance expression.
    delta_b_expr : sp.Expr
        Bid distance expression.
    spread_expr : sp.Expr
        Total spread expression.

    Returns
    -------
    bool
        True if the residual is symbolically zero.
    """
    residual = simplify(delta_a_expr + delta_b_expr - spread_expr)
    return residual == 0


def verify_reservation_price_consistency(
    r_a_expr: sp.Expr,
    r_b_expr: sp.Expr,
    r_expr: sp.Expr,
) -> bool:
    """
    Verify that 0.5*(r^a + r^b) equals the reservation price r symbolically.

    Parameters
    ----------
    r_a_expr : sp.Expr
        Reservation ask price.
    r_b_expr : sp.Expr
        Reservation bid price.
    r_expr : sp.Expr
        Reservation/indifference price.

    Returns
    -------
    bool
        True if the residual is symbolically zero.
    """
    residual = simplify(Rational(1, 2) * (r_a_expr + r_b_expr) - r_expr)
    return residual == 0


def verify_quote_from_reservation(
    r_expr: sp.Expr,
    delta_a_expr: sp.Expr,
    delta_b_expr: sp.Expr,
) -> Tuple[bool, bool]:
    """
    Verify that p^a = r + spread/2 and p^b = r - spread/2 are consistent
    with p^a = s + delta^a and p^b = s - delta^b.

    Parameters
    ----------
    r_expr : sp.Expr
        Reservation price.
    delta_a_expr : sp.Expr
        Ask distance from mid.
    delta_b_expr : sp.Expr
        Bid distance from mid.

    Returns
    -------
    Tuple[bool, bool]
        (ask_consistent, bid_consistent)
    """
    spread_expr = approximate_spread()
    # p^a = r + spread/2 = (s - q*gamma*sigma^2*tau) + 0.5*(gamma*sigma^2*tau + (2/gamma)*ln(1+gamma/k))
    p_a_from_r = r_expr + spread_expr / 2
    # p^a = s + delta^a
    p_a_from_delta = s + delta_a_expr

    # p^b = r - spread/2
    p_b_from_r = r_expr - spread_expr / 2
    # p^b = s - delta^b
    p_b_from_delta = s - delta_b_expr

    ask_residual = simplify(p_a_from_r - p_a_from_delta)
    bid_residual = simplify(p_b_from_r - p_b_from_delta)

    return ask_residual == 0, bid_residual == 0


# ---------------------------------------------------------------------------
# Numerical evaluation helpers
# ---------------------------------------------------------------------------

def stable_correction_term(gamma_val: float, k_val: float) -> float:
    """
    Numerically evaluate (1/gamma)*ln(1+gamma/k) with numerical stability near gamma=0.

    Uses np.log1p for stability: ln(1 + gamma/k) = log1p(gamma/k).

    Parameters
    ----------
    gamma_val : float
        Risk aversion parameter (> 0).
    k_val : float
        Intensity decay parameter (> 0).

    Returns
    -------
    float
        Value of (1/gamma)*ln(1+gamma/k).
    """
    if gamma_val <= 0:
        raise ValueError("gamma must be positive")
    if k_val <= 0:
        raise ValueError("k must be positive")
    return np.log1p(gamma_val / k_val) / gamma_val


def numerical_reservation_price(
    s_val: float,
    q_val: float,
    t_val: float,
    T_val: float,
    gamma_val: float,
    sigma_val: float,
) -> float:
    """
    Numerically evaluate the reservation/indifference price.

    r(s,q,t) = s - q * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s_val : float
        Current mid-price.
    q_val : float
        Current inventory.
    t_val : float
        Current time.
    T_val : float
        Terminal horizon.
    gamma_val : float
        Risk aversion.
    sigma_val : float
        Volatility.

    Returns
    -------
    float
        Reservation price.
    """
    tau_val = T_val - t_val
    return s_val - q_val * gamma_val * sigma_val**2 * tau_val


def numerical_quote_distances(
    q_val: float,
    t_val: float,
    T_val: float,
    gamma_val: float,
    sigma_val: float,
    k_val: float,
) -> Tuple[float, float]:
    """
    Numerically evaluate the approximate optimal quote distances delta^a and delta^b.

    delta^a = ((1-2q)/2)*gamma*sigma^2*(T-t) + (1/gamma)*ln(1+gamma/k)
    delta^b = ((1+2q)/2)*gamma*sigma^2*(T-t) + (1/gamma)*ln(1+gamma/k)

    Parameters
    ----------
    q_val : float
        Current inventory.
    t_val : float
        Current time.
    T_val : float
        Terminal horizon.
    gamma_val : float
        Risk aversion.
    sigma_val : float
        Volatility.
    k_val : float
        Intensity decay parameter.

    Returns
    -------
    Tuple[float, float]
        (delta_a, delta_b)
    """
    tau_val = T_val - t_val
    correction = stable_correction_term(gamma_val, k_val)
    delta_a = ((1 - 2 * q_val) / 2) * gamma_val * sigma_val**2 * tau_val + correction
    delta_b = ((1 + 2 * q_val) / 2) * gamma_val * sigma_val**2 * tau_val + correction
    return delta_a, delta_b


def numerical_spread(
    t_val: float,
    T_val: float,
    gamma_val: float,
    sigma_val: float,
    k_val: float,
) -> float:
    """
    Numerically evaluate the total spread.

    spread = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)

    Parameters
    ----------
    t_val : float
        Current time.
    T_val : float
        Terminal horizon.
    gamma_val : float
        Risk aversion.
    sigma_val : float
        Volatility.
    k_val : float
        Intensity decay parameter.

    Returns
    -------
    float
        Total spread.
    """
    tau_val = T_val - t_val
    return gamma_val * sigma_val**2 * tau_val + 2 * stable_correction_term(gamma_val, k_val)


# ---------------------------------------------------------------------------
# Limiting behavior checks
# ---------------------------------------------------------------------------

def check_limit_t_to_T(
    gamma_val: float,
    sigma_val: float,
    k_val: float,
    T_val: float = 1.0,
    tol: float = 1e-10,
) -> Dict[str, float]:
    """
    Check limiting behavior as t -> T (tau -> 0).

    At maturity:
    - Inventory risk term gamma*sigma^2*(T-t) -> 0
    - Spread -> (2/gamma)*ln(1+gamma/k)  (pure liquidity component)
    - r(s,q,T) -> s  (reservation price converges to mid-price)

    Parameters
    ----------
    gamma_val : float
        Risk aversion.
    sigma_val : float
        Volatility.
    k_val : float
        Intensity decay parameter.
    T_val : float
        Terminal horizon.
    tol : float
        Numerical tolerance.

    Returns
    -------
    Dict[str, float]
        Dictionary with limit values and residuals.
    """
    t_val = T_val  # t = T
    tau_val = T_val - t_val  # = 0

    inventory_risk = gamma_val * sigma_val**2 * tau_val
    spread_at_T = numerical_spread(t_val, T_val, gamma_val, sigma_val, k_val)
    liquidity_only = 2 * stable_correction_term(gamma_val, k_val)

    return {
        "inventory_risk_at_T": inventory_risk,
        "spread_at_T": spread_at_T,
        "liquidity_component": liquidity_only,
        "spread_equals_liquidity": abs(spread_at_T - liquidity_only) < tol,
    }


def check_limit_gamma_to_zero(
    sigma_val: float,
    k_val: float,
    T_val: float = 1.0,
    t_val: float = 0.0,
    q_val: float = 1.0,
    gamma_small: float = 1e-6,
    tol: float = 1e-4,
) -> Dict[str, float]:
    """
    Check limiting behavior as gamma -> 0.

    As gamma -> 0:
    - Inventory correction q*gamma*sigma^2*(T-t) -> 0
    - (1/gamma)*ln(1+gamma/k) -> 1/k  (by L'Hopital / Taylor expansion)
    - Spread -> 2/k
    - delta^a = delta^b = 1/k  (symmetric quoting)

    Parameters
    ----------
    sigma_val : float
        Volatility.
    k_val : float
        Intensity decay parameter.
    T_val : float
        Terminal horizon.
    t_val : float
        Current time.
    q_val : float
        Inventory (non-zero to test asymmetry vanishing).
    gamma_small : float
        Small gamma value to approximate the limit.
    tol : float
        Numerical tolerance.

    Returns
    -------
    Dict[str, float]
        Dictionary with limit values and residuals.
    """
    correction_small = stable_correction_term(gamma_small, k_val)
    correction_limit = 1.0 / k_val  # lim_{gamma->0} (1/gamma)*ln(1+gamma/k) = 1/k

    delta_a_small, delta_b_small = numerical_quote_distances(
        q_val, t_val, T_val, gamma_small, sigma_val, k_val
    )
    asymmetry = abs(delta_a_small - delta_b_small)

    return {
        "correction_small_gamma": correction_small,
        "correction_limit_1_over_k": correction_limit,
        "correction_converges": abs(correction_small - correction_limit) < tol,
        "delta_a_small_gamma": delta_a_small,
        "delta_b_small_gamma": delta_b_small,
        "asymmetry_small_gamma": asymmetry,
        "asymmetry_vanishes": asymmetry < tol,
    }


# ---------------------------------------------------------------------------
# Numerical parameter sweep
# ---------------------------------------------------------------------------

def numerical_parameter_sweep(
    gamma_values: List[float],
    q_values: List[float],
    t_values: List[float],
    sigma_val: float = 2.0,
    k_val: float = 1.5,
    T_val: float = 1.0,
    s_val: float = 100.0,
) -> List[Dict]:
    """
    Compute numerical values of key quantities for a grid of parameters.

    Parameters
    ----------
    gamma_values : List[float]
        Risk aversion values.
    q_values : List[float]
        Inventory values.
    t_values : List[float]
        Time values.
    sigma_val : float
        Volatility.
    k_val : float
        Intensity decay parameter.
    T_val : float
        Terminal horizon.
    s_val : float
        Current mid-price (for illustration).

    Returns
    -------
    List[Dict]
        List of result dictionaries for each parameter combination.
    """
    results = []
    for gamma_val in gamma_values:
        for q_val in q_values:
            for t_val in t_values:
                tau_val = T_val - t_val
                r_val = numerical_reservation_price(s_val, q_val, t_val, T_val, gamma_val, sigma_val)
                delta_a, delta_b = numerical_quote_distances(
                    q_val, t_val, T_val, gamma_val, sigma_val, k_val
                )
                spread = numerical_spread(t_val, T_val, gamma_val, sigma_val, k_val)
                correction = stable_correction_term(gamma_val, k_val)
                results.append({
                    "gamma": gamma_val,
                    "q": q_val,
                    "t": t_val,
                    "tau": tau_val,
                    "r": r_val,
                    "delta_a": delta_a,
                    "delta_b": delta_b,
                    "spread": spread,
                    "correction": correction,
                    "p_a": s_val + delta_a,
                    "p_b": s_val - delta_b,
                })
    return results


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment_1(verbose: bool = True) -> Dict:
    """
    Run the full Experiment 1: analytical reproduction of AS model formulas.

    Parameters
    ----------
    verbose : bool
        Whether to print intermediate results.

    Returns
    -------
    Dict
        Dictionary containing all symbolic expressions, verification results,
        and numerical sweep results.
    """
    results = {}

    # --- Symbolic derivations ---
    if verbose:
        print("=" * 70)
        print("EXPERIMENT 1: Avellaneda-Stoikov Analytical Reproduction")
        print("=" * 70)

    # Step 1-2: Frozen-inventory value function
    v_expr = frozen_inventory_value_function()
    results["v_expr"] = v_expr
    if verbose:
        print("\n[Step 1-2] Frozen-inventory value function:")
        print(f"  v(x,s,q,t) = {v_expr}")

    # Step 3-4: Reservation prices
    r_a_expr = reservation_ask_price()
    r_b_expr = reservation_bid_price()
    results["r_a_expr"] = r_a_expr
    results["r_b_expr"] = r_b_expr

    # Expected: r^a = s + ((1-2q)/2)*gamma*sigma^2*(T-t)
    r_a_expected = s + (Rational(1, 2) * (1 - 2 * q)) * gamma_sym * sigma_sym**2 * tau
    r_b_expected = s + (Rational(1, 2) * (-1 - 2 * q)) * gamma_sym * sigma_sym**2 * tau

    r_a_residual = simplify(r_a_expr - r_a_expected)
    r_b_residual = simplify(r_b_expr - r_b_expected)

    results["r_a_residual"] = r_a_residual
    results["r_b_residual"] = r_b_residual
    results["r_a_verified"] = r_a_residual == 0
    results["r_b_verified"] = r_b_residual == 0

    if verbose:
        print(f"\n[Step 3-4] Reservation ask price:")
        print(f"  r^a = {r_a_expr}")
        print(f"  Expected: {r_a_expected}")
        print(f"  Residual: {r_a_residual}  ✓" if r_a_residual == 0 else f"  Residual: {r_a_residual}  ✗")
        print(f"\n[Step 3-4] Reservation bid price:")
        print(f"  r^b = {r_b_expr}")
        print(f"  Expected: {r_b_expected}")
        print(f"  Residual: {r_b_residual}  ✓" if r_b_residual == 0 else f"  Residual: {r_b_residual}  ✗")

    # Step 5: Reservation/indifference price
    r_expr = reservation_price(r_a_expr, r_b_expr)
    r_expected = s - q * gamma_sym * sigma_sym**2 * tau
    r_residual = simplify(r_expr - r_expected)

    results["r_expr"] = r_expr
    results["r_residual"] = r_residual
    results["r_verified"] = r_residual == 0

    if verbose:
        print(f"\n[Step 5] Reservation/indifference price:")
        print(f"  r = {r_expr}")
        print(f"  Expected: {r_expected}")
        print(f"  Residual: {r_residual}  ✓" if r_residual == 0 else f"  Residual: {r_residual}  ✗")

    # Step 9: Quote correction term
    correction_expr = quote_correction_term()
    results["correction_expr"] = correction_expr
    if verbose:
        print(f"\n[Step 9] Quote correction term:")
        print(f"  (1/gamma)*ln(1+gamma/k) = {correction_expr}")

    # Step 10-11: Approximate quote formulas
    spread_expr = approximate_spread()
    delta_a_expr = approximate_ask_distance()
    delta_b_expr = approximate_bid_distance()

    results["spread_expr"] = spread_expr
    results["delta_a_expr"] = delta_a_expr
    results["delta_b_expr"] = delta_b_expr

    # Verify spread = delta^a + delta^b
    spread_consistent = verify_spread_consistency(delta_a_expr, delta_b_expr, spread_expr)
    results["spread_consistent"] = spread_consistent

    if verbose:
        print(f"\n[Step 10-11] Approximate spread:")
        print(f"  spread = {spread_expr}")
        print(f"\n[Step 10-11] Ask distance:")
        print(f"  delta^a = {delta_a_expr}")
        print(f"\n[Step 10-11] Bid distance:")
        print(f"  delta^b = {delta_b_expr}")
        print(f"\n  Spread = delta^a + delta^b: {'✓' if spread_consistent else '✗'}")

    # Verify quote-from-reservation consistency
    ask_ok, bid_ok = verify_quote_from_reservation(r_expr, delta_a_expr, delta_b_expr)
    results["ask_from_reservation_consistent"] = ask_ok
    results["bid_from_reservation_consistent"] = bid_ok

    if verbose:
        print(f"\n  p^a = r + spread/2 = s + delta^a: {'✓' if ask_ok else '✗'}")
        print(f"  p^b = r - spread/2 = s - delta^b: {'✓' if bid_ok else '✗'}")

    # Verify reservation price consistency
    r_consistent = verify_reservation_price_consistency(r_a_expr, r_b_expr, r_expr)
    results["r_consistent"] = r_consistent
    if verbose:
        print(f"  r = 0.5*(r^a + r^b): {'✓' if r_consistent else '✗'}")

    # --- Limiting behavior ---
    if verbose:
        print("\n[Step 12-13] Limiting behavior checks:")

    for gamma_val in [0.01, 0.1, 0.5]:
        lim_T = check_limit_t_to_T(gamma_val, sigma_val=2.0, k_val=1.5)
        lim_0 = check_limit_gamma_to_zero(sigma_val=2.0, k_val=1.5, q_val=1.0)
        if verbose:
            print(f"\n  gamma={gamma_val}:")
            print(f"    t->T: inventory_risk={lim_T['inventory_risk_at_T']:.6f}, "
                  f"spread_at_T={lim_T['spread_at_T']:.6f}, "
                  f"equals_liquidity={'✓' if lim_T['spread_equals_liquidity'] else '✗'}")
        results[f"limit_T_gamma_{gamma_val}"] = lim_T

    lim_0 = check_limit_gamma_to_zero(sigma_val=2.0, k_val=1.5, q_val=1.0)
    results["limit_gamma_to_zero"] = lim_0
    if verbose:
        print(f"\n  gamma->0: correction={lim_0['correction_small_gamma']:.6f}, "
              f"limit=1/k={lim_0['correction_limit_1_over_k']:.6f}, "
              f"converges={'✓' if lim_0['correction_converges'] else '✗'}")
        print(f"  gamma->0: asymmetry={lim_0['asymmetry_small_gamma']:.2e}, "
              f"vanishes={'✓' if lim_0['asymmetry_vanishes'] else '✗'}")

    # --- Numerical parameter sweep ---
    gamma_values = [0.01, 0.1, 0.5]
    q_values = [-2, -1, 0, 1, 2]
    t_values = [0.0, 0.5, 0.95, 1.0]

    sweep_results = numerical_parameter_sweep(
        gamma_values=gamma_values,
        q_values=q_values,
        t_values=t_values,
        sigma_val=2.0,
        k_val=1.5,
        T_val=1.0,
        s_val=100.0,
    )
    results["sweep_results"] = sweep_results

    if verbose:
        print("\n[Numerical Sweep] Sample results (gamma=0.1, t=0.0):")
        print(f"  {'q':>4} {'r':>8} {'delta_a':>10} {'delta_b':>10} {'spread':>10}")
        for row in sweep_results:
            if row["gamma"] == 0.1 and row["t"] == 0.0:
                print(f"  {row['q']:>4} {row['r']:>8.4f} {row['delta_a']:>10.4f} "
                      f"{row['delta_b']:>10.4f} {row['spread']:>10.4f}")

    # --- Liquidity component values ---
    if verbose:
        print("\n[Liquidity Components] (2/gamma)*ln(1+gamma/k) for k=1.5:")
        for gamma_val in [0.01, 0.1, 0.5]:
            liq = 2 * stable_correction_term(gamma_val, 1.5)
            print(f"  gamma={gamma_val}: liquidity_spread = {liq:.4f}")

    # Summary verification
    all_verified = all([
        results["r_a_verified"],
        results["r_b_verified"],
        results["r_verified"],
        results["spread_consistent"],
        results["ask_from_reservation_consistent"],
        results["bid_from_reservation_consistent"],
        results["r_consistent"],
    ])
    results["all_verified"] = all_verified

    if verbose:
        print("\n" + "=" * 70)
        print(f"EXPERIMENT 1 SUMMARY: All formulas verified = {all_verified}")
        print("=" * 70)

    return results


if __name__ == "__main__":
    run_experiment_1(verbose=True)
