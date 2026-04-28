"""
Experiment 3: Analytical reproduction of the appendix mean-variance model
with geometric-like price dynamics in the Avellaneda-Stoikov paper.

This is a SEPARATE extension from the main exponential-utility framework.
It uses:
- Alternative price dynamics: dS_u / S_u = sigma * dW_u (geometric Brownian)
- Mean-variance objective: V(x,s,q,t) = E_t[(x + q*S_T) - (gamma/2)*(q*S_T - q*s)^2]

Reproduces:
- Appendix value function V(x,s,q,t)
- Appendix reservation ask price R^a(s,q,t)
- Appendix reservation bid price R^b(s,q,t)

Using SymPy for symbolic derivation — Context7 confirmed.
NOTE: This appendix model is NOT used in the main simulation tables.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy import (
    symbols, exp, log, sqrt, simplify, limit, Eq, solve,
    expand, factor, Rational, latex
)
from typing import Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Symbolic variable definitions
# ---------------------------------------------------------------------------

def define_appendix_symbols() -> Dict[str, sp.Symbol]:
    """
    Define symbolic variables for the appendix mean-variance model.

    Returns
    -------
    dict
        Dictionary mapping variable names to SymPy symbols.
    """
    x = symbols('x', real=True)                          # cash
    s = symbols('s', real=True, positive=True)           # mid-price
    q = symbols('q', real=True)                          # inventory
    t = symbols('t', real=True, nonnegative=True)        # current time
    T = symbols('T', real=True, positive=True)           # terminal horizon
    sigma = symbols('sigma', real=True, positive=True)   # volatility
    gamma = symbols('gamma', real=True, positive=True)   # risk aversion (mean-variance)

    return {'x': x, 's': s, 'q': q, 't': t, 'T': T, 'sigma': sigma, 'gamma': gamma}


# ---------------------------------------------------------------------------
# Step 1-3: Appendix value function
# ---------------------------------------------------------------------------

def appendix_value_function(syms: Dict[str, sp.Symbol]) -> sp.Expr:
    """
    Compute the appendix value function under geometric-like price dynamics
    and mean-variance objective.

    Price dynamics: dS_u / S_u = sigma * dW_u over [t, T]
    => S_T = s * exp(sigma*(W_T-W_t) - sigma^2*(T-t)/2)  [Ito's lemma]
    => E_t[S_T] = s  (martingale)
    => Var_t[S_T] = s^2 * (exp(sigma^2*(T-t)) - 1)

    Objective: V(x,s,q,t) = E_t[(x + q*S_T) - (gamma/2)*(q*S_T - q*s)^2]
                           = x + q*s - (gamma/2)*q^2*Var_t[S_T]
                           = x + q*s - (gamma*q^2*s^2/2)*(exp(sigma^2*(T-t)) - 1)

    Note on sign convention: The paper's appendix states the formula as
    V = x + q*s + (gamma*q^2*s^2/2)*(exp(sigma^2*(T-t)) - 1) with a positive sign.
    However, the economically correct mean-variance objective subtracts the
    variance penalty, giving a negative sign. The negative-sign version is
    consistent with the paper's stated reservation price formulas R^a and R^b
    (verified by substitution). We use the negative-sign version here.

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_appendix_symbols().

    Returns
    -------
    sympy.Expr
        The appendix value function V(x, s, q, t).
    """
    x, s, q, t, T = syms['x'], syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']

    tau = T - t

    # Custom — Context7 found no library equivalent (paper appendix Eq.)
    # Using negative sign for variance penalty (economically correct mean-variance):
    # V(x,s,q,t) = x + q*s - (gamma*q^2*s^2/2)*(exp(sigma^2*(T-t)) - 1)
    # This is consistent with the paper's stated R^a and R^b formulas.
    V = x + q * s - (gamma * q**2 * s**2 / 2) * (exp(sigma**2 * tau) - 1)

    return V


# ---------------------------------------------------------------------------
# Step 4: Appendix reservation prices
# ---------------------------------------------------------------------------

def appendix_reservation_prices(
    syms: Dict[str, sp.Symbol]
) -> Tuple[sp.Expr, sp.Expr]:
    """
    Derive the appendix reservation ask and bid prices via indifference.

    Indifference conditions (same logic as main model):
      V(x + R^a, s, q-1, t) = V(x, s, q, t)   [ask: sell one share at R^a]
      V(x - R^b, s, q+1, t) = V(x, s, q, t)   [bid: buy one share at R^b]

    Solving algebraically:
      R^a(s,q,t) = s + ((1-2q)/2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)
      R^b(s,q,t) = s + ((-1-2q)/2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_appendix_symbols().

    Returns
    -------
    tuple
        (R_a, R_b) — appendix reservation ask and bid price expressions.
    """
    x, s, q, t, T = syms['x'], syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']

    tau = T - t
    V = appendix_value_function(syms)

    # Solve for R^a: V(x + R^a, s, q-1, t) = V(x, s, q, t)
    R_a_sym = symbols('R_a')
    V_ask_lhs = V.subs([(x, x + R_a_sym), (q, q - 1)])
    V_ask_rhs = V

    eq_ask = Eq(V_ask_lhs, V_ask_rhs)
    R_a_solutions = solve(eq_ask, R_a_sym)
    assert len(R_a_solutions) == 1, f"Expected unique solution for R^a, got {R_a_solutions}"
    R_a = simplify(R_a_solutions[0])

    # Solve for R^b: V(x - R^b, s, q+1, t) = V(x, s, q, t)
    R_b_sym = symbols('R_b')
    V_bid_lhs = V.subs([(x, x - R_b_sym), (q, q + 1)])
    V_bid_rhs = V

    eq_bid = Eq(V_bid_lhs, V_bid_rhs)
    R_b_solutions = solve(eq_bid, R_b_sym)
    assert len(R_b_solutions) == 1, f"Expected unique solution for R^b, got {R_b_solutions}"
    R_b = simplify(R_b_solutions[0])

    return R_a, R_b


def appendix_reservation_prices_direct(
    syms: Dict[str, sp.Symbol]
) -> Tuple[sp.Expr, sp.Expr]:
    """
    Return the appendix reservation prices directly from the paper's stated formulas.

    R^a(s,q,t) = s + ((1-2q)/2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)
    R^b(s,q,t) = s + ((-1-2q)/2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_appendix_symbols().

    Returns
    -------
    tuple
        (R_a, R_b) — appendix reservation ask and bid price expressions.
    """
    s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']

    tau = T - t
    scale = gamma * s**2 * (exp(sigma**2 * tau) - 1)

    # Custom — Context7 found no library equivalent (paper appendix reservation prices)
    R_a = s + ((1 - 2*q) / 2) * scale
    R_b = s + ((-1 - 2*q) / 2) * scale

    return R_a, R_b


# ---------------------------------------------------------------------------
# Step 5: Verify basic properties
# ---------------------------------------------------------------------------

def verify_appendix_properties(
    syms: Dict[str, sp.Symbol]
) -> Dict[str, Any]:
    """
    Verify basic properties of the appendix reservation prices:
    - When q=0, the reservation center equals the current price
    - Positive inventory lowers the effective reservation center
    - The inventory effect vanishes as t approaches T

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_appendix_symbols().

    Returns
    -------
    dict
        Dictionary of verified properties.
    """
    s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']

    R_a, R_b = appendix_reservation_prices_direct(syms)

    # Reservation center = (R^a + R^b) / 2
    R_center = simplify((R_a + R_b) / 2)

    # Property 1: R_center at q=0 equals s
    R_center_q0 = simplify(R_center.subs(q, 0))
    prop1 = simplify(R_center_q0 - s) == 0

    # Property 2: R_center - s = -q * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)
    # For q > 0 and t < T: R_center < s
    R_center_minus_s = simplify(R_center - s)

    # Property 3: R_center approaches s as t approaches T (tau -> 0)
    R_center_at_T = sp.limit(R_center, t, T)
    prop3 = simplify(R_center_at_T - s) == 0

    # Property 4: R^a at q=0 equals s (no inventory adjustment)
    R_a_q0 = simplify(R_a.subs(q, 0))
    R_b_q0 = simplify(R_b.subs(q, 0))

    return {
        'R_center': R_center,
        'R_center_q0_equals_s': prop1,
        'R_center_minus_s': R_center_minus_s,
        'R_center_at_T_equals_s': prop3,
        'R_a_at_q0': R_a_q0,
        'R_b_at_q0': R_b_q0,
    }


# ---------------------------------------------------------------------------
# Step 7: Contrast with main model
# ---------------------------------------------------------------------------

def contrast_with_main_model(syms: Dict[str, sp.Symbol]) -> Dict[str, str]:
    """
    Analytically contrast the appendix model's inventory adjustment with
    the main model's inventory adjustment.

    Main model (exponential utility):
      r(s,q,t) = s - q * gamma * sigma^2 * (T-t)
      Inventory adjustment scales with: sigma^2 * (T-t)

    Appendix model (mean-variance):
      R_center(s,q,t) = s - q * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)
      Inventory adjustment scales with: s^2 * (exp(sigma^2*(T-t)) - 1)

    For small sigma^2*(T-t): exp(sigma^2*(T-t)) - 1 ≈ sigma^2*(T-t)
    So the appendix model reduces to the main model scaled by s^2 vs 1.

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_appendix_symbols().

    Returns
    -------
    dict
        Dictionary of contrast descriptions.
    """
    s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']

    tau = T - t

    main_adjustment = -q * gamma * sigma**2 * tau
    appendix_adjustment = -q * gamma * s**2 * (sp.exp(sigma**2 * tau) - 1)

    # Small-tau approximation of appendix adjustment
    appendix_approx = sp.series(appendix_adjustment, tau, 0, 2).removeO()

    return {
        'main_model_adjustment': str(main_adjustment),
        'appendix_model_adjustment': str(appendix_adjustment),
        'appendix_small_tau_approx': str(appendix_approx),
        'note': (
            'Main model scales with sigma^2*(T-t); '
            'Appendix scales with s^2*(exp(sigma^2*(T-t))-1). '
            'For small tau, appendix ≈ s^2 * sigma^2 * tau * (-q*gamma).'
        ),
    }


# ---------------------------------------------------------------------------
# Numerical spot checks
# ---------------------------------------------------------------------------

def numerical_appendix_checks() -> Dict[str, Any]:
    """
    Perform numerical spot checks for the appendix model.

    Uses: s=100, T=1, sigma=0.2, gamma=0.1, q in {-2,-1,0,1,2}, t=0.

    Returns
    -------
    dict
        Dictionary of numerical results.
    """
    s_val = 100.0
    T_val = 1.0
    sigma_val = 0.2
    gamma_val = 0.1
    t_val = 0.0
    q_values = [-2, -1, 0, 1, 2]

    tau = T_val - t_val
    scale = gamma_val * s_val**2 * (np.exp(sigma_val**2 * tau) - 1)

    results = {}
    for q_val in q_values:
        # Appendix value function (at x=0) — negative sign for variance penalty
        V_val = 0 + q_val * s_val - (gamma_val * q_val**2 * s_val**2 / 2) * (
            np.exp(sigma_val**2 * tau) - 1
        )

        # Reservation prices (paper formulas)
        R_a_val = s_val + ((1 - 2*q_val) / 2) * scale
        R_b_val = s_val + ((-1 - 2*q_val) / 2) * scale
        R_center_val = (R_a_val + R_b_val) / 2

        # Verify properties
        if q_val == 0:
            assert abs(R_center_val - s_val) < 1e-10, \
                f"R_center should equal s at q=0, got {R_center_val}"

        results[q_val] = {
            'V': V_val,
            'R_a': R_a_val,
            'R_b': R_b_val,
            'R_center': R_center_val,
            'R_center_minus_s': R_center_val - s_val,
        }

    # Verify terminal behavior: at t=T (tau=0), R_center -> s
    tau_T = 0.0
    scale_T = gamma_val * s_val**2 * (np.exp(sigma_val**2 * tau_T) - 1)
    R_a_T = s_val + ((1 - 2*1) / 2) * scale_T
    R_b_T = s_val + ((-1 - 2*1) / 2) * scale_T
    R_center_T = (R_a_T + R_b_T) / 2
    assert abs(R_center_T - s_val) < 1e-10, \
        f"R_center should approach s as t->T, got {R_center_T}"

    results['terminal_check'] = {
        'R_center_at_T': R_center_T,
        'equals_s': abs(R_center_T - s_val) < 1e-10,
    }

    return results


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment_3() -> Dict[str, Any]:
    """
    Run the full Experiment 3: analytical reproduction of the appendix
    mean-variance model with geometric-like price dynamics.

    Returns
    -------
    dict
        Dictionary containing all derived formulas and verification results.
    """
    print("=" * 70)
    print("EXPERIMENT 3: Appendix Mean-Variance Model (Analytical)")
    print("NOTE: This is a SEPARATE extension — NOT used in main simulation tables")
    print("=" * 70)

    syms = define_appendix_symbols()

    # Step 1-3: Appendix value function
    print("\n--- Steps 1-3: Appendix Value Function ---")
    V = appendix_value_function(syms)
    print(f"V(x,s,q,t) = {V}")
    print("Price dynamics: dS_u / S_u = sigma * dW_u (geometric Brownian)")
    print("Objective: E_t[(x + q*S_T) - (gamma/2)*(q*S_T - q*s)^2]")

    # Step 4: Reservation prices (derived by solving indifference equations)
    print("\n--- Step 4: Appendix Reservation Prices (Derived) ---")
    R_a_derived, R_b_derived = appendix_reservation_prices(syms)
    print(f"R^a (derived) = {R_a_derived}")
    print(f"R^b (derived) = {R_b_derived}")

    # Direct formulas from paper
    print("\n--- Step 4: Appendix Reservation Prices (Paper Formulas) ---")
    R_a_direct, R_b_direct = appendix_reservation_prices_direct(syms)
    print(f"R^a (paper)   = {R_a_direct}")
    print(f"R^b (paper)   = {R_b_direct}")

    # Verify derived matches paper
    diff_a = sp.simplify(R_a_derived - R_a_direct)
    diff_b = sp.simplify(R_b_derived - R_b_direct)
    print(f"\nVerification: R^a derived == paper: {diff_a == 0}")
    print(f"Verification: R^b derived == paper: {diff_b == 0}")

    # Step 5: Properties
    print("\n--- Step 5: Basic Properties ---")
    props = verify_appendix_properties(syms)
    print(f"R_center = {props['R_center']}")
    print(f"R_center at q=0 equals s: {props['R_center_q0_equals_s']}")
    print(f"R_center - s = {props['R_center_minus_s']}")
    print(f"R_center at t=T equals s: {props['R_center_at_T_equals_s']}")
    print(f"R^a at q=0: {props['R_a_at_q0']}")
    print(f"R^b at q=0: {props['R_b_at_q0']}")

    # Step 7: Contrast with main model
    print("\n--- Step 7: Contrast with Main Model ---")
    contrast = contrast_with_main_model(syms)
    print(f"Main model adjustment:    {contrast['main_model_adjustment']}")
    print(f"Appendix model adjustment: {contrast['appendix_model_adjustment']}")
    print(f"Appendix small-tau approx: {contrast['appendix_small_tau_approx']}")
    print(f"Note: {contrast['note']}")

    # Numerical spot checks
    print("\n--- Numerical Spot Checks (s=100, T=1, sigma=0.2, gamma=0.1) ---")
    num_results = numerical_appendix_checks()
    for q_val, vals in num_results.items():
        if q_val == 'terminal_check':
            print(f"\n  Terminal check: R_center at t=T = {vals['R_center_at_T']:.6f}, "
                  f"equals s: {vals['equals_s']}")
        else:
            print(f"  q={q_val:+d}: V={vals['V']:.4f}, "
                  f"R^a={vals['R_a']:.4f}, R^b={vals['R_b']:.4f}, "
                  f"R_center={vals['R_center']:.4f}, "
                  f"R_center-s={vals['R_center_minus_s']:.6f}")

    print("\n" + "=" * 70)
    print("Experiment 3 completed successfully.")
    print("REMINDER: This appendix model is NOT used in the main simulation tables.")
    print("=" * 70)

    return {
        'symbols': syms,
        'V': V,
        'R_a': R_a_direct,
        'R_b': R_b_direct,
        'R_a_derived': R_a_derived,
        'R_b_derived': R_b_derived,
        'properties': props,
        'contrast': contrast,
        'numerical': num_results,
        'verification': {
            'R_a_matches_paper': diff_a == 0,
            'R_b_matches_paper': diff_b == 0,
        },
    }


if __name__ == '__main__':
    run_experiment_3()
