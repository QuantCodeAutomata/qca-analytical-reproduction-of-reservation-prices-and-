"""
Experiment 1: Analytical reproduction of reservation prices and optimal quote formulas
in the Avellaneda-Stoikov (AS) market-making model.

Reproduces the paper's analytical formulas for:
- Frozen-inventory value functions
- Finite-horizon reservation bid/ask prices
- Reservation/indifference price
- Infinite-horizon stationary reservation prices
- Optimal quote-distance conditions under exponential order-arrival intensities

Using SymPy for symbolic derivation — Context7 confirmed.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy import (
    symbols, exp, log, sqrt, simplify, limit, series, oo,
    Rational, latex, pprint, Eq, solve, diff, expand, factor
)
from typing import Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Symbolic variable definitions
# ---------------------------------------------------------------------------

def define_symbols() -> Dict[str, sp.Symbol]:
    """
    Define all symbolic variables used in the Avellaneda-Stoikov model.

    Returns
    -------
    dict
        Dictionary mapping variable names to SymPy symbols.
    """
    # State variables
    x = symbols('x', real=True)                          # cash
    s = symbols('s', real=True, positive=True)           # mid-price
    q = symbols('q', real=True)                          # inventory
    t = symbols('t', real=True, nonnegative=True)        # current time
    T = symbols('T', real=True, positive=True)           # terminal horizon

    # Model parameters
    sigma = symbols('sigma', real=True, positive=True)   # volatility
    gamma = symbols('gamma', real=True, positive=True)   # risk aversion
    omega = symbols('omega', real=True, positive=True)   # discount / inventory-bound parameter

    # Execution intensity parameters (exponential specification)
    A = symbols('A', real=True, positive=True)           # intensity scale
    k = symbols('k', real=True, positive=True)           # intensity decay

    # Power-law intensity parameters (alternative specification)
    B = symbols('B', real=True, positive=True)
    alpha = symbols('alpha', real=True, positive=True)
    beta = symbols('beta', real=True, positive=True)

    # Quote distances
    delta_a = symbols('delta^a', real=True, positive=True)   # ask distance
    delta_b = symbols('delta^b', real=True, positive=True)   # bid distance

    # Inventory bound for infinite-horizon admissibility
    q_max = symbols('q_max', real=True, positive=True)

    return {
        'x': x, 's': s, 'q': q, 't': t, 'T': T,
        'sigma': sigma, 'gamma': gamma, 'omega': omega,
        'A': A, 'k': k, 'B': B, 'alpha': alpha, 'beta': beta,
        'delta_a': delta_a, 'delta_b': delta_b, 'q_max': q_max,
    }


# ---------------------------------------------------------------------------
# Step 1-3: Frozen-inventory value function
# ---------------------------------------------------------------------------

def frozen_inventory_value_function(syms: Dict[str, sp.Symbol]) -> sp.Expr:
    """
    Compute the frozen-inventory value function under driftless Brownian mid-price
    dynamics and exponential terminal utility.

    Under dS_u = sigma dW_u, S_T | S_t=s is Gaussian with mean s and variance
    sigma^2 (T-t). The terminal utility is -exp(-gamma(x + q S_T)).

    E_t[-exp(-gamma(x + q S_T))]
      = -exp(-gamma x) * exp(-gamma q s) * exp((gamma^2 q^2 sigma^2 (T-t)) / 2)

    This follows from the moment-generating function of a Gaussian:
    E[exp(c Z)] = exp(c mu + c^2 var / 2) for Z ~ N(mu, var).

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    sympy.Expr
        The frozen-inventory value function v(x, s, q, t).
    """
    x, s, q, t, T = syms['x'], syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']

    # q*S_T ~ N(q*s, q^2 * sigma^2 * (T-t))
    # E[-exp(-gamma(x + q S_T))] = -exp(-gamma x) * E[exp(-gamma q S_T)]
    # Using MGF: E[exp(c Z)] = exp(c*mean + c^2*var/2) with c = -gamma
    # mean of q*S_T = q*s, var of q*S_T = q^2 * sigma^2 * (T-t)
    tau = T - t  # time to maturity

    v = -exp(-gamma * x) * exp(-gamma * q * s) * exp(
        (gamma**2 * q**2 * sigma**2 * tau) / 2
    )
    return v


# ---------------------------------------------------------------------------
# Step 4: Finite-horizon reservation ask and bid prices
# ---------------------------------------------------------------------------

def finite_horizon_reservation_prices(
    syms: Dict[str, sp.Symbol]
) -> Tuple[sp.Expr, sp.Expr]:
    """
    Derive the finite-horizon reservation ask price r^a and bid price r^b
    via exponential-utility indifference.

    Indifference conditions:
      v(x + r^a, s, q-1, t) = v(x, s, q, t)   [ask: sell one share at r^a]
      v(x - r^b, s, q+1, t) = v(x, s, q, t)   [bid: buy one share at r^b]

    Solving these algebraically yields:
      r^a(s, q, t) = s + ((1 - 2q) * gamma * sigma^2 * (T-t)) / 2
      r^b(s, q, t) = s + ((-1 - 2q) * gamma * sigma^2 * (T-t)) / 2

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    tuple
        (r_a, r_b) — reservation ask and bid price expressions.
    """
    s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']

    tau = T - t

    # Custom — Context7 found no library equivalent (paper Eq. for reservation prices)
    r_a = s + ((1 - 2*q) * gamma * sigma**2 * tau) / 2
    r_b = s + ((-1 - 2*q) * gamma * sigma**2 * tau) / 2

    return r_a, r_b


def verify_reservation_prices_by_substitution(
    syms: Dict[str, sp.Symbol]
) -> Dict[str, bool]:
    """
    Verify the reservation price formulas by substituting back into the
    frozen-inventory value function and checking the indifference equations.

    Uses logarithmic form for reliable symbolic simplification:
    ln(v(x+r^a, s, q-1, t)) - ln(v(x, s, q, t)) should equal 0.

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    dict
        Dictionary with keys 'ask_verified' and 'bid_verified' (bool).
    """
    x, s, q, t, T = syms['x'], syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']

    tau = T - t
    r_a, r_b = finite_horizon_reservation_prices(syms)

    # Work in log-space for reliable simplification.
    # ln(v) = -gamma*x - gamma*q*s + (gamma^2*q^2*sigma^2*tau)/2 + const
    def log_v(x_val, q_val):
        return -gamma * x_val - gamma * q_val * s + (gamma**2 * q_val**2 * sigma**2 * tau) / 2

    # Ask: v(x + r^a, s, q-1, t) = v(x, s, q, t)
    # => log_v(x + r^a, q-1) = log_v(x, q)
    diff_ask = simplify(log_v(x + r_a, q - 1) - log_v(x, q))
    ask_verified = diff_ask == 0

    # Bid: v(x - r^b, s, q+1, t) = v(x, s, q, t)
    # => log_v(x - r^b, q+1) = log_v(x, q)
    diff_bid = simplify(log_v(x - r_b, q + 1) - log_v(x, q))
    bid_verified = diff_bid == 0

    return {'ask_verified': ask_verified, 'bid_verified': bid_verified}


# ---------------------------------------------------------------------------
# Step 5: Reservation/indifference price
# ---------------------------------------------------------------------------

def reservation_price(syms: Dict[str, sp.Symbol]) -> sp.Expr:
    """
    Compute the reservation/indifference price r(s, q, t) as the average of
    the reservation ask and bid prices.

    r(s, q, t) = (r^a + r^b) / 2 = s - q * gamma * sigma^2 * (T-t)

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    sympy.Expr
        The reservation price expression.
    """
    r_a, r_b = finite_horizon_reservation_prices(syms)
    r = simplify((r_a + r_b) / 2)
    return r


# ---------------------------------------------------------------------------
# Step 6: Verify limiting and sign properties
# ---------------------------------------------------------------------------

def verify_reservation_price_properties(
    syms: Dict[str, sp.Symbol]
) -> Dict[str, Any]:
    """
    Verify the analytical properties of the reservation price:
    - r(s, 0, t) = s
    - q > 0 implies r < s
    - q < 0 implies r > s
    - r approaches s as t approaches T

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    dict
        Dictionary of verified properties.
    """
    s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']

    r = reservation_price(syms)

    # Property 1: r(s, 0, t) = s
    r_at_q0 = simplify(r.subs(q, 0))
    prop1 = simplify(r_at_q0 - s) == 0

    # Property 2: r - s = -q * gamma * sigma^2 * (T-t)
    # For q > 0 and t < T: r - s < 0, i.e., r < s
    r_minus_s = simplify(r - s)
    # The sign of r - s is the sign of -q * gamma * sigma^2 * (T-t)
    # which is negative when q > 0 (since gamma, sigma, T-t > 0)
    prop2_expr = r_minus_s  # = -q * gamma * sigma^2 * (T-t)

    # Property 3: limit as t -> T
    r_at_T = limit(r, t, T)
    prop3 = simplify(r_at_T - s) == 0

    return {
        'r_at_q0_equals_s': prop1,
        'r_minus_s_expression': prop2_expr,
        'r_at_T_equals_s': prop3,
        'r_formula': r,
    }


# ---------------------------------------------------------------------------
# Step 7: Infinite-horizon stationary reservation prices
# ---------------------------------------------------------------------------

def infinite_horizon_reservation_prices(
    syms: Dict[str, sp.Symbol]
) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Reproduce the infinite-horizon stationary reservation bid/ask price formulas.

    bar_r^a(s, q) = s + (1/gamma) * ln(1 + ((1-2q)*gamma^2*sigma^2) / (2*omega - gamma^2*q^2*sigma^2))
    bar_r^b(s, q) = s + (1/gamma) * ln(1 + ((-1-2q)*gamma^2*sigma^2) / (2*omega - gamma^2*q^2*sigma^2))

    Admissibility condition: omega > (1/2) * gamma^2 * sigma^2 * q^2
    Suggested boundedness: omega = (1/2) * gamma^2 * sigma^2 * (q_max + 1)^2

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    tuple
        (bar_r_a, bar_r_b, admissibility_condition)
    """
    s, q = syms['s'], syms['q']
    sigma, gamma, omega = syms['sigma'], syms['gamma'], syms['omega']
    q_max = syms['q_max']

    # Custom — Context7 found no library equivalent (paper infinite-horizon formulas)
    denom = 2 * omega - gamma**2 * q**2 * sigma**2

    bar_r_a = s + (1 / gamma) * log(
        1 + ((1 - 2*q) * gamma**2 * sigma**2) / denom
    )
    bar_r_b = s + (1 / gamma) * log(
        1 + ((-1 - 2*q) * gamma**2 * sigma**2) / denom
    )

    # Admissibility condition: omega > (1/2) * gamma^2 * sigma^2 * q^2
    admissibility = sp.Gt(omega, Rational(1, 2) * gamma**2 * sigma**2 * q**2)

    # Suggested boundedness choice
    omega_bound = Rational(1, 2) * gamma**2 * sigma**2 * (q_max + 1)**2

    return bar_r_a, bar_r_b, admissibility, omega_bound


# ---------------------------------------------------------------------------
# Steps 8-11: HJB equation and first-order conditions
# ---------------------------------------------------------------------------

def first_order_conditions_general(
    syms: Dict[str, sp.Symbol]
) -> Tuple[sp.Expr, sp.Expr]:
    """
    Derive the first-order conditions for optimal bid and ask quote distances
    from the HJB equation under the exponential-utility ansatz.

    The FOCs are:
      s - r^b = delta^b - (1/gamma) * ln(1 - gamma * lambda^b / lambda^{b,'})
      r^a - s = delta^a - (1/gamma) * ln(1 - gamma * lambda^a / lambda^{a,'})

    These are expressed symbolically in terms of lambda and its derivative.

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    tuple
        (foc_ask, foc_bid) — symbolic first-order condition expressions.
    """
    s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma = syms['sigma'], syms['gamma']
    delta_a, delta_b = syms['delta_a'], syms['delta_b']

    r_a, r_b = finite_horizon_reservation_prices(syms)

    # Symbolic lambda and its derivative (general)
    # Use simple names to avoid SymPy tuple-return from comma-containing strings
    lam_a = sp.Symbol('lambda_a', positive=True)
    lam_a_prime = sp.Symbol('lambda_a_prime', real=True, negative=True)
    lam_b = sp.Symbol('lambda_b', positive=True)
    lam_b_prime = sp.Symbol('lambda_b_prime', real=True, negative=True)

    # FOC for ask: r^a - s = delta^a - (1/gamma) * ln(1 - gamma * lam_a / lam_a_prime)
    foc_ask = Eq(
        r_a - s,
        delta_a - (1 / gamma) * log(1 - gamma * lam_a / lam_a_prime)
    )

    # FOC for bid: s - r^b = delta^b - (1/gamma) * ln(1 - gamma * lam_b / lam_b_prime)
    foc_bid = Eq(
        s - r_b,
        delta_b - (1 / gamma) * log(1 - gamma * lam_b / lam_b_prime)
    )

    return foc_ask, foc_bid


# ---------------------------------------------------------------------------
# Step 12: Exponential intensity specialization
# ---------------------------------------------------------------------------

def exponential_intensity_simplification(
    syms: Dict[str, sp.Symbol]
) -> Dict[str, sp.Expr]:
    """
    Specialize the FOCs to exponential execution intensity lambda(delta) = A * exp(-k * delta).

    Since lambda'(delta) = -k * lambda(delta), the logarithmic adjustment simplifies:
      1 - gamma * lambda / lambda' = 1 - gamma * lambda / (-k * lambda) = 1 + gamma/k

    This yields explicit quote distances:
      delta^a = r^a - s + (1/gamma) * ln(1 + gamma/k)
      delta^b = s - r^b + (1/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    dict
        Dictionary with keys 'lambda_expr', 'lambda_prime', 'log_adjustment',
        'delta_a_explicit', 'delta_b_explicit'.
    """
    s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma, k, A = syms['sigma'], syms['gamma'], syms['k'], syms['A']
    delta_a, delta_b = syms['delta_a'], syms['delta_b']

    # Exponential intensity: lambda(delta) = A * exp(-k * delta)
    # Using SymPy — Context7 confirmed
    lam = A * exp(-k * delta_a)
    lam_prime = diff(lam, delta_a)  # = -k * A * exp(-k * delta_a) = -k * lam

    # Verify: lambda' = -k * lambda
    ratio = simplify(lam_prime / lam)  # should be -k
    assert simplify(ratio + k) == 0, "lambda'/lambda should equal -k"

    # Logarithmic adjustment: 1 - gamma * lambda / lambda' = 1 + gamma/k
    log_arg = simplify(1 - gamma * lam / lam_prime)
    log_adjustment = (1 / gamma) * log(log_arg)
    log_adjustment_simplified = simplify(log_adjustment - (1 / gamma) * log(1 + gamma / k))
    assert log_adjustment_simplified == 0, "Log adjustment should simplify to (1/gamma)*ln(1+gamma/k)"

    r_a, r_b = finite_horizon_reservation_prices(syms)

    # Explicit quote distances
    spread_adj = (1 / gamma) * log(1 + gamma / k)
    delta_a_explicit = (r_a - s) + spread_adj
    delta_b_explicit = (s - r_b) + spread_adj

    return {
        'lambda_expr': lam,
        'lambda_prime': lam_prime,
        'log_adjustment': (1 / gamma) * log(1 + gamma / k),
        'delta_a_explicit': delta_a_explicit,
        'delta_b_explicit': delta_b_explicit,
        'spread_adjustment': spread_adj,
    }


# ---------------------------------------------------------------------------
# Step 13: Finite-horizon operational quoting rules
# ---------------------------------------------------------------------------

def finite_horizon_quote_distances(
    syms: Dict[str, sp.Symbol]
) -> Dict[str, sp.Expr]:
    """
    Substitute the finite-horizon reservation prices into the explicit quote
    distances to obtain the operational quoting rules.

    delta^a(s, q, t) = ((1-2q) * gamma * sigma^2 * (T-t)) / 2 + (1/gamma) * ln(1 + gamma/k)
    delta^b(s, q, t) = ((1+2q) * gamma * sigma^2 * (T-t)) / 2 + (1/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    dict
        Dictionary with 'delta_a', 'delta_b', 'p_a', 'p_b', 'Delta_t', 'r'.
    """
    s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma, k = syms['sigma'], syms['gamma'], syms['k']

    tau = T - t
    spread_adj = (1 / gamma) * log(1 + gamma / k)

    # Custom — Context7 found no library equivalent (paper Eq. for finite-horizon distances)
    delta_a = ((1 - 2*q) * gamma * sigma**2 * tau) / 2 + spread_adj
    delta_b = ((1 + 2*q) * gamma * sigma**2 * tau) / 2 + spread_adj

    # Total spread
    Delta_t = simplify(delta_a + delta_b)
    # Delta_t = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)

    # Reservation price
    r = reservation_price(syms)

    # Quotes in price space
    p_a = s + delta_a   # ask price = mid + ask distance
    p_b = s - delta_b   # bid price = mid - bid distance

    # Center/spread form
    p_a_center = r + Delta_t / 2
    p_b_center = r - Delta_t / 2

    return {
        'delta_a': delta_a,
        'delta_b': delta_b,
        'Delta_t': Delta_t,
        'r': r,
        'p_a': p_a,
        'p_b': p_b,
        'p_a_center_form': p_a_center,
        'p_b_center_form': p_b_center,
        'spread_adj': spread_adj,
    }


# ---------------------------------------------------------------------------
# Step 15: Gamma -> 0 limit
# ---------------------------------------------------------------------------

def gamma_to_zero_limit(syms: Dict[str, sp.Symbol]) -> Dict[str, sp.Expr]:
    """
    Demonstrate the convergence of the inventory-based strategy to the symmetric
    strategy as gamma approaches zero via Taylor expansion.

    As gamma -> 0:
      (1/gamma) * ln(1 + gamma/k) -> 1/k
      -q * gamma * sigma^2 * (T-t) -> 0

    So the reservation price r -> s and the spread adjustment -> 1/k,
    implying convergence to a symmetric strategy centered at the mid-price.

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    dict
        Dictionary with limit expressions.
    """
    s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
    sigma, gamma, k = syms['sigma'], syms['gamma'], syms['k']

    # Using SymPy limit — Context7 confirmed
    spread_adj = (1 / gamma) * log(1 + gamma / k)
    spread_adj_limit = limit(spread_adj, gamma, 0)  # -> 1/k

    r = reservation_price(syms)
    r_limit = limit(r, gamma, 0)  # -> s

    delta_a_formula = finite_horizon_quote_distances(syms)['delta_a']
    delta_b_formula = finite_horizon_quote_distances(syms)['delta_b']

    delta_a_limit = limit(delta_a_formula, gamma, 0)  # -> 1/k
    delta_b_limit = limit(delta_b_formula, gamma, 0)  # -> 1/k

    return {
        'spread_adj_limit': spread_adj_limit,
        'r_limit': r_limit,
        'delta_a_limit': delta_a_limit,
        'delta_b_limit': delta_b_limit,
        'symmetric_at_gamma_zero': simplify(delta_a_limit - delta_b_limit) == 0,
    }


# ---------------------------------------------------------------------------
# Step 16: Power-law intensity (alternative specification)
# ---------------------------------------------------------------------------

def power_law_intensity_specification(
    syms: Dict[str, sp.Symbol]
) -> Dict[str, sp.Expr]:
    """
    Document the alternative power-law execution intensity specification.

    lambda(delta) = B * delta^(-alpha/beta)

    This is an alternative analytical specification mentioned in the paper.
    It is NOT used in the main simulation tables.

    Parameters
    ----------
    syms : dict
        Dictionary of symbolic variables from define_symbols().

    Returns
    -------
    dict
        Dictionary with 'lambda_power_law' and 'lambda_prime_power_law'.
    """
    B, alpha, beta = syms['B'], syms['alpha'], syms['beta']
    delta_a = syms['delta_a']

    # Custom — Context7 found no library equivalent (paper alternative specification)
    lam_power = B * delta_a**(-alpha / beta)
    lam_power_prime = diff(lam_power, delta_a)

    return {
        'lambda_power_law': lam_power,
        'lambda_prime_power_law': lam_power_prime,
        'note': 'Alternative specification — not used in main simulation tables',
    }


# ---------------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------------

def numerical_spot_checks() -> Dict[str, Any]:
    """
    Perform numerical spot checks using the paper's suggested parameter values:
    s=100, T=1, sigma=2, gamma in {0.1, 0.01, 0.5}, k=1.5.

    Returns
    -------
    dict
        Dictionary of numerical results for each gamma value.
    """
    s_val = 100.0
    T_val = 1.0
    sigma_val = 2.0
    k_val = 1.5
    gamma_values = [0.1, 0.01, 0.5]
    q_values = [-2, -1, 0, 1, 2]
    t_val = 0.0  # at t=0

    results = {}

    for gamma_val in gamma_values:
        tau = T_val - t_val
        gamma_results = {}

        for q_val in q_values:
            # Reservation price
            r_val = s_val - q_val * gamma_val * sigma_val**2 * tau

            # Reservation ask and bid
            r_a_val = s_val + ((1 - 2*q_val) * gamma_val * sigma_val**2 * tau) / 2
            r_b_val = s_val + ((-1 - 2*q_val) * gamma_val * sigma_val**2 * tau) / 2

            # Spread adjustment
            spread_adj_val = (1 / gamma_val) * np.log(1 + gamma_val / k_val)

            # Quote distances
            delta_a_val = ((1 - 2*q_val) * gamma_val * sigma_val**2 * tau) / 2 + spread_adj_val
            delta_b_val = ((1 + 2*q_val) * gamma_val * sigma_val**2 * tau) / 2 + spread_adj_val

            # Total spread
            Delta_t_val = gamma_val * sigma_val**2 * tau + (2 / gamma_val) * np.log(1 + gamma_val / k_val)

            # Verify center/spread form
            p_a_val = r_val + Delta_t_val / 2
            p_b_val = r_val - Delta_t_val / 2

            # Verify consistency: p_a = s + delta_a, p_b = s - delta_b
            assert abs(p_a_val - (s_val + delta_a_val)) < 1e-10, \
                f"p_a consistency check failed for gamma={gamma_val}, q={q_val}"
            assert abs(p_b_val - (s_val - delta_b_val)) < 1e-10, \
                f"p_b consistency check failed for gamma={gamma_val}, q={q_val}"

            # Verify r(s, 0, t) = s
            if q_val == 0:
                assert abs(r_val - s_val) < 1e-10, "r(s,0,t) should equal s"

            # Verify sign of r - s
            if q_val > 0:
                assert r_val < s_val, f"r should be < s when q > 0 (q={q_val})"
            elif q_val < 0:
                assert r_val > s_val, f"r should be > s when q < 0 (q={q_val})"

            gamma_results[q_val] = {
                'r': r_val,
                'r_a': r_a_val,
                'r_b': r_b_val,
                'delta_a': delta_a_val,
                'delta_b': delta_b_val,
                'Delta_t': Delta_t_val,
                'p_a': p_a_val,
                'p_b': p_b_val,
                'spread_adj': spread_adj_val,
            }

        results[gamma_val] = gamma_results

    return results


def numerical_infinite_horizon_checks() -> Dict[str, Any]:
    """
    Numerical spot checks for the infinite-horizon stationary reservation prices.
    Verifies admissibility condition and guards against invalid log arguments.

    Returns
    -------
    dict
        Dictionary of numerical results.
    """
    s_val = 100.0
    sigma_val = 2.0
    gamma_values = [0.1, 0.01, 0.5]
    q_max_val = 5
    q_values = [-2, -1, 0, 1, 2]

    results = {}

    for gamma_val in gamma_values:
        # Suggested omega for admissibility
        omega_val = 0.5 * gamma_val**2 * sigma_val**2 * (q_max_val + 1)**2

        gamma_results = {}
        for q_val in q_values:
            denom = 2 * omega_val - gamma_val**2 * q_val**2 * sigma_val**2

            # Guard against invalid denominator
            if denom <= 0:
                gamma_results[q_val] = {'error': 'admissibility violated'}
                continue

            arg_a = 1 + ((1 - 2*q_val) * gamma_val**2 * sigma_val**2) / denom
            arg_b = 1 + ((-1 - 2*q_val) * gamma_val**2 * sigma_val**2) / denom

            # Guard against invalid log arguments
            if arg_a <= 0 or arg_b <= 0:
                gamma_results[q_val] = {'error': 'log argument non-positive'}
                continue

            bar_r_a = s_val + (1 / gamma_val) * np.log(arg_a)
            bar_r_b = s_val + (1 / gamma_val) * np.log(arg_b)

            gamma_results[q_val] = {
                'omega': omega_val,
                'bar_r_a': bar_r_a,
                'bar_r_b': bar_r_b,
                'admissibility_ok': omega_val > 0.5 * gamma_val**2 * sigma_val**2 * q_val**2,
            }

        results[gamma_val] = gamma_results

    return results


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment_1() -> Dict[str, Any]:
    """
    Run the full Experiment 1: analytical reproduction of the AS model formulas.

    Returns
    -------
    dict
        Dictionary containing all derived formulas and verification results.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Analytical Reproduction of AS Model Formulas")
    print("=" * 70)

    syms = define_symbols()

    # Step 1-3: Frozen-inventory value function
    print("\n--- Step 1-3: Frozen-Inventory Value Function ---")
    v = frozen_inventory_value_function(syms)
    print(f"v(x,s,q,t) = {v}")

    # Step 4: Reservation prices
    print("\n--- Step 4: Finite-Horizon Reservation Prices ---")
    r_a, r_b = finite_horizon_reservation_prices(syms)
    print(f"r^a(s,q,t) = {r_a}")
    print(f"r^b(s,q,t) = {r_b}")

    # Verify by substitution
    verification = verify_reservation_prices_by_substitution(syms)
    print(f"\nVerification by substitution:")
    print(f"  Ask indifference verified: {verification['ask_verified']}")
    print(f"  Bid indifference verified: {verification['bid_verified']}")

    # Step 5: Reservation price
    print("\n--- Step 5: Reservation/Indifference Price ---")
    r = reservation_price(syms)
    print(f"r(s,q,t) = {r}")

    # Step 6: Properties
    print("\n--- Step 6: Limiting and Sign Properties ---")
    props = verify_reservation_price_properties(syms)
    print(f"  r(s,0,t) = s: {props['r_at_q0_equals_s']}")
    print(f"  r - s = {props['r_minus_s_expression']}")
    print(f"  r -> s as t -> T: {props['r_at_T_equals_s']}")

    # Step 7: Infinite-horizon
    print("\n--- Step 7: Infinite-Horizon Stationary Reservation Prices ---")
    bar_r_a, bar_r_b, admissibility, omega_bound = infinite_horizon_reservation_prices(syms)
    print(f"bar_r^a(s,q) = {bar_r_a}")
    print(f"bar_r^b(s,q) = {bar_r_b}")
    print(f"Admissibility condition: {admissibility}")
    print(f"Suggested omega bound: {omega_bound}")

    # Steps 8-11: FOCs
    print("\n--- Steps 8-11: First-Order Conditions (General) ---")
    foc_ask, foc_bid = first_order_conditions_general(syms)
    print(f"FOC ask: {foc_ask}")
    print(f"FOC bid: {foc_bid}")

    # Step 12: Exponential intensity
    print("\n--- Step 12: Exponential Intensity Specialization ---")
    exp_results = exponential_intensity_simplification(syms)
    print(f"lambda(delta) = {exp_results['lambda_expr']}")
    print(f"lambda'(delta) = {exp_results['lambda_prime']}")
    print(f"Log adjustment = {exp_results['log_adjustment']}")
    print(f"delta^a = {exp_results['delta_a_explicit']}")
    print(f"delta^b = {exp_results['delta_b_explicit']}")

    # Step 13: Operational quoting rules
    print("\n--- Step 13: Finite-Horizon Operational Quoting Rules ---")
    quotes = finite_horizon_quote_distances(syms)
    print(f"delta^a(s,q,t) = {quotes['delta_a']}")
    print(f"delta^b(s,q,t) = {quotes['delta_b']}")
    print(f"Delta_t = {quotes['Delta_t']}")
    print(f"p^a = r + Delta_t/2 = {simplify(quotes['p_a_center_form'])}")
    print(f"p^b = r - Delta_t/2 = {simplify(quotes['p_b_center_form'])}")

    # Step 15: Gamma -> 0 limit
    print("\n--- Step 15: Gamma -> 0 Limit ---")
    limits = gamma_to_zero_limit(syms)
    print(f"lim(gamma->0) spread_adj = {limits['spread_adj_limit']}")
    print(f"lim(gamma->0) r = {limits['r_limit']}")
    print(f"lim(gamma->0) delta^a = {limits['delta_a_limit']}")
    print(f"lim(gamma->0) delta^b = {limits['delta_b_limit']}")
    print(f"Symmetric at gamma=0: {limits['symmetric_at_gamma_zero']}")

    # Step 16: Power-law intensity
    print("\n--- Step 16: Power-Law Intensity (Alternative Specification) ---")
    power_law = power_law_intensity_specification(syms)
    print(f"lambda_power(delta) = {power_law['lambda_power_law']}")
    print(f"lambda_power'(delta) = {power_law['lambda_prime_power_law']}")
    print(f"Note: {power_law['note']}")

    # Numerical spot checks
    print("\n--- Numerical Spot Checks (s=100, T=1, sigma=2, k=1.5) ---")
    num_results = numerical_spot_checks()
    for gamma_val, q_results in num_results.items():
        print(f"\n  gamma = {gamma_val}:")
        for q_val, vals in q_results.items():
            print(f"    q={q_val:+d}: r={vals['r']:.4f}, "
                  f"delta^a={vals['delta_a']:.4f}, delta^b={vals['delta_b']:.4f}, "
                  f"spread_adj={vals['spread_adj']:.4f}")

    print("\n--- Infinite-Horizon Numerical Checks ---")
    ih_results = numerical_infinite_horizon_checks()
    for gamma_val, q_results in ih_results.items():
        print(f"\n  gamma = {gamma_val}:")
        for q_val, vals in q_results.items():
            if 'error' in vals:
                print(f"    q={q_val:+d}: {vals['error']}")
            else:
                print(f"    q={q_val:+d}: bar_r^a={vals['bar_r_a']:.4f}, "
                      f"bar_r^b={vals['bar_r_b']:.4f}, "
                      f"admissibility_ok={vals['admissibility_ok']}")

    print("\n" + "=" * 70)
    print("Experiment 1 completed successfully.")
    print("=" * 70)

    return {
        'symbols': syms,
        'v': v,
        'r_a': r_a,
        'r_b': r_b,
        'r': r,
        'bar_r_a': bar_r_a,
        'bar_r_b': bar_r_b,
        'admissibility': admissibility,
        'omega_bound': omega_bound,
        'foc_ask': foc_ask,
        'foc_bid': foc_bid,
        'exp_intensity': exp_results,
        'quotes': quotes,
        'limits': limits,
        'power_law': power_law,
        'numerical': num_results,
        'infinite_horizon_numerical': ih_results,
        'verification': verification,
        'properties': props,
    }


if __name__ == '__main__':
    run_experiment_1()
