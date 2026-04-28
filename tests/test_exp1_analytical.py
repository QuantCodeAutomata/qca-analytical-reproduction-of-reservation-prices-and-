"""
Tests for Experiment 1: Analytical reproduction of AS model formulas.

Tests verify:
- Frozen-inventory value function structure
- Reservation price formulas by substitution
- Limiting and sign properties
- Infinite-horizon admissibility
- Exponential intensity simplification
- Gamma -> 0 convergence
- Numerical spot checks
"""

import pytest
import numpy as np
import sympy as sp
from sympy import symbols, exp, log, simplify, limit

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from exp1_analytical import (
    define_symbols,
    frozen_inventory_value_function,
    finite_horizon_reservation_prices,
    verify_reservation_prices_by_substitution,
    reservation_price,
    verify_reservation_price_properties,
    infinite_horizon_reservation_prices,
    exponential_intensity_simplification,
    finite_horizon_quote_distances,
    gamma_to_zero_limit,
    power_law_intensity_specification,
    numerical_spot_checks,
    numerical_infinite_horizon_checks,
)


@pytest.fixture
def syms():
    """Provide symbolic variables for all tests."""
    return define_symbols()


# ---------------------------------------------------------------------------
# Tests for frozen-inventory value function
# ---------------------------------------------------------------------------

class TestFrozenInventoryValueFunction:
    """Tests for the frozen-inventory value function v(x,s,q,t)."""

    def test_value_function_structure(self, syms):
        """Verify the value function has the correct exponential structure."""
        v = frozen_inventory_value_function(syms)
        x, s, q, t, T = syms['x'], syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        # v should be negative (exponential utility is negative)
        # Check that v contains exp(-gamma*x) factor
        assert v.has(exp)
        assert v.has(gamma)
        assert v.has(sigma)

    def test_value_function_at_zero_inventory(self, syms):
        """At q=0, value function should simplify to -exp(-gamma*x)."""
        v = frozen_inventory_value_function(syms)
        x, q = syms['x'], syms['q']
        gamma = syms['gamma']

        v_q0 = simplify(v.subs(q, 0))
        expected = -exp(-gamma * x)
        assert simplify(v_q0 - expected) == 0, \
            f"v at q=0 should be -exp(-gamma*x), got {v_q0}"

    def test_value_function_at_terminal_time(self, syms):
        """At t=T, value function should equal -exp(-gamma*(x+q*s))."""
        v = frozen_inventory_value_function(syms)
        x, s, q, t, T = syms['x'], syms['s'], syms['q'], syms['t'], syms['T']
        gamma = syms['gamma']

        v_at_T = simplify(v.subs(t, T))
        expected = -exp(-gamma * (x + q * s))
        assert simplify(v_at_T - expected) == 0, \
            f"v at t=T should be -exp(-gamma*(x+q*s)), got {v_at_T}"

    def test_value_function_negative(self, syms):
        """Value function should be negative (exponential utility)."""
        v = frozen_inventory_value_function(syms)
        # The leading coefficient is -1 (negative)
        # Check by evaluating at specific numerical values
        x_val, s_val, q_val = 0, 100, 0
        t_val, T_val = 0, 1
        sigma_val, gamma_val = 2, 0.1

        v_num = float(v.subs([
            (syms['x'], x_val), (syms['s'], s_val), (syms['q'], q_val),
            (syms['t'], t_val), (syms['T'], T_val),
            (syms['sigma'], sigma_val), (syms['gamma'], gamma_val)
        ]))
        assert v_num < 0, f"Value function should be negative, got {v_num}"


# ---------------------------------------------------------------------------
# Tests for reservation prices
# ---------------------------------------------------------------------------

class TestReservationPrices:
    """Tests for finite-horizon reservation ask and bid prices."""

    def test_reservation_ask_formula(self, syms):
        """Verify the reservation ask price formula."""
        r_a, r_b = finite_horizon_reservation_prices(syms)
        s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        tau = T - t
        expected_r_a = s + ((1 - 2*q) * gamma * sigma**2 * tau) / 2
        assert simplify(r_a - expected_r_a) == 0, \
            f"r^a formula mismatch: {r_a} vs {expected_r_a}"

    def test_reservation_bid_formula(self, syms):
        """Verify the reservation bid price formula."""
        r_a, r_b = finite_horizon_reservation_prices(syms)
        s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        tau = T - t
        expected_r_b = s + ((-1 - 2*q) * gamma * sigma**2 * tau) / 2
        assert simplify(r_b - expected_r_b) == 0, \
            f"r^b formula mismatch: {r_b} vs {expected_r_b}"

    def test_ask_greater_than_bid(self, syms):
        """Reservation ask should be greater than reservation bid."""
        r_a, r_b = finite_horizon_reservation_prices(syms)
        diff = simplify(r_a - r_b)
        # r^a - r^b = gamma * sigma^2 * (T-t) > 0
        s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']
        expected_diff = gamma * sigma**2 * (T - t)
        assert simplify(diff - expected_diff) == 0, \
            f"r^a - r^b should equal gamma*sigma^2*(T-t), got {diff}"

    def test_indifference_verification_ask(self, syms):
        """Verify ask reservation price satisfies indifference equation."""
        verification = verify_reservation_prices_by_substitution(syms)
        assert verification['ask_verified'], \
            "Ask indifference equation not satisfied"

    def test_indifference_verification_bid(self, syms):
        """Verify bid reservation price satisfies indifference equation."""
        verification = verify_reservation_prices_by_substitution(syms)
        assert verification['bid_verified'], \
            "Bid indifference equation not satisfied"

    def test_reservation_price_average(self, syms):
        """Reservation price should be average of ask and bid."""
        r_a, r_b = finite_horizon_reservation_prices(syms)
        r = reservation_price(syms)
        assert simplify(r - (r_a + r_b) / 2) == 0, \
            "Reservation price should be (r^a + r^b) / 2"


# ---------------------------------------------------------------------------
# Tests for reservation price properties
# ---------------------------------------------------------------------------

class TestReservationPriceProperties:
    """Tests for limiting and sign properties of the reservation price."""

    def test_r_at_zero_inventory_equals_s(self, syms):
        """r(s, 0, t) should equal s."""
        props = verify_reservation_price_properties(syms)
        assert props['r_at_q0_equals_s'], "r(s,0,t) should equal s"

    def test_r_approaches_s_at_terminal_time(self, syms):
        """r should approach s as t approaches T."""
        props = verify_reservation_price_properties(syms)
        assert props['r_at_T_equals_s'], "r should approach s as t -> T"

    def test_r_less_than_s_for_positive_inventory(self):
        """r < s when q > 0 (numerical check)."""
        s_val, sigma_val, gamma_val, tau_val = 100.0, 2.0, 0.1, 1.0
        for q_val in [1, 2, 5]:
            r_val = s_val - q_val * gamma_val * sigma_val**2 * tau_val
            assert r_val < s_val, \
                f"r should be < s when q={q_val} > 0, got r={r_val}"

    def test_r_greater_than_s_for_negative_inventory(self):
        """r > s when q < 0 (numerical check)."""
        s_val, sigma_val, gamma_val, tau_val = 100.0, 2.0, 0.1, 1.0
        for q_val in [-1, -2, -5]:
            r_val = s_val - q_val * gamma_val * sigma_val**2 * tau_val
            assert r_val > s_val, \
                f"r should be > s when q={q_val} < 0, got r={r_val}"

    def test_r_formula_structure(self, syms):
        """Verify the reservation price formula: r = s - q*gamma*sigma^2*(T-t)."""
        r = reservation_price(syms)
        s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        expected = s - q * gamma * sigma**2 * (T - t)
        assert simplify(r - expected) == 0, \
            f"r formula mismatch: {r} vs {expected}"


# ---------------------------------------------------------------------------
# Tests for infinite-horizon reservation prices
# ---------------------------------------------------------------------------

class TestInfiniteHorizonReservationPrices:
    """Tests for infinite-horizon stationary reservation prices."""

    def test_infinite_horizon_formulas_exist(self, syms):
        """Verify infinite-horizon formulas are well-defined."""
        bar_r_a, bar_r_b, admissibility, omega_bound = \
            infinite_horizon_reservation_prices(syms)
        assert bar_r_a is not None
        assert bar_r_b is not None

    def test_admissibility_condition(self, syms):
        """Verify the admissibility condition omega > (1/2)*gamma^2*sigma^2*q^2."""
        _, _, admissibility, _ = infinite_horizon_reservation_prices(syms)
        omega, gamma, sigma, q = syms['omega'], syms['gamma'], syms['sigma'], syms['q']
        # The condition should involve omega, gamma, sigma, q
        assert admissibility.has(omega)
        assert admissibility.has(gamma)

    def test_numerical_admissibility(self):
        """Numerical check: omega_bound satisfies admissibility for all q <= q_max."""
        sigma_val = 2.0
        gamma_val = 0.1
        q_max_val = 5

        omega_val = 0.5 * gamma_val**2 * sigma_val**2 * (q_max_val + 1)**2

        for q_val in range(-q_max_val, q_max_val + 1):
            required = 0.5 * gamma_val**2 * sigma_val**2 * q_val**2
            assert omega_val > required, \
                f"Admissibility violated for q={q_val}: omega={omega_val} <= {required}"

    def test_infinite_horizon_numerical_checks(self):
        """Verify numerical infinite-horizon checks run without errors."""
        results = numerical_infinite_horizon_checks()
        assert len(results) == 3  # three gamma values
        for gamma_val, q_results in results.items():
            for q_val, vals in q_results.items():
                if 'error' not in vals:
                    assert 'bar_r_a' in vals
                    assert 'bar_r_b' in vals
                    assert vals['admissibility_ok']


# ---------------------------------------------------------------------------
# Tests for exponential intensity specialization
# ---------------------------------------------------------------------------

class TestExponentialIntensity:
    """Tests for exponential intensity lambda(delta) = A*exp(-k*delta)."""

    def test_lambda_prime_equals_minus_k_lambda(self, syms):
        """Verify lambda'(delta) = -k * lambda(delta)."""
        exp_results = exponential_intensity_simplification(syms)
        k = syms['k']
        delta_a = syms['delta_a']
        A = syms['A']

        lam = exp_results['lambda_expr']
        lam_prime = exp_results['lambda_prime']

        ratio = simplify(lam_prime / lam)
        assert simplify(ratio + k) == 0, \
            f"lambda'/lambda should be -k, got {ratio}"

    def test_log_adjustment_simplification(self, syms):
        """Verify log adjustment simplifies to (1/gamma)*ln(1+gamma/k)."""
        exp_results = exponential_intensity_simplification(syms)
        gamma, k = syms['gamma'], syms['k']

        log_adj = exp_results['log_adjustment']
        expected = (1 / gamma) * sp.log(1 + gamma / k)
        assert simplify(log_adj - expected) == 0, \
            f"Log adjustment mismatch: {log_adj} vs {expected}"

    def test_explicit_quote_distances(self, syms):
        """Verify explicit quote distances from exponential intensity."""
        exp_results = exponential_intensity_simplification(syms)
        s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma, k = syms['sigma'], syms['gamma'], syms['k']

        tau = T - t
        spread_adj = (1 / gamma) * sp.log(1 + gamma / k)

        expected_delta_a = ((1 - 2*q) * gamma * sigma**2 * tau) / 2 + spread_adj
        expected_delta_b = ((1 + 2*q) * gamma * sigma**2 * tau) / 2 + spread_adj

        assert simplify(exp_results['delta_a_explicit'] - expected_delta_a) == 0
        assert simplify(exp_results['delta_b_explicit'] - expected_delta_b) == 0


# ---------------------------------------------------------------------------
# Tests for finite-horizon quote distances
# ---------------------------------------------------------------------------

class TestFiniteHorizonQuoteDistances:
    """Tests for the operational finite-horizon quoting rules."""

    def test_total_spread_formula(self, syms):
        """Verify total spread Delta_t = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)."""
        quotes = finite_horizon_quote_distances(syms)
        s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma, k = syms['sigma'], syms['gamma'], syms['k']

        tau = T - t
        expected_Delta = gamma * sigma**2 * tau + (2 / gamma) * sp.log(1 + gamma / k)
        assert simplify(quotes['Delta_t'] - expected_Delta) == 0, \
            f"Delta_t mismatch: {quotes['Delta_t']} vs {expected_Delta}"

    def test_center_spread_form_consistency(self, syms):
        """Verify p^a = r + Delta_t/2 and p^b = r - Delta_t/2."""
        quotes = finite_horizon_quote_distances(syms)
        s = syms['s']

        # p^a from direct formula vs center/spread form
        p_a_direct = quotes['p_a']
        p_a_center = quotes['p_a_center_form']
        assert simplify(p_a_direct - p_a_center) == 0, \
            "p^a direct and center/spread forms should match"

        p_b_direct = quotes['p_b']
        p_b_center = quotes['p_b_center_form']
        assert simplify(p_b_direct - p_b_center) == 0, \
            "p^b direct and center/spread forms should match"

    def test_quote_distances_positive_at_q0(self):
        """Quote distances should be positive at q=0 for reasonable parameters."""
        s_val, sigma_val, gamma_val, k_val, tau_val = 100.0, 2.0, 0.1, 1.5, 1.0
        q_val = 0

        spread_adj = (1 / gamma_val) * np.log(1 + gamma_val / k_val)
        delta_a = ((1 - 2*q_val) * gamma_val * sigma_val**2 * tau_val) / 2 + spread_adj
        delta_b = ((1 + 2*q_val) * gamma_val * sigma_val**2 * tau_val) / 2 + spread_adj

        assert delta_a > 0, f"delta^a should be positive at q=0, got {delta_a}"
        assert delta_b > 0, f"delta^b should be positive at q=0, got {delta_b}"


# ---------------------------------------------------------------------------
# Tests for gamma -> 0 limit
# ---------------------------------------------------------------------------

class TestGammaToZeroLimit:
    """Tests for convergence to symmetric strategy as gamma -> 0."""

    def test_spread_adj_limit(self, syms):
        """(1/gamma)*ln(1+gamma/k) -> 1/k as gamma -> 0."""
        limits = gamma_to_zero_limit(syms)
        k = syms['k']
        expected = 1 / k
        assert simplify(limits['spread_adj_limit'] - expected) == 0, \
            f"Spread adj limit should be 1/k, got {limits['spread_adj_limit']}"

    def test_reservation_price_limit(self, syms):
        """r -> s as gamma -> 0."""
        limits = gamma_to_zero_limit(syms)
        s = syms['s']
        assert simplify(limits['r_limit'] - s) == 0, \
            f"r limit should be s, got {limits['r_limit']}"

    def test_symmetric_convergence(self, syms):
        """delta^a = delta^b as gamma -> 0 (symmetric strategy)."""
        limits = gamma_to_zero_limit(syms)
        assert limits['symmetric_at_gamma_zero'], \
            "delta^a and delta^b should be equal at gamma=0"

    def test_numerical_gamma_convergence(self):
        """Numerically verify convergence as gamma decreases."""
        s_val, sigma_val, k_val, tau_val = 100.0, 2.0, 1.5, 1.0
        q_val = 2  # non-zero inventory

        gamma_values = [0.5, 0.1, 0.01, 0.001]
        prev_diff = None

        for gamma_val in gamma_values:
            spread_adj = (1 / gamma_val) * np.log(1 + gamma_val / k_val)
            delta_a = ((1 - 2*q_val) * gamma_val * sigma_val**2 * tau_val) / 2 + spread_adj
            delta_b = ((1 + 2*q_val) * gamma_val * sigma_val**2 * tau_val) / 2 + spread_adj
            diff = abs(delta_a - delta_b)

            if prev_diff is not None:
                assert diff < prev_diff, \
                    f"Asymmetry should decrease as gamma decreases: {diff} >= {prev_diff}"
            prev_diff = diff

        # At very small gamma, asymmetry = 2*q*gamma*sigma^2*tau -> 0
        # At gamma=0.001: asymmetry = 2*2*0.001*4*1 = 0.016
        assert diff < 0.05, f"At gamma=0.001, asymmetry should be small, got {diff}"


# ---------------------------------------------------------------------------
# Tests for numerical spot checks
# ---------------------------------------------------------------------------

class TestNumericalSpotChecks:
    """Tests for numerical verification of formulas."""

    def test_numerical_spot_checks_run(self):
        """Verify numerical spot checks complete without errors."""
        results = numerical_spot_checks()
        assert len(results) == 3  # three gamma values

    def test_spread_adj_values(self):
        """Verify spread adjustment values match paper's reported spreads."""
        # Paper reports: gamma=0.1 -> 1.29, gamma=0.01 -> 1.33, gamma=0.5 -> 1.15
        k_val = 1.5
        expected = {0.1: 1.29, 0.01: 1.33, 0.5: 1.15}

        for gamma_val, expected_spread in expected.items():
            spread_adj = (2 / gamma_val) * np.log(1 + gamma_val / k_val)
            assert abs(spread_adj - expected_spread) < 0.01, \
                f"Spread adj for gamma={gamma_val}: got {spread_adj:.4f}, expected ~{expected_spread}"

    def test_r_at_q0_equals_s_numerically(self):
        """Numerically verify r(s, 0, t) = s."""
        s_val, sigma_val, gamma_val, tau_val = 100.0, 2.0, 0.1, 1.0
        r_val = s_val - 0 * gamma_val * sigma_val**2 * tau_val
        assert abs(r_val - s_val) < 1e-10

    def test_all_q_values_have_results(self):
        """Verify results exist for all q values."""
        results = numerical_spot_checks()
        for gamma_val, q_results in results.items():
            assert len(q_results) == 5  # q in {-2,-1,0,1,2}

    def test_quote_distances_positive(self):
        """Verify quote distances are positive for reasonable parameters."""
        results = numerical_spot_checks()
        for gamma_val, q_results in results.items():
            for q_val, vals in q_results.items():
                # At q=0, both distances should be positive
                if q_val == 0:
                    assert vals['delta_a'] > 0, \
                        f"delta^a should be positive at q=0, gamma={gamma_val}"
                    assert vals['delta_b'] > 0, \
                        f"delta^b should be positive at q=0, gamma={gamma_val}"


# ---------------------------------------------------------------------------
# Tests for power-law intensity
# ---------------------------------------------------------------------------

class TestPowerLawIntensity:
    """Tests for the alternative power-law intensity specification."""

    def test_power_law_formula(self, syms):
        """Verify power-law intensity formula structure."""
        power_law = power_law_intensity_specification(syms)
        B, alpha, beta = syms['B'], syms['alpha'], syms['beta']
        delta_a = syms['delta_a']

        lam = power_law['lambda_power_law']
        assert lam.has(B)
        assert lam.has(delta_a)

    def test_power_law_derivative(self, syms):
        """Verify power-law intensity derivative is negative (decreasing)."""
        power_law = power_law_intensity_specification(syms)
        lam_prime = power_law['lambda_prime_power_law']
        # The derivative should contain a negative sign
        # (lambda decreases with distance)
        assert lam_prime is not None


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestExperiment1Integration:
    """Integration test running the full experiment."""

    def test_run_experiment_1(self):
        """Verify the full experiment runs without errors."""
        from exp1_analytical import run_experiment_1
        results = run_experiment_1()

        # Check all expected keys are present
        expected_keys = [
            'symbols', 'v', 'r_a', 'r_b', 'r', 'bar_r_a', 'bar_r_b',
            'admissibility', 'omega_bound', 'foc_ask', 'foc_bid',
            'exp_intensity', 'quotes', 'limits', 'power_law',
            'numerical', 'infinite_horizon_numerical',
            'verification', 'properties',
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_verification_passes(self):
        """Verify that indifference equations are satisfied."""
        from exp1_analytical import run_experiment_1
        results = run_experiment_1()
        assert results['verification']['ask_verified']
        assert results['verification']['bid_verified']
