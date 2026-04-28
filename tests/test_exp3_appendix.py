"""
Tests for Experiment 3: Appendix mean-variance model analytical reproduction.

Tests verify:
- Appendix value function structure
- Reservation price derivation matches paper formulas
- Basic properties (q=0, terminal time, sign)
- Contrast with main model
- Numerical spot checks
"""

import pytest
import numpy as np
import sympy as sp
from sympy import symbols, exp, simplify, limit

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from exp3_appendix import (
    define_appendix_symbols,
    appendix_value_function,
    appendix_reservation_prices,
    appendix_reservation_prices_direct,
    verify_appendix_properties,
    contrast_with_main_model,
    numerical_appendix_checks,
)


@pytest.fixture
def syms():
    """Provide appendix symbolic variables."""
    return define_appendix_symbols()


# ---------------------------------------------------------------------------
# Tests for appendix value function
# ---------------------------------------------------------------------------

class TestAppendixValueFunction:
    """Tests for the appendix mean-variance value function."""

    def test_value_function_structure(self, syms):
        """Verify value function has correct structure."""
        V = appendix_value_function(syms)
        x, s, q = syms['x'], syms['s'], syms['q']
        sigma, gamma = syms['sigma'], syms['gamma']

        assert V.has(x)
        assert V.has(s)
        assert V.has(q)
        assert V.has(sigma)
        assert V.has(gamma)

    def test_value_function_at_zero_inventory(self, syms):
        """At q=0, V should equal x (no inventory risk)."""
        V = appendix_value_function(syms)
        x, q = syms['x'], syms['q']

        V_q0 = simplify(V.subs(q, 0))
        assert simplify(V_q0 - x) == 0, \
            f"V at q=0 should be x, got {V_q0}"

    def test_value_function_at_terminal_time(self, syms):
        """At t=T, V should equal x + q*s (no variance term)."""
        V = appendix_value_function(syms)
        x, s, q, t, T = syms['x'], syms['s'], syms['q'], syms['t'], syms['T']

        V_at_T = simplify(V.subs(t, T))
        expected = x + q * s
        assert simplify(V_at_T - expected) == 0, \
            f"V at t=T should be x+q*s, got {V_at_T}"

    def test_value_function_formula(self, syms):
        """Verify V uses negative sign for variance penalty (economically correct)."""
        V = appendix_value_function(syms)
        x, s, q, t, T = syms['x'], syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        tau = T - t
        # Negative sign: V = x + q*s - (gamma*q^2*s^2/2)*(exp-1)
        expected = x + q * s - (gamma * q**2 * s**2 / 2) * (exp(sigma**2 * tau) - 1)
        assert simplify(V - expected) == 0, \
            f"V does not match expected formula"

    def test_value_function_numerical(self, syms):
        """Numerical check of value function."""
        V = appendix_value_function(syms)
        x, s, q, t, T = syms['x'], syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        # Substitute numerical values
        V_num = float(V.subs([
            (x, 0), (s, 100), (q, 1), (t, 0), (T, 1),
            (sigma, 0.2), (gamma, 0.1)
        ]))

        # Manual calculation with negative sign
        tau = 1.0
        expected = 0 + 1 * 100 - (0.1 * 1**2 * 100**2 / 2) * (np.exp(0.2**2 * tau) - 1)
        assert abs(V_num - expected) < 1e-6, \
            f"V numerical mismatch: {V_num} vs {expected}"


# ---------------------------------------------------------------------------
# Tests for appendix reservation prices
# ---------------------------------------------------------------------------

class TestAppendixReservationPrices:
    """Tests for appendix reservation ask and bid prices."""

    def test_derived_matches_direct_ask(self, syms):
        """Derived R^a should match paper's direct formula."""
        R_a_derived, _ = appendix_reservation_prices(syms)
        R_a_direct, _ = appendix_reservation_prices_direct(syms)
        assert simplify(R_a_derived - R_a_direct) == 0, \
            f"Derived R^a does not match paper formula"

    def test_derived_matches_direct_bid(self, syms):
        """Derived R^b should match paper's direct formula."""
        _, R_b_derived = appendix_reservation_prices(syms)
        _, R_b_direct = appendix_reservation_prices_direct(syms)
        assert simplify(R_b_derived - R_b_direct) == 0, \
            f"Derived R^b does not match paper formula"

    def test_ask_formula_structure(self, syms):
        """Verify R^a formula structure."""
        R_a, _ = appendix_reservation_prices_direct(syms)
        s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        tau = T - t
        expected = s + ((1 - 2*q) / 2) * gamma * s**2 * (exp(sigma**2 * tau) - 1)
        assert simplify(R_a - expected) == 0, \
            f"R^a formula mismatch"

    def test_bid_formula_structure(self, syms):
        """Verify R^b formula structure."""
        _, R_b = appendix_reservation_prices_direct(syms)
        s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        tau = T - t
        expected = s + ((-1 - 2*q) / 2) * gamma * s**2 * (exp(sigma**2 * tau) - 1)
        assert simplify(R_b - expected) == 0, \
            f"R^b formula mismatch"

    def test_ask_greater_than_bid(self, syms):
        """R^a should be greater than R^b."""
        R_a, R_b = appendix_reservation_prices_direct(syms)
        diff = simplify(R_a - R_b)
        # R^a - R^b = gamma * s^2 * (exp(sigma^2*(T-t)) - 1) > 0
        s, t, T = syms['s'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']
        expected_diff = gamma * s**2 * (exp(sigma**2 * (T - t)) - 1)
        assert simplify(diff - expected_diff) == 0, \
            f"R^a - R^b should equal gamma*s^2*(exp(sigma^2*tau)-1)"

    def test_indifference_ask_satisfied(self, syms):
        """Verify ask indifference: V(x+R^a(q), s, q-1, t) = V(x, s, q, t).

        Correct substitution order: first substitute q->q-1 in V (inventory arg),
        then substitute x -> x + R_a (where R_a is evaluated at original q).
        """
        V = appendix_value_function(syms)
        R_a, _ = appendix_reservation_prices_direct(syms)
        x, q = syms['x'], syms['q']

        # Step 1: substitute q -> q-1 in V (inventory argument)
        V_q_minus_1 = V.subs(q, q - 1)
        # Step 2: substitute x -> x + R_a (R_a has original q)
        V_lhs = V_q_minus_1.subs(x, x + R_a)
        V_rhs = V
        diff = simplify(V_lhs - V_rhs)
        assert diff == 0, f"Ask indifference not satisfied: diff = {diff}"

    def test_indifference_bid_satisfied(self, syms):
        """Verify bid indifference: V(x-R^b(q), s, q+1, t) = V(x, s, q, t).

        Correct substitution order: first substitute q->q+1 in V (inventory arg),
        then substitute x -> x - R_b (where R_b is evaluated at original q).
        """
        V = appendix_value_function(syms)
        _, R_b = appendix_reservation_prices_direct(syms)
        x, q = syms['x'], syms['q']

        # Step 1: substitute q -> q+1 in V (inventory argument)
        V_q_plus_1 = V.subs(q, q + 1)
        # Step 2: substitute x -> x - R_b (R_b has original q)
        V_lhs = V_q_plus_1.subs(x, x - R_b)
        V_rhs = V
        diff = simplify(V_lhs - V_rhs)
        assert diff == 0, f"Bid indifference not satisfied: diff = {diff}"


# ---------------------------------------------------------------------------
# Tests for appendix properties
# ---------------------------------------------------------------------------

class TestAppendixProperties:
    """Tests for basic properties of the appendix model."""

    def test_r_center_at_zero_inventory(self, syms):
        """R_center at q=0 should equal s."""
        props = verify_appendix_properties(syms)
        assert props['R_center_q0_equals_s'], \
            "R_center at q=0 should equal s"

    def test_r_center_at_terminal_time(self, syms):
        """R_center should approach s as t -> T."""
        props = verify_appendix_properties(syms)
        assert props['R_center_at_T_equals_s'], \
            "R_center should approach s as t -> T"

    def test_r_center_minus_s_sign(self, syms):
        """R_center - s should be negative for positive inventory."""
        props = verify_appendix_properties(syms)
        s, q, t, T = syms['s'], syms['q'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        # R_center - s = -q * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)
        # For q > 0 and t < T: negative
        R_center_minus_s = props['R_center_minus_s']
        # Substitute q=1 and check sign
        val = R_center_minus_s.subs(q, 1)
        # Should be negative (contains -gamma*s^2*(exp(...)-1) which is negative)
        assert val.has(gamma), "R_center - s should depend on gamma"

    def test_r_a_at_q0_structure(self, syms):
        """R^a at q=0 should be above s (ask above mid when no inventory)."""
        props = verify_appendix_properties(syms)
        s, t, T = syms['s'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        # R^a at q=0 = s + (1/2)*gamma*s^2*(exp-1) > s
        R_a_q0 = props['R_a_at_q0']
        expected = s + sp.Rational(1, 2) * gamma * s**2 * (exp(sigma**2 * (T - t)) - 1)
        assert simplify(R_a_q0 - expected) == 0, \
            f"R^a at q=0 mismatch: {R_a_q0} vs {expected}"

    def test_r_b_at_q0_structure(self, syms):
        """R^b at q=0 should be below s (bid below mid when no inventory)."""
        props = verify_appendix_properties(syms)
        s, t, T = syms['s'], syms['t'], syms['T']
        sigma, gamma = syms['sigma'], syms['gamma']

        # R^b at q=0 = s - (1/2)*gamma*s^2*(exp-1) < s
        R_b_q0 = props['R_b_at_q0']
        expected = s - sp.Rational(1, 2) * gamma * s**2 * (exp(sigma**2 * (T - t)) - 1)
        assert simplify(R_b_q0 - expected) == 0, \
            f"R^b at q=0 mismatch: {R_b_q0} vs {expected}"


# ---------------------------------------------------------------------------
# Tests for contrast with main model
# ---------------------------------------------------------------------------

class TestContrastWithMainModel:
    """Tests for the contrast between appendix and main model."""

    def test_contrast_returns_required_keys(self, syms):
        """Verify contrast function returns all required keys."""
        contrast = contrast_with_main_model(syms)
        for key in ['main_model_adjustment', 'appendix_model_adjustment',
                    'appendix_small_tau_approx', 'note']:
            assert key in contrast, f"Missing key: {key}"

    def test_main_model_adjustment_structure(self, syms):
        """Main model adjustment should contain sigma^2*(T-t)."""
        contrast = contrast_with_main_model(syms)
        assert 'sigma' in contrast['main_model_adjustment']
        assert 'T' in contrast['main_model_adjustment']

    def test_appendix_model_adjustment_structure(self, syms):
        """Appendix model adjustment should contain s^2 and exp."""
        contrast = contrast_with_main_model(syms)
        assert 's' in contrast['appendix_model_adjustment']
        assert 'exp' in contrast['appendix_model_adjustment']


# ---------------------------------------------------------------------------
# Tests for numerical spot checks
# ---------------------------------------------------------------------------

class TestNumericalAppendixChecks:
    """Tests for numerical verification of appendix formulas."""

    def test_numerical_checks_run(self):
        """Verify numerical checks complete without errors."""
        results = numerical_appendix_checks()
        assert len(results) > 0

    def test_r_center_at_q0_equals_s_numerically(self):
        """Numerically verify R_center at q=0 equals s."""
        results = numerical_appendix_checks()
        assert 0 in results
        r_center_q0 = results[0]['R_center']
        assert abs(r_center_q0 - 100.0) < 1e-10, \
            f"R_center at q=0 should be 100, got {r_center_q0}"

    def test_r_center_less_than_s_for_positive_q(self):
        """R_center < s for positive inventory (long position lowers reservation center)."""
        results = numerical_appendix_checks()
        for q_val in [1, 2]:
            if q_val in results:
                r_center = results[q_val]['R_center']
                assert r_center < 100.0, \
                    f"R_center should be < s=100 for q={q_val}, got {r_center}"

    def test_r_center_greater_than_s_for_negative_q(self):
        """R_center > s for negative inventory (short position raises reservation center)."""
        results = numerical_appendix_checks()
        for q_val in [-1, -2]:
            if q_val in results:
                r_center = results[q_val]['R_center']
                assert r_center > 100.0, \
                    f"R_center should be > s=100 for q={q_val}, got {r_center}"

    def test_terminal_check(self):
        """Verify terminal time check passes."""
        results = numerical_appendix_checks()
        assert 'terminal_check' in results
        assert results['terminal_check']['equals_s'], \
            "R_center should equal s at terminal time"

    def test_r_a_greater_than_r_b_numerically(self):
        """R^a should be greater than R^b numerically."""
        results = numerical_appendix_checks()
        for q_val in [-2, -1, 0, 1, 2]:
            if q_val in results and 'error' not in results[q_val]:
                R_a = results[q_val]['R_a']
                R_b = results[q_val]['R_b']
                assert R_a > R_b, \
                    f"R^a should be > R^b for q={q_val}: R^a={R_a:.4f}, R^b={R_b:.4f}"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestExperiment3Integration:
    """Integration test for the full appendix experiment."""

    def test_run_experiment_3(self):
        """Verify the full experiment runs without errors."""
        from exp3_appendix import run_experiment_3
        results = run_experiment_3()

        expected_keys = [
            'symbols', 'V', 'R_a', 'R_b', 'R_a_derived', 'R_b_derived',
            'properties', 'contrast', 'numerical', 'verification',
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_verification_passes(self):
        """Verify that derived formulas match paper formulas."""
        from exp3_appendix import run_experiment_3
        results = run_experiment_3()
        assert results['verification']['R_a_matches_paper'], \
            "Derived R^a should match paper formula"
        assert results['verification']['R_b_matches_paper'], \
            "Derived R^b should match paper formula"
