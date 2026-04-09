"""
Tests for Experiment 1: Analytical reproduction of AS model formulas.

Verifies:
- Frozen-inventory value function closed form
- Reservation ask/bid price derivations
- Reservation/indifference price formula
- Quote distance formulas
- Spread consistency
- Limiting behavior
- Numerical stability
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from sympy import Rational, simplify, symbols

from exp1_analytical import (
    approximate_ask_distance,
    approximate_bid_distance,
    approximate_spread,
    check_limit_gamma_to_zero,
    check_limit_t_to_T,
    frozen_inventory_value_function,
    numerical_parameter_sweep,
    numerical_quote_distances,
    numerical_reservation_price,
    numerical_spread,
    quote_correction_term,
    reservation_ask_price,
    reservation_bid_price,
    reservation_price,
    run_experiment_1,
    stable_correction_term,
    verify_quote_from_reservation,
    verify_reservation_price_consistency,
    verify_spread_consistency,
)

# Shared symbols
x, s, q, t, T = symbols("x s q t T", real=True)
gamma_sym = symbols("gamma", positive=True)
sigma_sym = symbols("sigma", positive=True)
k_sym = symbols("k", positive=True)
tau = T - t


# ---------------------------------------------------------------------------
# Tests for frozen-inventory value function
# ---------------------------------------------------------------------------

class TestFrozenInventoryValueFunction:
    """Tests for the closed-form frozen-inventory value function."""

    def test_returns_sympy_expression(self) -> None:
        """Value function should return a SymPy expression."""
        v = frozen_inventory_value_function()
        assert isinstance(v, sp.Expr)

    def test_contains_exponential_terms(self) -> None:
        """Value function should contain exponential terms."""
        v = frozen_inventory_value_function()
        assert v.has(sp.exp)

    def test_negative_value(self) -> None:
        """Value function should be negative (exponential utility)."""
        v = frozen_inventory_value_function()
        # Substitute positive values and check sign
        v_num = v.subs([
            (x, 0), (s, 100), (q, 0), (t, 0), (T, 1),
            (gamma_sym, 0.1), (sigma_sym, 2.0)
        ])
        assert float(v_num) < 0

    def test_matches_paper_formula(self) -> None:
        """Value function should match: -exp(-gamma*x)*exp(-gamma*q*s)*exp(0.5*gamma^2*q^2*sigma^2*(T-t))."""
        v = frozen_inventory_value_function()
        expected = (
            -sp.exp(-gamma_sym * x)
            * sp.exp(-gamma_sym * q * s)
            * sp.exp(Rational(1, 2) * gamma_sym**2 * q**2 * sigma_sym**2 * tau)
        )
        residual = simplify(v - expected)
        assert residual == 0, f"Value function mismatch, residual: {residual}"

    def test_zero_inventory_simplification(self) -> None:
        """At q=0, value function should reduce to -exp(-gamma*(x+q*s)) = -exp(-gamma*x)."""
        v = frozen_inventory_value_function()
        v_q0 = simplify(v.subs(q, 0))
        expected_q0 = -sp.exp(-gamma_sym * x)
        residual = simplify(v_q0 - expected_q0)
        assert residual == 0

    def test_terminal_condition(self) -> None:
        """At t=T (tau=0), value function should equal -exp(-gamma*(x+q*s))."""
        v = frozen_inventory_value_function()
        v_terminal = simplify(v.subs(t, T))
        expected = -sp.exp(-gamma_sym * (x + q * s))
        residual = simplify(v_terminal - expected)
        assert residual == 0


# ---------------------------------------------------------------------------
# Tests for reservation prices
# ---------------------------------------------------------------------------

class TestReservationPrices:
    """Tests for reservation ask and bid price derivations."""

    def test_reservation_ask_formula(self) -> None:
        """r^a should equal s + ((1-2q)/2)*gamma*sigma^2*(T-t)."""
        r_a = reservation_ask_price()
        expected = s + (Rational(1, 2) * (1 - 2 * q)) * gamma_sym * sigma_sym**2 * tau
        residual = simplify(r_a - expected)
        assert residual == 0, f"r^a mismatch, residual: {residual}"

    def test_reservation_bid_formula(self) -> None:
        """r^b should equal s + ((-1-2q)/2)*gamma*sigma^2*(T-t)."""
        r_b = reservation_bid_price()
        expected = s + (Rational(1, 2) * (-1 - 2 * q)) * gamma_sym * sigma_sym**2 * tau
        residual = simplify(r_b - expected)
        assert residual == 0, f"r^b mismatch, residual: {residual}"

    def test_ask_above_bid_for_zero_inventory(self) -> None:
        """At q=0, r^a > s > r^b (ask above mid, bid below mid)."""
        r_a = reservation_ask_price()
        r_b = reservation_bid_price()
        # At q=0: r^a = s + 0.5*gamma*sigma^2*tau, r^b = s - 0.5*gamma*sigma^2*tau
        r_a_q0 = simplify(r_a.subs(q, 0))
        r_b_q0 = simplify(r_b.subs(q, 0))
        spread_q0 = simplify(r_a_q0 - r_b_q0)
        # Spread should be gamma*sigma^2*tau > 0
        expected_spread = gamma_sym * sigma_sym**2 * tau
        residual = simplify(spread_q0 - expected_spread)
        assert residual == 0

    def test_ask_bid_symmetry_at_zero_inventory(self) -> None:
        """At q=0, r^a and r^b should be symmetric around s."""
        r_a = reservation_ask_price()
        r_b = reservation_bid_price()
        r_a_q0 = r_a.subs(q, 0)
        r_b_q0 = r_b.subs(q, 0)
        midpoint = simplify(Rational(1, 2) * (r_a_q0 + r_b_q0))
        residual = simplify(midpoint - s)
        assert residual == 0

    def test_utility_indifference_ask(self) -> None:
        """Verify v(x+r^a, s, q-1, t) = v(x, s, q, t) numerically."""
        r_a = reservation_ask_price()
        v = frozen_inventory_value_function()

        # Substitute numeric values
        subs = {gamma_sym: 0.1, sigma_sym: 2.0, T: 1.0, t: 0.0, s: 100.0, q: 0.0}
        r_a_num = float(r_a.subs(subs))

        v_lhs = float(v.subs({**subs, x: 0.0 + r_a_num, q: subs[q] - 1}))
        v_rhs = float(v.subs({**subs, x: 0.0}))

        assert abs(v_lhs - v_rhs) < 1e-10, f"Utility indifference violated: {v_lhs} != {v_rhs}"

    def test_utility_indifference_bid(self) -> None:
        """Verify v(x-r^b, s, q+1, t) = v(x, s, q, t) numerically."""
        r_b = reservation_bid_price()
        v = frozen_inventory_value_function()

        subs = {gamma_sym: 0.1, sigma_sym: 2.0, T: 1.0, t: 0.0, s: 100.0, q: 0.0}
        r_b_num = float(r_b.subs(subs))

        v_lhs = float(v.subs({**subs, x: 0.0 - r_b_num, q: subs[q] + 1}))
        v_rhs = float(v.subs({**subs, x: 0.0}))

        assert abs(v_lhs - v_rhs) < 1e-10, f"Utility indifference violated: {v_lhs} != {v_rhs}"


# ---------------------------------------------------------------------------
# Tests for reservation/indifference price
# ---------------------------------------------------------------------------

class TestReservationIndifferencePrice:
    """Tests for the reservation/indifference price r(s,q,t)."""

    def test_reservation_price_formula(self) -> None:
        """r should equal s - q*gamma*sigma^2*(T-t)."""
        r_a = reservation_ask_price()
        r_b = reservation_bid_price()
        r = reservation_price(r_a, r_b)
        expected = s - q * gamma_sym * sigma_sym**2 * tau
        residual = simplify(r - expected)
        assert residual == 0, f"r mismatch, residual: {residual}"

    def test_reservation_price_midpoint(self) -> None:
        """r should be the midpoint of r^a and r^b."""
        r_a = reservation_ask_price()
        r_b = reservation_bid_price()
        r = reservation_price(r_a, r_b)
        assert verify_reservation_price_consistency(r_a, r_b, r)

    def test_reservation_price_at_zero_inventory(self) -> None:
        """At q=0, reservation price should equal mid-price s."""
        r_a = reservation_ask_price()
        r_b = reservation_bid_price()
        r = reservation_price(r_a, r_b)
        r_q0 = simplify(r.subs(q, 0))
        residual = simplify(r_q0 - s)
        assert residual == 0

    def test_reservation_price_at_maturity(self) -> None:
        """At t=T, reservation price should equal s regardless of inventory."""
        r_a = reservation_ask_price()
        r_b = reservation_bid_price()
        r = reservation_price(r_a, r_b)
        r_terminal = simplify(r.subs(t, T))
        residual = simplify(r_terminal - s)
        assert residual == 0

    def test_reservation_price_inventory_adjustment_sign(self) -> None:
        """Positive inventory should push reservation price below mid-price."""
        # r = s - q*gamma*sigma^2*tau
        # For q > 0, gamma > 0, sigma > 0, tau > 0: r < s
        r_num = numerical_reservation_price(100.0, 1.0, 0.0, 1.0, 0.1, 2.0)
        assert r_num < 100.0, "Positive inventory should lower reservation price"

    def test_reservation_price_negative_inventory(self) -> None:
        """Negative inventory should push reservation price above mid-price."""
        r_num = numerical_reservation_price(100.0, -1.0, 0.0, 1.0, 0.1, 2.0)
        assert r_num > 100.0, "Negative inventory should raise reservation price"


# ---------------------------------------------------------------------------
# Tests for quote correction term
# ---------------------------------------------------------------------------

class TestQuoteCorrectionTerm:
    """Tests for the quote correction term (1/gamma)*ln(1+gamma/k)."""

    def test_correction_term_formula(self) -> None:
        """Correction term should equal (1/gamma)*ln(1+gamma/k)."""
        correction = quote_correction_term()
        expected = (1 / gamma_sym) * sp.ln(1 + gamma_sym / k_sym)
        residual = simplify(correction - expected)
        assert residual == 0

    def test_correction_term_positive(self) -> None:
        """Correction term should be positive for gamma, k > 0."""
        for gamma_val in [0.01, 0.1, 0.5, 1.0]:
            val = stable_correction_term(gamma_val, 1.5)
            assert val > 0, f"Correction term should be positive, got {val}"

    def test_correction_term_limit_gamma_zero(self) -> None:
        """As gamma -> 0, (1/gamma)*ln(1+gamma/k) -> 1/k."""
        k_val = 1.5
        limit = 1.0 / k_val
        for gamma_val in [1e-4, 1e-5, 1e-6]:
            val = stable_correction_term(gamma_val, k_val)
            assert abs(val - limit) < 1e-3, (
                f"Correction term should approach 1/k={limit:.4f}, got {val:.6f}"
            )

    def test_correction_term_decreasing_in_gamma(self) -> None:
        """Correction term (1/gamma)*ln(1+gamma/k) is decreasing in gamma."""
        k_val = 1.5
        gammas = [0.01, 0.1, 0.5]
        vals = [stable_correction_term(g, k_val) for g in gammas]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1], (
                f"Correction term should decrease with gamma: {vals}"
            )

    def test_stable_correction_term_numerical_stability(self) -> None:
        """stable_correction_term should not produce NaN or Inf for small gamma."""
        for gamma_val in [1e-10, 1e-8, 1e-6, 1e-4]:
            val = stable_correction_term(gamma_val, 1.5)
            assert np.isfinite(val), f"Non-finite value for gamma={gamma_val}: {val}"

    def test_stable_correction_term_invalid_inputs(self) -> None:
        """Should raise ValueError for non-positive inputs."""
        with pytest.raises(ValueError):
            stable_correction_term(-0.1, 1.5)
        with pytest.raises(ValueError):
            stable_correction_term(0.1, -1.5)


# ---------------------------------------------------------------------------
# Tests for approximate quote formulas
# ---------------------------------------------------------------------------

class TestApproximateQuoteFormulas:
    """Tests for the approximate optimal quote distance formulas."""

    def test_spread_equals_sum_of_distances(self) -> None:
        """Total spread should equal delta^a + delta^b."""
        delta_a = approximate_ask_distance()
        delta_b = approximate_bid_distance()
        spread = approximate_spread()
        assert verify_spread_consistency(delta_a, delta_b, spread)

    def test_ask_distance_formula(self) -> None:
        """delta^a should equal ((1-2q)/2)*gamma*sigma^2*tau + (1/gamma)*ln(1+gamma/k)."""
        delta_a = approximate_ask_distance()
        expected = (
            (Rational(1, 2) * (1 - 2 * q)) * gamma_sym * sigma_sym**2 * tau
            + (1 / gamma_sym) * sp.ln(1 + gamma_sym / k_sym)
        )
        residual = simplify(delta_a - expected)
        assert residual == 0

    def test_bid_distance_formula(self) -> None:
        """delta^b should equal ((1+2q)/2)*gamma*sigma^2*tau + (1/gamma)*ln(1+gamma/k)."""
        delta_b = approximate_bid_distance()
        expected = (
            (Rational(1, 2) * (1 + 2 * q)) * gamma_sym * sigma_sym**2 * tau
            + (1 / gamma_sym) * sp.ln(1 + gamma_sym / k_sym)
        )
        residual = simplify(delta_b - expected)
        assert residual == 0

    def test_symmetric_quotes_at_zero_inventory(self) -> None:
        """At q=0, delta^a should equal delta^b (symmetric quoting)."""
        delta_a = approximate_ask_distance()
        delta_b = approximate_bid_distance()
        diff = simplify(delta_a.subs(q, 0) - delta_b.subs(q, 0))
        assert diff == 0

    def test_ask_distance_positive_inventory(self) -> None:
        """For q > 0, delta^a < delta^b (ask closer, bid farther to reduce inventory)."""
        gamma_val, sigma_val, k_val = 0.1, 2.0, 1.5
        for q_val in [1, 2]:
            da, db = numerical_quote_distances(q_val, 0.0, 1.0, gamma_val, sigma_val, k_val)
            assert da < db, f"For q={q_val}>0, delta^a should be < delta^b"

    def test_bid_distance_negative_inventory(self) -> None:
        """For q < 0, delta^b < delta^a (bid closer, ask farther to increase inventory)."""
        gamma_val, sigma_val, k_val = 0.1, 2.0, 1.5
        for q_val in [-1, -2]:
            da, db = numerical_quote_distances(q_val, 0.0, 1.0, gamma_val, sigma_val, k_val)
            assert db < da, f"For q={q_val}<0, delta^b should be < delta^a"

    def test_spread_at_maturity(self) -> None:
        """At t=T (tau=0), spread should equal pure liquidity component (2/gamma)*ln(1+gamma/k)."""
        gamma_val, sigma_val, k_val = 0.1, 2.0, 1.5
        spread_T = numerical_spread(1.0, 1.0, gamma_val, sigma_val, k_val)
        liquidity = 2.0 * stable_correction_term(gamma_val, k_val)
        assert abs(spread_T - liquidity) < 1e-12

    def test_quote_from_reservation_consistency(self) -> None:
        """p^a = r + spread/2 = s + delta^a and p^b = r - spread/2 = s - delta^b."""
        r_a = reservation_ask_price()
        r_b = reservation_bid_price()
        r = reservation_price(r_a, r_b)
        delta_a = approximate_ask_distance()
        delta_b = approximate_bid_distance()
        ask_ok, bid_ok = verify_quote_from_reservation(r, delta_a, delta_b)
        assert ask_ok, "Ask quote consistency failed"
        assert bid_ok, "Bid quote consistency failed"

    def test_spread_formula(self) -> None:
        """Spread should equal gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)."""
        spread = approximate_spread()
        expected = (
            gamma_sym * sigma_sym**2 * tau
            + (2 / gamma_sym) * sp.ln(1 + gamma_sym / k_sym)
        )
        residual = simplify(spread - expected)
        assert residual == 0


# ---------------------------------------------------------------------------
# Tests for numerical functions
# ---------------------------------------------------------------------------

class TestNumericalFunctions:
    """Tests for numerical evaluation functions."""

    def test_numerical_reservation_price_zero_inventory(self) -> None:
        """At q=0, numerical reservation price should equal mid-price."""
        r = numerical_reservation_price(100.0, 0.0, 0.0, 1.0, 0.1, 2.0)
        assert abs(r - 100.0) < 1e-12

    def test_numerical_reservation_price_at_maturity(self) -> None:
        """At t=T, numerical reservation price should equal mid-price."""
        r = numerical_reservation_price(100.0, 2.0, 1.0, 1.0, 0.1, 2.0)
        assert abs(r - 100.0) < 1e-12

    def test_numerical_quote_distances_symmetric_at_zero_q(self) -> None:
        """At q=0, delta^a should equal delta^b."""
        da, db = numerical_quote_distances(0.0, 0.0, 1.0, 0.1, 2.0, 1.5)
        assert abs(da - db) < 1e-12

    def test_numerical_spread_positive(self) -> None:
        """Spread should always be positive."""
        for gamma_val in [0.01, 0.1, 0.5]:
            for t_val in [0.0, 0.5, 0.95]:
                spread = numerical_spread(t_val, 1.0, gamma_val, 2.0, 1.5)
                assert spread > 0, f"Spread should be positive, got {spread}"

    def test_numerical_spread_decreasing_to_maturity(self) -> None:
        """Spread should decrease as t approaches T (inventory risk term vanishes)."""
        gamma_val, sigma_val, k_val = 0.1, 2.0, 1.5
        spread_t0 = numerical_spread(0.0, 1.0, gamma_val, sigma_val, k_val)
        spread_t05 = numerical_spread(0.5, 1.0, gamma_val, sigma_val, k_val)
        spread_tT = numerical_spread(1.0, 1.0, gamma_val, sigma_val, k_val)
        assert spread_t0 > spread_t05 > spread_tT

    def test_liquidity_spread_values(self) -> None:
        """Liquidity spread (2/gamma)*ln(1+gamma/k) should match paper reference values."""
        k_val = 1.5
        # Paper reference: ~1.33 for gamma=0.01, ~1.29 for gamma=0.1, ~1.15 for gamma=0.5
        expected = {0.01: 1.33, 0.1: 1.29, 0.5: 1.15}
        for gamma_val, ref in expected.items():
            liq = 2.0 * stable_correction_term(gamma_val, k_val)
            assert abs(liq - ref) < 0.02, (
                f"Liquidity spread for gamma={gamma_val}: got {liq:.4f}, expected ~{ref}"
            )

    def test_parameter_sweep_shape(self) -> None:
        """Parameter sweep should return correct number of results."""
        gamma_values = [0.01, 0.1]
        q_values = [-1, 0, 1]
        t_values = [0.0, 0.5]
        results = numerical_parameter_sweep(gamma_values, q_values, t_values)
        expected_count = len(gamma_values) * len(q_values) * len(t_values)
        assert len(results) == expected_count

    def test_parameter_sweep_keys(self) -> None:
        """Each sweep result should contain required keys."""
        results = numerical_parameter_sweep([0.1], [0], [0.0])
        required_keys = {"gamma", "q", "t", "tau", "r", "delta_a", "delta_b", "spread", "correction"}
        assert required_keys.issubset(set(results[0].keys()))


# ---------------------------------------------------------------------------
# Tests for limiting behavior
# ---------------------------------------------------------------------------

class TestLimitingBehavior:
    """Tests for limiting behavior as t->T and gamma->0."""

    def test_limit_t_to_T_inventory_risk_vanishes(self) -> None:
        """At t=T, inventory risk term should be zero."""
        for gamma_val in [0.01, 0.1, 0.5]:
            result = check_limit_t_to_T(gamma_val, sigma_val=2.0, k_val=1.5)
            assert result["inventory_risk_at_T"] == 0.0

    def test_limit_t_to_T_spread_equals_liquidity(self) -> None:
        """At t=T, spread should equal pure liquidity component."""
        for gamma_val in [0.01, 0.1, 0.5]:
            result = check_limit_t_to_T(gamma_val, sigma_val=2.0, k_val=1.5)
            assert result["spread_equals_liquidity"], (
                f"Spread at T should equal liquidity for gamma={gamma_val}"
            )

    def test_limit_gamma_to_zero_correction_converges(self) -> None:
        """As gamma->0, correction term should converge to 1/k."""
        result = check_limit_gamma_to_zero(sigma_val=2.0, k_val=1.5)
        assert result["correction_converges"], (
            f"Correction term should converge to 1/k: "
            f"got {result['correction_small_gamma']:.6f}, "
            f"expected {result['correction_limit_1_over_k']:.6f}"
        )

    def test_limit_gamma_to_zero_asymmetry_vanishes(self) -> None:
        """As gamma->0, inventory asymmetry delta^a - delta^b should vanish."""
        result = check_limit_gamma_to_zero(sigma_val=2.0, k_val=1.5, q_val=1.0)
        assert result["asymmetry_vanishes"], (
            f"Asymmetry should vanish as gamma->0: got {result['asymmetry_small_gamma']:.2e}"
        )

    def test_inventory_correction_proportional_to_gamma(self) -> None:
        """Inventory correction q*gamma*sigma^2*tau should scale linearly with gamma."""
        q_val, sigma_val, tau_val = 1.0, 2.0, 1.0
        gamma1, gamma2 = 0.1, 0.2
        corr1 = q_val * gamma1 * sigma_val**2 * tau_val
        corr2 = q_val * gamma2 * sigma_val**2 * tau_val
        assert abs(corr2 / corr1 - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestRunExperiment1:
    """Integration test for the full experiment 1 runner."""

    def test_run_experiment_1_all_verified(self) -> None:
        """Full experiment 1 should pass all symbolic verifications."""
        results = run_experiment_1(verbose=False)
        assert results["all_verified"], (
            "Not all formulas verified in experiment 1"
        )

    def test_run_experiment_1_returns_required_keys(self) -> None:
        """Experiment 1 results should contain all required keys."""
        results = run_experiment_1(verbose=False)
        required_keys = {
            "v_expr", "r_a_expr", "r_b_expr", "r_expr",
            "spread_expr", "delta_a_expr", "delta_b_expr",
            "correction_expr", "sweep_results", "all_verified",
        }
        assert required_keys.issubset(set(results.keys()))

    def test_run_experiment_1_sweep_results_nonempty(self) -> None:
        """Sweep results should be non-empty."""
        results = run_experiment_1(verbose=False)
        assert len(results["sweep_results"]) > 0

    def test_run_experiment_1_residuals_zero(self) -> None:
        """All symbolic residuals should be zero."""
        results = run_experiment_1(verbose=False)
        assert results["r_a_residual"] == 0
        assert results["r_b_residual"] == 0
        assert results["r_residual"] == 0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_time_to_maturity(self) -> None:
        """At tau=0, quote distances should equal pure correction term."""
        gamma_val, k_val = 0.1, 1.5
        correction = stable_correction_term(gamma_val, k_val)
        for q_val in [-2, -1, 0, 1, 2]:
            da, db = numerical_quote_distances(q_val, 1.0, 1.0, gamma_val, 2.0, k_val)
            assert abs(da - correction) < 1e-12, f"delta^a at tau=0 should equal correction"
            assert abs(db - correction) < 1e-12, f"delta^b at tau=0 should equal correction"

    def test_large_inventory_quote_asymmetry(self) -> None:
        """Large positive inventory should create large ask-bid asymmetry."""
        gamma_val, sigma_val, k_val = 0.5, 2.0, 1.5
        da_large, db_large = numerical_quote_distances(5.0, 0.0, 1.0, gamma_val, sigma_val, k_val)
        da_zero, db_zero = numerical_quote_distances(0.0, 0.0, 1.0, gamma_val, sigma_val, k_val)
        # Large positive q: delta^a should be much smaller than at q=0
        assert da_large < da_zero
        # Large positive q: delta^b should be much larger than at q=0
        assert db_large > db_zero

    def test_spread_independent_of_inventory(self) -> None:
        """Total spread delta^a + delta^b should be independent of inventory."""
        gamma_val, sigma_val, k_val = 0.1, 2.0, 1.5
        t_val, T_val = 0.0, 1.0
        spread_ref = numerical_spread(t_val, T_val, gamma_val, sigma_val, k_val)
        for q_val in [-3, -1, 0, 1, 3]:
            da, db = numerical_quote_distances(q_val, t_val, T_val, gamma_val, sigma_val, k_val)
            total = da + db
            assert abs(total - spread_ref) < 1e-12, (
                f"Spread should be independent of q, got {total:.6f} vs {spread_ref:.6f}"
            )

    def test_numerical_values_finite(self) -> None:
        """All numerical outputs should be finite."""
        for gamma_val in [0.01, 0.1, 0.5]:
            for q_val in [-5, 0, 5]:
                for t_val in [0.0, 0.5, 0.99]:
                    r = numerical_reservation_price(100.0, q_val, t_val, 1.0, gamma_val, 2.0)
                    da, db = numerical_quote_distances(q_val, t_val, 1.0, gamma_val, 2.0, 1.5)
                    spread = numerical_spread(t_val, 1.0, gamma_val, 2.0, 1.5)
                    assert np.isfinite(r)
                    assert np.isfinite(da)
                    assert np.isfinite(db)
                    assert np.isfinite(spread)
