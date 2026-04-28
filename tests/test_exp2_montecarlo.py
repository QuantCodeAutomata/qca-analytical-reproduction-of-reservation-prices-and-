"""
Tests for Experiment 2: Monte Carlo replication of AS market-making strategies.

Tests verify:
- Core formula functions (reservation price, spread, intensity)
- Single-path simulation correctness
- Monte Carlo statistical properties
- Hypothesis verification (inventory strategy reduces risk)
- Gamma -> 0 convergence behavior
"""

import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from exp2_montecarlo import (
    PARAMS,
    reservation_price,
    spread_adjustment,
    total_spread,
    execution_intensity,
    simulate_path,
    run_monte_carlo,
    compute_summary_stats,
    run_diagnostics,
)


@pytest.fixture
def params():
    """Provide simulation parameters."""
    return PARAMS.copy()


@pytest.fixture
def rng():
    """Provide a seeded random generator."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Tests for core formula functions
# ---------------------------------------------------------------------------

class TestCoreFormulas:
    """Tests for reservation price, spread, and intensity formulas."""

    def test_reservation_price_at_zero_inventory(self):
        """r(S, 0, gamma, sigma, tau) should equal S."""
        S, gamma, sigma, tau = 100.0, 0.1, 2.0, 1.0
        r = reservation_price(S, 0, gamma, sigma, tau)
        assert abs(r - S) < 1e-10, f"r at q=0 should be S={S}, got {r}"

    def test_reservation_price_positive_inventory(self):
        """r < S when q > 0."""
        S, gamma, sigma, tau = 100.0, 0.1, 2.0, 1.0
        for q in [1, 2, 5]:
            r = reservation_price(S, q, gamma, sigma, tau)
            assert r < S, f"r should be < S when q={q} > 0, got r={r}"

    def test_reservation_price_negative_inventory(self):
        """r > S when q < 0."""
        S, gamma, sigma, tau = 100.0, 0.1, 2.0, 1.0
        for q in [-1, -2, -5]:
            r = reservation_price(S, q, gamma, sigma, tau)
            assert r > S, f"r should be > S when q={q} < 0, got r={r}"

    def test_reservation_price_at_terminal_time(self):
        """r -> S as tau -> 0."""
        S, gamma, sigma = 100.0, 0.1, 2.0
        r = reservation_price(S, 5, gamma, sigma, 0.0)
        assert abs(r - S) < 1e-10, f"r should equal S at tau=0, got {r}"

    def test_spread_adjustment_positive(self):
        """Spread adjustment should be positive."""
        for gamma in [0.01, 0.1, 0.5]:
            adj = spread_adjustment(gamma, 1.5)
            assert adj > 0, f"Spread adjustment should be positive, got {adj}"

    def test_spread_adjustment_matches_paper(self):
        """Verify spread adjustment matches paper's reported values."""
        k = 1.5
        expected = {0.1: 1.29, 0.01: 1.33, 0.5: 1.15}
        for gamma, exp_val in expected.items():
            adj = 2 * spread_adjustment(gamma, k)  # full spread = 2 * half-spread
            assert abs(adj - exp_val) < 0.01, \
                f"Full spread for gamma={gamma}: got {adj:.4f}, expected ~{exp_val}"

    def test_total_spread_positive(self):
        """Total spread should be positive."""
        for gamma in [0.01, 0.1, 0.5]:
            Delta = total_spread(gamma, 2.0, 1.0, 1.5)
            assert Delta > 0, f"Total spread should be positive, got {Delta}"

    def test_execution_intensity_decreasing(self):
        """Execution intensity should decrease with quote distance."""
        A, k = 140.0, 1.5
        deltas = [0.1, 0.5, 1.0, 2.0]
        intensities = [execution_intensity(d, A, k) for d in deltas]
        for i in range(len(intensities) - 1):
            assert intensities[i] > intensities[i + 1], \
                f"Intensity should decrease: {intensities[i]} <= {intensities[i+1]}"

    def test_execution_intensity_at_zero_distance(self):
        """At delta=0, intensity should equal A."""
        A, k = 140.0, 1.5
        lam = execution_intensity(0.0, A, k)
        assert abs(lam - A) < 1e-10, f"Intensity at delta=0 should be A={A}, got {lam}"

    def test_execution_probability_bounded(self):
        """Execution probability lambda*dt should be in [0, 1]."""
        A, k, dt = 140.0, 1.5, 0.005
        for delta in [0.1, 0.5, 1.0, 2.0]:
            prob = execution_intensity(delta, A, k) * dt
            assert 0 <= prob <= 1, \
                f"Execution probability should be in [0,1], got {prob} for delta={delta}"


# ---------------------------------------------------------------------------
# Tests for single-path simulation
# ---------------------------------------------------------------------------

class TestSinglePathSimulation:
    """Tests for single-path simulation correctness."""

    def test_simulate_path_returns_required_keys(self, params, rng):
        """Verify simulate_path returns all required keys."""
        result = simulate_path(0.1, 'inventory', params, rng)
        for key in ['Pi_T', 'q_T', 'X_T', 'S_T']:
            assert key in result, f"Missing key: {key}"

    def test_simulate_path_trajectory_keys(self, params, rng):
        """Verify trajectory keys when store_trajectory=True."""
        result = simulate_path(0.1, 'inventory', params, rng, store_trajectory=True)
        for key in ['S_traj', 'r_traj', 'p_a_traj', 'p_b_traj', 'q_traj']:
            assert key in result, f"Missing trajectory key: {key}"

    def test_trajectory_length(self, params, rng):
        """Verify trajectory length equals N+1."""
        result = simulate_path(0.1, 'inventory', params, rng, store_trajectory=True)
        N = params['N']
        assert len(result['S_traj']) == N + 1
        assert len(result['q_traj']) == N + 1

    def test_terminal_profit_formula(self, params, rng):
        """Verify Pi_T = X_T + q_T * S_T."""
        result = simulate_path(0.1, 'inventory', params, rng)
        Pi_T = result['Pi_T']
        X_T = result['X_T']
        q_T = result['q_T']
        S_T = result['S_T']
        assert abs(Pi_T - (X_T + q_T * S_T)) < 1e-10, \
            f"Pi_T should equal X_T + q_T*S_T"

    def test_initial_price_in_trajectory(self, params, rng):
        """Verify trajectory starts at S0."""
        result = simulate_path(0.1, 'inventory', params, rng, store_trajectory=True)
        assert abs(result['S_traj'][0] - params['S0']) < 1e-10

    def test_both_strategies_run(self, params, rng):
        """Verify both strategies simulate without errors."""
        for strategy in ['inventory', 'symmetric']:
            result = simulate_path(0.1, strategy, params, rng)
            assert 'Pi_T' in result

    def test_price_steps_are_binomial(self, params):
        """Verify mid-price follows binomial steps of ±sigma*sqrt(dt)."""
        sigma = params['sigma']
        dt = params['dt']
        step_size = sigma * np.sqrt(dt)

        rng = np.random.default_rng(123)
        result = simulate_path(0.1, 'inventory', params, rng, store_trajectory=True)
        S_traj = result['S_traj']

        # Check that all price changes are ±step_size
        diffs = np.diff(S_traj)
        for diff in diffs:
            assert abs(abs(diff) - step_size) < 1e-10, \
                f"Price step should be ±{step_size:.4f}, got {diff:.6f}"


# ---------------------------------------------------------------------------
# Tests for Monte Carlo statistics
# ---------------------------------------------------------------------------

class TestMonteCarloStatistics:
    """Tests for Monte Carlo simulation statistical properties."""

    def test_monte_carlo_returns_arrays(self, params, rng):
        """Verify run_monte_carlo returns numpy arrays."""
        results = run_monte_carlo(0.1, 'inventory', params, rng)
        assert isinstance(results['profits'], np.ndarray)
        assert isinstance(results['inventories'], np.ndarray)

    def test_monte_carlo_array_length(self, params, rng):
        """Verify arrays have correct length."""
        results = run_monte_carlo(0.1, 'inventory', params, rng)
        assert len(results['profits']) == params['n_paths']
        assert len(results['inventories']) == params['n_paths']

    def test_summary_stats_keys(self, params, rng):
        """Verify summary stats contain all required keys."""
        results = run_monte_carlo(0.1, 'inventory', params, rng)
        stats = compute_summary_stats(
            results['profits'], results['inventories'], 0.1, params
        )
        for key in ['spread', 'mean_profit', 'std_profit', 'mean_q', 'std_q']:
            assert key in stats, f"Missing stats key: {key}"

    def test_std_profit_positive(self, params, rng):
        """Standard deviation of profit should be positive."""
        results = run_monte_carlo(0.1, 'inventory', params, rng)
        stats = compute_summary_stats(
            results['profits'], results['inventories'], 0.1, params
        )
        assert stats['std_profit'] > 0

    def test_mean_profit_positive(self, params, rng):
        """Mean profit should be positive (market maker earns spread)."""
        results = run_monte_carlo(0.1, 'inventory', params, rng)
        stats = compute_summary_stats(
            results['profits'], results['inventories'], 0.1, params
        )
        assert stats['mean_profit'] > 0, \
            f"Mean profit should be positive, got {stats['mean_profit']}"


# ---------------------------------------------------------------------------
# Tests for hypothesis verification
# ---------------------------------------------------------------------------

class TestHypothesisVerification:
    """Tests for the paper's main hypotheses."""

    @pytest.fixture(scope='class')
    def simulation_results(self):
        """Run a small simulation for hypothesis testing."""
        test_params = PARAMS.copy()
        test_params['n_paths'] = 200  # reduced for test speed

        results = {}
        for gamma in [0.1, 0.5]:
            results[gamma] = {}
            for strategy in ['inventory', 'symmetric']:
                rng = np.random.default_rng(42 + int(gamma * 100))
                mc = run_monte_carlo(gamma, strategy, test_params, rng)
                stats = compute_summary_stats(
                    mc['profits'], mc['inventories'], gamma, test_params
                )
                results[gamma][strategy] = stats
        return results

    def test_inventory_lower_std_profit_gamma01(self, simulation_results):
        """Inventory strategy should have lower std(profit) than symmetric at gamma=0.1."""
        inv_std = simulation_results[0.1]['inventory']['std_profit']
        sym_std = simulation_results[0.1]['symmetric']['std_profit']
        assert inv_std < sym_std, \
            f"Inventory std_profit ({inv_std:.2f}) should be < symmetric ({sym_std:.2f})"

    def test_inventory_lower_std_q_gamma01(self, simulation_results):
        """Inventory strategy should have lower std(q) than symmetric at gamma=0.1."""
        inv_std_q = simulation_results[0.1]['inventory']['std_q']
        sym_std_q = simulation_results[0.1]['symmetric']['std_q']
        assert inv_std_q < sym_std_q, \
            f"Inventory std_q ({inv_std_q:.2f}) should be < symmetric ({sym_std_q:.2f})"

    def test_inventory_lower_std_profit_gamma05(self, simulation_results):
        """Inventory strategy should have lower std(profit) than symmetric at gamma=0.5."""
        inv_std = simulation_results[0.5]['inventory']['std_profit']
        sym_std = simulation_results[0.5]['symmetric']['std_profit']
        assert inv_std < sym_std, \
            f"Inventory std_profit ({inv_std:.2f}) should be < symmetric ({sym_std:.2f})"

    def test_larger_gamma_lower_inventory_profit(self, simulation_results):
        """Larger gamma should reduce mean profit for inventory strategy."""
        profit_01 = simulation_results[0.1]['inventory']['mean_profit']
        profit_05 = simulation_results[0.5]['inventory']['mean_profit']
        assert profit_05 < profit_01, \
            f"gamma=0.5 profit ({profit_05:.2f}) should be < gamma=0.1 ({profit_01:.2f})"


# ---------------------------------------------------------------------------
# Tests for diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """Tests for diagnostic checks."""

    def test_diagnostics_pass(self, params):
        """Verify all diagnostic checks pass."""
        run_diagnostics(params)  # Should not raise

    def test_lambda_dt_below_one(self, params):
        """Verify lambda*dt < 1 for typical quote distances."""
        A, k, dt = params['A'], params['k'], params['dt']
        # At delta=0 (worst case), lambda = A
        max_prob = A * dt
        assert max_prob < 1, \
            f"lambda*dt at delta=0 should be < 1, got {max_prob}"


# ---------------------------------------------------------------------------
# Tests for edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_time_to_maturity(self):
        """At tau=0, spread should equal static component only."""
        gamma, sigma, k = 0.1, 2.0, 1.5
        tau = 0.0
        Delta = total_spread(gamma, sigma, tau, k)
        static = (2 / gamma) * np.log(1 + gamma / k)
        assert abs(Delta - static) < 1e-10, \
            f"At tau=0, spread should equal static component {static:.4f}, got {Delta:.4f}"

    def test_large_inventory_reservation_price(self):
        """Large inventory should shift reservation price significantly."""
        S, gamma, sigma, tau = 100.0, 0.5, 2.0, 1.0
        r_large_q = reservation_price(S, 10, gamma, sigma, tau)
        r_small_q = reservation_price(S, 1, gamma, sigma, tau)
        assert r_large_q < r_small_q, \
            "Larger positive inventory should give lower reservation price"

    def test_single_path_no_crash(self):
        """Single path simulation should not crash for extreme gamma."""
        params = PARAMS.copy()
        rng = np.random.default_rng(0)
        for gamma in [0.001, 1.0]:
            result = simulate_path(gamma, 'inventory', params, rng)
            assert np.isfinite(result['Pi_T']), \
                f"Pi_T should be finite for gamma={gamma}"

    def test_spread_adjustment_limit_at_small_gamma(self):
        """(2/gamma)*ln(1+gamma/k) -> 2/k as gamma -> 0."""
        k = 1.5
        expected_limit = 2 / k
        gamma_small = 1e-6
        adj = (2 / gamma_small) * np.log(1 + gamma_small / k)
        assert abs(adj - expected_limit) < 1e-4, \
            f"Spread adj limit should be 2/k={expected_limit:.4f}, got {adj:.6f}"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestExperiment2Integration:
    """Integration test for the full experiment."""

    def test_run_experiment_2_small(self, tmp_path):
        """Run experiment with reduced paths to verify end-to-end."""
        from exp2_montecarlo import run_experiment_2

        test_params = PARAMS.copy()
        test_params['n_paths'] = 50
        test_params['gamma_values'] = [0.1]

        results = run_experiment_2(params=test_params, output_dir=str(tmp_path))

        assert 'summary' in results
        assert isinstance(results['summary'], pd.DataFrame)
        assert len(results['summary']) == 2  # inventory + symmetric

    def test_summary_csv_created(self, tmp_path):
        """Verify summary CSV is created."""
        from exp2_montecarlo import run_experiment_2

        test_params = PARAMS.copy()
        test_params['n_paths'] = 50
        test_params['gamma_values'] = [0.1]

        run_experiment_2(params=test_params, output_dir=str(tmp_path))

        csv_path = tmp_path / 'exp2_summary.csv'
        assert csv_path.exists(), "Summary CSV should be created"
