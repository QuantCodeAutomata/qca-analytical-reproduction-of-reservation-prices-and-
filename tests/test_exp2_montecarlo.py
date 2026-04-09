"""
Tests for Experiment 2: Monte Carlo market-making simulation.

Verifies:
- Simulation mechanics (fill probabilities, cash/inventory updates)
- Strategy implementations (inventory vs symmetric)
- Statistical properties of simulation outputs
- Convergence behavior as gamma -> 0
- Paper-consistent qualitative results
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from exp2_montecarlo import (
    DEFAULT_PARAMS,
    GAMMA_VALUES,
    compute_liquidity_spread,
    compute_quote_distances_inventory,
    compute_quote_distances_symmetric,
    compute_reservation_price,
    compute_spread,
    compute_statistics,
    run_experiment_2,
    simulate_paths,
    stable_correction_term,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_params() -> Dict:
    """Small parameter set for fast tests."""
    return {
        "S0": 100.0,
        "q0": 0.0,
        "X0": 0.0,
        "T": 1.0,
        "dt": 0.005,
        "N": 200,
        "sigma": 2.0,
        "A": 140.0,
        "k": 1.5,
        "n_paths": 100,  # Fewer paths for speed
    }


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed random generator for reproducibility."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Tests for utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    """Tests for utility/helper functions."""

    def test_stable_correction_term_positive(self) -> None:
        """Correction term should be positive."""
        for gamma in [0.01, 0.1, 0.5]:
            val = stable_correction_term(gamma, 1.5)
            assert val > 0

    def test_stable_correction_term_limit(self) -> None:
        """As gamma -> 0, correction term -> 1/k."""
        k = 1.5
        val = stable_correction_term(1e-8, k)
        assert abs(val - 1.0 / k) < 1e-5

    def test_compute_spread_positive(self) -> None:
        """Spread should always be positive."""
        for tau in [0.0, 0.5, 1.0]:
            spread = compute_spread(tau, 0.1, 2.0, 1.5)
            assert spread >= 0

    def test_compute_spread_at_zero_tau(self) -> None:
        """At tau=0, spread should equal liquidity component only."""
        gamma, k = 0.1, 1.5
        spread = compute_spread(0.0, gamma, 2.0, k)
        liquidity = 2.0 * stable_correction_term(gamma, k)
        assert abs(spread - liquidity) < 1e-12

    def test_compute_reservation_price_array(self) -> None:
        """Reservation price should work with numpy arrays."""
        S = np.array([100.0, 101.0, 99.0])
        q = np.array([0.0, 1.0, -1.0])
        r = compute_reservation_price(S, q, 0.5, 0.1, 2.0)
        assert r.shape == (3,)
        assert np.all(np.isfinite(r))

    def test_compute_reservation_price_zero_q(self) -> None:
        """At q=0, reservation price should equal mid-price."""
        S = np.array([100.0])
        q = np.array([0.0])
        r = compute_reservation_price(S, q, 0.5, 0.1, 2.0)
        assert abs(r[0] - 100.0) < 1e-12

    def test_compute_quote_distances_inventory_symmetry(self) -> None:
        """At q=0, inventory strategy should give symmetric quotes."""
        q = np.array([0.0])
        da, db = compute_quote_distances_inventory(q, 0.5, 0.1, 2.0, 1.5)
        assert abs(da[0] - db[0]) < 1e-12

    def test_compute_quote_distances_symmetric_equal(self) -> None:
        """Symmetric strategy should always give equal quote distances."""
        da, db = compute_quote_distances_symmetric(0.5, 0.1, 2.0, 1.5)
        assert abs(da - db) < 1e-12

    def test_compute_liquidity_spread_paper_values(self) -> None:
        """Liquidity spread should match paper reference values."""
        k = 1.5
        expected = {0.01: 1.33, 0.1: 1.29, 0.5: 1.15}
        for gamma, ref in expected.items():
            liq = compute_liquidity_spread(gamma, k)
            assert abs(liq - ref) < 0.02, (
                f"Liquidity spread for gamma={gamma}: got {liq:.4f}, expected ~{ref}"
            )


# ---------------------------------------------------------------------------
# Tests for simulate_paths
# ---------------------------------------------------------------------------

class TestSimulatePaths:
    """Tests for the core simulation function."""

    def test_output_shapes(self, small_params: Dict, rng: np.random.Generator) -> None:
        """Output arrays should have correct shapes."""
        n_paths = small_params["n_paths"]
        N = small_params["N"]
        result = simulate_paths("inventory", 0.1, n_paths, small_params, rng)

        assert result["terminal_profit"].shape == (n_paths,)
        assert result["terminal_inventory"].shape == (n_paths,)
        assert result["sample_S"].shape == (N + 1,)
        assert result["sample_r"].shape == (N + 1,)
        assert result["sample_pa"].shape == (N,)
        assert result["sample_pb"].shape == (N,)
        assert result["sample_q"].shape == (N + 1,)

    def test_initial_conditions(self, small_params: Dict, rng: np.random.Generator) -> None:
        """Sample path should start at initial conditions."""
        result = simulate_paths("inventory", 0.1, 10, small_params, rng)
        assert abs(result["sample_S"][0] - small_params["S0"]) < 1e-12
        assert abs(result["sample_q"][0] - small_params["q0"]) < 1e-12

    def test_terminal_profit_finite(self, small_params: Dict, rng: np.random.Generator) -> None:
        """All terminal profits should be finite."""
        result = simulate_paths("inventory", 0.1, small_params["n_paths"], small_params, rng)
        assert np.all(np.isfinite(result["terminal_profit"]))

    def test_terminal_inventory_integer_valued(
        self, small_params: Dict, rng: np.random.Generator
    ) -> None:
        """Terminal inventory should be integer-valued (unit fills)."""
        result = simulate_paths("inventory", 0.1, small_params["n_paths"], small_params, rng)
        q_T = result["terminal_inventory"]
        # Check that all values are close to integers
        assert np.all(np.abs(q_T - np.round(q_T)) < 1e-10)

    def test_both_strategies_run(self, small_params: Dict, rng: np.random.Generator) -> None:
        """Both inventory and symmetric strategies should run without error."""
        for strategy in ["inventory", "symmetric"]:
            result = simulate_paths(strategy, 0.1, 10, small_params, rng)
            assert "terminal_profit" in result

    def test_invalid_strategy_raises(self, small_params: Dict, rng: np.random.Generator) -> None:
        """Invalid strategy name should raise AssertionError."""
        with pytest.raises(AssertionError):
            simulate_paths("invalid_strategy", 0.1, 10, small_params, rng)

    def test_ask_above_bid_in_sample_path(
        self, small_params: Dict, rng: np.random.Generator
    ) -> None:
        """Ask price should generally be above bid price in sample path."""
        result = simulate_paths("inventory", 0.1, 10, small_params, rng)
        # Check that ask > bid for most steps (may not hold for extreme inventory)
        ask_above_bid = result["sample_pa"] > result["sample_pb"]
        assert np.mean(ask_above_bid) > 0.9, "Ask should be above bid for most steps"

    def test_symmetric_strategy_equal_distances(
        self, small_params: Dict, rng: np.random.Generator
    ) -> None:
        """Symmetric strategy should have equal ask/bid distances from mid-price."""
        result = simulate_paths("symmetric", 0.1, 10, small_params, rng)
        # For symmetric strategy, p^a - S = S - p^b at each step
        S = result["sample_S"][:-1]  # S at each step (before update)
        pa = result["sample_pa"]
        pb = result["sample_pb"]
        ask_dist = pa - S
        bid_dist = S - pb
        # Should be equal (within floating point)
        assert np.allclose(ask_dist, bid_dist, atol=1e-10)

    def test_inventory_strategy_zero_q_symmetric(
        self, small_params: Dict, rng: np.random.Generator
    ) -> None:
        """At q=0, inventory strategy should give same distances as symmetric."""
        # Run a single path with q=0 throughout (no fills)
        params_no_fills = {**small_params, "A": 0.0}  # Zero intensity = no fills
        rng_inv = np.random.default_rng(123)
        rng_sym = np.random.default_rng(123)

        result_inv = simulate_paths("inventory", 0.1, 1, params_no_fills, rng_inv)
        result_sym = simulate_paths("symmetric", 0.1, 1, params_no_fills, rng_sym)

        # With no fills, q stays at 0, so inventory strategy = symmetric
        assert np.allclose(result_inv["sample_pa"], result_sym["sample_pa"], atol=1e-10)
        assert np.allclose(result_inv["sample_pb"], result_sym["sample_pb"], atol=1e-10)

    def test_reproducibility_with_same_seed(self, small_params: Dict) -> None:
        """Same seed should produce identical results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result1 = simulate_paths("inventory", 0.1, 50, small_params, rng1)
        result2 = simulate_paths("inventory", 0.1, 50, small_params, rng2)
        assert np.allclose(result1["terminal_profit"], result2["terminal_profit"])


# ---------------------------------------------------------------------------
# Tests for compute_statistics
# ---------------------------------------------------------------------------

class TestComputeStatistics:
    """Tests for the statistics computation function."""

    def test_returns_required_keys(self) -> None:
        """Statistics should contain required keys."""
        profits = np.random.randn(100)
        inventories = np.random.randn(100)
        stats = compute_statistics(profits, inventories)
        assert "mean_profit" in stats
        assert "std_profit" in stats
        assert "mean_final_q" in stats
        assert "std_final_q" in stats

    def test_mean_profit_correct(self) -> None:
        """Mean profit should match numpy mean."""
        profits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_statistics(profits, np.zeros(5))
        assert abs(stats["mean_profit"] - 3.0) < 1e-12

    def test_std_profit_correct(self) -> None:
        """Std profit should use ddof=1 (sample std)."""
        profits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_statistics(profits, np.zeros(5))
        expected_std = np.std(profits, ddof=1)
        assert abs(stats["std_profit"] - expected_std) < 1e-12

    def test_single_path(self) -> None:
        """Should handle single path without error."""
        stats = compute_statistics(np.array([42.0]), np.array([0.0]))
        assert stats["mean_profit"] == 42.0
        assert np.isnan(stats["std_profit"])  # std of single value is NaN with ddof=1


# ---------------------------------------------------------------------------
# Tests for qualitative properties
# ---------------------------------------------------------------------------

class TestQualitativeProperties:
    """Tests for qualitative properties matching paper expectations."""

    @pytest.fixture(autouse=True)
    def run_simulation(self) -> None:
        """Run a small simulation for qualitative tests."""
        params = {**DEFAULT_PARAMS, "n_paths": 500}
        self.results = run_experiment_2(
            gamma_values=[0.1, 0.5],
            params=params,
            seed=42,
            verbose=False,
        )

    def test_inventory_lower_profit_std_than_symmetric(self) -> None:
        """Inventory strategy should have lower profit std than symmetric."""
        for gamma in [0.1, 0.5]:
            inv_std = self.results[(gamma, "inventory")]["std_profit"]
            sym_std = self.results[(gamma, "symmetric")]["std_profit"]
            assert inv_std < sym_std, (
                f"gamma={gamma}: inventory std_profit={inv_std:.2f} should be < "
                f"symmetric std_profit={sym_std:.2f}"
            )

    def test_inventory_lower_inventory_std_than_symmetric(self) -> None:
        """Inventory strategy should have lower terminal inventory std than symmetric."""
        for gamma in [0.1, 0.5]:
            inv_std = self.results[(gamma, "inventory")]["std_final_q"]
            sym_std = self.results[(gamma, "symmetric")]["std_final_q"]
            assert inv_std < sym_std, (
                f"gamma={gamma}: inventory std_q={inv_std:.2f} should be < "
                f"symmetric std_q={sym_std:.2f}"
            )

    def test_higher_gamma_lower_inventory_std(self) -> None:
        """Higher risk aversion should reduce inventory dispersion for inventory strategy."""
        inv_std_01 = self.results[(0.1, "inventory")]["std_final_q"]
        inv_std_05 = self.results[(0.5, "inventory")]["std_final_q"]
        assert inv_std_05 < inv_std_01, (
            f"Higher gamma should reduce inventory std: "
            f"gamma=0.1: {inv_std_01:.2f}, gamma=0.5: {inv_std_05:.2f}"
        )

    def test_symmetric_profit_higher_or_comparable(self) -> None:
        """Symmetric strategy should have higher or comparable mean profit."""
        for gamma in [0.1, 0.5]:
            inv_mean = self.results[(gamma, "inventory")]["mean_profit"]
            sym_mean = self.results[(gamma, "symmetric")]["mean_profit"]
            # Symmetric should have higher mean profit (especially at high gamma)
            # Allow some tolerance for statistical noise
            assert sym_mean >= inv_mean - 5.0, (
                f"gamma={gamma}: symmetric mean_profit={sym_mean:.2f} should be >= "
                f"inventory mean_profit={inv_mean:.2f} (with tolerance)"
            )

    def test_summary_dataframe_shape(self) -> None:
        """Summary DataFrame should have correct shape."""
        summary = self.results["summary"]
        assert isinstance(summary, pd.DataFrame)
        # 2 gamma values * 2 strategies = 4 rows
        assert len(summary) == 4

    def test_summary_dataframe_columns(self) -> None:
        """Summary DataFrame should have required columns."""
        summary = self.results["summary"]
        required_cols = {
            "gamma", "strategy", "liquidity_spread",
            "mean_profit", "std_profit", "mean_final_q", "std_final_q"
        }
        assert required_cols.issubset(set(summary.columns))


# ---------------------------------------------------------------------------
# Tests for convergence as gamma -> 0
# ---------------------------------------------------------------------------

class TestConvergenceGammaToZero:
    """Tests for convergence of inventory and symmetric strategies as gamma -> 0."""

    def test_strategies_converge_at_small_gamma(self) -> None:
        """At small gamma, inventory and symmetric strategies should be more similar."""
        params = {**DEFAULT_PARAMS, "n_paths": 500}
        results = run_experiment_2(
            gamma_values=[0.01, 0.5],
            params=params,
            seed=42,
            verbose=False,
        )

        # Ratio of std(profit) inventory/symmetric should be closer to 1 at small gamma
        ratio_small = (
            results[(0.01, "inventory")]["std_profit"]
            / results[(0.01, "symmetric")]["std_profit"]
        )
        ratio_large = (
            results[(0.5, "inventory")]["std_profit"]
            / results[(0.5, "symmetric")]["std_profit"]
        )

        assert ratio_small > ratio_large, (
            f"Strategies should converge at small gamma: "
            f"ratio at gamma=0.01: {ratio_small:.3f}, "
            f"ratio at gamma=0.5: {ratio_large:.3f}"
        )

    def test_inventory_correction_small_at_small_gamma(self) -> None:
        """At small gamma, inventory correction q*gamma*sigma^2*tau should be small."""
        gamma_small = 0.01
        q_val = np.array([1.0])
        tau = 0.5
        sigma = 2.0
        k = 1.5

        da_inv, db_inv = compute_quote_distances_inventory(q_val, tau, gamma_small, sigma, k)
        da_sym, db_sym = compute_quote_distances_symmetric(tau, gamma_small, sigma, k)

        # Inventory asymmetry should be small
        asymmetry = abs(da_inv[0] - db_inv[0])
        assert asymmetry < 0.1, f"Inventory asymmetry should be small at gamma=0.01: {asymmetry:.4f}"


# ---------------------------------------------------------------------------
# Tests for full experiment runner
# ---------------------------------------------------------------------------

class TestRunExperiment2:
    """Integration tests for the full experiment 2 runner."""

    def test_run_experiment_2_returns_all_keys(self) -> None:
        """Results should contain keys for all gamma/strategy combinations."""
        params = {**DEFAULT_PARAMS, "n_paths": 50}
        results = run_experiment_2(
            gamma_values=[0.1],
            params=params,
            seed=42,
            verbose=False,
        )
        assert (0.1, "inventory") in results
        assert (0.1, "symmetric") in results
        assert "summary" in results

    def test_run_experiment_2_summary_correct_rows(self) -> None:
        """Summary should have one row per gamma/strategy combination."""
        params = {**DEFAULT_PARAMS, "n_paths": 50}
        results = run_experiment_2(
            gamma_values=[0.01, 0.1, 0.5],
            params=params,
            seed=42,
            verbose=False,
        )
        assert len(results["summary"]) == 6  # 3 gamma * 2 strategies

    def test_run_experiment_2_profits_positive(self) -> None:
        """Mean terminal profits should be positive (market maker earns spread)."""
        params = {**DEFAULT_PARAMS, "n_paths": 200}
        results = run_experiment_2(
            gamma_values=[0.1],
            params=params,
            seed=42,
            verbose=False,
        )
        for strategy in ["inventory", "symmetric"]:
            mean_profit = results[(0.1, strategy)]["mean_profit"]
            assert mean_profit > 0, (
                f"Mean profit should be positive for {strategy}: {mean_profit:.2f}"
            )

    def test_run_experiment_2_sample_path_length(self) -> None:
        """Sample path should have correct length."""
        params = {**DEFAULT_PARAMS, "n_paths": 50}
        results = run_experiment_2(
            gamma_values=[0.1],
            params=params,
            seed=42,
            verbose=False,
        )
        N = params["N"]
        assert len(results[(0.1, "inventory")]["sample_S"]) == N + 1
        assert len(results[(0.1, "inventory")]["sample_pa"]) == N


# ---------------------------------------------------------------------------
# Regression tests with known expected outcomes
# ---------------------------------------------------------------------------

class TestRegressionValues:
    """Regression tests with known expected outcomes."""

    def test_liquidity_spread_gamma_001(self) -> None:
        """Liquidity spread for gamma=0.01, k=1.5 should be approximately 1.33."""
        liq = compute_liquidity_spread(0.01, 1.5)
        assert abs(liq - 1.33) < 0.01

    def test_liquidity_spread_gamma_01(self) -> None:
        """Liquidity spread for gamma=0.1, k=1.5 should be approximately 1.29."""
        liq = compute_liquidity_spread(0.1, 1.5)
        assert abs(liq - 1.29) < 0.01

    def test_liquidity_spread_gamma_05(self) -> None:
        """Liquidity spread for gamma=0.5, k=1.5 should be approximately 1.15."""
        liq = compute_liquidity_spread(0.5, 1.5)
        assert abs(liq - 1.15) < 0.01

    def test_spread_at_t0_gamma_01(self) -> None:
        """Total spread at t=0 for gamma=0.1, sigma=2, k=1.5, T=1 should be ~1.69."""
        # spread = 0.1 * 4 * 1 + 2 * stable_correction_term(0.1, 1.5)
        # = 0.4 + 2 * 0.6454 = 0.4 + 1.2908 = 1.6908
        spread = compute_spread(1.0, 0.1, 2.0, 1.5)
        assert abs(spread - 1.6908) < 0.001

    def test_reservation_price_known_value(self) -> None:
        """Reservation price for known inputs should match expected value."""
        # r = 100 - 1 * 0.1 * 4 * 0.5 = 100 - 0.2 = 99.8
        S = np.array([100.0])
        q = np.array([1.0])
        r = compute_reservation_price(S, q, 0.5, 0.1, 2.0)
        assert abs(r[0] - 99.8) < 1e-10

    def test_quote_distances_known_values(self) -> None:
        """Quote distances for known inputs should match expected values."""
        # gamma=0.1, sigma=2, k=1.5, q=0, tau=1
        # correction = log1p(0.1/1.5) / 0.1 = log1p(0.0667) / 0.1
        # delta^a = delta^b = correction (at q=0)
        gamma, sigma, k = 0.1, 2.0, 1.5
        q = np.array([0.0])
        correction = stable_correction_term(gamma, k)
        da, db = compute_quote_distances_inventory(q, 1.0, gamma, sigma, k)
        # At q=0, tau=1: delta^a = 0.5*gamma*sigma^2*tau + correction = 0.2 + correction
        # Wait: ((1-2*0)/2)*gamma*sigma^2*tau = 0.5*0.1*4*1 = 0.2
        expected_da = 0.5 * gamma * sigma**2 * 1.0 + correction
        assert abs(da[0] - expected_da) < 1e-10
