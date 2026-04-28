"""
Main runner for Avellaneda-Stoikov model experiments.

Executes both experiments and saves results to results/RESULTS.md.

Usage:
    python run_experiments.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from exp1_analytical import run_experiment_1, stable_correction_term
from exp2_montecarlo import (
    DEFAULT_PARAMS,
    GAMMA_VALUES,
    compute_liquidity_spread,
    run_experiment_2,
)
from visualize import generate_all_plots

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def format_exp1_report(exp1_results: Dict) -> str:
    """
    Format Experiment 1 results as a Markdown section.

    Parameters
    ----------
    exp1_results : Dict
        Results from run_experiment_1.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = []
    lines.append("## Experiment 1: Analytical Reproduction of AS Model Formulas")
    lines.append("")
    lines.append("### Symbolic Verification Results")
    lines.append("")
    lines.append("| Formula | Verified |")
    lines.append("|---------|----------|")

    checks = [
        ("Reservation ask price r^a = s + ((1-2q)/2)·γ·σ²·τ", exp1_results["r_a_verified"]),
        ("Reservation bid price r^b = s + ((-1-2q)/2)·γ·σ²·τ", exp1_results["r_b_verified"]),
        ("Reservation price r = s - q·γ·σ²·τ", exp1_results["r_verified"]),
        ("Spread = δᵃ + δᵇ (consistency)", exp1_results["spread_consistent"]),
        ("p^a = r + spread/2 = s + δᵃ", exp1_results["ask_from_reservation_consistent"]),
        ("p^b = r - spread/2 = s - δᵇ", exp1_results["bid_from_reservation_consistent"]),
        ("r = 0.5·(r^a + r^b)", exp1_results["r_consistent"]),
    ]

    for name, verified in checks:
        status = "✓ PASS" if verified else "✗ FAIL"
        lines.append(f"| {name} | {status} |")

    lines.append("")
    lines.append(f"**All formulas verified: {exp1_results['all_verified']}**")
    lines.append("")

    # Symbolic expressions
    lines.append("### Key Symbolic Expressions")
    lines.append("")
    lines.append("```")
    lines.append(f"v(x,s,q,t)  = {exp1_results['v_expr']}")
    lines.append(f"r^a(s,q,t)  = {exp1_results['r_a_expr']}")
    lines.append(f"r^b(s,q,t)  = {exp1_results['r_b_expr']}")
    lines.append(f"r(s,q,t)    = {exp1_results['r_expr']}")
    lines.append(f"spread      = {exp1_results['spread_expr']}")
    lines.append(f"delta^a     = {exp1_results['delta_a_expr']}")
    lines.append(f"delta^b     = {exp1_results['delta_b_expr']}")
    lines.append(f"correction  = {exp1_results['correction_expr']}")
    lines.append("```")
    lines.append("")

    # Limiting behavior
    lines.append("### Limiting Behavior Checks")
    lines.append("")
    lines.append("#### As t → T (τ → 0)")
    lines.append("")
    lines.append("| γ | Inventory Risk at T | Spread at T | Equals Liquidity |")
    lines.append("|---|---------------------|-------------|-----------------|")
    for gamma_val in [0.01, 0.1, 0.5]:
        lim = exp1_results[f"limit_T_gamma_{gamma_val}"]
        lines.append(
            f"| {gamma_val} | {lim['inventory_risk_at_T']:.6f} | "
            f"{lim['spread_at_T']:.6f} | "
            f"{'✓' if lim['spread_equals_liquidity'] else '✗'} |"
        )

    lines.append("")
    lines.append("#### As γ → 0")
    lines.append("")
    lim0 = exp1_results["limit_gamma_to_zero"]
    lines.append(f"- Correction term at γ=1e-6: {lim0['correction_small_gamma']:.6f}")
    lines.append(f"- Limit 1/k: {lim0['correction_limit_1_over_k']:.6f}")
    lines.append(f"- Converges: {'✓' if lim0['correction_converges'] else '✗'}")
    lines.append(f"- Asymmetry at γ=1e-6 (q=1): {lim0['asymmetry_small_gamma']:.2e}")
    lines.append(f"- Asymmetry vanishes: {'✓' if lim0['asymmetry_vanishes'] else '✗'}")
    lines.append("")

    # Numerical sweep table
    lines.append("### Numerical Parameter Sweep (s=100, σ=2, k=1.5, T=1)")
    lines.append("")
    lines.append("Selected results for t=0.0:")
    lines.append("")
    lines.append("| γ | q | τ | r | δᵃ | δᵇ | Spread |")
    lines.append("|---|---|---|---|----|----|--------|")

    for row in exp1_results["sweep_results"]:
        if row["t"] == 0.0:
            lines.append(
                f"| {row['gamma']} | {row['q']:+d} | {row['tau']:.2f} | "
                f"{row['r']:.4f} | {row['delta_a']:.4f} | "
                f"{row['delta_b']:.4f} | {row['spread']:.4f} |"
            )

    lines.append("")
    lines.append("### Liquidity Spread Components (2/γ)·ln(1+γ/k) for k=1.5")
    lines.append("")
    lines.append("| γ | Liquidity Spread | Paper Reference |")
    lines.append("|---|-----------------|-----------------|")
    paper_refs = {0.01: "≈1.33", 0.1: "≈1.29", 0.5: "≈1.15"}
    for gamma_val in [0.01, 0.1, 0.5]:
        liq = 2.0 * stable_correction_term(gamma_val, 1.5)
        lines.append(f"| {gamma_val} | {liq:.4f} | {paper_refs[gamma_val]} |")

    lines.append("")
    return "\n".join(lines)


def format_exp2_report(exp2_results: Dict) -> str:
    """
    Format Experiment 2 results as a Markdown section.

    Parameters
    ----------
    exp2_results : Dict
        Results from run_experiment_2.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = []
    lines.append("## Experiment 2: Monte Carlo Market-Making Simulation")
    lines.append("")
    lines.append("### Simulation Parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for k, v in DEFAULT_PARAMS.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Summary statistics table
    lines.append("### Summary Statistics")
    lines.append("")
    lines.append(
        "| γ | Strategy | Liquidity Spread | Mean Profit | Std(Profit) | "
        "Mean Final q | Std(Final q) |"
    )
    lines.append(
        "|---|----------|-----------------|-------------|-------------|"
        "-------------|-------------|"
    )

    summary_df = exp2_results["summary"]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['gamma']} | {row['strategy']} | {row['liquidity_spread']:.4f} | "
            f"{row['mean_profit']:.2f} | {row['std_profit']:.2f} | "
            f"{row['mean_final_q']:.3f} | {row['std_final_q']:.3f} |"
        )

    lines.append("")

    # Paper reference values
    lines.append("### Paper Reference Values (for qualitative comparison)")
    lines.append("")
    lines.append(
        "| γ | Strategy | Mean Profit (paper) | Std(Profit) (paper) | "
        "Mean q (paper) | Std(q) (paper) |"
    )
    lines.append(
        "|---|----------|---------------------|---------------------|"
        "----------------|----------------|"
    )
    paper_refs = [
        (0.01, "inventory", 66.78, 8.76, -0.02, 4.70),
        (0.01, "symmetric", 67.36, 13.40, -0.31, 8.65),
        (0.1, "inventory", 62.94, 5.89, 0.10, 2.80),
        (0.1, "symmetric", 67.21, 13.43, -0.018, 8.66),
        (0.5, "inventory", 33.92, 4.72, -0.02, 1.88),
        (0.5, "symmetric", 66.20, 14.53, 0.25, 9.06),
    ]
    for gamma, strategy, mp, sp, mq, sq in paper_refs:
        lines.append(f"| {gamma} | {strategy} | {mp} | {sp} | {mq} | {sq} |")

    lines.append("")

    # Qualitative analysis
    lines.append("### Qualitative Analysis")
    lines.append("")

    for gamma in GAMMA_VALUES:
        inv_stats = {
            k: v for k, v in exp2_results[(gamma, "inventory")].items()
            if k in ("mean_profit", "std_profit", "mean_final_q", "std_final_q")
        }
        sym_stats = {
            k: v for k, v in exp2_results[(gamma, "symmetric")].items()
            if k in ("mean_profit", "std_profit", "mean_final_q", "std_final_q")
        }

        profit_risk_reduction = (
            (sym_stats["std_profit"] - inv_stats["std_profit"]) / sym_stats["std_profit"] * 100
        )
        inv_risk_reduction = (
            (sym_stats["std_final_q"] - inv_stats["std_final_q"]) / sym_stats["std_final_q"] * 100
        )
        profit_diff = sym_stats["mean_profit"] - inv_stats["mean_profit"]

        lines.append(f"**γ = {gamma}:**")
        lines.append(
            f"- Profit risk reduction (inventory vs symmetric): {profit_risk_reduction:.1f}%"
        )
        lines.append(
            f"- Inventory risk reduction: {inv_risk_reduction:.1f}%"
        )
        lines.append(
            f"- Symmetric strategy mean profit advantage: {profit_diff:.2f}"
        )
        lines.append("")

    # Convergence check
    lines.append("### Convergence Check (γ → 0)")
    lines.append("")
    gamma_01 = GAMMA_VALUES[0]  # 0.01
    gamma_05 = GAMMA_VALUES[2]  # 0.5

    inv_std_01 = exp2_results[(gamma_01, "inventory")]["std_profit"]
    sym_std_01 = exp2_results[(gamma_01, "symmetric")]["std_profit"]
    inv_std_05 = exp2_results[(gamma_05, "inventory")]["std_profit"]
    sym_std_05 = exp2_results[(gamma_05, "symmetric")]["std_profit"]

    ratio_01 = inv_std_01 / sym_std_01
    ratio_05 = inv_std_05 / sym_std_05

    lines.append(
        f"- std(Profit) ratio inventory/symmetric at γ=0.01: {ratio_01:.3f} "
        f"(closer to 1 = more similar)"
    )
    lines.append(
        f"- std(Profit) ratio inventory/symmetric at γ=0.5: {ratio_05:.3f} "
        f"(further from 1 = more different)"
    )
    lines.append(
        f"- Strategies converge as γ→0: {'✓' if ratio_01 > ratio_05 else '✗'}"
    )
    lines.append("")

    return "\n".join(lines)


def save_results_markdown(
    exp1_results: Dict,
    exp2_results: Dict,
    elapsed_exp1: float,
    elapsed_exp2: float,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    Save all experiment results to results/RESULTS.md.

    Parameters
    ----------
    exp1_results : Dict
        Results from run_experiment_1.
    exp2_results : Dict
        Results from run_experiment_2.
    elapsed_exp1 : float
        Runtime for experiment 1 in seconds.
    elapsed_exp2 : float
        Runtime for experiment 2 in seconds.
    results_dir : Path
        Directory to save results.
    """
    lines = []
    lines.append("# Avellaneda-Stoikov Model: Experiment Results")
    lines.append("")
    lines.append(
        "Analytical reproduction and Monte Carlo replication of the "
        "Avellaneda-Stoikov (2008) market-making model."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append(format_exp1_report(exp1_results))
    lines.append("")
    lines.append(f"*Experiment 1 runtime: {elapsed_exp1:.2f}s*")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append(format_exp2_report(exp2_results))
    lines.append("")
    lines.append(f"*Experiment 2 runtime: {elapsed_exp2:.2f}s*")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Generated Plots")
    lines.append("")
    lines.append("### Experiment 1")
    lines.append("- `exp1_reservation_price.png` — Reservation price vs τ for different q and γ")
    lines.append("- `exp1_quote_distances.png` — Optimal quote distances δᵃ and δᵇ vs τ")
    lines.append("- `exp1_spread_components.png` — Total spread and its components vs τ")
    lines.append("- `exp1_limiting_behavior.png` — Limiting behavior as γ → 0")
    lines.append("")
    lines.append("### Experiment 2")
    lines.append("- `exp2_sample_path_gamma0.1.png` — Sample path with quotes and reservation price")
    lines.append("- `exp2_profit_histograms.png` — Terminal profit distributions")
    lines.append("- `exp2_inventory_histograms.png` — Terminal inventory distributions")
    lines.append("- `exp2_summary_comparison.png` — Bar chart comparison of strategies")
    lines.append("")

    md_path = results_dir / "RESULTS.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all experiments and save results."""
    print("=" * 70)
    print("Avellaneda-Stoikov Model: Full Experiment Suite")
    print("=" * 70)

    # --- Experiment 1 ---
    print("\nRunning Experiment 1: Analytical Reproduction...")
    t0 = time.time()
    exp1_results = run_experiment_1(verbose=True)
    elapsed_exp1 = time.time() - t0
    print(f"\nExperiment 1 completed in {elapsed_exp1:.2f}s")

    # --- Experiment 2 ---
    print("\nRunning Experiment 2: Monte Carlo Simulation...")
    t0 = time.time()
    exp2_results = run_experiment_2(verbose=True, seed=42)
    elapsed_exp2 = time.time() - t0
    print(f"\nExperiment 2 completed in {elapsed_exp2:.2f}s")

    # --- Generate plots ---
    generate_all_plots(
        exp1_results=exp1_results,
        exp2_results=exp2_results,
        results_dir=RESULTS_DIR,
    )

    # --- Save results ---
    save_results_markdown(
        exp1_results=exp1_results,
        exp2_results=exp2_results,
        elapsed_exp1=elapsed_exp1,
        elapsed_exp2=elapsed_exp2,
    )

    print("\n" + "=" * 70)
    print("All experiments completed successfully.")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
