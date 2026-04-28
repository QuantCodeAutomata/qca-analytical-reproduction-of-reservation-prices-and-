"""
Visualization module for Avellaneda-Stoikov model experiments.

Generates all plots for Experiment 1 (analytical) and Experiment 2 (Monte Carlo).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from exp1_analytical import (
    numerical_parameter_sweep,
    numerical_quote_distances,
    numerical_reservation_price,
    numerical_spread,
    stable_correction_term,
)
from exp2_montecarlo import DEFAULT_PARAMS, GAMMA_VALUES, compute_liquidity_spread


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment 1 plots
# ---------------------------------------------------------------------------

def plot_exp1_reservation_price(
    gamma_values: List[float],
    q_values: List[float],
    sigma: float = 2.0,
    s_val: float = 100.0,
    T: float = 1.0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot reservation price r(s,q,t) as a function of time-to-maturity tau
    for different inventory levels and risk aversion values.

    Parameters
    ----------
    gamma_values : List[float]
        Risk aversion values.
    q_values : List[float]
        Inventory values.
    sigma : float
        Volatility.
    s_val : float
        Current mid-price.
    T : float
        Terminal horizon.
    save_path : Optional[Path]
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    tau_grid = np.linspace(0, T, 200)
    fig, axes = plt.subplots(1, len(gamma_values), figsize=(15, 5), sharey=False)

    for ax, gamma in zip(axes, gamma_values):
        for q in q_values:
            r_vals = [
                numerical_reservation_price(s_val, q, T - tau, T, gamma, sigma)
                for tau in tau_grid
            ]
            ax.plot(tau_grid, r_vals, label=f"q={q}")
        ax.axhline(s_val, color="black", linestyle="--", linewidth=0.8, label="mid-price s")
        ax.set_xlabel("Time to maturity τ = T - t")
        ax.set_ylabel("Reservation price r(s,q,t)")
        ax.set_title(f"γ = {gamma}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Reservation Price r(s,q,t) = s - q·γ·σ²·τ\n"
        f"(s={s_val}, σ={sigma}, T={T})",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_exp1_quote_distances(
    gamma_values: List[float],
    q_values: List[float],
    sigma: float = 2.0,
    k: float = 1.5,
    T: float = 1.0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot optimal quote distances delta^a and delta^b as functions of tau
    for different inventory levels and risk aversion values.

    Parameters
    ----------
    gamma_values : List[float]
        Risk aversion values.
    q_values : List[float]
        Inventory values.
    sigma : float
        Volatility.
    k : float
        Intensity decay parameter.
    T : float
        Terminal horizon.
    save_path : Optional[Path]
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    tau_grid = np.linspace(0, T, 200)
    fig, axes = plt.subplots(
        len(gamma_values), 2, figsize=(14, 4 * len(gamma_values)), sharex=True
    )

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(q_values)))

    for row, gamma in enumerate(gamma_values):
        ax_a = axes[row, 0]
        ax_b = axes[row, 1]

        for q, color in zip(q_values, colors):
            delta_a_vals = []
            delta_b_vals = []
            for tau in tau_grid:
                t_val = T - tau
                da, db = numerical_quote_distances(q, t_val, T, gamma, sigma, k)
                delta_a_vals.append(da)
                delta_b_vals.append(db)

            ax_a.plot(tau_grid, delta_a_vals, color=color, label=f"q={q}")
            ax_b.plot(tau_grid, delta_b_vals, color=color, label=f"q={q}")

        for ax, label in [(ax_a, "δᵃ (ask distance)"), (ax_b, "δᵇ (bid distance)")]:
            ax.set_ylabel(label)
            ax.set_title(f"γ = {gamma}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.5)

    for ax in axes[-1]:
        ax.set_xlabel("Time to maturity τ = T - t")

    fig.suptitle(
        "Optimal Quote Distances δᵃ and δᵇ vs Time to Maturity\n"
        f"(σ={sigma}, k={k}, T={T})",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_exp1_spread_components(
    gamma_values: List[float],
    sigma: float = 2.0,
    k: float = 1.5,
    T: float = 1.0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot total spread and its components (inventory risk + liquidity) vs tau.

    Parameters
    ----------
    gamma_values : List[float]
        Risk aversion values.
    sigma : float
        Volatility.
    k : float
        Intensity decay parameter.
    T : float
        Terminal horizon.
    save_path : Optional[Path]
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    tau_grid = np.linspace(0, T, 200)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "orange", "green"]
    for gamma, color in zip(gamma_values, colors):
        spread_vals = [numerical_spread(T - tau, T, gamma, sigma, k) for tau in tau_grid]
        liq_component = 2.0 * stable_correction_term(gamma, k)
        inv_risk_vals = [gamma * sigma**2 * tau for tau in tau_grid]

        ax.plot(tau_grid, spread_vals, color=color, linewidth=2,
                label=f"Total spread (γ={gamma})")
        ax.plot(tau_grid, inv_risk_vals, color=color, linewidth=1.5,
                linestyle="--", label=f"Inventory risk γσ²τ (γ={gamma})")
        ax.axhline(liq_component, color=color, linewidth=1.0,
                   linestyle=":", label=f"Liquidity (2/γ)ln(1+γ/k) (γ={gamma})")

    ax.set_xlabel("Time to maturity τ = T - t")
    ax.set_ylabel("Spread")
    ax.set_title(
        f"Total Spread = γσ²τ + (2/γ)ln(1+γ/k)\n(σ={sigma}, k={k}, T={T})"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_exp1_limiting_behavior(
    sigma: float = 2.0,
    k: float = 1.5,
    T: float = 1.0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot limiting behavior: correction term (1/gamma)*ln(1+gamma/k) vs gamma,
    showing convergence to 1/k as gamma -> 0.

    Parameters
    ----------
    sigma : float
        Volatility.
    k : float
        Intensity decay parameter.
    T : float
        Terminal horizon.
    save_path : Optional[Path]
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    gamma_grid = np.logspace(-4, 0, 300)
    correction_vals = [stable_correction_term(g, k) for g in gamma_grid]
    limit_val = 1.0 / k

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: correction term vs gamma
    ax = axes[0]
    ax.semilogx(gamma_grid, correction_vals, "b-", linewidth=2,
                label=r"$(1/\gamma)\ln(1+\gamma/k)$")
    ax.axhline(limit_val, color="red", linestyle="--", linewidth=1.5,
               label=f"Limit = 1/k = {limit_val:.4f}")
    ax.set_xlabel("γ (log scale)")
    ax.set_ylabel(r"$(1/\gamma)\ln(1+\gamma/k)$")
    ax.set_title(f"Correction Term vs γ (k={k})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: inventory asymmetry delta^a - delta^b vs gamma for q=1
    q_val = 1.0
    tau_val = 0.5
    asymmetry_vals = []
    for g in gamma_grid:
        da, db = numerical_quote_distances(q_val, T - tau_val, T, g, sigma, k)
        asymmetry_vals.append(da - db)

    ax2 = axes[1]
    ax2.semilogx(gamma_grid, asymmetry_vals, "g-", linewidth=2,
                 label=f"δᵃ - δᵇ (q={q_val}, τ={tau_val})")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("γ (log scale)")
    ax2.set_ylabel("δᵃ - δᵇ (inventory asymmetry)")
    ax2.set_title(f"Inventory Asymmetry vs γ\n(q={q_val}, τ={tau_val}, σ={sigma})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Limiting Behavior as γ → 0", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Experiment 2 plots
# ---------------------------------------------------------------------------

def plot_exp2_sample_path(
    sim_results: Dict,
    gamma: float,
    params: Dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a sample path showing mid-price, reservation price, and bid/ask quotes.

    Parameters
    ----------
    sim_results : Dict
        Simulation results from simulate_paths for the inventory strategy.
    gamma : float
        Risk aversion used in simulation.
    params : Dict
        Model parameters.
    save_path : Optional[Path]
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    N = params["N"]
    T = params["T"]
    dt = params["dt"]

    time_grid = np.linspace(0, T, N + 1)
    time_grid_steps = np.linspace(0, T - dt, N)

    sample_S = sim_results["sample_S"]
    sample_r = sim_results["sample_r"]
    sample_pa = sim_results["sample_pa"]
    sample_pb = sim_results["sample_pb"]
    sample_q = sim_results["sample_q"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: prices
    ax1 = axes[0]
    ax1.plot(time_grid, sample_S, "k-", linewidth=1.2, label="Mid-price S_t", alpha=0.8)
    ax1.plot(time_grid, sample_r, "b--", linewidth=1.5, label="Reservation price r_t")
    ax1.step(time_grid_steps, sample_pa, "r-", linewidth=1.0, label="Ask quote p^a_t",
             alpha=0.7, where="post")
    ax1.step(time_grid_steps, sample_pb, "g-", linewidth=1.0, label="Bid quote p^b_t",
             alpha=0.7, where="post")
    ax1.set_ylabel("Price")
    ax1.set_title(
        f"Sample Path — Inventory Strategy (γ={gamma})\n"
        f"σ={params['sigma']}, A={params['A']}, k={params['k']}"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: inventory
    ax2 = axes[1]
    ax2.step(time_grid, sample_q, "purple", linewidth=1.5, label="Inventory q_t",
             where="post")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Inventory q_t")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_exp2_profit_histograms(
    all_results: Dict,
    gamma_values: List[float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot overlaid profit histograms for inventory vs symmetric strategies
    for each gamma value.

    Parameters
    ----------
    all_results : Dict
        Results dictionary from run_experiment_2.
    gamma_values : List[float]
        Risk aversion values.
    save_path : Optional[Path]
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, len(gamma_values), figsize=(15, 5), sharey=False)

    for ax, gamma in zip(axes, gamma_values):
        inv_profits = all_results[(gamma, "inventory")]["terminal_profit"]
        sym_profits = all_results[(gamma, "symmetric")]["terminal_profit"]

        # Determine common bin range
        all_profits = np.concatenate([inv_profits, sym_profits])
        bins = np.linspace(np.percentile(all_profits, 1), np.percentile(all_profits, 99), 50)

        ax.hist(inv_profits, bins=bins, alpha=0.6, color="blue",
                label=f"Inventory\nμ={np.mean(inv_profits):.1f}, σ={np.std(inv_profits, ddof=1):.1f}")
        ax.hist(sym_profits, bins=bins, alpha=0.6, color="orange",
                label=f"Symmetric\nμ={np.mean(sym_profits):.1f}, σ={np.std(sym_profits, ddof=1):.1f}")

        ax.axvline(np.mean(inv_profits), color="blue", linewidth=2, linestyle="--")
        ax.axvline(np.mean(sym_profits), color="orange", linewidth=2, linestyle="--")

        ax.set_xlabel("Terminal Profit X_T + q_T·S_T")
        ax.set_ylabel("Frequency")
        ax.set_title(f"γ = {gamma}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Terminal Profit Distribution: Inventory vs Symmetric Strategy\n"
        "(1000 Monte Carlo paths each)",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_exp2_inventory_histograms(
    all_results: Dict,
    gamma_values: List[float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot overlaid terminal inventory histograms for inventory vs symmetric strategies.

    Parameters
    ----------
    all_results : Dict
        Results dictionary from run_experiment_2.
    gamma_values : List[float]
        Risk aversion values.
    save_path : Optional[Path]
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, len(gamma_values), figsize=(15, 5), sharey=False)

    for ax, gamma in zip(axes, gamma_values):
        inv_q = all_results[(gamma, "inventory")]["terminal_inventory"]
        sym_q = all_results[(gamma, "symmetric")]["terminal_inventory"]

        all_q = np.concatenate([inv_q, sym_q])
        q_range = max(abs(np.percentile(all_q, 1)), abs(np.percentile(all_q, 99)))
        bins = np.linspace(-q_range, q_range, 50)

        ax.hist(inv_q, bins=bins, alpha=0.6, color="blue",
                label=f"Inventory\nμ={np.mean(inv_q):.2f}, σ={np.std(inv_q, ddof=1):.2f}")
        ax.hist(sym_q, bins=bins, alpha=0.6, color="orange",
                label=f"Symmetric\nμ={np.mean(sym_q):.2f}, σ={np.std(sym_q, ddof=1):.2f}")

        ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Terminal Inventory q_T")
        ax.set_ylabel("Frequency")
        ax.set_title(f"γ = {gamma}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Terminal Inventory Distribution: Inventory vs Symmetric Strategy\n"
        "(1000 Monte Carlo paths each)",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_exp2_summary_comparison(
    summary_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot bar charts comparing mean profit and std(profit) across strategies and gamma values.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics DataFrame from run_experiment_2.
    save_path : Optional[Path]
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    gamma_values = sorted(summary_df["gamma"].unique())
    strategies = ["inventory", "symmetric"]
    x = np.arange(len(gamma_values))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"inventory": "steelblue", "symmetric": "darkorange"}

    for metric, ax, title, ylabel in [
        ("mean_profit", axes[0], "Mean Terminal Profit", "Mean Profit"),
        ("std_profit", axes[1], "Std Dev of Terminal Profit", "Std(Profit)"),
    ]:
        for i, strategy in enumerate(strategies):
            vals = [
                summary_df.loc[
                    (summary_df["gamma"] == g) & (summary_df["strategy"] == strategy),
                    metric,
                ].values[0]
                for g in gamma_values
            ]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=strategy.capitalize(),
                          color=colors[strategy], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"γ={g}" for g in gamma_values])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Strategy Comparison: Inventory vs Symmetric", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Main visualization runner
# ---------------------------------------------------------------------------

def generate_all_plots(
    exp1_results: Optional[Dict] = None,
    exp2_results: Optional[Dict] = None,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    Generate and save all plots for both experiments.

    Parameters
    ----------
    exp1_results : Optional[Dict]
        Results from run_experiment_1. If None, skips exp1 plots.
    exp2_results : Optional[Dict]
        Results from run_experiment_2. If None, skips exp2 plots.
    results_dir : Path
        Directory to save plots.
    """
    results_dir.mkdir(exist_ok=True)
    print("\nGenerating plots...")

    gamma_values = [0.01, 0.1, 0.5]
    q_values = [-2, -1, 0, 1, 2]

    # --- Experiment 1 plots ---
    print("\n[Exp 1] Generating analytical plots...")

    plot_exp1_reservation_price(
        gamma_values=gamma_values,
        q_values=q_values,
        save_path=results_dir / "exp1_reservation_price.png",
    )
    plt.close("all")

    plot_exp1_quote_distances(
        gamma_values=gamma_values,
        q_values=q_values,
        save_path=results_dir / "exp1_quote_distances.png",
    )
    plt.close("all")

    plot_exp1_spread_components(
        gamma_values=gamma_values,
        save_path=results_dir / "exp1_spread_components.png",
    )
    plt.close("all")

    plot_exp1_limiting_behavior(
        save_path=results_dir / "exp1_limiting_behavior.png",
    )
    plt.close("all")

    # --- Experiment 2 plots ---
    if exp2_results is not None:
        print("\n[Exp 2] Generating Monte Carlo plots...")

        # Sample path for gamma=0.1
        gamma_sample = 0.1
        plot_exp2_sample_path(
            sim_results=exp2_results[(gamma_sample, "inventory")],
            gamma=gamma_sample,
            params=DEFAULT_PARAMS,
            save_path=results_dir / f"exp2_sample_path_gamma{gamma_sample}.png",
        )
        plt.close("all")

        plot_exp2_profit_histograms(
            all_results=exp2_results,
            gamma_values=GAMMA_VALUES,
            save_path=results_dir / "exp2_profit_histograms.png",
        )
        plt.close("all")

        plot_exp2_inventory_histograms(
            all_results=exp2_results,
            gamma_values=GAMMA_VALUES,
            save_path=results_dir / "exp2_inventory_histograms.png",
        )
        plt.close("all")

        plot_exp2_summary_comparison(
            summary_df=exp2_results["summary"],
            save_path=results_dir / "exp2_summary_comparison.png",
        )
        plt.close("all")

    print("\nAll plots generated.")
