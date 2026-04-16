"""
Charting utilities for backtest results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from config.settings import STRESS_PERIODS


def plot_equity_curves(results: dict, title: str = "Equity Curves", save_path: str = None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1, ax2 = axes

    colors = ["#00c176", "#f5a623", "#4a90d9", "#e74c3c", "#9b59b6"]

    for (name, r), color in zip(results.items(), colors):
        ec = r["equity_curve"]
        ax1.plot(ec.index, ec["equity"], label=name, color=color, linewidth=1.5)
        ax2.fill_between(ec.index, -ec["drawdown"] * 100, 0, alpha=0.4, color=color, label=name)

    # Shade stress periods
    for label, (start, end) in STRESS_PERIODS.items():
        for ax in axes:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color="red")

    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trade_distribution(trades_df: pd.DataFrame, title: str = "Trade P&L Distribution"):
    if trades_df.empty:
        print("No trades to plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    wins  = trades_df[trades_df["pnl"] > 0]["pnl_pct"] * 100
    losses = trades_df[trades_df["pnl"] <= 0]["pnl_pct"] * 100

    axes[0].hist(wins, bins=30, color="#00c176", alpha=0.7, label=f"Wins ({len(wins)})")
    axes[0].hist(losses, bins=30, color="#e74c3c", alpha=0.7, label=f"Losses ({len(losses)})")
    axes[0].axvline(0, color="white", linewidth=1)
    axes[0].set_title(f"{title} — P&L %")
    axes[0].set_xlabel("Return (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Trade duration
    if "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
        duration = (trades_df["exit_date"] - trades_df["entry_date"]).dt.days
        axes[1].hist(duration, bins=30, color="#4a90d9", alpha=0.8)
        axes[1].set_title("Trade Duration (days)")
        axes[1].set_xlabel("Days held")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
