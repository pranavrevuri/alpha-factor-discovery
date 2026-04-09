"""charts.py - Visualisation for single-ticker and comparison mode."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np


def plot_single_stock(
    df: pd.DataFrame,
    indicators: pd.DataFrame,
    ticker: str,
    importance: pd.Series | None = None,
    save_path: str | None = None,
) -> None:
    """
    Three-panel chart:
      Panel 1: Close price with SMA, EMA, Bollinger Bands
      Panel 2: RSI
      Panel 3: Volume Ratio

    If importance is provided, append a fourth panel with a bar chart.
    """
    close = df["close"]
    volume = df["volume"]

    if importance is not None:
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        ax_price = fig.add_subplot(gs[0])
        ax_rsi   = fig.add_subplot(gs[1])
        ax_vr    = fig.add_subplot(gs[2])
        ax_imp   = fig.add_subplot(gs[3])
    else:
        fig, (ax_price, ax_rsi, ax_vr) = plt.subplots(
            3, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.3}
        )
        ax_imp = None

    # Panel 1: Price + SMA + EMA + Bollinger Bands
    ax_price.plot(close.index, close, label="Close", color="black", linewidth=1)
    ax_price.plot(close.index, indicators["SMA"], label="SMA(20)", color="blue", linewidth=1)
    ax_price.plot(close.index, indicators["EMA"], label="EMA(20)", color="orange", linewidth=1)
    ax_price.plot(close.index, indicators["BB_Upper"], label="BB Upper", color="red", linestyle="--", linewidth=0.8)
    ax_price.plot(close.index, indicators["BB_Middle"], label="BB Middle", color="gray", linestyle=":", linewidth=0.8)
    ax_price.plot(close.index, indicators["BB_Lower"], label="BB Lower", color="red", linestyle="--", linewidth=0.8)
    ax_price.fill_between(
        indicators.index,
        indicators["BB_Lower"],
        indicators["BB_Upper"],
        alpha=0.1,
        color="red",
        label="BB Band",
    )
    ax_price.set_title(f"{ticker} – Price, SMA/EMA, Bollinger Bands", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=7)
    ax_price.set_ylabel("Price ($)")

    # Panel 2: RSI
    ax_rsi.plot(indicators.index, indicators["RSI"], color="purple", linewidth=1)
    ax_rsi.axhline(y=70, color="red", linestyle="--", linewidth=0.8, label="Overbought (70)")
    ax_rsi.axhline(y=30, color="green", linestyle="--", linewidth=0.8, label="Oversold (30)")
    ax_rsi.fill_between(indicators.index, 70, 100, alpha=0.1, color="red")
    ax_rsi.fill_between(indicators.index, 0, 30, alpha=0.1, color="green")
    ax_rsi.set_title("RSI (14-day)", fontsize=10)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(fontsize=7, loc="upper left")

    # Panel 3: Volume Ratio
    ax_vr.bar(indicators.index, indicators["VolumeRatio"], color="steelblue", alpha=0.6, width=1)
    ax_vr.axhline(y=1, color="black", linestyle="-", linewidth=0.8)
    ax_vr.set_title("Volume Ratio (20-day avg)", fontsize=10)
    ax_vr.set_ylabel("Ratio")
    ax_vr.set_xlabel("Date")

    # Panel 4: Feature Importances
    if ax_imp is not None and importance is not None:
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance)))[::-1]
        importance.plot.barh(ax=ax_imp, color=colors)
        ax_imp.set_title("Feature Importances (Random Forest)", fontsize=10)
        ax_imp.set_xlabel("Importance")
        ax_imp.invert_yaxis()

    plt.suptitle(f"Alpha Factor Analysis – {ticker}", fontsize=14, y=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison(
    ticker_a: str, df_a: pd.DataFrame, indicators_a: pd.DataFrame,
    ticker_b: str, df_b: pd.DataFrame, indicators_b: pd.DataFrame,
    importance_a: pd.Series, importance_b: pd.Series,
    save_dir: str = ".",
) -> None:
    """Create side-by-side comparison charts for two tickers."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Price comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(df_a.index, df_a["close"], label=f"{ticker_a} Close", color="blue")
    axes[0].plot(df_b.index, df_b["close"], label=f"{ticker_b} Close", color="orange")
    axes[0].set_title("Close Price Comparison")
    axes[0].legend()
    axes[0].set_ylabel("Price ($)")

    axes[1].plot(indicators_a.index, indicators_a["RSI"], label=f"{ticker_a} RSI", color="blue")
    axes[1].plot(indicators_b.index, indicators_b["RSI"], label=f"{ticker_b} RSI", color="orange")
    axes[1].axhline(y=70, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(y=30, color="green", linestyle="--", linewidth=0.8)
    axes[1].set_title("RSI Comparison")
    axes[1].legend()
    axes[1].set_ylabel("RSI")
    axes[1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_rsi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Comparison chart saved to {path}")
    plt.close()

    # Feature importance comparison (side-by-side bar chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    importance_a.sort_values().plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title(f"{ticker_a} Feature Importances")
    axes[0].set_xlabel("Importance")

    importance_b.sort_values().plot.barh(ax=axes[1], color="darkorange")
    axes[1].set_title(f"{ticker_b} Feature Importances")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_importances.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Feature importance comparison saved to {path}")
    plt.close()
