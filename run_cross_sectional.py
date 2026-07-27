"""
Two fixes for the two problems the ML experiment exposed.

PROBLEM 1: ~98 effective samples is not enough to learn anything.
FIX: go cross-sectional. Instead of one timing decision per period, rank
assets against each other. N_assets x N_periods samples instead of N_periods.

PROBLEM 2: the universe has never contained a single failure. Every coin in
it survived to 2026. That is survivorship bias, and it inflates every number.
FIX: add pairs that collapsed. LUNA went from ~$100 to fractions of a cent in
May 2022. FTT collapsed with FTX in November 2022. Binance keeps the archives
for delisted pairs, so we can actually measure what their absence was worth.

Run:
    python run_cross_sectional.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
import strategies as st
from data import load_universe
from run_real import COSTS

RESULTS = Path("results")
START = "2019-01"
END = "2025-12"

# Coins that survived to today. This is the biased universe.
SURVIVORS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "XRPUSDT", "ADAUSDT",
    "LINKUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "ATOMUSDT",
    "UNIUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "FILUSDT", "NEARUSDT",
    "ALGOUSDT", "VETUSDT",
]

# Coins that collapsed. Any that fail to download are skipped automatically.
CASUALTIES = ["LUNAUSDT", "FTTUSDT", "SRMUSDT", "WAVESUSDT", "ANTUSDT"]


def cross_sectional_momentum(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = (30, 90, 180),
    n_long: int = 5,
    vol_window: int = 60,
    target_vol: float = 0.20,
    ann: int = 365,
) -> pd.DataFrame:
    """
    Rank assets by trailing return, hold the top n_long, weight by inverse
    volatility. All trailing, no lookahead.

    This is relative momentum, not absolute. It asks 'which of these is
    strongest' rather than 'is the market going up', which is a different and
    much better-sampled question.
    """
    scores = sum(prices.pct_change(lb).rank(axis=1, pct=True) for lb in lookbacks)
    scores = scores / len(lookbacks)

    # Hold only the top-ranked names that actually have a price today.
    valid = prices.notna()
    ranked = scores.where(valid)
    threshold = ranked.rank(axis=1, ascending=False) <= n_long
    selected = threshold.astype(float)

    vol = prices.pct_change().rolling(vol_window).std() * np.sqrt(ann)
    w = selected * (target_vol / vol.replace(0, np.nan))
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    gross = w.abs().sum(axis=1)
    over = gross > 1.0
    w.loc[over] = w.loc[over].div(gross[over], axis=0)
    return w.fillna(0.0)


def _evaluate(prices: pd.DataFrame, label: str) -> dict:
    print(f"\n  {label}: {prices.shape[1]} assets, {len(prices)} bars")
    out = {}
    out["buy_and_hold"] = bt.run(prices, st.buy_and_hold(prices), **COSTS)
    out["ts_momentum"] = bt.run(prices, st.ts_momentum(prices), **COSTS)
    out["cross_sectional"] = bt.run(prices, cross_sectional_momentum(prices), **COSTS)
    for name, res in out.items():
        s = res.stats
        print(f"    {name:<20}Sharpe {s['sharpe_annual']:>6.2f}   "
              f"CAGR {s['cagr']:>8.1%}   maxDD {s['max_drawdown']:>7.1%}")
    return out


def main() -> None:
    print("Downloading survivor universe...")
    surv = load_universe(SURVIVORS, "1d", START, END).dropna(how="all")

    print("\nDownloading collapsed pairs...")
    dead = load_universe(CASUALTIES, "1d", START, END)
    dead = dead.dropna(axis=1, how="all")

    if dead.empty:
        print("  none available — survivorship comparison skipped")
        full = surv
    else:
        print(f"\n  Recovered {dead.shape[1]} collapsed pairs: {list(dead.columns)}")
        for col in dead.columns:
            series = dead[col].dropna()
            if len(series) > 10:
                peak, last = series.max(), series.iloc[-1]
                print(f"    {col:<12} peak {peak:>12,.4f} -> last {last:>12,.4f}"
                      f"   ({last / peak - 1:>7.1%})")
        full = pd.concat([surv, dead], axis=1).sort_index()

    common = surv.index.intersection(full.index)
    surv, full = surv.loc[common], full.loc[common]

    print("\n" + "=" * 64)
    print("SURVIVORSHIP BIAS, MEASURED")
    print("=" * 64)
    biased = _evaluate(surv, "Survivors only (biased)")
    honest = _evaluate(full, "Including collapsed pairs (honest)")

    print("\n" + "=" * 64)
    print("THE COST OF PRETENDING NOTHING EVER DIES")
    print("=" * 64)
    print(f"  {'strategy':<22}{'biased':>10}{'honest':>10}{'inflation':>12}")
    print("  " + "-" * 54)
    for name in biased:
        b = biased[name].stats["sharpe_annual"]
        h = honest[name].stats["sharpe_annual"]
        print(f"  {name:<22}{b:>10.2f}{h:>10.2f}{b - h:>+12.2f}")

    print()
    print("  The 'inflation' column is Sharpe you would have reported to")
    print("  yourself purely by choosing a universe of survivors. It is not")
    print("  corrected by the deflated Sharpe, walk-forward, or any amount of")
    print("  model sophistication. It is a data problem, and data problems")
    print("  can only be fixed with data.")

    print("\n" + "=" * 64)
    print("EFFECTIVE SAMPLE SIZE")
    print("=" * 64)
    n_assets = full.shape[1]
    periods = len(full) // 21
    print(f"  Timing question  (is the market trending?):   ~{periods} samples")
    print(f"  Ranking question (which asset is strongest?): ~{periods * n_assets} samples")
    print(f"  Going cross-sectional multiplied usable data by {n_assets}x.")
    print("  THAT is how you make machine learning viable here — not by")
    print("  making the model bigger.")

    _plot(honest)
    print(f"\n  Chart written to {RESULTS.resolve()}")


def _plot(results: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=name, linewidth=1.3)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log scale)")
    ax.set_title("Honest universe, including assets that went to zero")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "cross_sectional.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
