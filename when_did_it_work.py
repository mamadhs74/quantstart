"""
Resolving the contradiction in robustness.py.

The parameter surface said the effect is real. The walk-forward said it beats
buy-and-hold out of sample with essentially zero overfitting gap. And the
sub-period test said it lost to buy-and-hold in all three sub-periods.

Those cannot all be true unless the outperformance is concentrated in a
narrow slice of time that the sub-period boundaries happened to split.

This script finds out exactly when the strategy earned its edge, and whether
it has earned anything since.

WHY THIS MATTERS MORE THAN ANY OTHER TEST HERE
----------------------------------------------
A flat parameter surface proves the result is not an artifact of the
parameters you chose. It says NOTHING about whether the effect persists in
time. Those are different failure modes, and the second one is what actually
destroys traders: they find something that worked in one regime, deploy it,
and bleed for years while the backtest keeps insisting it works.

Run:
    python when_did_it_work.py
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
from run_cross_sectional import CASUALTIES, SURVIVORS, cross_sectional_momentum
from run_real import COSTS

RESULTS = Path("results")
START, END = "2019-01", "2025-12"
WINDOW = 252  # rolling year


def main() -> None:
    surv = load_universe(SURVIVORS, "1d", START, END)
    dead = load_universe(CASUALTIES, "1d", START, END).dropna(axis=1, how="all")
    prices = pd.concat([surv, dead], axis=1).sort_index().dropna(how="all")

    cs = bt.run(prices, cross_sectional_momentum(prices), **COSTS)
    bh = bt.run(prices, st.buy_and_hold(prices), **COSTS)

    rel = cs.returns - bh.returns              # daily excess over the benchmark
    cum_rel = (1 + cs.returns).cumprod() / (1 + bh.returns).cumprod()

    print("=" * 66)
    print("ROLLING 12-MONTH SHARPE, CROSS-SECTIONAL MINUS BUY-AND-HOLD")
    print("=" * 66)
    roll_cs = (cs.returns.rolling(WINDOW).mean()
               / cs.returns.rolling(WINDOW).std() * np.sqrt(365))
    roll_bh = (bh.returns.rolling(WINDOW).mean()
               / bh.returns.rolling(WINDOW).std() * np.sqrt(365))
    diff = (roll_cs - roll_bh).dropna()

    print(f"  {'12 months ending':<22}{'cross-sec':>12}{'buy&hold':>12}{'diff':>10}")
    print("  " + "-" * 56)
    for ts in pd.date_range(diff.index[0], diff.index[-1], freq="2QS"):
        nearest = diff.index[diff.index.get_indexer([ts], method="nearest")[0]]
        print(f"  {nearest.date()!s:<22}{roll_cs[nearest]:>12.2f}"
              f"{roll_bh[nearest]:>12.2f}{diff[nearest]:>10.2f}")

    print("\n" + "=" * 66)
    print("WHERE THE EDGE ACTUALLY CAME FROM")
    print("=" * 66)

    by_year = {}
    for year, chunk in rel.groupby(rel.index.year):
        excess = (1 + cs.returns[chunk.index]).prod() - (1 + bh.returns[chunk.index]).prod()
        by_year[year] = excess

    print(f"  {'year':<10}{'excess return vs buy-and-hold':>34}")
    print("  " + "-" * 44)
    for year, excess in by_year.items():
        bar = "#" * min(int(abs(excess) * 20), 30)
        sign = "+" if excess >= 0 else "-"
        print(f"  {year:<10}{excess:>18.1%}   {sign}{bar}")

    best_year = max(by_year, key=by_year.get)
    total_excess = cum_rel.iloc[-1] - 1
    without_best = total_excess - by_year[best_year]

    print()
    print(f"  Total excess over the whole history: {total_excess:>8.1%}")
    print(f"  Contributed by {best_year} alone:            {by_year[best_year]:>8.1%}")
    print(f"  Everything else combined:            {without_best:>8.1%}")

    print("\n" + "=" * 66)
    print("PERSISTENCE")
    print("=" * 66)
    frac = (diff > 0).mean()
    print(f"  Rolling 12m windows where it beat buy-and-hold: {frac:>6.1%}")

    recent = diff.last("730D") if hasattr(diff, "last") else diff.iloc[-730:]
    print(f"  Last 24 months, mean Sharpe difference:         {recent.mean():>+6.2f}")
    print(f"  Last 24 months, windows won:                    {(recent > 0).mean():>6.1%}")

    peak_date = cum_rel.idxmax()
    print(f"\n  Relative equity peaked on {peak_date.date()}")
    since = cum_rel.iloc[-1] / cum_rel.max() - 1
    days = (cum_rel.index[-1] - peak_date).days
    print(f"  Since then: {since:>+.1%} relative, over {days} days "
          f"({days / 365:.1f} years)")

    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    concentrated = by_year[best_year] > total_excess * 0.7
    still_working = recent.mean() > 0

    if concentrated and not still_working:
        print(f"  The edge is ONE YEAR ({best_year}) and it stopped working.")
        print("  The parameter surface was flat because every parameter")
        print("  setting captured the same windfall, not because the effect")
        print("  is durable. The sub-period test was right and the aggregate")
        print("  walk-forward Sharpe was misleading.")
        print()
        print("  This is the single most common way backtests kill people:")
        print("  one regime, long since gone, averaged into a number that")
        print("  still looks good years later.")
    elif still_working:
        print("  Outperformance is not confined to one year and the recent")
        print("  window is still positive. Weak but real evidence. Keep")
        print("  measuring; do not size up yet.")
    else:
        print("  Mixed. Not a single-year artifact, but not currently")
        print("  working either. Treat as dormant, not disproven.")

    _plot(cum_rel, diff)
    print(f"\n  Chart written to {RESULTS.resolve()}")


def _plot(cum_rel: pd.Series, diff: pd.Series) -> None:
    RESULTS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(cum_rel.index, cum_rel.values, linewidth=1.4)
    ax1.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_yscale("log")
    ax1.set_ylabel("Relative equity")
    ax1.set_title("Cross-sectional vs buy-and-hold — rising means winning")
    ax1.grid(alpha=0.3)

    ax2.fill_between(diff.index, diff.values, 0,
                     where=diff.values > 0, alpha=0.6, label="beating benchmark")
    ax2.fill_between(diff.index, diff.values, 0,
                     where=diff.values <= 0, alpha=0.6, label="losing to benchmark")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Rolling 12m Sharpe difference")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "when_did_it_work.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
