"""
When does the strategy work, and when does it hurt you?

An aggregate Sharpe over seven years hides everything that matters. A strategy
with a Sharpe of 1.1 that made all its money in one year and bled in the other
six is not a strategy, it is a lucky bet with extra steps.

This script breaks the same backtests down by calendar year and by market
regime, so you can see where the number actually came from.

Run:
    python analyze_regimes.py
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
from run_real import COSTS, END, START, UNIVERSE

RESULTS = Path("results")


def _annual_table(returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Calendar-year return and Sharpe for each strategy."""
    rows = {}
    for name, r in returns.items():
        by_year = {}
        for year, chunk in r.groupby(r.index.year):
            total = (1 + chunk).prod() - 1
            sharpe = (chunk.mean() / chunk.std() * np.sqrt(365)
                      if chunk.std() > 0 else np.nan)
            by_year[year] = (total, sharpe)
        rows[name] = by_year
    return pd.DataFrame(rows)


def _regime_split(returns: dict[str, pd.Series], benchmark: pd.Series) -> None:
    """
    Split by whether the market itself was rising or falling.

    A trend follower is supposed to earn its keep in the down months. If it
    doesn't, it is just leveraged beta wearing a costume.
    """
    monthly_bench = (1 + benchmark).resample("ME").prod() - 1
    up = monthly_bench > 0
    print(f"\n  Market up months:   {up.sum()}")
    print(f"  Market down months: {(~up).sum()}\n")

    print(f"  {'strategy':<26}{'up months':>14}{'down months':>16}")
    print("  " + "-" * 56)
    for name, r in returns.items():
        monthly = (1 + r).resample("ME").prod() - 1
        monthly = monthly.reindex(monthly_bench.index).dropna()
        u = monthly[up.reindex(monthly.index).fillna(False)]
        d = monthly[~up.reindex(monthly.index).fillna(False)]
        print(f"  {name:<26}{u.mean():>13.2%}{d.mean():>16.2%}")


def main() -> None:
    prices = load_universe(UNIVERSE, "1d", START, END).dropna(how="all")

    # Rebuild the same three strategies from run_real.py.
    wf = bt.walk_forward(
        prices,
        build_weights=lambda train, test: st.ts_momentum(
            pd.concat([train.tail(400), test])
        ).loc[test.index],
        train_bars=500,
        test_bars=180,
        embargo_bars=10,
        **COSTS,
    )
    window = wf.returns.index
    p = prices.loc[window]

    bh = bt.run(p, st.buy_and_hold(p), **COSTS)

    returns = {
        "buy_and_hold": bh.returns,
        "momentum_walk_forward": wf.returns,
    }

    print("=" * 64)
    print("ANNUAL BREAKDOWN")
    print("=" * 64)
    print(f"  Period: {window[0].date()} to {window[-1].date()}\n")

    table = _annual_table(returns)
    print(f"  {'year':<8}{'buy & hold':>22}{'momentum':>22}")
    print(f"  {'':<8}{'return':>12}{'Sharpe':>10}{'return':>12}{'Sharpe':>10}")
    print("  " + "-" * 54)
    for year in table.index:
        bh_r, bh_s = table.loc[year, "buy_and_hold"]
        mo_r, mo_s = table.loc[year, "momentum_walk_forward"]
        print(f"  {year:<8}{bh_r:>11.1%}{bh_s:>10.2f}{mo_r:>12.1%}{mo_s:>10.2f}")

    print("\n" + "=" * 64)
    print("CONDITIONAL ON MARKET DIRECTION")
    print("=" * 64)
    _regime_split(returns, bh.returns)

    print("\n" + "=" * 64)
    print("CONCENTRATION OF RETURNS")
    print("=" * 64)
    print("  If removing the best few days destroys the result, you do not")
    print("  have a strategy — you have a handful of lucky days.\n")
    for name, r in returns.items():
        total = (1 + r).prod() - 1
        for drop in (5, 10, 20):
            trimmed = r.drop(r.nlargest(drop).index)
            t = (1 + trimmed).prod() - 1
            print(f"  {name:<24} minus best {drop:>2} days: "
                  f"{total:>9.1%} -> {t:>9.1%}")
        print()

    _plot_annual(table)
    print(f"  Chart written to {RESULTS.resolve()}")


def _plot_annual(table: pd.DataFrame) -> None:
    RESULTS.mkdir(exist_ok=True)
    years = list(table.index)
    bh = [table.loc[y, "buy_and_hold"][0] for y in years]
    mo = [table.loc[y, "momentum_walk_forward"][0] for y in years]

    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, bh, 0.4, label="buy and hold")
    ax.bar(x + 0.2, mo, 0.4, label="momentum (walk-forward)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Calendar year return")
    ax.set_title("Where the performance actually came from")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "annual_returns.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
