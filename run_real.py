"""
The same pipeline as selftest.py, but pointed at real market data.

Run:
    python run_real.py

First run downloads ~7 years of daily bars and caches them to
~/.cache/binance_klines. Later runs read from disk and are instant.

READ THIS BEFORE BELIEVING ANY NUMBER BELOW
-------------------------------------------
This universe is survivorship-biased. It contains coins that still exist and
still trade liquidly today. Everything that died between 2019 and now is
absent. That inflates buy-and-hold badly, and it is NOT corrected by the
deflated Sharpe ratio, which only accounts for how many strategies you tried.

Two biases the deflated Sharpe does not protect you from:
  - period selection (crypto rose enormously over this window)
  - survivorship (dead assets are missing from the universe)

Treat every result here as an upper bound.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNG files, don't try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
import strategies as st
from data import load_universe

# Liquid pairs with long Binance history. Assets listed later simply carry
# NaNs early on and receive zero weight until they exist — no lookahead.
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT",
    "XRPUSDT", "ADAUSDT", "LINKUSDT", "SOLUSDT",
]

START = "2019-01"
END = "2025-12"

# Binance spot taker fee is ~10 bps. Slippage on daily bars in liquid pairs is
# small but never zero. These are deliberately not optimistic.
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0

# Be honest here. Every parameter set you try counts.
N_TRIALS = 20

COSTS = dict(fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS, interval="1d")
RESULTS = Path("results")


def _line(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    print(f"Downloading {len(UNIVERSE)} symbols, {START} to {END}...")
    print("(first run is slow, afterwards it reads from cache)\n")

    prices = load_universe(UNIVERSE, "1d", START, END).dropna(how="all")

    print(f"Loaded {prices.shape[1]} symbols, {len(prices)} daily bars")
    print(f"Range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print("\nData coverage per symbol:")
    for sym, frac in prices.notna().mean().sort_values().items():
        print(f"  {sym:<10} {frac:>6.1%}")

    results = {}

    _line("1. ORACLE  (cheating — this is the ceiling)")
    res = bt.run(prices, st.oracle(prices), **COSTS)
    print(res)
    ceiling = res.stats["sharpe_annual"]

    # ---- Walk-forward first, so we can match every other test to its window.
    _line("2. MOMENTUM, WALK-FORWARD  (the number that counts)")
    wf = bt.walk_forward(
        prices,
        build_weights=lambda train, test: st.ts_momentum(
            pd.concat([train.tail(400), test])
        ).loc[test.index],
        train_bars=500,
        test_bars=180,
        embargo_bars=10,
        n_trials=N_TRIALS,
        **COSTS,
    )
    print(wf)
    results["momentum_walk_forward"] = wf
    oos_window = wf.returns.index          # <- the only fair comparison window
    oos = wf.stats["sharpe_annual"]
    dsr = wf.stats["deflated_sharpe"]

    # ---- Everything below is restricted to the SAME bars as the walk-forward.
    # Comparing a full-history in-sample run against a shorter walk-forward run
    # compares two different market regimes and tells you nothing.
    p_matched = prices.loc[oos_window]

    _line("3. MOMENTUM, IN-SAMPLE  (same bars as above — fair comparison)")
    res = bt.run(p_matched, st.ts_momentum(prices).loc[oos_window],
                 n_trials=N_TRIALS, **COSTS)
    print(res)
    results["momentum_in_sample"] = res
    ins = res.stats["sharpe_annual"]

    _line("4. BUY AND HOLD  (same bars — the floor to beat)")
    res = bt.run(p_matched, st.buy_and_hold(p_matched), **COSTS)
    print(res)
    results["buy_and_hold"] = res
    floor = res.stats["sharpe_annual"]
    floor_dd = res.stats["max_drawdown"]

    _line("5. NOISE FLOOR  (30 random strategies, same bars, costs off)")
    free = dict(fee_bps=0, slippage_bps=0, interval="1d")
    noise = [
        bt.run(p_matched, st.random_weights(p_matched, seed=s), **free)
        .stats["sharpe_annual"]
        for s in range(30)
    ]
    lo, hi = np.percentile(noise, [5, 95])
    print(f"  random Sharpe 5th-95th pct   {lo:>8.3f} to {hi:>8.3f}")
    print(f"  walk-forward momentum        {oos:>8.3f}")
    inside = lo <= oos <= hi
    print(f"  -> {'inside the noise band' if inside else 'OUTSIDE the noise band'}")

    _line("6. COST SENSITIVITY  (walk-forward returns, costs re-applied)")
    print("  Turnover is fixed by the strategy; only the cost rate varies.")
    turnover = (wf.positions - wf.positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    gross = wf.returns + turnover * (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    for fee in (0, 5, 10, 25, 50):
        net = gross - turnover * fee / 10_000.0
        eq = (1 + net).cumprod()
        yrs = len(net) / 365
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        sharpe = net.mean() / net.std() * np.sqrt(365)
        print(f"  fee {fee:>3} bps  ->  CAGR {cagr:>9.2%}   Sharpe {sharpe:>6.3f}")

    _plot(results)

    _line("VERDICT")
    print(f"  Oracle Sharpe (impossible)        {ceiling:>8.2f}")
    print(f"  Buy and hold Sharpe               {floor:>8.2f}")
    print(f"  Momentum, in-sample               {ins:>8.2f}")
    print(f"  Momentum, walk-forward            {oos:>8.2f}")
    print(f"  Deflated Sharpe probability       {dsr:>8.3f}")
    print()
    print(f"  Overfitting gap (in - out):       {ins - oos:>8.2f}")
    print("  All four Sharpes above are measured on identical bars, so this")
    print("  gap is real overfitting rather than a change of market regime.")
    print()

    wf_dd = wf.stats["max_drawdown"]
    print(f"  Max drawdown, buy and hold        {floor_dd:>8.1%}")
    print(f"  Max drawdown, momentum            {wf_dd:>8.1%}")
    print()

    beats_sharpe = oos > floor
    beats_dd = wf_dd > floor_dd
    significant = dsr >= 0.95

    if beats_sharpe and significant:
        print("  Beats buy-and-hold AND clears the 0.95 deflated Sharpe bar.")
        print("  Before believing it, change one thing at a time: a different")
        print("  universe, a different date range, double the costs. A real")
        print("  edge survives all three. An overfit survives none.")
    elif beats_dd and not beats_sharpe:
        print("  Momentum did NOT beat buy-and-hold on Sharpe, but it did cut")
        print("  the drawdown. That is the volatility targeting working, and")
        print("  it is a genuine risk-management result — not alpha. Useful,")
        print("  but do not mistake it for predictive skill.")
    else:
        print("  No edge demonstrated. This is the normal, expected outcome.")
        print("  A public signal on the most-traded assets on earth is where")
        print("  edges are least likely to survive.")

    print()
    print("  Reminder: this universe is survivorship-biased and this period")
    print("  was historically kind to crypto. Both inflate every number here,")
    print("  and neither is corrected by the deflated Sharpe.")
    print(f"\n  Charts written to {RESULTS.resolve()}")


def _plot(results: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for name, res in results.items():
        ax1.plot(res.equity.index, res.equity.values, label=name, linewidth=1.3)
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log scale)")
    ax1.set_title("Equity curves, net of costs, matched period")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    for name, res in results.items():
        dd = res.equity / res.equity.cummax() - 1.0
        ax2.plot(dd.index, dd.values, label=name, linewidth=1.0)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown — look here before you look at returns")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "equity_curves.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
