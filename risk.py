"""
Position sizing and risk of ruin.

Signal research answers 'which way'. This answers 'how much', and how much
matters more. A mediocre edge sized correctly compounds. A good edge sized
badly goes to zero, and going to zero is not recoverable by being right later.

Your strategy's deflated Sharpe is 0.937 -- promising, not proven. This module
asks the question that follows from that honestly: if the edge is real, how
much should you bet? And if it is NOT real, how badly does that hurt?

Run:
    python risk.py
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
N_PATHS = 10_000
HORIZON_DAYS = 365
RUIN_LEVEL = -0.50        # losing half is, practically, ruin
BLOCK = 20                # bootstrap block length, preserves autocorrelation


def kelly_fraction(returns: pd.Series, ann: int = 365) -> float:
    """
    Continuous Kelly: f* = mu / sigma^2, in annualised terms.

    This is the growth-optimal bet size IF your estimate of the edge is exact.
    It never is. Kelly is extremely sensitive to estimation error, which is
    why practitioners use a fraction of it -- typically a quarter or a half.
    """
    mu = returns.mean() * ann
    var = returns.var() * ann
    return mu / var if var > 0 else 0.0


def block_bootstrap(returns: np.ndarray, n_paths: int, horizon: int,
                    block: int = BLOCK, seed: int = 0) -> np.ndarray:
    """
    Resample in blocks, not individual days.

    Independent resampling destroys volatility clustering and autocorrelation,
    which makes drawdowns look far milder than they are. Blocks preserve the
    local structure that actually produces bad runs.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n - block, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :])
    return returns[idx].reshape(n_paths, -1)[:, :horizon]


def simulate(returns: pd.Series, leverage: float, n_paths: int = N_PATHS,
             horizon: int = HORIZON_DAYS) -> dict:
    """Monte Carlo the next year at a given bet size."""
    paths = block_bootstrap(returns.values, n_paths, horizon) * leverage
    equity = np.cumprod(1.0 + paths, axis=1)

    # Guard against paths that mathematically wipe out under leverage.
    equity = np.where(equity <= 0, 1e-12, equity)

    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = equity / running_max - 1.0
    worst = drawdowns.min(axis=1)
    final = equity[:, -1] - 1.0

    return {
        "leverage": leverage,
        "median_return": float(np.median(final)),
        "p5_return": float(np.percentile(final, 5)),
        "p95_return": float(np.percentile(final, 95)),
        "prob_loss": float((final < 0).mean()),
        "prob_dd_30": float((worst < -0.30).mean()),
        "prob_dd_50": float((worst < RUIN_LEVEL).mean()),
        "median_worst_dd": float(np.median(worst)),
    }


def main() -> None:
    surv = load_universe(SURVIVORS, "1d", START, END)
    dead = load_universe(CASUALTIES, "1d", START, END).dropna(axis=1, how="all")
    prices = pd.concat([surv, dead], axis=1).sort_index().dropna(how="all")

    wf = bt.walk_forward(
        prices,
        build_weights=lambda train, test: cross_sectional_momentum(
            pd.concat([train.tail(400), test])
        ).loc[test.index],
        train_bars=500, test_bars=180, embargo_bars=10, **COSTS,
    )
    r = wf.returns

    print("=" * 68)
    print("THE STRATEGY, AS MEASURED")
    print("=" * 68)
    print(f"  annualised return      {r.mean() * 365:>10.1%}")
    print(f"  annualised volatility  {r.std() * np.sqrt(365):>10.1%}")
    print(f"  Sharpe                 {wf.stats['sharpe_annual']:>10.2f}")
    print(f"  realised max drawdown  {wf.stats['max_drawdown']:>10.1%}")

    k = kelly_fraction(r)
    print("\n" + "=" * 68)
    print("KELLY")
    print("=" * 68)
    print(f"  Full Kelly leverage:        {k:>6.2f}x")
    print(f"  Half Kelly:                 {k / 2:>6.2f}x")
    print(f"  Quarter Kelly:              {k / 4:>6.2f}x")
    print()
    print("  Full Kelly is growth-optimal only if your edge estimate is exact.")
    print("  Yours has a deflated Sharpe of 0.937, meaning the edge might not")
    print("  exist at all. Betting full Kelly on an uncertain edge is how")
    print("  mathematically sophisticated people go broke.")

    print("\n" + "=" * 68)
    print(f"MONTE CARLO: {N_PATHS:,} paths, one year, block bootstrap")
    print("=" * 68)
    levs = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    rows = [simulate(r, lv) for lv in levs]

    print(f"  {'lev':>5}{'median':>10}{'5th pct':>10}{'95th pct':>11}"
          f"{'P(loss)':>10}{'P(-30%)':>10}{'P(-50%)':>10}")
    print("  " + "-" * 66)
    for row in rows:
        print(f"  {row['leverage']:>4.2f}x{row['median_return']:>10.1%}"
              f"{row['p5_return']:>10.1%}{row['p95_return']:>11.1%}"
              f"{row['prob_loss']:>10.1%}{row['prob_dd_30']:>10.1%}"
              f"{row['prob_dd_50']:>10.1%}")

    print("\n  Note how P(-50%) grows far faster than median return does.")
    print("  That asymmetry is the whole argument for betting small.")

    print("\n" + "=" * 68)
    print("IF THE EDGE IS NOT REAL")
    print("=" * 68)
    print("  Same volatility, same costs, zero expected return. This is the")
    print("  world where the deflated Sharpe was right and you were wrong.\n")
    r_noalpha = r - r.mean()
    print(f"  {'lev':>5}{'median':>10}{'5th pct':>10}{'P(loss)':>11}{'P(-50%)':>11}")
    print("  " + "-" * 48)
    for lv in levs:
        s = simulate(r_noalpha, lv)
        print(f"  {lv:>4.2f}x{s['median_return']:>10.1%}{s['p5_return']:>10.1%}"
              f"{s['prob_loss']:>11.1%}{s['prob_dd_50']:>11.1%}")
    print("\n  Costs alone make a no-edge strategy lose money reliably.")
    print("  Leverage does not fix that. It accelerates it.")

    print("\n" + "=" * 68)
    print("KILL SWITCH SIZING")
    print("=" * 68)
    print("  Probability of hitting a given drawdown limit within one year:\n")
    print(f"  {'limit':>8}" + "".join(f"{lv:>9.2f}x" for lv in (0.25, 0.5, 1.0, 1.5)))
    print("  " + "-" * 48)
    for limit in (0.15, 0.20, 0.25, 0.30):
        probs = []
        for lv in (0.25, 0.5, 1.0, 1.5):
            paths = block_bootstrap(r.values, N_PATHS, HORIZON_DAYS, seed=1) * lv
            eq = np.cumprod(1.0 + paths, axis=1)
            eq = np.where(eq <= 0, 1e-12, eq)
            dd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
            probs.append((dd < -limit).mean())
        print(f"  {limit:>7.0%} " + "".join(f"{p:>10.0%}" for p in probs))
    print()
    print("  A limit you will breach 80% of the time is not a limit, it is a")
    print("  guarantee you will turn the system off. Choose one you can")
    print("  actually live with, then size leverage to fit it.")
    print()
    print("  Pick the limit BEFORE you deploy, write it into the code, and")
    print("  make it automatic. A drawdown limit you have to decide to honour")
    print("  in the moment is not a risk control, it is a hope.")

    _plot(r, levs)
    print(f"\n  Chart written to {RESULTS.resolve()}")


def _plot(returns: pd.Series, levs: list[float]) -> None:
    RESULTS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    paths = block_bootstrap(returns.values, 300, HORIZON_DAYS)
    eq = np.cumprod(1 + paths, axis=1)
    for i in range(300):
        ax1.plot(eq[i], alpha=0.06, color="tab:blue", linewidth=0.8)
    ax1.plot(np.median(eq, axis=0), color="black", linewidth=2, label="median")
    ax1.plot(np.percentile(eq, 5, axis=0), color="tab:red",
             linewidth=1.5, linestyle="--", label="5th percentile")
    ax1.axhline(1.0, color="grey", linewidth=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("trading days")
    ax1.set_ylabel("equity multiple")
    ax1.set_title("300 possible next years at 1x leverage")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ruin = [simulate(returns, lv)["prob_dd_50"] for lv in levs]
    med = [simulate(returns, lv)["median_return"] for lv in levs]
    ax2.plot(levs, ruin, marker="o", label="P(50% drawdown)")
    ax2.plot(levs, med, marker="s", label="median return")
    ax2.set_xlabel("leverage")
    ax2.set_title("Risk grows faster than reward")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "risk.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
