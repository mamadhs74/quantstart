"""
The 'is this real?' battery, applied to the cross-sectional result.

Cross-sectional momentum beat buy-and-hold in-sample. That is exactly when
you should be most suspicious, because a promising in-sample number is the
raw material every overfit is made from.

Four tests, in order of how hard they are to fake:

  1. PARAMETER SURFACE. Does performance sit on a broad plateau, or a knife
     edge at the one setting that was tried first? A real effect is robust to
     small parameter changes. An overfit is not. This is the single best
     overfitting diagnostic there is, and almost nobody runs it.

  2. SUB-PERIOD STABILITY. Split the history in three. Does it work in all
     three, or did one lucky stretch carry the whole number?

  3. WALK-FORWARD. Out-of-sample, refit as it goes.

  4. DEFLATED SHARPE with n_trials set to the honest number of combinations
     tried in test 1 — not 1.

Run:
    python robustness.py
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

N_LONG_GRID = [3, 5, 8, 10]
LOOKBACK_GRID = {
    "short (30)": (30,),
    "mid (30,90)": (30, 90),
    "base (30,90,180)": (30, 90, 180),
    "long (90,180,360)": (90, 180, 360),
}


def _load() -> pd.DataFrame:
    surv = load_universe(SURVIVORS, "1d", START, END)
    dead = load_universe(CASUALTIES, "1d", START, END).dropna(axis=1, how="all")
    full = pd.concat([surv, dead], axis=1) if not dead.empty else surv
    return full.sort_index().dropna(how="all")


def parameter_surface(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Every combination. A plateau is evidence; a spike is a warning."""
    grid = pd.DataFrame(index=list(LOOKBACK_GRID), columns=N_LONG_GRID, dtype=float)
    trials = 0
    for lb_name, lookbacks in LOOKBACK_GRID.items():
        for n_long in N_LONG_GRID:
            w = cross_sectional_momentum(prices, lookbacks=lookbacks, n_long=n_long)
            grid.loc[lb_name, n_long] = bt.run(prices, w, **COSTS).stats["sharpe_annual"]
            trials += 1
    return grid, trials


def sub_periods(prices: pd.DataFrame, n_splits: int = 3) -> pd.DataFrame:
    """Does it work everywhere, or did one stretch carry the whole number?"""
    bounds = np.linspace(0, len(prices), n_splits + 1).astype(int)
    rows = []
    for i in range(n_splits):
        chunk = prices.iloc[bounds[i]:bounds[i + 1]]
        if len(chunk) < 200:
            continue
        cs = bt.run(chunk, cross_sectional_momentum(chunk), **COSTS)
        bh = bt.run(chunk, st.buy_and_hold(chunk), **COSTS)
        rows.append({
            "period": f"{chunk.index[0].date()} to {chunk.index[-1].date()}",
            "cross_sectional": cs.stats["sharpe_annual"],
            "buy_and_hold": bh.stats["sharpe_annual"],
            "cs_cagr": cs.stats["cagr"],
            "bh_cagr": bh.stats["cagr"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    prices = _load()
    print(f"Universe: {prices.shape[1]} assets, {len(prices)} bars, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    print("\n" + "=" * 66)
    print("1. PARAMETER SURFACE  (in-sample Sharpe)")
    print("=" * 66)
    grid, trials = parameter_surface(prices)
    print(f"  {'lookbacks':<20}" + "".join(f"{n:>10}" for n in N_LONG_GRID)
          + f"{'spread':>10}")
    print("  " + "-" * 62)
    for name in grid.index:
        row = grid.loc[name]
        print(f"  {name:<20}" + "".join(f"{v:>10.2f}" for v in row)
              + f"{row.max() - row.min():>10.2f}")

    flat = grid.values.flatten()
    print()
    print(f"  best {flat.max():.2f}   worst {flat.min():.2f}   "
          f"mean {flat.mean():.2f}   std {flat.std():.2f}")
    ratio = flat.std() / abs(flat.mean()) if flat.mean() != 0 else np.inf
    print(f"  coefficient of variation: {ratio:.2f}")
    print()
    if ratio < 0.25 and flat.min() > 0:
        print("  PLATEAU. Every parameter choice works and none dominates.")
        print("  That is what a real effect looks like.")
    elif flat.min() < 0 < flat.max():
        print("  UNSTABLE. Some settings profit, others lose. The result")
        print("  depends on choices you had no principled reason to make.")
    else:
        print("  MIXED. Positive throughout but sensitive to parameters.")
        print("  Weaker evidence than a flat surface, stronger than a spike.")

    print("\n" + "=" * 66)
    print("2. SUB-PERIOD STABILITY")
    print("=" * 66)
    sp = sub_periods(prices)
    print(f"  {'period':<28}{'cross-sec':>12}{'buy&hold':>12}{'cs CAGR':>12}")
    print("  " + "-" * 62)
    for _, r in sp.iterrows():
        print(f"  {r['period']:<28}{r['cross_sectional']:>12.2f}"
              f"{r['buy_and_hold']:>12.2f}{r['cs_cagr']:>12.1%}")
    wins = (sp["cross_sectional"] > sp["buy_and_hold"]).sum()
    print(f"\n  Beat buy-and-hold in {wins} of {len(sp)} sub-periods.")

    print("\n" + "=" * 66)
    print("3. WALK-FORWARD  (out of sample, refit as it goes)")
    print("=" * 66)
    wf = bt.walk_forward(
        prices,
        build_weights=lambda train, test: cross_sectional_momentum(
            pd.concat([train.tail(400), test])
        ).loc[test.index],
        train_bars=500,
        test_bars=180,
        embargo_bars=10,
        n_trials=trials,
        **COSTS,
    )
    print(wf)

    window = wf.returns.index
    p = prices.loc[window]
    bh = bt.run(p, st.buy_and_hold(p), **COSTS)
    ins = bt.run(p, cross_sectional_momentum(prices).loc[window],
                 n_trials=trials, **COSTS)

    print("\n" + "=" * 66)
    print("4. VERDICT  (all on identical bars)")
    print("=" * 66)
    print(f"  {'':<28}{'Sharpe':>10}{'CAGR':>11}{'max DD':>10}")
    print("  " + "-" * 60)
    for label, r in (("buy and hold", bh),
                     ("cross-sectional, in-sample", ins),
                     ("cross-sectional, walk-fwd", wf)):
        s = r.stats
        print(f"  {label:<28}{s['sharpe_annual']:>10.2f}"
              f"{s['cagr']:>11.1%}{s['max_drawdown']:>10.1%}")

    oos = wf.stats["sharpe_annual"]
    dsr = wf.stats["deflated_sharpe"]
    print()
    print(f"  Overfitting gap (in - out):  {ins.stats['sharpe_annual'] - oos:>+7.2f}")
    print(f"  Deflated Sharpe (n_trials={trials}): {dsr:>7.3f}")
    print()

    beats = oos > bh.stats["sharpe_annual"]
    if beats and dsr >= 0.95 and wins >= len(sp) - 1:
        print("  Survived all four tests. That is rare and worth taking")
        print("  seriously. Next: a different exchange, a different asset")
        print("  class, and double the costs. Then paper trade for a year.")
    elif beats and dsr < 0.95:
        print("  Beats the benchmark but does not clear the deflated Sharpe")
        print("  bar once the parameter search is accounted for. Promising,")
        print("  not proven. More data would settle it; more tuning would not.")
    else:
        print("  Did not survive. The in-sample result was the parameter")
        print("  search talking. This is the engine doing its job — it cost")
        print("  you an afternoon instead of a year of live trading.")

    _plot(grid)
    print(f"\n  Chart written to {RESULTS.resolve()}")


def _plot(grid: pd.DataFrame) -> None:
    RESULTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid.values.astype(float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("number of assets held")
    ax.set_title("Parameter surface — flat is good, spiky is overfit")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="in-sample Sharpe")
    plt.tight_layout()
    plt.savefig(RESULTS / "parameter_surface.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
