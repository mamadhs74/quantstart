"""
The first actual machine learning in this project — and the first test that
the walk-forward machinery can genuinely fail.

THE QUESTION
------------
The regime analysis showed momentum wins in trending markets (2022: -1.2% vs
buy-and-hold's -69.6%) and bleeds in choppy ones (2023: -31.2% vs +152.2%).

So: can we predict, from information available today, whether the next month
favours trend following? If yes, scale exposure up. If no, scale down.

This is a legitimate ML problem because it has an economic reason to work.
"Neural network reads candlesticks and predicts price" does not.

WHY THIS WILL PROBABLY FAIL
---------------------------
You have ~1800 daily bars. After a 500-bar warmup, roughly 1300 usable
samples, and the monthly targets overlap heavily so the effective independent
sample count is closer to 60. Sixty samples. That is not a machine learning
dataset, it is a rounding error.

This is THE fundamental constraint of finance ML and no model architecture
fixes it. A bigger network does not create information that isn't there.

Run:
    python ml_regime.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

import backtest as bt
import strategies as st
from data import load_universe
from run_real import COSTS, END, START, UNIVERSE

RESULTS = Path("results")
HORIZON = 21          # predict roughly one month ahead
TRAIN_BARS = 500
TEST_BARS = 180
EMBARGO = HORIZON     # embargo MUST be >= horizon or labels leak backwards


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Features describing the market regime. All trailing, all cheap to compute,
    all with a plausible reason to relate to trend persistence.

    Deliberately few. With ~60 effective samples, every extra feature is
    another chance to fit noise.
    """
    ret = prices.pct_change()
    eq = ret.mean(axis=1)                    # equal-weight market proxy

    f = pd.DataFrame(index=prices.index)

    # Realised volatility, short and long. Trends tend to persist in calm
    # markets and break in violent ones.
    f["vol_20"] = eq.rolling(20).std() * np.sqrt(365)
    f["vol_ratio"] = (eq.rolling(20).std() / eq.rolling(90).std())

    # Trend strength: how much of the total path length was net direction.
    # Near 1 means a clean trend, near 0 means chop.
    net = eq.rolling(60).sum().abs()
    path = eq.abs().rolling(60).sum()
    f["efficiency"] = net / path.replace(0, np.nan)

    # Cross-sectional dispersion. When assets move together, trends are
    # usually market-wide and more durable.
    f["dispersion"] = ret.std(axis=1).rolling(20).mean()

    # Autocorrelation of daily returns. Positive means momentum-friendly.
    f["autocorr"] = eq.rolling(60).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )

    # Drawdown depth of the market proxy.
    proxy = (1 + eq).cumprod()
    f["drawdown"] = proxy / proxy.cummax() - 1.0

    return f.replace([np.inf, -np.inf], np.nan)


def build_target(strategy_returns: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """
    What we're predicting: the Sharpe of the BASE STRATEGY over the next
    `horizon` days. Positive means trend following is about to work.

    Note this is forward-looking by construction. It is a label, never a
    feature, and the embargo below keeps it from leaking into training.
    """
    fwd_mean = strategy_returns.shift(-horizon).rolling(horizon).mean()
    fwd_std = strategy_returns.shift(-horizon).rolling(horizon).std()
    return (fwd_mean / fwd_std.replace(0, np.nan)).rename("target")


def walk_forward_ml(
    prices: pd.DataFrame,
    base_weights: pd.DataFrame,
    base_returns: pd.Series,
    model_name: str,
):
    """
    Refit the model on each training window, predict on the test window, and
    use the prediction to scale the base strategy's exposure between 0 and 1.

    Embargo of HORIZON bars between train and test. Without it, the last
    training labels overlap the first test bars and the model sees the future.
    """
    features = build_features(prices)
    target = build_target(base_returns)

    scaled_pieces = []
    start = TRAIN_BARS

    while start + EMBARGO + TEST_BARS <= len(prices):
        tr_idx = prices.index[:start]
        te_idx = prices.index[start + EMBARGO : start + EMBARGO + TEST_BARS]

        Xtr = features.loc[tr_idx]
        ytr = target.loc[tr_idx]
        ok = Xtr.notna().all(axis=1) & ytr.notna()

        Xte = features.loc[te_idx].ffill()

        if ok.sum() < 100 or Xte.isna().all(axis=None):
            exposure = pd.Series(1.0, index=te_idx)
        else:
            model = (
                Ridge(alpha=10.0) if model_name == "ridge"
                else GradientBoostingRegressor(
                    n_estimators=60, max_depth=2, learning_rate=0.05,
                    subsample=0.8, random_state=0,
                )
            )
            model.fit(Xtr[ok], ytr[ok])
            pred = pd.Series(model.predict(Xte.fillna(0.0)), index=te_idx)
            # Map prediction to exposure in [0, 1]. Positive forecast -> on.
            exposure = (pred > 0).astype(float).clip(0.0, 1.0)

        scaled_pieces.append(base_weights.loc[te_idx].mul(exposure, axis=0))
        start = start + EMBARGO + TEST_BARS

    weights = pd.concat(scaled_pieces)
    return bt.run(prices.loc[weights.index], weights, **COSTS), weights.index


def main() -> None:
    prices = load_universe(UNIVERSE, "1d", START, END).dropna(how="all")
    base_weights = st.ts_momentum(prices)
    base_all = bt.run(prices, base_weights, **COSTS)

    print("=" * 64)
    print("ML REGIME FILTER")
    print("=" * 64)
    print(f"  Bars available:        {len(prices)}")
    print(f"  Prediction horizon:    {HORIZON} days")
    print(f"  Embargo:               {EMBARGO} days")
    eff = (len(prices) - TRAIN_BARS) / HORIZON
    print(f"  Effective independent samples: ~{eff:.0f}")
    print("  (overlapping labels mean the real number is roughly this, not")
    print("   the raw bar count — this is the whole problem)")

    results = {}
    for name in ("ridge", "gbm"):
        res, window = walk_forward_ml(prices, base_weights, base_all.returns, name)
        results[name] = res
        print("\n" + "=" * 64)
        print(f"MOMENTUM + {name.upper()} REGIME FILTER  (walk-forward)")
        print("=" * 64)
        print(res)

    # Same window, unfiltered, for a fair comparison.
    window = list(results.values())[0].returns.index
    p = prices.loc[window]
    plain = bt.run(p, base_weights.loc[window], **COSTS)
    bh = bt.run(p, st.buy_and_hold(p), **COSTS)

    print("\n" + "=" * 64)
    print("MOMENTUM, NO FILTER  (same bars — the thing to beat)")
    print("=" * 64)
    print(plain)

    results["momentum_plain"] = plain
    results["buy_and_hold"] = bh
    _plot(results)

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"  {'strategy':<28}{'Sharpe':>10}{'CAGR':>11}{'max DD':>11}")
    print("  " + "-" * 58)
    order = ["buy_and_hold", "momentum_plain", "ridge", "gbm"]
    for k in order:
        s = results[k].stats
        print(f"  {k:<28}{s['sharpe_annual']:>10.2f}"
              f"{s['cagr']:>11.1%}{s['max_drawdown']:>11.1%}")

    plain_sh = plain.stats["sharpe_annual"]
    best_ml = max(results["ridge"].stats["sharpe_annual"],
                  results["gbm"].stats["sharpe_annual"])

    print()
    print(f"  Best ML filter minus plain momentum: {best_ml - plain_sh:>+7.2f} Sharpe")
    print()
    if best_ml > plain_sh + 0.2:
        print("  The filter helped, out of sample. Do not celebrate yet —")
        print("  re-run with a different horizon, a different universe, and")
        print("  a different random seed. If the gain vanishes, it was noise.")
    else:
        print("  The filter did not help. This is the expected result, and it")
        print("  is the most valuable thing this script can tell you.")
        print()
        print("  Sixty effective samples cannot support learning a reliable")
        print("  regime signal. The honest response is NOT a bigger model.")
        print("  It is more data: a longer history, more assets, or a shorter")
        print("  prediction horizon that yields more independent samples.")

    print(f"\n  Chart written to {RESULTS.resolve()}")


def _plot(results: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=name, linewidth=1.3)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log scale)")
    ax.set_title("Does an ML regime filter add anything? (walk-forward, net of costs)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "ml_regime.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
