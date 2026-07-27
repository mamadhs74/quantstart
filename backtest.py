"""
A backtester built to tell you the truth rather than to make you feel good.

Three rules are enforced structurally, not by convention:

  1. A position held over bar t may only depend on data up to bar t-1.
     The engine applies the shift itself. You cannot forget to.
  2. Turnover always costs money. Fees and slippage are mandatory arguments.
  3. Volatility scaling uses trailing volatility only, never full-sample.

If a strategy looks good here, it might be real. If it looks good only after
you relax one of these, it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Bars per year, for annualising. Extend as needed.
ANNUALISER = {"1d": 365, "1h": 365 * 24, "4h": 365 * 6, "1w": 52}


@dataclass
class Result:
    returns: pd.Series          # net per-bar portfolio returns
    positions: pd.DataFrame     # weights actually held each bar
    equity: pd.Series
    stats: dict

    def __repr__(self) -> str:
        lines = ["Backtest result"]
        for k, v in self.stats.items():
            lines.append(f"  {k:<22} {v:>10.4f}" if isinstance(v, float) else f"  {k:<22} {v:>10}")
        return "\n".join(lines)


def run(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    interval: str = "1d",
    n_trials: int = 1,
) -> Result:
    """
    prices          wide frame of close prices, one column per asset
    target_weights  desired weights, indexed identically to prices.
                    Computed from information available AT bar t; the engine
                    lags them by one bar before applying, so you trade at t+1.
    fee_bps         exchange fee per side, in basis points (Binance spot ~10)
    slippage_bps    assumed adverse fill, in basis points
    n_trials        how many strategy variants you tried before this one.
                    Be honest. It feeds the deflated Sharpe ratio.
    """
    prices = prices.sort_index()
    target_weights = target_weights.reindex(prices.index).fillna(0.0)

    # --- The single most important line in this file -------------------------
    # Weights decided using bar t's close can only be held from bar t+1 onward.
    held = target_weights.shift(1).fillna(0.0)
    # ------------------------------------------------------------------------

    asset_returns = prices.pct_change().fillna(0.0)
    gross = (held * asset_returns).sum(axis=1)

    turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    net = gross - turnover * cost_rate

    equity = (1.0 + net).cumprod()

    return Result(
        returns=net,
        positions=held,
        equity=equity,
        stats=_summarise(net, turnover, interval, n_trials),
    )


def _summarise(r: pd.Series, turnover: pd.Series, interval: str, n_trials: int) -> dict:
    ann = ANNUALISER.get(interval, 365)
    r = r.dropna()
    if r.std() == 0 or len(r) < 2:
        return {"n_bars": len(r), "note": "degenerate"}

    sharpe_per_bar = r.mean() / r.std()
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = len(r) / ann

    return {
        "n_bars": len(r),
        "years": years,
        "cagr": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "vol_annual": r.std() * np.sqrt(ann),
        "sharpe_annual": sharpe_per_bar * np.sqrt(ann),
        "max_drawdown": drawdown.min(),
        "hit_rate": (r > 0).mean(),
        "turnover_annual": turnover.mean() * ann,
        "deflated_sharpe": deflated_sharpe(r, n_trials),
    }


def deflated_sharpe(returns: pd.Series, n_trials: int = 1) -> float:
    """
    Bailey & Lopez de Prado (2014), 'The Deflated Sharpe Ratio'.

    Answers: given that I tried n_trials strategies, and given that my returns
    are skewed and fat-tailed, what is the probability this Sharpe is real
    rather than the best of n_trials coin flips?

    Output is a probability. Below ~0.95, treat the result as noise.
    """
    r = returns.dropna().values
    T = len(r)
    if T < 10 or r.std() == 0:
        return np.nan

    sr = r.mean() / r.std()                     # per-bar, NOT annualised
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)      # non-excess

    # Expected maximum Sharpe achievable by pure luck across n_trials.
    if n_trials > 1:
        gamma = 0.5772156649  # Euler-Mascheroni
        e = np.e
        sr0 = np.sqrt(1.0 / T) * (
            (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
            + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * e))
        )
    else:
        sr0 = 0.0

    denom = np.sqrt(1 - skew * sr + ((kurt - 1) / 4.0) * sr**2)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan

    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def walk_forward(
    prices: pd.DataFrame,
    build_weights,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 0,
    **run_kwargs,
) -> Result:
    """
    Anchored walk-forward with an optional embargo gap.

    build_weights(train_df, test_df) -> weights for the test window only.
    It never sees anything past the end of its test window. The embargo drops
    bars between train and test so that overlapping-label leakage cannot
    carry information backwards.
    """
    pieces = []
    start = train_bars

    while start + embargo_bars + test_bars <= len(prices):
        train = prices.iloc[:start]
        test_start = start + embargo_bars
        test = prices.iloc[test_start : test_start + test_bars]

        w = build_weights(train, test)
        pieces.append(w.reindex(test.index).fillna(0.0))
        start = test_start + test_bars

    if not pieces:
        raise ValueError("Not enough data for even one train/test split.")

    weights = pd.concat(pieces)
    return run(prices.loc[weights.index], weights, **run_kwargs)
