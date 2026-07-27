"""
Baselines first. Always baselines first.

A strategy result means nothing on its own. It only means something next to
the trivial thing it is supposed to beat. These three are your yardsticks:

  buy_and_hold   the floor. If you cannot beat this, stop.
  random_weights the noise level. Run it 100 times and look at the spread of
                 Sharpes. Anything inside that spread is not a discovery.
  ts_momentum    the published, decades-tested baseline
                 (Moskowitz, Ooi & Pedersen 2012).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal weight, never rebalanced away from target."""
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def random_weights(prices: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Your null hypothesis, with the same turnover profile as a real strategy."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=prices.shape)
    w = pd.DataFrame(raw, index=prices.index, columns=prices.columns)
    return w.div(w.abs().sum(axis=1), axis=0)


def ts_momentum(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = (30, 90, 180, 360),
    vol_window: int = 60,
    target_vol: float = 0.20,
    ann: int = 365,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """
    Time-series momentum, ensembled across lookbacks and volatility-targeted.

    Ensembling is not decoration. A single lookback is a free parameter you
    will end up fitting to your sample; averaging several removes most of that
    temptation and costs almost nothing in performance.

    Vol targeting is where most of the risk-adjusted improvement actually comes
    from, not from the signal. Everything below uses trailing windows only.
    """
    returns = prices.pct_change()

    # Sign of trailing return over each horizon, averaged into [-1, 1].
    signals = [np.sign(prices.pct_change(lb)) for lb in lookbacks]
    signal = sum(signals) / len(signals)

    # Inverse-volatility sizing, trailing estimate only.
    vol = returns.rolling(vol_window).std() * np.sqrt(ann)
    scaled = signal * (target_vol / vol.replace(0, np.nan))
    scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Cap gross exposure so a low-vol regime cannot lever you into oblivion.
    gross = scaled.abs().sum(axis=1)
    over = gross > max_leverage
    scaled.loc[over] = scaled.loc[over].div(gross[over], axis=0) * max_leverage

    return scaled.fillna(0.0)


# --------------------------------------------------------------------------
# The one strategy you must run, and must never believe.
# --------------------------------------------------------------------------
def oracle(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Perfect foresight: hold tomorrow's winner, known today. Cheating.

    Run it once. Note the Sharpe. That number is the ceiling no real strategy
    can approach. If one of your honest strategies ever produces a Sharpe in
    the same neighbourhood, you have a lookahead bug, not an edge.
    """
    fwd = prices.pct_change().shift(-1)
    fwd = fwd.dropna(how="all")          # first and last bars are undefined
    best = fwd.idxmax(axis=1)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for t, col in best.dropna().items():
        w.loc[t, col] = 1.0
    return w
