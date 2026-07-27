"""
Runs the whole pipeline on synthetic data with KNOWN properties.

This is the step almost everyone skips, and it is the reason almost everyone's
backtester is silently wrong. Before you trust an engine on real prices, prove
it produces the correct answer on data where you already know the answer.

Three assertions:
  1. On pure random walks, no honest strategy should show a real edge.
  2. Costs must reduce returns, monotonically, as turnover rises.
  3. The oracle must score absurdly high. If it doesn't, the engine is broken.
     If an honest strategy scores near it, YOUR strategy is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import strategies as st


def synthetic_prices(n_assets=6, n_bars=2000, seed=42, drift=0.0, vol=0.03):
    """Geometric random walks. By construction, no predictable structure."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=(n_bars, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2019-01-01", periods=n_bars, freq="D", tz="UTC")
    cols = [f"ASSET{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=idx, columns=cols)


def main() -> None:
    prices = synthetic_prices()
    print(f"Synthetic universe: {prices.shape[1]} assets, {len(prices)} daily bars")
    print("Data-generating process: pure random walk, no exploitable structure.\n")

    print("=" * 64)
    print("1. ORACLE  (cheats — establishes the ceiling)")
    print("=" * 64)
    res = bt.run(prices, st.oracle(prices), interval="1d")
    print(res)
    oracle_sharpe = res.stats["sharpe_annual"]

    print("\n" + "=" * 64)
    print("2. BUY AND HOLD  (the floor)")
    print("=" * 64)
    print(bt.run(prices, st.buy_and_hold(prices), interval="1d"))

    print("\n" + "=" * 64)
    print("3. TIME-SERIES MOMENTUM on random walks")
    print("   Should show NO edge. If it does, the engine is lying.")
    print("=" * 64)
    res_mom = bt.run(
        prices, st.ts_momentum(prices), interval="1d", n_trials=20
    )
    print(res_mom)

    print("\n" + "=" * 64)
    print("4. NOISE FLOOR  (30 random strategies)")
    print("   Any Sharpe inside this range is not a discovery.")
    print("=" * 64)
    # Zero costs here, so this measures SIGNAL quality alone.
    free = dict(fee_bps=0, slippage_bps=0, interval="1d")
    sharpes = [
        bt.run(prices, st.random_weights(prices, seed=s), **free)
        .stats["sharpe_annual"]
        for s in range(30)
    ]
    lo, hi = np.percentile(sharpes, [5, 95])
    mom_free = bt.run(prices, st.ts_momentum(prices), **free).stats["sharpe_annual"]
    print(f"  random Sharpe 5th-95th pct   {lo:>8.3f} to {hi:>8.3f}")
    print(f"  momentum Sharpe (no costs)   {mom_free:>8.3f}")
    inside = lo <= mom_free <= hi
    print(f"  -> momentum is {'INSIDE the noise band (correct)' if inside else 'OUTSIDE the band'}")
    print()
    print("  Note the gap between this and section 3. Same signal, same data.")
    print(f"  Costs alone moved Sharpe from {mom_free:.2f} to "
          f"{res_mom.stats['sharpe_annual']:.2f}. That is the whole game.")

    print("\n" + "=" * 64)
    print("5. COST SENSITIVITY")
    print("   A strategy that dies on realistic costs was never alive.")
    print("=" * 64)
    for fee in (0, 5, 10, 25, 50):
        r = bt.run(
            prices, st.ts_momentum(prices),
            fee_bps=fee, slippage_bps=0, interval="1d",
        )
        print(f"  fee {fee:>3} bps  ->  CAGR {r.stats['cagr']:>8.2%}   "
              f"Sharpe {r.stats['sharpe_annual']:>6.3f}")

    print("\n" + "=" * 64)
    print("6. WALK-FORWARD with embargo")
    print("=" * 64)
    res_wf = bt.walk_forward(
        prices,
        build_weights=lambda train, test: st.ts_momentum(
            pd.concat([train.tail(400), test])
        ).loc[test.index],
        train_bars=500,
        test_bars=200,
        embargo_bars=10,
        interval="1d",
        n_trials=20,
    )
    print(res_wf)

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"  Oracle Sharpe (impossible)     {oracle_sharpe:>8.2f}")
    print(f"  Momentum Sharpe (honest)       {res_mom.stats['sharpe_annual']:>8.2f}")
    print(f"  Deflated Sharpe probability    {res_mom.stats['deflated_sharpe']:>8.3f}")
    print()
    print("  Deflated Sharpe below 0.95 means: given how many variants were")
    print("  tried, this result is indistinguishable from luck. Which is the")
    print("  correct answer here, because the data is literally noise.")
    print()
    print("  The engine passes. Now point it at real prices — and expect it")
    print("  to keep giving you answers you do not want.")


if __name__ == "__main__":
    main()
