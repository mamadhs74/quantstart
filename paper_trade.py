"""
The end of backtesting and the start of evidence.

The robustness battery reached a deflated Sharpe of 0.937 against a 0.95 bar.
That gap cannot be closed by more backtesting — every additional variant you
test raises n_trials and pushes the deflated Sharpe DOWN. The historical data
is spent.

The only thing that generates new information is time.

WHAT THIS DOES
--------------
Run it once a day. It:
  1. downloads the latest prices
  2. computes today's target weights from the frozen strategy
  3. appends them to a log with a timestamp
  4. scores every past signal against what the market actually did
  5. reports the live track record next to the backtest expectation

No money involved. The point is to find out whether live execution matches
simulation before that question costs anything.

WHAT TO WATCH FOR
-----------------
The failure mode is not "live loses money". It is "live and backtest diverge".
If your realised returns differ materially from what the backtest said the
same signals should have produced, the backtest is wrong somewhere -- stale
prices, wrong fill assumptions, a lookahead you did not catch.

Freeze the parameters before you start. Changing them mid-run destroys the
track record, because you are then measuring a strategy that did not exist
when the signals were generated.

Run:
    python paper_trade.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from data import load_universe
from run_cross_sectional import CASUALTIES, SURVIVORS, cross_sectional_momentum
from run_real import COSTS

LOG = Path("paper_log.csv")
META = Path("paper_meta.json")

# FROZEN. Do not change these once logging has started. If you must change
# them, archive the old log and start a new one -- do not mix.
PARAMS = dict(lookbacks=(30, 90, 180), n_long=5, vol_window=60, target_vol=0.20)
START = "2019-01"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ensure_meta() -> dict:
    if META.exists():
        return json.loads(META.read_text())
    meta = {
        "started": _today(),
        "params": {k: list(v) if isinstance(v, tuple) else v
                   for k, v in PARAMS.items()},
        "universe": SURVIVORS + CASUALTIES,
        "costs": COSTS,
    }
    META.write_text(json.dumps(meta, indent=2))
    print(f"  Started a new paper trading log on {meta['started']}")
    print("  Parameters are now FROZEN. Changing them invalidates the record.")
    return meta


def main() -> None:
    meta = _ensure_meta()
    end = datetime.now(timezone.utc).strftime("%Y-%m")

    print("Downloading latest prices...")
    surv = load_universe(SURVIVORS, "1d", START, end)
    dead = load_universe(CASUALTIES, "1d", START, end).dropna(axis=1, how="all")
    prices = pd.concat([surv, dead], axis=1).sort_index().dropna(how="all")

    weights = cross_sectional_momentum(prices, **PARAMS)
    latest = weights.iloc[-1]
    latest_date = weights.index[-1]

    print("\n" + "=" * 60)
    print(f"TARGET POSITIONS FOR {latest_date.date()}")
    print("=" * 60)
    held = latest[latest.abs() > 1e-6].sort_values(ascending=False)
    if held.empty:
        print("  Flat. No positions today.")
    else:
        for sym, w in held.items():
            print(f"  {sym:<12}{w:>8.1%}")
        print(f"  {'TOTAL':<12}{held.sum():>8.1%}")

    # --- Append today's signal, one row per day, no duplicates.
    row = latest.to_frame().T
    row.index = [latest_date]
    if LOG.exists():
        log = pd.read_csv(LOG, index_col=0, parse_dates=True)
        log = pd.concat([log[~log.index.isin(row.index)], row]).sort_index()
    else:
        log = row
    log.to_csv(LOG)
    print(f"\n  Logged. {len(log)} signals recorded since {meta['started']}.")

    # --- Score the live record.
    if len(log) < 5:
        print(f"\n  Need at least 5 days to score. Come back in "
              f"{5 - len(log)} more runs.")
        return

    live_prices = prices.reindex(columns=log.columns).loc[log.index[0]:]
    live = bt.run(live_prices, log.reindex(live_prices.index).ffill().fillna(0.0),
                  **COSTS)

    print("\n" + "=" * 60)
    print("LIVE PAPER RECORD")
    print("=" * 60)
    print(live)

    # --- Compare against what the backtest expected over the same window.
    hist = bt.run(prices.loc[:log.index[0]],
                  cross_sectional_momentum(prices.loc[:log.index[0]], **PARAMS),
                  **COSTS)

    print("\n" + "=" * 60)
    print("LIVE vs BACKTEST")
    print("=" * 60)
    print(f"  {'':<22}{'backtest':>12}{'live':>12}")
    print("  " + "-" * 46)
    for key, label in (("sharpe_annual", "Sharpe"),
                       ("cagr", "CAGR"),
                       ("max_drawdown", "max drawdown"),
                       ("turnover_annual", "turnover")):
        b, l = hist.stats.get(key, np.nan), live.stats.get(key, np.nan)
        fmt = "{:>12.1%}" if key in ("cagr", "max_drawdown") else "{:>12.2f}"
        print(f"  {label:<22}" + fmt.format(b) + fmt.format(l))

    years = live.stats.get("years", 0)
    print()
    if years < 0.25:
        print(f"  {years * 12:.1f} months of live data. Far too early to")
        print("  conclude anything. Keep running it daily.")
    elif years < 1:
        print(f"  {years * 12:.1f} months live. Watch whether turnover and")
        print("  drawdown match the backtest — those diverge before returns do.")
    else:
        print(f"  {years:.1f} years live. Now the comparison means something.")
        print("  A live Sharpe within about 0.5 of backtest is normal decay.")
        print("  A much larger gap means the backtest is wrong, not unlucky.")

    print("\n  Run this daily. Do not change PARAMS.")


if __name__ == "__main__":
    main()
