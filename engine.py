"""
Production trading engine. Runs in PAPER mode by default.

This is the code that would eventually touch real money, which is why it is
written now and run for months with none. Execution bugs are found by running,
not by reading. Every safety control below exists because it has cost somebody
real money at some point.

TO GO LIVE you must do three separate things:
    1. set LIVE = True below
    2. set the environment variable QUANTSTART_CONFIRM_LIVE=yes
    3. implement place_orders(), which deliberately raises today

Three steps, not one, because nobody has ever gone live by accident with three.

SAFETY CONTROLS
    kill switch      halts permanently past a drawdown limit
    leverage cap     hard maximum regardless of what sizing suggests
    staleness check  refuses to trade on old data
    turnover guard   refuses implausibly large position changes
    concentration    caps any single position
    audit log        every decision recorded, append only

Run:
    python engine.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_universe
from run_cross_sectional import CASUALTIES, SURVIVORS, cross_sectional_momentum

# ============================================================================
# CONFIGURATION. Change deliberately, never mid-run.
# ============================================================================

LIVE = False                      # <- the only switch that matters

STATE_FILE = Path("engine_state.json")
AUDIT_LOG = Path("engine_audit.jsonl")


@dataclass
class Config:
    # Sizing. Quarter Kelly. risk.py showed full Kelly (2.11x) carries a 93%
    # chance of a 50% drawdown, and 75% chance of losing half if the edge is
    # not real. 0.25x carries 0.1%.
    leverage: float = 0.25
    max_leverage: float = 0.50    # hard cap, never exceeded whatever sizing says

    # Kill switch. From risk.py: at 0.25x, a 25% limit is breached in ~3% of
    # simulated years. Tight enough to protect, loose enough to survive.
    max_drawdown: float = 0.25

    # Refuse to trade on stale data. Binance daily archives lag 1-2 days.
    max_data_age_days: int = 4

    # A daily strategy rebalancing more than this has probably malfunctioned.
    max_daily_turnover: float = 1.20

    # No single asset may exceed this share of the book.
    max_position: float = 0.40

    starting_equity: float = 10_000.0
    strategy_params: dict = field(default_factory=lambda: dict(
        lookbacks=(30, 90, 180), n_long=5, vol_window=60, target_vol=0.20,
    ))


CFG = Config()


# ============================================================================
# STATE
# ============================================================================

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    state = {
        "created": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if LIVE else "paper",
        "equity": CFG.starting_equity,
        "peak_equity": CFG.starting_equity,
        "positions": {},          # symbol -> weight
        "last_bar": None,
        "killed": False,
        "kill_reason": None,
        "history": [],            # [{date, equity, drawdown}]
    }
    save_state(state)
    print(f"  Initialised new {state['mode']} state at "
          f"{CFG.starting_equity:,.0f} notional.")
    return state


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def audit(event: str, **fields) -> None:
    """Append-only log. Never rewritten, so the record cannot be tidied up."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event,
              "mode": "live" if LIVE else "paper", **fields}
    with AUDIT_LOG.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ============================================================================
# SAFETY CHECKS. Each returns (ok, message).
# ============================================================================

def check_kill_switch(state: dict) -> tuple[bool, str]:
    if state["killed"]:
        return False, f"HALTED: {state['kill_reason']}"
    dd = state["equity"] / state["peak_equity"] - 1.0
    if dd < -CFG.max_drawdown:
        state["killed"] = True
        state["kill_reason"] = (f"drawdown {dd:.1%} breached limit "
                                f"{-CFG.max_drawdown:.0%} on "
                                f"{datetime.now(timezone.utc).date()}")
        save_state(state)
        audit("kill_switch", drawdown=dd, limit=-CFG.max_drawdown)
        return False, f"KILL SWITCH TRIPPED: {state['kill_reason']}"
    return True, f"drawdown {dd:.1%}, limit {-CFG.max_drawdown:.0%}"


def check_staleness(prices: pd.DataFrame) -> tuple[bool, str]:
    age = (pd.Timestamp.now(tz="UTC").normalize() - prices.index[-1]).days
    if age > CFG.max_data_age_days:
        return False, f"data is {age} days old, limit {CFG.max_data_age_days}"
    return True, f"data {age} days old"


def check_turnover(current: dict, target: pd.Series) -> tuple[bool, str]:
    cur = pd.Series(current, dtype=float).reindex(target.index).fillna(0.0)
    turnover = (target - cur).abs().sum()
    if turnover > CFG.max_daily_turnover:
        return False, (f"turnover {turnover:.2f} exceeds limit "
                       f"{CFG.max_daily_turnover:.2f}")
    return True, f"turnover {turnover:.2f}"


def check_concentration(target: pd.Series) -> tuple[bool, str]:
    worst = target.abs().max()
    if worst > CFG.max_position:
        return False, f"position {worst:.1%} exceeds limit {CFG.max_position:.0%}"
    return True, f"largest position {worst:.1%}"


# ============================================================================
# EXECUTION
# ============================================================================

def place_orders(orders: pd.Series, prices: pd.Series) -> None:
    """
    Live execution. Deliberately not implemented.

    When you do implement it, the exchange API key must be created with
    TRADE permission only. Never enable withdrawal. A key that cannot move
    coins off the exchange cannot be used to steal them, however badly the
    rest of this code fails.
    """
    raise NotImplementedError(
        "Live execution is not implemented. This is intentional. "
        "Implement it only after a full paper trading period, and only with "
        "a trade-only API key that has withdrawals disabled."
    )


def simulate_fills(state: dict, target: pd.Series, prices: pd.DataFrame) -> float:
    """
    Paper fills. Marks the book forward using the realised return since the
    last run, then applies costs on the traded amount.

    Costs are charged at the same 15 bps used throughout the backtests, so
    paper results stay comparable to simulated ones.
    """
    last = state.get("last_bar")
    if last is None:
        return 0.0

    last_ts = pd.Timestamp(last)
    window = prices.loc[last_ts:]
    if len(window) < 2:
        return 0.0

    held = pd.Series(state["positions"], dtype=float)
    if held.empty:
        return 0.0

    period_ret = (window.iloc[-1] / window.iloc[0] - 1.0)
    held = held.reindex(period_ret.index).fillna(0.0)
    gross = float((held * period_ret.fillna(0.0)).sum())

    traded = float((target.reindex(held.index).fillna(0.0) - held).abs().sum())
    cost = traded * 15 / 10_000.0
    return gross - cost


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    mode = "LIVE" if LIVE else "PAPER"
    print("=" * 66)
    print(f"QUANTSTART ENGINE  [{mode}]  "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 66)

    if LIVE and os.environ.get("QUANTSTART_CONFIRM_LIVE") != "yes":
        print("  LIVE is set but QUANTSTART_CONFIRM_LIVE is not 'yes'. Refusing.")
        sys.exit(1)

    state = load_state()

    ok, msg = check_kill_switch(state)
    print(f"\n  kill switch    {msg}")
    if not ok:
        print("\n  No trading. Investigate before restarting, and reset state")
        print("  deliberately — do not delete the file to make this go away.")
        audit("halted", reason=state["kill_reason"])
        return

    print("\n  Loading prices...")
    surv = load_universe(SURVIVORS, "1d", "2019-01",
                         datetime.now(timezone.utc).strftime("%Y-%m"),
                         include_tail=True)
    dead = load_universe(CASUALTIES, "1d", "2019-01",
                         datetime.now(timezone.utc).strftime("%Y-%m"),
                         include_tail=True).dropna(axis=1, how="all")
    prices = pd.concat([surv, dead], axis=1).sort_index().dropna(how="all")

    ok, msg = check_staleness(prices)
    print(f"  staleness      {msg}")
    if not ok:
        print("\n  Refusing to trade on stale data.")
        audit("skipped", reason=msg)
        return

    # --- Mark the book forward before computing anything new.
    weights = cross_sectional_momentum(prices, **CFG.strategy_params)
    raw_target = weights.iloc[-1]
    bar = weights.index[-1]

    lev = min(CFG.leverage, CFG.max_leverage)
    target = (raw_target * lev).round(6)

    ok_t, msg_t = check_turnover(state["positions"], target)
    ok_c, msg_c = check_concentration(target)
    print(f"  turnover       {msg_t}")
    print(f"  concentration  {msg_c}")

    if not (ok_t and ok_c):
        print("\n  Sanity check failed. Not trading today.")
        audit("skipped", turnover_ok=ok_t, concentration_ok=ok_c,
              detail=f"{msg_t}; {msg_c}")
        return

    pnl_pct = simulate_fills(state, target, prices)
    state["equity"] *= (1.0 + pnl_pct)
    state["peak_equity"] = max(state["peak_equity"], state["equity"])
    drawdown = state["equity"] / state["peak_equity"] - 1.0

    print("\n" + "-" * 66)
    print(f"  bar            {bar.date()}")
    print(f"  period P&L     {pnl_pct:>+8.2%}")
    print(f"  equity         {state['equity']:>12,.2f}")
    print(f"  drawdown       {drawdown:>+8.2%}")
    print(f"  leverage       {lev:>8.2f}x")
    print("-" * 66)

    held = target[target.abs() > 1e-6].sort_values(ascending=False)
    cur = pd.Series(state["positions"], dtype=float)
    print("\n  TARGET BOOK")
    if held.empty:
        print("    flat")
    else:
        for sym, w in held.items():
            prev = float(cur.get(sym, 0.0))
            delta = w - prev
            arrow = "+" if delta > 1e-6 else ("-" if delta < -1e-6 else " ")
            print(f"    {sym:<12}{w:>8.2%}   ({arrow}{abs(delta):.2%})")
        print(f"    {'GROSS':<12}{held.abs().sum():>8.2%}")

    if LIVE:
        place_orders(target - cur.reindex(target.index).fillna(0.0),
                     prices.iloc[-1])

    state["positions"] = {k: float(v) for k, v in target.items() if abs(v) > 1e-6}
    state["last_bar"] = str(bar)
    state["history"].append({
        "date": str(bar.date()), "equity": state["equity"],
        "drawdown": drawdown, "pnl_pct": pnl_pct,
    })
    save_state(state)
    audit("traded", bar=str(bar.date()), equity=state["equity"],
          drawdown=drawdown, pnl_pct=pnl_pct, positions=state["positions"])

    n = len(state["history"])
    print(f"\n  {n} sessions recorded. State in {STATE_FILE}, "
          f"audit trail in {AUDIT_LOG}.")
    if n >= 20:
        hist = pd.DataFrame(state["history"])
        rets = hist["pnl_pct"]
        if rets.std() > 0:
            sharpe = rets.mean() / rets.std() * np.sqrt(365)
            print(f"  Live Sharpe so far: {sharpe:>.2f}  "
                  f"(backtest expectation ~1.50)")
            print("  Watch for divergence, not for losses. A live Sharpe far")
            print("  from backtest means the simulation is wrong somewhere.")
    else:
        print(f"  Need {20 - n} more sessions before live stats mean anything.")


if __name__ == "__main__":
    main()
