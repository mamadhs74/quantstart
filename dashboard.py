"""
Monitoring dashboard for the trading engine.

    pip install streamlit
    streamlit run dashboard.py

DELIBERATELY READ-ONLY
----------------------
There is no button here that starts trading, changes mode, adjusts leverage,
or resets the kill switch. That is not an oversight.

engine.py requires three separate deliberate acts to go live: edit LIVE,
set an environment variable, and implement place_orders(). A toggle in a web
UI would collapse all three into one click reachable by a misclick, a stale
browser tab, or anyone who can reach localhost:8501.

This dashboard shows you what the system is doing. Changing what it does
stays in the code, under version control, where it leaves a git history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

STATE_FILE = Path("engine_state.json")
AUDIT_LOG = Path("engine_audit.jsonl")

# Expectations from the backtest, for comparison. From robustness.py.
BACKTEST_SHARPE = 1.50
BACKTEST_TURNOVER = 79.3
BACKTEST_MAX_DD = -0.735
MAX_DRAWDOWN_LIMIT = 0.25

st.set_page_config(page_title="quantstart", layout="wide")


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text())


def load_audit() -> pd.DataFrame:
    if not AUDIT_LOG.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in AUDIT_LOG.read_text().splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts", ascending=False)


state = load_state()
if state is None:
    st.error("No engine state found. Run `python engine.py` first.")
    st.stop()

# ---------------------------------------------------------------- header
mode = state.get("mode", "paper").upper()
if mode == "LIVE":
    st.error(f"### MODE: {mode} — REAL MONEY AT RISK", icon="🔴")
else:
    st.success(f"### MODE: {mode} — no money at risk", icon="🟢")

if state.get("killed"):
    st.error(f"**ENGINE HALTED.** {state.get('kill_reason')}", icon="⛔")
    st.caption("Trading has stopped permanently. Investigate before resetting "
               "state — do not delete the file to make this go away.")

# ---------------------------------------------------------------- metrics
hist = pd.DataFrame(state.get("history", []))
equity = state["equity"]
peak = state["peak_equity"]
drawdown = equity / peak - 1.0
sessions = len(hist)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", f"{equity:,.2f}",
          f"{equity / 10_000 - 1:+.2%}" if equity != 10_000 else None)
c2.metric("Drawdown", f"{drawdown:.2%}",
          f"limit {-MAX_DRAWDOWN_LIMIT:.0%}", delta_color="off")
c3.metric("Sessions", sessions,
          f"{max(0, 20 - sessions)} to meaningful stats" if sessions < 20 else "sufficient")
c4.metric("Last bar", str(state.get("last_bar", "—"))[:10])

# Distance to the kill switch, as a bar.
used = min(abs(drawdown) / MAX_DRAWDOWN_LIMIT, 1.0)
st.progress(used, text=f"Kill switch budget used: {used:.0%}")

st.divider()

# ---------------------------------------------------------------- charts
left, right = st.columns([2, 1])

with left:
    st.subheader("Equity")
    if sessions >= 2:
        h = hist.copy()
        h["date"] = pd.to_datetime(h["date"])
        h = h.set_index("date")
        st.line_chart(h["equity"], height=260)
        st.subheader("Drawdown")
        st.area_chart(h["drawdown"], height=180, color="#d9534f")
    else:
        st.info("Two or more sessions needed before charting. "
                "The engine runs once per day.")

with right:
    st.subheader("Current book")
    positions = state.get("positions", {})
    if not positions:
        st.write("Flat — no positions.")
    else:
        pos = (pd.Series(positions, name="weight")
               .sort_values(ascending=False)
               .to_frame())
        pos["weight"] = pos["weight"].map("{:.2%}".format)
        st.dataframe(pos, use_container_width=True)
        st.caption(f"Gross exposure: {sum(abs(v) for v in positions.values()):.1%}")

st.divider()

# ---------------------------------------------------------------- live vs backtest
st.subheader("Live vs backtest")

if sessions < 20:
    st.info(
        f"**{sessions} of 20 sessions.** Live statistics are meaningless "
        "before roughly 20 observations. The comparison below unlocks then.\n\n"
        "What matters is not whether live makes money — it is whether live "
        "*matches* the simulation. Turnover and drawdown diverge before "
        "returns do, so those are the early warning."
    )
else:
    rets = hist["pnl_pct"]
    live_sharpe = (rets.mean() / rets.std() * (365 ** 0.5)) if rets.std() > 0 else float("nan")
    live_dd = hist["drawdown"].min()

    comp = pd.DataFrame({
        "backtest": [BACKTEST_SHARPE, BACKTEST_MAX_DD, BACKTEST_TURNOVER],
        "live": [live_sharpe, live_dd, float("nan")],
    }, index=["Sharpe (annualised)", "Max drawdown", "Turnover (annual)"])
    st.dataframe(comp.style.format("{:.2f}"), use_container_width=True)

    gap = abs(live_sharpe - BACKTEST_SHARPE)
    if gap < 0.5:
        st.success("Live is tracking the backtest within normal decay.")
    elif gap < 1.0:
        st.warning("Live is drifting from backtest. Worth watching closely.")
    else:
        st.error("Live and backtest disagree materially. The simulation is "
                 "probably wrong somewhere — check fills, costs, and timing "
                 "before drawing conclusions about the strategy.")

st.divider()

# ---------------------------------------------------------------- audit
st.subheader("Audit trail")
audit = load_audit()
if audit.empty:
    st.write("No audit records yet.")
else:
    cols = [c for c in ["ts", "event", "mode", "bar", "equity", "drawdown", "reason"]
            if c in audit.columns]
    st.dataframe(audit[cols].head(50), use_container_width=True, height=280)
    st.caption(f"{len(audit)} records. Append-only — this file is never "
               "rewritten, so the history cannot be quietly tidied up.")

st.caption(
    "Read-only by design. Mode, leverage, and the kill switch are changed in "
    "engine.py, under version control, not from a browser."
)
