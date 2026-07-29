"""
Monitoring dashboard for the trading engine.

    streamlit run dashboard.py

DELIBERATELY READ-ONLY
----------------------
No button here starts trading, changes mode, adjusts leverage, or resets the
kill switch. engine.py requires three separate deliberate acts to go live.
A toggle in a browser would collapse all three into one click reachable by a
misclick, a stale tab, or anyone who can reach this port.

Freqtrade's own documentation warns that a trading UI should never be exposed
directly to the internet: no HTTPS by default, and every endpoint returns
sensitive information. Keep this on localhost. If you ever need remote access,
put it behind a VPN or an authenticating reverse proxy -- not a port forward.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

STATE_FILE = Path("engine_state.json")
AUDIT_LOG = Path("engine_audit.jsonl")
LOG_DIR = Path("logs")

# Backtest expectations, from robustness.py. The point of comparison.
BT = {
    "sharpe": 1.50,
    "cagr": 1.255,
    "max_dd": -0.735,
    "turnover": 79.3,
    "dsr": 0.937,
}
DD_LIMIT = 0.25
START_EQUITY = 10_000.0

st.set_page_config(page_title="quantstart", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------- loading
def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text())


def load_audit() -> pd.DataFrame:
    if not AUDIT_LOG.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in AUDIT_LOG.read_text().splitlines() if l.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts", ascending=False)


@st.cache_data(ttl=3600, show_spinner="Loading market data (cached for an hour)...")
def load_signals() -> pd.DataFrame | None:
    """
    Current momentum ranking across the whole universe, not just holdings.

    Cached aggressively: a universe load takes ~45 seconds and the signal only
    changes once a day. Use the sidebar button to force a refresh.
    """
    try:
        from data import load_universe
        from run_cross_sectional import CASUALTIES, SURVIVORS
        from engine import CFG

        end = datetime.now(timezone.utc).strftime("%Y-%m")
        surv = load_universe(SURVIVORS, "1d", "2019-01", end, include_tail=True)
        dead = load_universe(CASUALTIES, "1d", "2019-01", end,
                             include_tail=True).dropna(axis=1, how="all")
        prices = pd.concat([surv, dead], axis=1).sort_index().dropna(how="all")

        lookbacks = CFG.strategy_params["lookbacks"]
        scores = sum(prices.pct_change(lb).rank(axis=1, pct=True)
                     for lb in lookbacks) / len(lookbacks)
        latest = scores.iloc[-1].dropna().sort_values(ascending=False)

        out = pd.DataFrame({"score": latest})
        out["rank"] = range(1, len(out) + 1)
        for lb in lookbacks:
            out[f"{lb}d return"] = prices.pct_change(lb).iloc[-1].reindex(out.index)
        vol = prices.pct_change().rolling(60).std().iloc[-1] * np.sqrt(365)
        out["volatility"] = vol.reindex(out.index)
        return out
    except Exception as exc:  # noqa: BLE001
        st.session_state["signal_error"] = str(exc)
        return None


state = load_state()
if state is None:
    st.error("No engine state found. Run `python engine.py` first.")
    st.stop()

hist = pd.DataFrame(state.get("history", []))
if not hist.empty:
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.set_index("date").sort_index()

equity = state["equity"]
peak = state["peak_equity"]
drawdown = equity / peak - 1.0
sessions = len(hist)
mode = state.get("mode", "paper").upper()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("quantstart")
    if mode == "LIVE":
        st.error("**LIVE** — real money", icon="🔴")
    else:
        st.success("**PAPER** — no money at risk", icon="🟢")

    st.metric("Equity", f"{equity:,.2f}", f"{equity / START_EQUITY - 1:+.2%}")
    st.metric("Sessions", sessions)

    st.divider()
    st.caption("**Health**")

    # Did it actually run today?
    last_bar = str(state.get("last_bar", ""))[:10]
    if last_bar:
        age = (datetime.now(timezone.utc).date()
               - datetime.fromisoformat(last_bar).date()).days
        if age <= 2:
            st.success(f"Data current ({age}d old)")
        else:
            st.error(f"Data {age} days old — scheduler may be down")

    today_log = LOG_DIR / f"engine_{datetime.now(timezone.utc):%Y%m%d}.log"
    if today_log.exists():
        st.success("Ran today")
    else:
        st.warning("No run logged today")

    if state.get("killed"):
        st.error("HALTED")
    else:
        st.success("Kill switch armed")

    st.divider()
    if st.button("Clear data cache", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Read-only. Mode and leverage live in engine.py, under version "
               "control — not in a browser.")


# ---------------------------------------------------------------- header
if state.get("killed"):
    st.error(f"**ENGINE HALTED.** {state.get('kill_reason')}", icon="⛔")
    st.caption("Trading stopped permanently. Investigate before resetting "
               "state — do not delete the file to make this go away.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Equity", f"{equity:,.2f}", f"{equity / START_EQUITY - 1:+.2%}")
c2.metric("Drawdown", f"{drawdown:.2%}", f"limit {-DD_LIMIT:.0%}",
          delta_color="off")
c3.metric("Peak", f"{peak:,.2f}")
c4.metric("Sessions", sessions,
          f"{max(0, 20 - sessions)} to stats" if sessions < 20 else "sufficient",
          delta_color="off")
c5.metric("Last bar", last_bar or "—")

used = min(abs(drawdown) / DD_LIMIT, 1.0)
st.progress(used, text=f"Kill switch budget used: {used:.0%} "
                       f"({abs(drawdown):.1%} of {DD_LIMIT:.0%})")

tabs = st.tabs(["Performance", "Positions & signals", "Risk",
                "Live vs backtest", "Audit & logs"])


# ---------------------------------------------------------------- performance
with tabs[0]:
    if sessions < 2:
        st.info("Two or more sessions needed before charting. "
                "The engine runs once per day.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Equity")
            st.line_chart(hist["equity"], height=260)
        with right:
            st.subheader("Drawdown")
            dd = hist["drawdown"].to_frame("drawdown")
            dd["limit"] = -DD_LIMIT
            st.line_chart(dd, height=260, color=["#d9534f", "#999999"])

        st.subheader("Daily returns")
        st.bar_chart(hist["pnl_pct"], height=200)

        r = hist["pnl_pct"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best day", f"{r.max():+.2%}")
        m2.metric("Worst day", f"{r.min():+.2%}")
        m3.metric("Win rate", f"{(r > 0).mean():.0%}")
        m4.metric("Daily vol", f"{r.std():.2%}")

        if sessions >= 30:
            st.subheader("Monthly returns")
            monthly = (1 + r).resample("ME").prod() - 1
            st.bar_chart(monthly, height=200)


# ---------------------------------------------------------------- positions
with tabs[1]:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Current book")
        positions = state.get("positions", {})
        if not positions:
            st.write("Flat — no positions.")
        else:
            pos = (pd.Series(positions, name="weight")
                   .sort_values(ascending=False).to_frame())
            st.dataframe(pos.style.format("{:.2%}"), width="stretch")
            gross = sum(abs(v) for v in positions.values())
            st.metric("Gross exposure", f"{gross:.1%}")
            st.caption(f"Largest: {max(positions.values()):.1%}")

    with right:
        st.subheader("Universe ranking")
        st.caption("Why these five and not others. Rank is the blended "
                   "percentile across all momentum lookbacks.")
        signals = load_signals()
        if signals is None:
            st.warning(f"Could not compute signals: "
                       f"{st.session_state.get('signal_error', 'unknown')}")
        else:
            held = set(state.get("positions", {}))
            show = signals.copy()
            show.insert(0, "held", [s in held for s in show.index])
            fmt = {"score": "{:.3f}", "volatility": "{:.0%}"}
            for c in show.columns:
                if "return" in c:
                    fmt[c] = "{:+.1%}"
            st.dataframe(show.style.format(fmt), width="stretch", height=420)


# ---------------------------------------------------------------- risk
with tabs[2]:
    st.subheader("Where you sit versus the kill switch")

    a, b, c = st.columns(3)
    a.metric("Current drawdown", f"{drawdown:.2%}")
    b.metric("Room remaining", f"{DD_LIMIT - abs(drawdown):.2%}")
    c.metric("Loss to trip", f"{peak * (1 - DD_LIMIT) - equity:,.0f}")

    st.divider()
    st.subheader("Forward projection")
    st.caption("Block bootstrap from backtest returns at the configured "
               "0.25x leverage. This is the distribution of plausible next "
               "years, not a forecast.")

    proj = pd.DataFrame({
        "outcome": ["5th percentile", "median", "95th percentile"],
        "one-year return": ["-7.9%", "+27.3%", "+80.4%"],
    }).set_index("outcome")
    st.dataframe(proj, width="stretch")

    st.markdown(
        "**At 0.25x** (current setting): 11.4% chance of a losing year, "
        "0.6% chance of a 30% drawdown, ~0% chance of losing half.\n\n"
        "**At 1.0x**: 89% chance of a 30% drawdown, 32% chance of losing half. "
        "And if the edge is not real, 75% chance of losing half.\n\n"
        "That asymmetry is why leverage is set where it is. Full Kelly here "
        "would be 2.11x."
    )


# ---------------------------------------------------------------- live vs bt
with tabs[3]:
    if sessions < 20:
        st.info(
            f"**{sessions} of 20 sessions.** Live statistics are meaningless "
            "before roughly 20 observations.\n\n"
            "What matters is not whether live makes money — it is whether "
            "live *matches* simulation. Turnover and drawdown diverge before "
            "returns do, so those are the early warning."
        )
        st.progress(sessions / 20, text=f"{sessions}/20 sessions")
    else:
        r = hist["pnl_pct"]
        live_sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
        live_dd = hist["drawdown"].min()
        years = sessions / 365
        live_cagr = (equity / START_EQUITY) ** (1 / years) - 1 if years > 0 else np.nan

        comp = pd.DataFrame({
            "backtest": [BT["sharpe"], BT["cagr"], BT["max_dd"]],
            "live": [live_sharpe, live_cagr, live_dd],
        }, index=["Sharpe", "CAGR", "Max drawdown"])
        comp["difference"] = comp["live"] - comp["backtest"]
        st.dataframe(comp.style.format("{:.2f}"), width="stretch")

        gap = abs(live_sharpe - BT["sharpe"])
        if gap < 0.5:
            st.success("Tracking the backtest within normal decay.")
        elif gap < 1.0:
            st.warning("Drifting from backtest. Watch closely.")
        else:
            st.error("Live and backtest disagree materially. The simulation "
                     "is probably wrong — check fills, costs, and timing "
                     "before drawing conclusions about the strategy.")

    st.divider()
    st.caption("**Backtest baseline** (robustness.py, walk-forward, 25 assets, "
               "2019–2025)")
    st.dataframe(pd.DataFrame({
        "value": [f"{BT['sharpe']:.2f}", f"{BT['cagr']:.1%}",
                  f"{BT['max_dd']:.1%}", f"{BT['turnover']:.0f}x",
                  f"{BT['dsr']:.3f}"],
    }, index=["Sharpe", "CAGR", "Max drawdown", "Annual turnover",
              "Deflated Sharpe"]), width="stretch")
    st.caption("Deflated Sharpe of 0.937 is below the 0.95 significance bar. "
               "The edge is promising, not proven — which is the entire "
               "reason for the paper period.")


# ---------------------------------------------------------------- audit
with tabs[4]:
    st.subheader("Audit trail")
    audit = load_audit()
    if audit.empty:
        st.write("No audit records yet.")
    else:
        events = st.multiselect("Filter events",
                                sorted(audit["event"].unique()),
                                default=sorted(audit["event"].unique()))
        view = audit[audit["event"].isin(events)]
        cols = [c for c in ["ts", "event", "mode", "bar", "equity",
                            "drawdown", "pnl_pct", "reason"]
                if c in view.columns]
        st.dataframe(view[cols], width="stretch", height=300)
        st.caption(f"{len(audit)} records. Append-only — never rewritten, so "
                   "the history cannot be quietly tidied up.")

    st.divider()
    st.subheader("Run logs")
    if not LOG_DIR.exists():
        st.write("No logs yet.")
    else:
        files = sorted(LOG_DIR.glob("engine_*.log"), reverse=True)
        if not files:
            st.write("No logs yet.")
        else:
            pick = st.selectbox("Log file", [f.name for f in files])
            st.code((LOG_DIR / pick).read_text(), language="text")
