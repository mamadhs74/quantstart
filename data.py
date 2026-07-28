"""
Bulk loader for Binance public market data.

Uses data.binance.vision, which serves ZIP archives over plain HTTPS.
No account, no API key, no KYC. This is public reference data.

    from data import load_klines, load_universe
    df = load_klines("BTCUSDT", "1d", "2019-01", "2025-12")

MONTHLY vs DAILY ARCHIVES
-------------------------
Binance publishes two archive sets. Monthly files are compact and ideal for
history, but a month only appears AFTER it closes -- so on 27 July the newest
monthly file is June. A backtest does not care. A live system does: it would
log the same stale signal every day for a month.

So this loader pulls monthly archives for history and then fills the tail from
daily archives. Daily files also publish with a lag of a day or two, which is
fine -- the backtest engine trades on the next bar anyway.
"""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_MONTHLY = "https://data.binance.vision/data/spot/monthly/klines"
BASE_DAILY = "https://data.binance.vision/data/spot/daily/klines"
CACHE = Path.home() / ".cache" / "binance_klines"

# Column layout of Binance kline CSVs (they ship without a header row).
_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

# How many trailing days to try from daily archives. 45 comfortably covers the
# gap left by an unpublished current month plus the publication lag.
TAIL_DAYS = 45

# Concurrent downloads. These are I/O-bound, so threads are the right tool.
# Keep it modest -- this is a free public endpoint, not something to hammer.
MAX_WORKERS = 12


def _parse_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            df = pd.read_csv(fh, header=None, names=_COLS)

    # Binance switched open_time from ms to us epoch partway through 2025.
    # Detect by magnitude rather than by date, so this keeps working.
    unit = "us" if df["open_time"].iloc[0] > 1e15 else "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)

    df = df[["open_time", "open", "high", "low", "close", "volume", "trades"]]
    return df.set_index("open_time").astype(float)


def _fetch(url: str, cached: Path, allow_negative: bool = False) -> pd.DataFrame | None:
    """
    Download once, then read from disk forever. Published files never change.

    allow_negative: also remember that a file does NOT exist, so we stop
    re-requesting it. Only safe for days old enough that publication is
    certain -- a recent 404 may just mean 'not published yet'.
    """
    if cached.exists():
        return pd.read_parquet(cached)

    missing = cached.with_suffix(".missing")
    if allow_negative and missing.exists():
        return None

    try:
        resp = requests.get(url, timeout=60)
    except requests.RequestException:
        return None

    if resp.status_code == 404:
        if allow_negative:
            missing.parent.mkdir(parents=True, exist_ok=True)
            missing.touch()
        return None

    resp.raise_for_status()
    df = _parse_zip(resp.content)
    cached.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached)
    return df


def _fetch_month(symbol: str, interval: str, month: str) -> pd.DataFrame | None:
    return _fetch(
        f"{BASE_MONTHLY}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip",
        CACHE / symbol / interval / f"{month}.parquet",
    )


def _fetch_day(symbol: str, interval: str, day: str,
               allow_negative: bool = False) -> pd.DataFrame | None:
    return _fetch(
        f"{BASE_DAILY}/{symbol}/{interval}/{symbol}-{interval}-{day}.zip",
        CACHE / symbol / interval / "daily" / f"{day}.parquet",
        allow_negative=allow_negative,
    )


def load_klines(
    symbol: str,
    interval: str = "1d",
    start: str = "2019-01",
    end: str = "2025-12",
    include_tail: bool = False,
) -> pd.DataFrame:
    """
    Return a UTC-indexed OHLCV frame.

    Index is bar OPEN time. That matters: a bar stamped 2024-03-01 contains
    information that is only complete at the END of that day. Every lookahead
    bug in retail backtesting starts by forgetting this.

    include_tail=True also pulls the last ~45 days from daily archives, so the
    frame reaches up to yesterday instead of the last closed month. Use it for
    live signals; leave it off for reproducible backtests.
    """
    months = [str(m) for m in pd.period_range(start=start, end=end, freq="M")]
    frames = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_fetch_month, symbol, interval, m) for m in months]
        for fut in as_completed(futures):
            part = fut.result()
            if part is not None:
                frames.append(part)

    if include_tail:
        today = date.today()
        # A day more than 3 days old that is still missing will never appear --
        # the symbol was delisted or was not trading then. Cache that fact.
        # Recent misses stay uncached so they get retried tomorrow.
        jobs = [(str(today - timedelta(days=i)), i > 3)
                for i in range(TAIL_DAYS, 0, -1)]

        # These are I/O-bound HTTPS round trips. Done sequentially, 45 days
        # across 25 symbols takes ~10 minutes. In parallel it takes seconds.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_fetch_day, symbol, interval, day, neg)
                       for day, neg in jobs]
            for fut in as_completed(futures):
                part = fut.result()
                if part is not None:
                    frames.append(part)

    if not frames:
        raise RuntimeError(f"No data returned for {symbol} {interval}")

    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def load_universe(
    symbols: list[str],
    interval: str = "1d",
    start: str = "2019-01",
    end: str = "2025-12",
    include_tail: bool = False,
) -> pd.DataFrame:
    """Wide frame of close prices, one column per symbol, aligned on time."""
    closes = {}
    for sym in symbols:
        try:
            closes[sym] = load_klines(sym, interval, start, end,
                                      include_tail=include_tail)["close"]
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {sym}: {exc}")
    return pd.DataFrame(closes).sort_index()


if __name__ == "__main__":
    hist = load_klines("BTCUSDT", "1d", "2024-01", "2024-03")
    print(f"monthly only: {len(hist)} bars, ends {hist.index[-1].date()}")

    live = load_klines("BTCUSDT", "1d", "2024-01",
                       date.today().strftime("%Y-%m"), include_tail=True)
    print(f"with tail:    {len(live)} bars, ends {live.index[-1].date()}")
