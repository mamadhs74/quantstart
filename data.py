"""
Bulk loader for Binance public market data.

Uses data.binance.vision, which serves monthly ZIP archives over plain HTTPS.
No account, no API key, no KYC. This is public reference data.

    from data import load_klines
    df = load_klines("BTCUSDT", "1d", "2019-01", "2025-12")
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/spot/monthly/klines"
CACHE = Path.home() / ".cache" / "binance_klines"

# Column layout of Binance kline CSVs (they ship without a header row).
_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings."""
    months = pd.period_range(start=start, end=end, freq="M")
    return [str(m) for m in months]


def _fetch_month(symbol: str, interval: str, month: str) -> pd.DataFrame | None:
    """Download one month, with an on-disk cache so you only pay for it once."""
    cache_dir = CACHE / symbol / interval
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{month}.parquet"

    if cached.exists():
        return pd.read_parquet(cached)

    url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None  # month not published (before listing, or not yet closed)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=_COLS)

    # Binance switched open_time from ms to us epoch partway through 2025.
    # Detect by magnitude rather than by date, so this keeps working.
    unit = "us" if df["open_time"].iloc[0] > 1e15 else "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)

    df = df[["open_time", "open", "high", "low", "close", "volume", "trades"]]
    df = df.set_index("open_time").astype(float)
    df.to_parquet(cached)
    return df


def load_klines(
    symbol: str,
    interval: str = "1d",
    start: str = "2019-01",
    end: str = "2025-12",
) -> pd.DataFrame:
    """
    Return a UTC-indexed OHLCV frame.

    Index is bar OPEN time. That matters: a bar stamped 2024-03-01 contains
    information that is only complete at the END of that day. Every lookahead
    bug in retail backtesting starts by forgetting this.
    """
    frames = []
    for month in _month_range(start, end):
        part = _fetch_month(symbol, interval, month)
        if part is not None:
            frames.append(part)

    if not frames:
        raise RuntimeError(f"No data returned for {symbol} {interval}")

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_universe(
    symbols: list[str],
    interval: str = "1d",
    start: str = "2019-01",
    end: str = "2025-12",
) -> pd.DataFrame:
    """Wide frame of close prices, one column per symbol, aligned on time."""
    closes = {}
    for sym in symbols:
        try:
            closes[sym] = load_klines(sym, interval, start, end)["close"]
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {sym}: {exc}")
    return pd.DataFrame(closes).sort_index()


if __name__ == "__main__":
    df = load_klines("BTCUSDT", "1d", "2024-01", "2024-03")
    print(df.head())
    print(f"\n{len(df)} bars, {df.index[0]} to {df.index[-1]}")
