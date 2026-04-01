import sys
import os
from typing import Literal

import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BINANCE_BASE_URL
VALID_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}

MIN_ROWS_DEFAULT = 210


def fetch_klines(
    symbol: str,
    interval: str = "1d",
    limit: int = MIN_ROWS_DEFAULT,
) -> pd.DataFrame:
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Interval '{interval}' tidak valid. "
            f"Pilihan: {sorted(VALID_INTERVALS)}"
        )

    limit = min(max(limit, 1), 1500)  # clamp 1–1500

    url = f"{BINANCE_BASE_URL}/fapi/v1/klines"
    params = {
        "symbol":   symbol.upper(),
        "interval": interval,
        "limit":    limit,
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    if not raw or not isinstance(raw, list):
        raise ValueError(f"Response kline kosong untuk {symbol} {interval}")

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "transactions", "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    for col in ["open_time", "open", "high", "low", "close", "volume", "transactions"]:
        df[col] = pd.to_numeric(df[col])

    df = df[["open_time", "open", "high", "low", "close", "volume", "transactions"]]
    return df.sort_values("open_time").reset_index(drop=True)


def get_df(
    symbol: str,
    interval: str = "1d",
    limit: int = MIN_ROWS_DEFAULT,
) -> pd.DataFrame:
    return fetch_klines(symbol, interval, limit)
