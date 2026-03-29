"""
indicators/binance_fetcher.py — Centralized Binance Kline Fetcher

Satu-satunya modul yang konek ke Binance API untuk kebutuhan indikator.
Semua indikator import dari sini, bukan langsung ke requests.

Timeframe yang didukung (Binance interval):
  Menit  : 1m, 3m, 5m, 15m, 30m
  Jam    : 1h, 2h, 4h, 6h, 8h, 12h
  Harian : 1d, 3d
  Mingguan: 1w
  Bulanan : 1M

Kolom yang dikembalikan:
  open_time, open, high, low, close, volume, transactions
"""

import sys
import os
from typing import Literal

import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BINANCE_BASE_URL

# Semua interval valid Binance Futures
VALID_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}

# Minimum rows yang dibutuhkan indikator paling berat (MA 200)
MIN_ROWS_DEFAULT = 210


def fetch_klines(
    symbol: str,
    interval: str = "1d",
    limit: int = MIN_ROWS_DEFAULT,
) -> pd.DataFrame:
    """
    Ambil kline dari Binance USDⓈ-M Futures dan return sebagai DataFrame.

    Args:
        symbol   : contoh 'BTCUSDT', 'ETHUSDT'
        interval : timeframe Binance — '1m', '15m', '1h', '4h', '1d', dst
        limit    : jumlah candle yang diambil (max 1500 per Binance)

    Returns:
        DataFrame dengan kolom:
            open_time (int ms), open, high, low, close,
            volume, transactions (float, sudah di-cast)
        Diurutkan ascending (candle lama → baru).

    Raises:
        ValueError       : interval tidak valid atau response kosong
        requests.HTTPError: HTTP error dari Binance
    """
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

    # Binance kline index:
    # 0=open_time, 1=open, 2=high, 3=low, 4=close,
    # 5=volume, 6=close_time, 7=quote_vol,
    # 8=transactions, 9=taker_buy_base, 10=taker_buy_quote, 11=ignore
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "transactions", "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    # Cast ke numerik
    for col in ["open_time", "open", "high", "low", "close", "volume", "transactions"]:
        df[col] = pd.to_numeric(df[col])

    df = df[["open_time", "open", "high", "low", "close", "volume", "transactions"]]
    return df.sort_values("open_time").reset_index(drop=True)


def get_df(
    symbol: str,
    interval: str = "1d",
    limit: int = MIN_ROWS_DEFAULT,
) -> pd.DataFrame:
    """
    Alias publik dari fetch_klines — ini yang dipakai oleh semua indikator.

    Contoh:
        from indicators.binance_fetcher import get_df
        df = get_df("BTCUSDT", "4h", limit=300)
    """
    return fetch_klines(symbol, interval, limit)
