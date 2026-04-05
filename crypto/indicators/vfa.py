import sys
import os

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BINANCE_BASE_URL

def fetch_kline_df(symbol: str, days: int = 8) -> pd.DataFrame:

    url = f"{BINANCE_BASE_URL}/fapi/v1/klines"
    params = {
        "symbol":   symbol.upper(),
        "interval": "1d",
        "limit":    days,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    if not raw or not isinstance(raw, list):
        raise ValueError(f"Response kline tidak valid untuk {symbol}")
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "transactions", "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["open_time"]     = pd.to_numeric(df["open_time"])
    df["volume"]        = pd.to_numeric(df["volume"])
    df["transactions"]  = pd.to_numeric(df["transactions"])

    return df[["open_time", "volume", "transactions"]].sort_values("open_time").reset_index(drop=True)

def _avg_pct_change(arr: np.ndarray, n: int = 7) -> float:
    """
    Hitung rata-rata % perubahan harian dari n hari terakhir.
    Menghasilkan n-1 selisih, lalu dirata-rata.
    Return 0.0 jika data tidak cukup atau semua nilai nol.
    """
    tail = arr[-(n + 1):] 
    if len(tail) < 2:
        return 0.0

    changes = []
    for i in range(1, len(tail)):
        prev = tail[i - 1]
        if prev == 0:
            continue
        changes.append((tail[i] - prev) / prev * 100.0)

    if not changes:
        return 0.0

    return float(np.mean(changes))


def score_vfa(df: pd.DataFrame) -> int:
    if "transactions" not in df.columns:
        return 0

    if len(df) < 8:  
        return 0

    vol_arr  = df["volume"].values
    freq_arr = df["transactions"].values

    avg_vol  = _avg_pct_change(vol_arr)
    avg_freq = _avg_pct_change(freq_arr)

    if avg_vol < 0 and avg_freq < 0:
        return -3

    if avg_vol < 0:
        return 2   
    if avg_freq < 0:
        return 1   

    # Keduanya positif → logic penuh
    if avg_vol == 0 and avg_freq == 0:
        return 0

    if avg_freq == 0:
        return -1   
    if avg_vol == 0:
        return 2    

    vol_to_freq = avg_vol / avg_freq
    freq_to_vol = avg_freq / avg_vol

    if vol_to_freq >= 2.0:
        return -1
    elif avg_vol > avg_freq:
        return 1
    elif freq_to_vol >= 2.0:
        return 3
    else:
        return 2


def get_vfa_detail(df: pd.DataFrame) -> dict:
    if "transactions" not in df.columns or len(df) < 8:
        return {"avg_vol": 0.0, "avg_freq": 0.0, "score": 0}

    vol_arr  = df["volume"].values
    freq_arr = df["transactions"].values

    avg_vol  = _avg_pct_change(vol_arr)
    avg_freq = _avg_pct_change(freq_arr)
    score    = score_vfa(df)

    return {
        "avg_vol":  round(avg_vol,  2),
        "avg_freq": round(avg_freq, 2),
        "score":    score,
    }

def analyze(symbol: str, days: int = 8) -> dict:

    days = max(days, 8)   # minimal 8 agar scoring valid
    df = fetch_kline_df(symbol, days=days)
    detail = get_vfa_detail(df)

    return {
        "symbol":   symbol.upper(),
        "avg_vol":  detail["avg_vol"],
        "avg_freq": detail["avg_freq"],
        "score":    detail["score"],
        "df":       df,
    }

def analyze(symbol: str, interval: str = "1d", limit: int = 210) -> dict:  # noqa: F811
    """Fetch data Binance lalu return hasil VFA lengkap."""
    from indicators.binance_fetcher import get_df
    limit = max(limit, 8)
    df = get_df(symbol, interval, limit)
    detail = get_vfa_detail(df)
    return {
        "symbol":   symbol.upper(),
        "interval": interval,
        "avg_vol":  detail["avg_vol"],
        "avg_freq": detail["avg_freq"],
        "score":    detail["score"],
        "df":       df,
    }
