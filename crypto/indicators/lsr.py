import logging
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BINANCE_BASE_URL, DEFAULT_INTERVAL

logger = logging.getLogger(__name__)

_LSR_ENDPOINT = "/futures/data/globalLongShortAccountRatio"

# Tabel scoring simetris di sekitar zona balance (1.2–0.85 = 0)
# Makin ekstrem ke kedua arah → penalti
# Zona sehat (tidak terlalu berat ke satu arah) → reward


def fetch_lsr(symbol: str, interval: str = "30m", limit: int = 96) -> pd.DataFrame:

    url    = f"{BINANCE_BASE_URL}{_LSR_ENDPOINT}"
    params = {
        "symbol":   symbol.upper(),
        "period":   interval,
        "limit":    min(limit, 500),
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        return pd.DataFrame(columns=["timestamp", "longShortRatio", "longAccount", "shortAccount"])

    df = pd.DataFrame(raw)
    df["timestamp"]      = pd.to_numeric(df["timestamp"])
    df["longShortRatio"] = pd.to_numeric(df["longShortRatio"])
    df["longAccount"]    = pd.to_numeric(df["longAccount"])
    df["shortAccount"]   = pd.to_numeric(df["shortAccount"])
    return (
        df[["timestamp", "longShortRatio", "longAccount", "shortAccount"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def score_lsr(df: pd.DataFrame) -> float:
    """
    Scoring simetris berdasarkan Long/Short Ratio.

    Tabel:
      > 3.5          → -1.0  (ekstrem banyak long, market overheated)
      3.5 – 2.5      →  0.5  (agak banyak long, mulai waspada)
      2.5 – 1.2      →  1.0  (long sedikit dominan, sehat)
      1.2 – 0.85     →  0.0  (balance — netral)
      0.85 – 0.6     →  1.0  (short sedikit dominan, sehat)
      0.6 – 0.45     →  0.5  (agak banyak short, mulai waspada)
      < 0.45         → -1.0  (ekstrem banyak short, market overheated)

    Tidak ada bonus tambahan.
    """
    if df.empty or "longShortRatio" not in df.columns:
        return 0.0

    latest = float(df["longShortRatio"].iloc[-1])

    if latest > 3.5:
        score = -1.0
    elif latest > 2.5:
        score = 0.5
    elif latest > 1.2:
        score = 1.0
    elif latest >= 0.85:
        score = 0.0
    elif latest >= 0.6:
        score = 1.0
    elif latest >= 0.45:
        score = 0.5
    else:
        score = -1.0

    return float(np.clip(score, -2.0, 2.0))


def get_lsr_detail(df: pd.DataFrame) -> dict:
    """Return detail L/S ratio untuk konteks AI."""
    if df.empty:
        return {"latest_ratio": 0.0, "long_pct": 0.0, "short_pct": 0.0, "score": 0.0}

    latest = df.iloc[-1]
    return {
        "latest_ratio": round(float(latest["longShortRatio"]), 4),
        "long_pct":     round(float(latest["longAccount"]) * 100, 2),
        "short_pct":    round(float(latest["shortAccount"]) * 100, 2),
        "score":        score_lsr(df),
    }


def analyze(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = 96) -> dict:
    """Fetch L/S ratio lalu return skor + detail."""
    df     = fetch_lsr(symbol, interval=interval, limit=limit)
    detail = get_lsr_detail(df)
    return {
        "symbol":   symbol.upper(),
        "interval": interval,
        "score":    detail["score"],
        "detail":   detail,
        "df":       df,
    }
