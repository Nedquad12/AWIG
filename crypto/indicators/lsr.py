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

THRESHOLD_VERY_LONG  = 2.0
THRESHOLD_LONG       = 1.3
THRESHOLD_SHORT      = 0.77
THRESHOLD_VERY_SHORT = 0.50


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

def score_lsr(df: pd.DataFrame) -> int:

    if df.empty or "longShortRatio" not in df.columns:
        return 0

    latest = float(df["longShortRatio"].iloc[-1])

    # Base score (contrarian)
    if latest > THRESHOLD_VERY_LONG:
        score = -2
    elif latest > THRESHOLD_LONG:
        score = -1
    elif latest < THRESHOLD_VERY_SHORT:
        score = 2
    elif latest < THRESHOLD_SHORT:
        score = 1
    else:
        score = 0

    if len(df) >= 3:
        last3 = df["longShortRatio"].values[-3:]
        trending_up   = all(last3[i] < last3[i + 1] for i in range(2))
        trending_down = all(last3[i] > last3[i + 1] for i in range(2))

        if trending_up and latest > THRESHOLD_LONG:
            score -= 1   # makin long → makin bearish signal
        elif trending_down and latest < THRESHOLD_SHORT:
            score += 1   # makin short → makin bullish signal

    return int(np.clip(score, -3, 3))


def get_lsr_detail(df: pd.DataFrame) -> dict:
    """Return detail L/S ratio untuk konteks AI."""
    if df.empty:
        return {"latest_ratio": 0.0, "long_pct": 0.0, "short_pct": 0.0, "score": 0}

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
