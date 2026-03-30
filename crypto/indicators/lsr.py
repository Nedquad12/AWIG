"""
indicators/lsr.py — Long/Short Ratio Scoring (Rule-Based)

Pakai Global Long/Short Account Ratio (semua trader, bukan hanya top).
Endpoint: GET /futures/data/globalLongShortAccountRatio

Logika — contrarian:
  Ketika mayoritas long → pasar bisa reversal turun (terlalu crowded)
  Ketika mayoritas short → pasar bisa reversal naik (short squeeze)

  longShortRatio > 2.0   →  -2  (sangat long-heavy → waspada)
  longShortRatio > 1.3   →  -1  (long dominan)
  longShortRatio 0.77–1.3 →  0  (seimbang)
  longShortRatio < 0.77  →  +1  (short dominan → potensi squeeze)
  longShortRatio < 0.5   →  +2  (sangat short-heavy → squeeze kuat)

  Bonus momentum:
    Jika ratio naik 3 data terakhir dan sudah > 1.3 → tambah -1
    Jika ratio turun 3 data terakhir dan sudah < 0.77 → tambah +1

  Final score di-clamp ke [-3, +3]
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BINANCE_BASE_URL, DEFAULT_INTERVAL

logger = logging.getLogger(__name__)

# Binance endpoint untuk global L/S ratio
_LSR_ENDPOINT = "/futures/data/globalLongShortAccountRatio"

THRESHOLD_VERY_LONG  = 2.0
THRESHOLD_LONG       = 1.3
THRESHOLD_SHORT      = 0.77
THRESHOLD_VERY_SHORT = 0.50


# ------------------------------------------------------------------
# Fetch L/S ratio history
# ------------------------------------------------------------------

def fetch_lsr(symbol: str, interval: str = "30m", limit: int = 96) -> pd.DataFrame:
    """
    Ambil riwayat global long/short account ratio.
    Binance simpan max 30 hari.

    interval: '5m','15m','30m','1h','2h','4h','6h','12h','1d'
    limit   : max 500 per request (Binance limit)

    Returns:
        DataFrame dengan kolom: timestamp (int ms), longShortRatio,
        longAccount (float %), shortAccount (float %)
        Diurutkan ascending.
    """
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


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def score_lsr(df: pd.DataFrame) -> int:
    """
    Hitung skor L/S ratio dari DataFrame.

    Args:
        df: DataFrame dengan kolom 'longShortRatio', diurutkan ascending.

    Returns:
        Skor integer antara -3 dan +3
    """
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

    # Bonus momentum dari 3 data terakhir
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


# ------------------------------------------------------------------
# Binance entry point
# ------------------------------------------------------------------

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
