"""
indicators/funding.py — Funding Rate Scoring (Rule-Based)

Logika:
  Funding rate adalah biaya yang dibayar long ke short (positif) atau
  sebaliknya (negatif). Nilainya mencerminkan sentimen pasar.

  Nilai funding rate Binance biasanya dalam range ±0.01% hingga ±0.1%.
  Threshold (per 8 jam):

  rate > +0.05%   →  -2  (pasar terlalu long → potensi reversal turun)
  rate > +0.01%   →  -1  (long dominan, waspada)
  rate ~  0       →   0  (netral)
  rate < -0.01%   →  +1  (short dominan → potensi reversal naik)
  rate < -0.05%   →  +2  (pasar terlalu short → potensi squeeze naik)

  Bonus trend:
    Jika 3 data terakhir semua positif naik → tambah -1 (makin bearish signal)
    Jika 3 data terakhir semua negatif turun → tambah +1 (makin bullish signal)

  Final score di-clamp ke [-3, +3]
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BINANCE_BASE_URL

logger = logging.getLogger(__name__)

# Threshold funding rate (dalam desimal, bukan %)
THRESHOLD_STRONG  = 0.0005   # 0.05%
THRESHOLD_WEAK    = 0.0001   # 0.01%


# ------------------------------------------------------------------
# Fetch funding rate history
# ------------------------------------------------------------------

def fetch_funding_rate(symbol: str, limit: int = 90) -> pd.DataFrame:
    """
    Ambil riwayat funding rate dari Binance.
    Max 1000 per request, Binance simpan ~30 hari (tiap 8 jam = ~90 data).

    Returns:
        DataFrame dengan kolom: fundingTime (int ms), fundingRate (float)
        Diurutkan ascending.
    """
    url    = f"{BINANCE_BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol.upper(), "limit": min(limit, 1000)}
    resp   = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    df = pd.DataFrame(raw)
    df["fundingTime"] = pd.to_numeric(df["fundingTime"])
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    return df[["fundingTime", "fundingRate"]].sort_values("fundingTime").reset_index(drop=True)


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def score_funding(df: pd.DataFrame) -> int:
    """
    Hitung skor funding rate dari DataFrame.

    Args:
        df: DataFrame dengan kolom 'fundingRate', diurutkan ascending.

    Returns:
        Skor integer antara -3 dan +3
    """
    if df.empty or "fundingRate" not in df.columns:
        return 0

    latest = float(df["fundingRate"].iloc[-1])

    # Base score dari nilai terbaru
    if latest > THRESHOLD_STRONG:
        score = -2
    elif latest > THRESHOLD_WEAK:
        score = -1
    elif latest < -THRESHOLD_STRONG:
        score = 2
    elif latest < -THRESHOLD_WEAK:
        score = 1
    else:
        score = 0

    # Bonus trend dari 3 data terakhir
    if len(df) >= 3:
        last3 = df["fundingRate"].values[-3:]
        if all(last3[i] < last3[i + 1] for i in range(2)) and last3[-1] > 0:
            # Naik terus dan positif → makin bearish
            score -= 1
        elif all(last3[i] > last3[i + 1] for i in range(2)) and last3[-1] < 0:
            # Turun terus dan negatif → makin bullish
            score += 1

    return int(np.clip(score, -3, 3))


def get_funding_detail(df: pd.DataFrame) -> dict:
    """Return detail funding rate untuk konteks AI."""
    if df.empty:
        return {"latest": 0.0, "mean_7d": 0.0, "score": 0}

    latest  = float(df["fundingRate"].iloc[-1])
    # 7 hari = 21 data (3 per hari)
    tail_21 = df["fundingRate"].tail(21)
    mean_7d = float(tail_21.mean()) if len(tail_21) > 0 else 0.0

    return {
        "latest":  round(latest * 100, 6),   # dalam %
        "mean_7d": round(mean_7d * 100, 6),
        "score":   score_funding(df),
    }


# ------------------------------------------------------------------
# Binance entry point
# ------------------------------------------------------------------

def analyze(symbol: str, limit: int = 90) -> dict:
    """Fetch funding rate lalu return skor + detail."""
    df     = fetch_funding_rate(symbol, limit=limit)
    detail = get_funding_detail(df)
    return {
        "symbol": symbol.upper(),
        "score":  detail["score"],
        "detail": detail,
        "df":     df,
    }
