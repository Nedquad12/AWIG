import logging
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BINANCE_BASE_URL

logger = logging.getLogger(__name__)

# Threshold funding rate
THRESHOLD_EXTREME = 0.002   # 0.2% — overheated (positif atau negatif)
THRESHOLD_MILD    = 0.0005  # 0.05% — zona netral / noise


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


def score_funding(df: pd.DataFrame) -> float:
    """
    Scoring kontarian berbasis mean funding 3 periode terakhir.

    Logika dasar:
      Funding positif → longs dominan → bias reversal DOWN → skor NEGATIF
      Funding negatif → shorts dominan → bias reversal UP   → skor POSITIF

    Skala skor berdasarkan mean funding 3 periode:
      mean >  0.2%        → -1.5  (sangat overheated long, reversal down kuat)
      mean  0.05% – 0.2%  → -1.0  (longs dominan, bias turun)
      mean  0   – 0.05%   →  0.0  (zona netral, noise)
      mean -0.05% – 0     →  0.0  (zona netral, noise)
      mean -0.2% – -0.05% → +1.0  (shorts dominan, bias naik)
      mean < -0.2%        → +1.5  (sangat overheated short, reversal up kuat)

    Bonus ±0.5 jika kondisi konsisten (ketiga periode searah dan bukan noise):
      3 periode semua positif di atas noise → tambah -0.5 (makin bearish)
      3 periode semua negatif di atas noise → tambah +0.5 (makin bullish)
    """
    if df.empty or "fundingRate" not in df.columns:
        return 0.0

    last3 = df["fundingRate"].values[-3:] if len(df) >= 3 else df["fundingRate"].values
    mean3 = float(np.mean(last3))

    # Base score — contrarian
    if mean3 > THRESHOLD_EXTREME:
        score = -1.5
    elif mean3 > THRESHOLD_MILD:
        score = -1.0
    elif mean3 >= -THRESHOLD_MILD:
        score = 0.0   # zona netral / noise
    elif mean3 >= -THRESHOLD_EXTREME:
        score = 1.0
    else:
        score = 1.5

    # Bonus konsistensi — hanya di luar zona noise
    if len(last3) == 3:
        all_positive = all(x > THRESHOLD_MILD for x in last3)
        all_negative = all(x < -THRESHOLD_MILD for x in last3)
        if all_positive:
            score -= 0.5   # konsisten longs → makin bearish
        elif all_negative:
            score += 0.5   # konsisten shorts → makin bullish

    return float(np.clip(score, -2.0, 2.0))


def get_funding_detail(df: pd.DataFrame) -> dict:
    """Return detail funding rate untuk konteks AI."""
    if df.empty:
        return {"latest": 0.0, "mean_7d": 0.0, "score": 0.0}

    latest  = float(df["fundingRate"].iloc[-1])
    tail_21 = df["fundingRate"].tail(21)
    mean_7d = float(tail_21.mean()) if len(tail_21) > 0 else 0.0

    return {
        "latest":  round(latest * 100, 6),   # dalam %
        "mean_7d": round(mean_7d * 100, 6),
        "score":   score_funding(df),
    }


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
