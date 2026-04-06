import logging
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BINANCE_BASE_URL

logger = logging.getLogger(__name__)

# Threshold baru — simetris, tidak bias ke satu arah
THRESHOLD_EXTREME = 0.002   # 0.2% — terlalu ekstrem (positif atau negatif)
THRESHOLD_STABLE  = 0.002   # batas "stabil" untuk bonus


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
    Scoring simetris — tidak bias ke SHORT maupun LONG.

    Logic:
      - |funding| > 0.2%  → -1.0  (market overheated ke satu arah)
      - |funding| <= 0.2% → +1.0  (funding sehat / seimbang)

    Bonus +0.5 jika funding stabil di zona positif (0 < f <= 0.2% selama 3 periode)
    atau stabil di zona negatif (-0.2% <= f < 0 selama 3 periode),
    dan tidak pernah jebol batas THRESHOLD_EXTREME di 3 periode tsb.
    """
    if df.empty or "fundingRate" not in df.columns:
        return 0.0

    latest = float(df["fundingRate"].iloc[-1])
    score: float

    if abs(latest) > THRESHOLD_EXTREME:
        score = -1.0
    else:
        score = 1.0

    # Bonus stabilitas: funding konsisten di satu zona kecil tanpa pernah jebol
    if len(df) >= 3:
        last3 = df["fundingRate"].values[-3:]
        none_jebol = all(abs(x) <= THRESHOLD_EXTREME for x in last3)

        if none_jebol:
            all_positive = all(x > 0 for x in last3)
            all_negative = all(x < 0 for x in last3)
            if all_positive or all_negative:
                score += 0.5

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
