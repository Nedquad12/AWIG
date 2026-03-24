"""
indicators/macd.py — Moving Average Convergence Divergence (12, 26, 9)

Logika per kondisi:
  1. MACD > Signal  →  +1  |  MACD < Signal  →  -1
  2. MACD > 0       →  +1  |  MACD < 0       →  -1
"""

import pandas as pd


def _compute_macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float]:
    """
    Hitung nilai MACD line dan Signal line terakhir.

    Returns:
        (macd_value, signal_value)
    """
    ema_fast    = closes.ewm(span=fast,   adjust=False).mean()
    ema_slow    = closes.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def score_macd(df: pd.DataFrame) -> int:
    """
    Hitung skor MACD dari data close harian.

    Args:
        df: DataFrame dengan kolom 'close', diurutkan ascending

    Returns:
        Skor integer antara -2 dan +2
    """
    if len(df) < 35:   # butuh minimal 26 + 9 hari
        return 0

    macd_val, signal_val = _compute_macd(df["close"])

    score = 0

    # Kondisi 1: posisi MACD terhadap Signal
    if macd_val > signal_val:
        score += 1
    elif macd_val < signal_val:
        score -= 1

    # Kondisi 2: nilai MACD positif/negatif
    if macd_val > 0:
        score += 1
    elif macd_val < 0:
        score -= 1

    return score
