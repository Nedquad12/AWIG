"""
indicators/ip.py — Indikator Poin (IP)

IP = rata-rata dari:
  - IP Daily   (MACD score + Stochastic score pada data harian)
  - IP Weekly  (MACD score + Stochastic score pada data 5-harian)
  - IP Monthly (MACD score + Stochastic score pada data 25-harian)

Konversi IP → skor poin:
  IP >= 4   →  +2.0
  IP >= 3   →  +1.5
  IP >= 2   →  +1.0
  IP  0–1   →   0.0
  (sebaliknya untuk negatif)
"""

import numpy as np
import pandas as pd


# ── Internal helpers ─────────────────────────────────────────────────────────

def _ema(series: list[float], span: int) -> list[float]:
    """EMA sederhana tanpa pandas."""
    k   = 2 / (span + 1)
    out = [series[0]]
    for v in series[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _macd_score(closes: list[float]) -> int:
    """Hitung skor MACD (±2) dari list close."""
    if len(closes) < 35:
        return 0

    ema12  = _ema(closes, 12)
    ema26  = _ema(closes, 26)
    macd   = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema(macd, 9)

    m = macd[-1]
    s = signal[-1]

    score  = 0
    score += 1 if m > s  else (-1 if m < s  else 0)
    score += 1 if m > 0  else (-1 if m < 0  else 0)
    return score


def _stoch_score(highs: list[float], lows: list[float], closes: list[float],
                 k_period: int = 14, d_period: int = 3) -> int:
    """Hitung skor Stochastic (±3) dari list OHLC."""
    if len(closes) < k_period + d_period:
        return 0

    k_vals = []
    for i in range(len(closes) - k_period + 1):
        lo = min(lows[i:i + k_period])
        hi = max(highs[i:i + k_period])
        if hi != lo:
            k_vals.append((closes[i + k_period - 1] - lo) / (hi - lo) * 100)
        else:
            k_vals.append(50.0)

    if len(k_vals) < d_period:
        return 0

    k = k_vals[-1]
    d = float(np.mean(k_vals[-d_period:]))

    score  = 0
    if k < 25:
        score += 1
    elif k > 85:
        score -= 1
    score += 1 if k > d else (-1 if k < d else 0)
    return score


def _aggregate(closes: list, highs: list, lows: list, period: int):
    """Resampling daily → period-bar (ambil close terakhir tiap window)."""
    wc, wh, wl = [], [], []
    for i in range(0, len(closes), period):
        c = closes[i:i + period]
        h = highs[i:i + period]
        l = lows[i:i + period]
        if c:
            wc.append(c[-1])
            wh.append(max(h))
            wl.append(min(l))
    return wc, wh, wl


# ── Public functions ──────────────────────────────────────────────────────────

def calculate_ip(df: pd.DataFrame) -> float:
    """
    Hitung nilai IP (Indikator Poin) lintas timeframe.

    Args:
        df: DataFrame dengan kolom close/high/low, diurutkan ascending

    Returns:
        Nilai IP float (rata-rata IPd + IPw + IPm)
    """
    closes = df["close"].tolist()
    highs  = df["high"].tolist()
    lows   = df["low"].tolist()

    # Daily
    ipd = _macd_score(closes) + _stoch_score(highs, lows, closes)

    # Weekly (5 hari)
    wc, wh, wl = _aggregate(closes, highs, lows, 5)
    ipw = _macd_score(wc) + _stoch_score(wh, wl, wc)

    # Monthly (25 hari)
    mc, mh, ml = _aggregate(closes, highs, lows, 25)
    ipm = _macd_score(mc) + _stoch_score(mh, ml, mc)

    return (ipd + ipw + ipm) / 3.0


def score_ip(ip_value: float) -> float:
    """
    Konversi nilai IP menjadi poin skor.

    Args:
        ip_value: hasil dari calculate_ip()

    Returns:
        Skor float antara -2.0 dan +2.0
    """
    if ip_value >= 3:
        return 4.0
    elif ip_value >= 1.3:
        return 2.0
    elif ip_value >= 0.3:
        return 1.0
    elif ip_value >= 0:
        return 0.0
    elif ip_value <= -3:
        return -4.0
    elif ip_value <= -1.3:
        return -2.0
    elif ip_value <= -0.3:
        return -1.0
    else:
        return 0.0
