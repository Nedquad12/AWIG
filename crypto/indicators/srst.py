"""
indicators/srst.py — Support & Resistance Smart Tool (SRST)

Menggunakan detect_sr() dari sr.py (Donchian default) untuk mendeteksi
level aktif, lalu menghitung skor berdasarkan kedekatan harga ke zona.

Logika Support (ambil level support aktif yang paling dekat ke close):
  Jarak = (close - top_zona_support) / close * 100
  Jarak <= 5%   → +1
  entries > 2   → +1 lagi  (kumulatif)
  entries > 4   → +1 lagi  (kumulatif)
  Max: +3

Logika Resistance (ambil level resistance aktif yang paling dekat ke close):
  Jarak = (base_price_resis - close) / close * 100
  Jarak <= 5%   → -1
  entries > 2   → -1 lagi  (kumulatif)
  entries > 4   → -2 lagi  (kumulatif)
  Max: -4

Keduanya dihitung & dijumlahkan (bisa saling cancel).
Tidak ada level aktif → 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from indicators.sr import detect_sr, SRLevel


def _score_support(close: float, active_levels: list[SRLevel]) -> int:
    """
    Cari level support aktif yang top-nya paling dekat ke close (dari bawah).
    Hanya pertimbangkan level yang top-nya <= close (harga di atas zona).
    """
    sup_levels = [l for l in active_levels if l.is_support and l.top <= close]
    if not sup_levels:
        return 0

    # Ambil yang top-nya paling dekat ke close
    nearest = min(sup_levels, key=lambda l: close - l.top)

    dist_pct = (close - nearest.top) / close * 100
    if dist_pct > 5.0:
        return 0

    score = 1  # dekat support
    if nearest.entries > 2:
        score += 1
    if nearest.entries > 4:
        score += 1

    return score


def _score_resistance(close: float, active_levels: list[SRLevel]) -> int:
    """
    Cari level resistance aktif yang base_price-nya paling dekat ke close (dari atas).
    Hanya pertimbangkan level yang base_price-nya >= close (harga di bawah zona).
    """
    res_levels = [l for l in active_levels if not l.is_support and l.base_price >= close]
    if not res_levels:
        return 0

    # Ambil yang base_price-nya paling dekat ke close
    nearest = min(res_levels, key=lambda l: l.base_price - close)

    dist_pct = (nearest.base_price - close) / close * 100
    if dist_pct > 5.0:
        return 0

    score = -1  # dekat resistance
    if nearest.entries > 2:
        score -= 1
    if nearest.entries > 4:
        score -= 2

    return score


def score_srst(df: pd.DataFrame) -> int:
    """
    Hitung skor SRST berdasarkan kedekatan harga ke level S&R aktif.

    Args:
        df: DataFrame dengan kolom open/high/low/close/volume, diurutkan ascending

    Returns:
        Skor integer antara -4 dan +3
        0 jika tidak ada level aktif atau data tidak cukup
    """
    if len(df) < 30:
        return 0

    try:
        active_levels, _ = detect_sr(df)
    except Exception:
        return 0

    if not active_levels:
        return 0

    close = float(df["close"].iloc[-1])

    sup_score = _score_support(close, active_levels)
    res_score = _score_resistance(close, active_levels)

    return sup_score + res_score


def get_srst_detail(df: pd.DataFrame) -> dict:
    """
    Return detail SRST untuk keperluan debugging / display.

    Returns:
        dict: nearest_sup, sup_dist_pct, sup_entries,
              nearest_res, res_dist_pct, res_entries, score
    """
    empty = {
        "nearest_sup":   None,
        "sup_dist_pct":  None,
        "sup_entries":   None,
        "nearest_res":   None,
        "res_dist_pct":  None,
        "res_entries":   None,
        "score":         0,
    }

    if len(df) < 30:
        return empty

    try:
        active_levels, _ = detect_sr(df)
    except Exception:
        return empty

    if not active_levels:
        return empty

    close = float(df["close"].iloc[-1])

    # Support
    sup_levels = [l for l in active_levels if l.is_support and l.top <= close]
    nearest_sup = min(sup_levels, key=lambda l: close - l.top) if sup_levels else None

    # Resistance
    res_levels = [l for l in active_levels if not l.is_support and l.base_price >= close]
    nearest_res = min(res_levels, key=lambda l: l.base_price - close) if res_levels else None

    return {
        "nearest_sup":  round(nearest_sup.top, 4)          if nearest_sup else None,
        "sup_dist_pct": round((close - nearest_sup.top) / close * 100, 2) if nearest_sup else None,
        "sup_entries":  nearest_sup.entries                 if nearest_sup else None,
        "nearest_res":  round(nearest_res.base_price, 4)   if nearest_res else None,
        "res_dist_pct": round((nearest_res.base_price - close) / close * 100, 2) if nearest_res else None,
        "res_entries":  nearest_res.entries                 if nearest_res else None,
        "score":        score_srst(df),
    }
